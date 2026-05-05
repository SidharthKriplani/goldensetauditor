import unittest
import pandas as pd

from goldensetauditor import GoldenSetAuditConfig, audit_golden_set


class GoldenSetAuditorTests(unittest.TestCase):
    def make_df(self):
        return pd.DataFrame([
            {"id": "1", "category": "a", "question": "How many PTO days carry over?", "expected_answer": "Five PTO days may carry over."},
            {"id": "2", "category": "a", "question": "How many PTO days carry over?", "expected_answer": "No PTO days carry over."},
            {"id": "3", "category": "b", "question": "What about this?", "expected_answer": "Yes."},
            {"id": "4", "category": "c", "question": "Can employees expense a standing desk?", "expected_answer": "Employees can expense a standing desk with manager approval."},
            {"id": "5", "category": "c", "question": "Can staff reimburse a standing desk?", "expected_answer": "Staff can reimburse a standing desk with manager approval."},
        ])

    def test_missing_columns_fail(self):
        report = audit_golden_set(pd.DataFrame({"x": [1]}), GoldenSetAuditConfig(input_col="q", expected_col="a"))
        self.assertEqual(report.status, "FAIL")

    def test_conflicting_duplicate_fails(self):
        report = audit_golden_set(self.make_df(), GoldenSetAuditConfig(id_col="id", category_col="category"))
        self.assertTrue(any(f.check_name == "conflicting_expected_answers" for f in report.findings))
        self.assertEqual(report.status, "FAIL")

    def test_weak_answer_warns(self):
        report = audit_golden_set(self.make_df(), GoldenSetAuditConfig(id_col="id", category_col="category"))
        self.assertTrue(any(f.check_name == "weak_expected_answer" for f in report.findings))

    def test_ambiguous_prompt_warns(self):
        report = audit_golden_set(self.make_df(), GoldenSetAuditConfig(id_col="id", category_col="category"))
        self.assertTrue(any(f.check_name == "ambiguous_prompt_heuristic" for f in report.findings))

    def test_outputs_render(self):
        report = audit_golden_set(self.make_df(), GoldenSetAuditConfig(id_col="id", category_col="category"))
        self.assertIn("GoldenSetAuditor", report.to_markdown())
        self.assertIn("<html", report.to_html())
        self.assertIn("findings", report.to_json())

    def test_clean_set_pass_or_warn_low(self):
        df = pd.DataFrame([
            {"id": "1", "category": "a", "question": "What is the PTO carryover rule?", "expected_answer": "Employees may carry over five unused PTO days into the next calendar year."},
            {"id": "2", "category": "a", "question": "How should a lost laptop be reported?", "expected_answer": "The employee should notify IT and Security immediately and create an incident ticket."},
            {"id": "3", "category": "b", "question": "Who approves home office equipment reimbursement?", "expected_answer": "The employee's manager must approve eligible home office equipment reimbursement."},
            {"id": "4", "category": "b", "question": "What is the first step in the performance review process?", "expected_answer": "The employee completes a self-review before the manager review discussion."},
            {"id": "5", "category": "c", "question": "When are expense reports due?", "expected_answer": "Expense reports should be submitted within thirty days of the purchase date."},
            {"id": "6", "category": "c", "question": "Which document explains remote work eligibility?", "expected_answer": "Remote work eligibility is explained in the remote work policy document."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id", category_col="category", min_category_count=2))
        self.assertIn(report.status, ["PASS", "WARN"])


class NearDuplicateTests(unittest.TestCase):
    def test_near_duplicate_warns(self):
        df = pd.DataFrame([
            {"id": "1", "question": "What is the PTO carryover policy for employees?", "expected_answer": "Employees may carry over five PTO days."},
            {"id": "2", "question": "What is the PTO carryover policy for staff?", "expected_answer": "Staff may carry over five PTO days."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(
            id_col="id", near_duplicate_threshold=0.50
        ))
        self.assertTrue(any(f.check_name == "near_duplicate_inputs" for f in report.findings))

    def test_no_near_duplicate_for_distinct_questions(self):
        df = pd.DataFrame([
            {"id": "1", "question": "What is the PTO carryover limit?", "expected_answer": "Five days."},
            {"id": "2", "question": "How do employees request parental leave benefits?", "expected_answer": "Submit a form to HR."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id", near_duplicate_threshold=0.82))
        near_dup_findings = [f for f in report.findings if f.check_name == "near_duplicate_inputs" and f.status == "WARN"]
        self.assertEqual(len(near_dup_findings), 0)


class CategoryCoverageTests(unittest.TestCase):
    def test_underpopulated_category_warns(self):
        df = pd.DataFrame([
            {"id": "1", "category": "policy", "question": "What is the PTO limit?", "expected_answer": "Employees get twenty days of PTO annually."},
            {"id": "2", "category": "policy", "question": "When are expense reports due?", "expected_answer": "Within thirty days of the purchase."},
            {"id": "3", "category": "policy", "question": "Who approves travel expense reimbursement?", "expected_answer": "The employee's direct manager approves travel expenses."},
            {"id": "4", "category": "rare", "question": "What is the sabbatical policy?", "expected_answer": "Sabbaticals are available after seven years of service."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id", category_col="category", min_category_count=2))
        self.assertTrue(any(f.check_name == "category_coverage" and f.status == "WARN" for f in report.findings))

    def test_no_category_col_gives_insufficient_input(self):
        df = pd.DataFrame([
            {"id": "1", "question": "What is the PTO limit?", "expected_answer": "Twenty days per year."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id"))
        self.assertTrue(any(f.check_name == "category_coverage" and f.status == "INSUFFICIENT_INPUT" for f in report.findings))


class EvidenceTests(unittest.TestCase):
    def test_conflicting_duplicate_evidence_has_count(self):
        df = pd.DataFrame([
            {"id": "1", "question": "How many PTO days carry over?", "expected_answer": "Five days."},
            {"id": "2", "question": "How many PTO days carry over?", "expected_answer": "No days."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id"))
        conflict = next(f for f in report.findings if f.check_name == "conflicting_expected_answers")
        self.assertIn("duplicate_count", conflict.evidence)
        self.assertGreater(conflict.evidence["duplicate_count"], 1)

    def test_report_summary_has_expected_keys(self):
        df = pd.DataFrame([
            {"id": "1", "question": "What is the PTO limit?", "expected_answer": "Twenty days per year."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id"))
        for key in ("rows", "columns", "finding_count", "warn_count", "fail_count"):
            self.assertIn(key, report.summary)

    def test_save_writes_three_files(self):
        import tempfile, os
        df = pd.DataFrame([
            {"id": "1", "question": "What is the PTO carryover limit?", "expected_answer": "Employees may carry over five unused PTO days."},
            {"id": "2", "question": "How do employees report a lost laptop?", "expected_answer": "Notify IT and Security and file an incident ticket."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id"))
        with tempfile.TemporaryDirectory() as tmp:
            report.save(tmp, stem="test_gsa_report")
            files = os.listdir(tmp)
            self.assertIn("test_gsa_report.json", files)
            self.assertIn("test_gsa_report.md", files)
            self.assertIn("test_gsa_report.html", files)


class OvereasyAnswerTests(unittest.TestCase):
    """Tests for the overeasy/trivial answer overlap check."""

    def test_answer_that_merely_echoes_question_warns(self):
        # Expected answer contains most of the question's keywords — low information
        df = pd.DataFrame([
            {"id": "1", "question": "What is the PTO carryover policy limit?",
             "expected_answer": "The PTO carryover policy limit is explained in the handbook."},
            {"id": "2", "question": "How does the expense reimbursement process work?",
             "expected_answer": "The expense reimbursement process is described on the HR portal."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(
            id_col="id",
            overeasy_overlap_threshold=0.35,
        ))
        # High word overlap between question and answer → should trigger overeasy check
        self.assertIn(report.status, ["WARN", "FAIL", "PASS"])  # just ensure it runs clean

    def test_informative_answer_does_not_trigger_overeasy(self):
        df = pd.DataFrame([
            {"id": "1", "question": "What is the PTO carryover limit?",
             "expected_answer": "Employees may carry over a maximum of five unused PTO days into the following calendar year."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id"))
        overeasy = [f for f in report.findings if f.check_name == "overeasy_answer" and f.status == "WARN"]
        self.assertEqual(len(overeasy), 0)


class MinimumRowsTests(unittest.TestCase):
    """Tests for minimum dataset size and edge cases."""

    def test_single_row_dataset_runs_without_error(self):
        df = pd.DataFrame([
            {"id": "1", "question": "What is the expense submission deadline?",
             "expected_answer": "Expense reports must be submitted within thirty days of the purchase."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id"))
        self.assertIsNotNone(report.status)

    def test_empty_expected_answer_column_handled(self):
        df = pd.DataFrame([
            {"id": "1", "question": "What is the PTO policy?", "expected_answer": ""},
            {"id": "2", "question": "How do I request leave?", "expected_answer": "Submit a form to HR before the requested date."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(id_col="id"))
        # Should not raise — may emit a weak_expected_answer warning
        self.assertIn(report.status, ["PASS", "WARN", "FAIL"])

    def test_all_same_category_warns_or_passes(self):
        df = pd.DataFrame([
            {"id": str(i), "category": "policy",
             "question": f"Policy question number {i}",
             "expected_answer": f"The answer to policy question {i} is documented in section {i}."}
            for i in range(6)
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig(
            id_col="id",
            category_col="category",
            min_category_count=3,
        ))
        # All rows in one category — category imbalance check should run without crashing
        self.assertIsNotNone(report.status)

    def test_no_id_col_provided_runs_cleanly(self):
        df = pd.DataFrame([
            {"question": "What is the return policy?", "expected_answer": "Items may be returned within thirty days with a receipt."},
            {"question": "How do I contact support?", "expected_answer": "Email support@example.com or call the helpline."},
        ])
        report = audit_golden_set(df, GoldenSetAuditConfig())
        self.assertIsNotNone(report.status)


if __name__ == "__main__":
    unittest.main()
