from pathlib import Path
import pandas as pd

from goldensetauditor import GoldenSetAuditConfig, audit_golden_set

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

rows = [
    {"id": "Q001", "category": "policy", "question": "How many PTO days can employees carry over?", "expected_answer": "Employees may carry over up to five unused PTO days into the next calendar year."},
    {"id": "Q002", "category": "policy", "question": "How many PTO days can employees carry over?", "expected_answer": "No PTO days can be carried over after year end."},
    {"id": "Q003", "category": "benefits", "question": "Can I expense a standing desk for my home office?", "expected_answer": "Standing desks may be expensed with manager approval under the equipment policy."},
    {"id": "Q004", "category": "benefits", "question": "Can I reimburse a standing desk for my remote office?", "expected_answer": "A standing desk for a remote office can be reimbursed if the manager approves it."},
    {"id": "Q005", "category": "disciplinary", "question": "What happens after a first disciplinary offense?", "expected_answer": "The manager issues a written warning and documents the incident in the HR system."},
    {"id": "Q006", "category": "remote", "question": "What about this?", "expected_answer": "It depends."},
    {"id": "Q007", "category": "expense", "question": "What is the meal reimbursement limit and can alcohol be reimbursed?", "expected_answer": "The meal reimbursement limit is fifty dollars and alcohol is not reimbursable."},
    {"id": "Q008", "category": "expense", "question": "The VPN reimbursement limit is 30 dollars", "expected_answer": "The VPN reimbursement limit is 30 dollars."},
    {"id": "Q009", "category": "security", "question": "What is the process for reporting a lost laptop?", "expected_answer": "Report the lost laptop to IT and Security immediately, then file an incident ticket."},
    {"id": "Q010", "category": "security", "question": "When should employees report a lost laptop?", "expected_answer": "Immediately after discovering the loss."},
]

df = pd.DataFrame(rows)
data_path = DATA / "demo_golden_set.csv"
df.to_csv(data_path, index=False)

report = audit_golden_set(df, GoldenSetAuditConfig(
    input_col="question",
    expected_col="expected_answer",
    category_col="category",
    id_col="id",
    min_category_count=2,
))
report.save(OUT)
print(report.to_markdown())
print(f"\nSaved demo data to {data_path}")
print(f"Saved reports to {OUT}")
