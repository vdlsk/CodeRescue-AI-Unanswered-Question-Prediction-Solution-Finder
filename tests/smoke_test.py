"""End-to-end run on SYNTHETIC inspections and a SYNTHETIC mini food code. Proves the code works;
numbers are meaningless and must not be reported.   Usage:  python tests/smoke_test.py"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VIOLS = {
    2: "CITY OF CHICAGO FOOD SERVICE SANITATION CERTIFICATE",
    5: "PROCEDURES FOR RESPONDING TO VOMITING AND DIARRHEAL EVENTS",
    10: "ADEQUATE HANDWASHING SINKS PROPERLY SUPPLIED AND ACCESSIBLE",
    38: "INSECTS, RODENTS, & ANIMALS NOT PRESENT",
    47: "FOOD & NON-FOOD CONTACT SURFACES CLEANABLE, PROPERLY DESIGNED, CONSTRUCTED & USED",
    55: "PHYSICAL FACILITIES INSTALLED, MAINTAINED & CLEAN",
    37: "FOOD PROPERLY LABELED; ORIGINAL CONTAINER",
}
REGS = """
2-102.12 Certified Food Protection Manager
At least one employee with supervisory responsibility shall be a certified food protection manager.
2-201.11 Responsibility of Permit Holder, Person in Charge, and Conditional Employees
Employees shall report symptoms of vomiting, diarrhea, jaundice and sore throat with fever.
2-301.14 When to Wash
Food employees shall wash their hands after touching bare human body parts and after using the toilet room.
3-501.16 Time/Temperature Control for Safety Food, Hot and Cold Holding
Food shall be maintained at 57°C (135°F) or above, or at 5°C (41°F) or less.
3-501.17 Ready-to-Eat Food, Date Marking
Food shall be marked to indicate the date by which it shall be consumed, which is 7 calendar days.
5-205.11 Using a Handwashing Sink
A handwashing sink shall be maintained so that it is accessible at all times for employee use.
6-501.111 Controlling Pests
The premises shall be maintained free of insects, rodents, and other pests.
7-201.11 Separation
Poisonous or toxic materials shall be stored so they cannot contaminate food.
4-602.11 Equipment Food-Contact Surfaces and Utensils
Surfaces shall be cleaned throughout the day at least every 4 hours.
""" * 2


def make_csv(path, n_est=600, seed=0):
    rng = np.random.default_rng(seed)
    rows, iid = [], 1_000_000
    for lic in range(1, n_est + 1):
        quality = rng.beta(2, 6)                       # persistent establishment weakness
        ftype = rng.choice(["Restaurant", "Grocery Store", "School", "Bakery"])
        risk = rng.choice(["Risk 1 (High)", "Risk 2 (Medium)", "Risk 3 (Low)"])
        d = pd.Timestamp("2016-01-01") + pd.Timedelta(days=int(rng.integers(0, 200)))
        prev_fail = False
        while d < pd.Timestamp("2024-12-31"):
            p = min(0.95, quality + (0.08 if prev_fail else 0))
            fail = rng.random() < p
            vs = rng.choice(list(VIOLS), size=int(rng.integers(0, 4)) + (2 if fail else 0), replace=True)
            vtxt = " | ".join(f"{v}. {VIOLS[v]} - Comments: observed issue {v}" for v in sorted(set(vs)))
            iid += 1
            rows.append({"Inspection ID": iid, "DBA Name": f"Place {lic}", "AKA Name": "", "License #": lic,
                         "Facility Type": ftype, "Risk": risk, "Address": f"{lic} W TEST ST", "City": "CHICAGO",
                         "State": "IL", "Zip": "60601", "Inspection Date": d.strftime("%m/%d/%Y"),
                         "Inspection Type": "Canvass", "Results": "Fail" if fail else rng.choice(["Pass", "Pass w/ Conditions"]),
                         "Violations": vtxt, "Latitude": 41.88, "Longitude": -87.63, "Location": ""})
            if fail:
                iid += 1
                rd = d + pd.Timedelta(days=14)
                rows.append({**rows[-1], "Inspection ID": iid, "Inspection Date": rd.strftime("%m/%d/%Y"),
                             "Inspection Type": "Canvass Re-Inspection", "Results": "Pass", "Violations": ""})
            prev_fail = fail
            d += pd.Timedelta(days=int(rng.integers(150, 400)))
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="p3_smoke_"))
    for sub in ("raw", "regs", "out"):
        (tmp / sub).mkdir()
    make_csv(tmp / "raw" / "food_inspections.csv")
    (tmp / "regs" / "mini_food_code.txt").write_text(REGS)
    env = {**os.environ, "P3_RAW_DIR": str(tmp / "raw"), "P3_REG_DIR": str(tmp / "regs"),
           "P3_DB_PATH": str(tmp / "t.duckdb"), "P3_OUT_DIR": str(tmp / "out"),
           "P3_INDEX_PATH": str(tmp / "idx.joblib"), "DENSE_MODEL": "none-forces-fallback"}
    env.pop("ANTHROPIC_API_KEY", None)
    for s in ("build_db.py", "stats_analysis.py", "regs.py", "eval_rag.py"):
        print(f"\n>>> {s}")
        subprocess.run([sys.executable, str(ROOT / "src" / s)], check=True, env=env)
    print("\n>>> assistant")
    code = "from assistant import FoodSafe; f=FoodSafe(); o=f.ask(5,'why failing?'); print(o['answer']); print(o['grounding'])"
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=ROOT / "src")
    print("\nSMOKE TEST PASSED (synthetic data, not real results)")


if __name__ == "__main__":
    main()
