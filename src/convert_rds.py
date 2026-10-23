"""Convert the City's 13_food_inspections.Rds into the portal CSV layout used by build_db.py.
Usage:  python src/convert_rds.py /path/to/13_food_inspections.Rds"""
import sys

from config import CSV_NAME, RAW_DIR
from rds_reader import read_rds

df = read_rds(sys.argv[1]).rename(columns={
    "Inspection_ID": "Inspection ID", "DBA_Name": "DBA Name", "AKA_Name": "AKA Name", "License": "License #",
    "Facility_Type": "Facility Type", "Inspection_Date": "Inspection Date", "Inspection_Type": "Inspection Type"})
for c in ("Inspection ID", "License #", "Zip"):
    df[c] = df[c].astype("Int64")
df["Inspection Date"] = df["Inspection Date"].dt.strftime("%m/%d/%Y")
RAW_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(RAW_DIR / CSV_NAME, index=False)
print(df.shape, "->", RAW_DIR / CSV_NAME)
