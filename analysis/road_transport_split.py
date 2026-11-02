"""
Step 1: How is Danish road transport energy split between cars, buses, vans and trucks?

Save as:  pypsa-eur/analysis/road_transport_split.py
Run with: pixi run python analysis/road_transport_split.py

Reads data PyPSA-Eur already built from JRC-IDEES (no model run needed).
"""

from pathlib import Path

import pandas as pd

RUN_NAME = "dk-base-2030-jun"  # any finished run for Denmark
folder = Path("resources") / RUN_NAME

# --- Energy demand per node (TWh/a), summed over all Danish nodes -----------
energy = pd.read_csv(folder / "pop_weighted_energy_totals.csv", index_col=0)
dk = energy.sum()  # all nodes in the model are Danish

columns = {
    "Passenger cars": "total passenger cars",
    "Buses & coaches": "total other road passenger",
    "Vans (light freight)": "total light duty road freight",
    "Trucks (heavy freight)": "total heavy duty road freight",
    "Two-wheelers": "total two-wheel",
}
missing = [c for c in columns.values() if c not in dk.index]
if missing:
    print("Columns not found:", missing)
    print("Available road/transport columns:")
    print([c for c in dk.index if "road" in c or "car" in c or "two" in c])
    raise SystemExit

split = pd.Series({name: dk[col] for name, col in columns.items()})
total_road = dk["total road"]
non_electric_rail = dk.get("total rail", 0.0) - dk.get("electricity rail", 0.0)

print("\n=== Danish road transport energy (fuel use, TWh/a) ===")
table = pd.DataFrame({"TWh/a": split, "share of road [%]": 100 * split / total_road})
print(table.round(2).to_string())
print(f"\nSum of categories:  {split.sum():.2f} TWh/a")
print(f"'total road':       {total_road:.2f} TWh/a  (difference = other/rounding)")
print(f"Non-electric rail:  {non_electric_rail:.2f} TWh/a  (also added to the EV demand in PyPSA-Eur!)")

# --- Number of cars in the model ------------------------------------------
td = folder / "transport_data.csv"
if td.exists():
    transport = pd.read_csv(td, index_col=0)
    if "number cars" in transport.columns:
        print(f"\nNumber of passenger cars (all drivetrains): {transport['number cars'].sum():,.0f}")

# --- Heavy-duty share to use in the model ---------------------------------
heavy = split["Buses & coaches"] + split["Trucks (heavy freight)"]
print("\n=== Shares of the land transport demand (road + non-electric rail) ===")
base = total_road + non_electric_rail
for name in split.index:
    print(f"{name:25s} {100 * split[name] / base:5.1f} %")
print(f"{'Rail (non-electric)':25s} {100 * non_electric_rail / base:5.1f} %")
print(f"\nBuses + trucks together: {100 * heavy / base:.1f} % of the 'EV' demand in the model")