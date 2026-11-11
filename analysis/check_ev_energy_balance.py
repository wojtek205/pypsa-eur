"""
Check the EV energy balance for the baseline and VIPV runs.

Save as:  pypsa-eur/analysis/check_ev_energy_balance.py
Run with: pixi run python analysis/check_ev_energy_balance.py

Everything entering the EV battery bus must equal everything leaving it
(plus the change in stored energy). If the 'mismatch' is ~0, the results are consistent.
"""

from pathlib import Path

import pandas as pd
import pypsa

RUNS = {"Baseline": "dk-base-2030-jun", "VIPV": "dk-vipv-2030-jun-d075-e095"}
rows = {}

for label, run in RUNS.items():
    folder = Path("results") / run / "networks"
    files = sorted(folder.glob("solved_*.nc")) or sorted(folder.glob("*.nc"))
    n = pypsa.Network(files[-1])
    w = n.snapshot_weightings.generators  # hours per snapshot

    chargers = n.links.index[n.links.carrier == "BEV charger"]
    v2g = n.links.index[n.links.carrier == "V2G"]
    stores = n.stores.index[n.stores.carrier == "EV battery"]
    loads = n.loads.index[n.loads.carrier == "land transport EV"]
    vipv = n.generators.index[n.generators.carrier == "VIPV"]

    gwh = lambda df: df.mul(w, axis=0).sum().sum() / 1e3

    grid_drawn = gwh(n.links_t.p0[chargers])                 # taken from the grid
    into_bus_from_grid = gwh(-n.links_t.p1[chargers])        # arrives in the cars (after losses)
    vipv_in = gwh(n.generators_t.p[vipv]) if len(vipv) else 0.0
    driving = gwh(n.loads_t.p[loads])
    v2g_from_bus = gwh(n.links_t.p0[v2g])                    # taken out of the cars
    v2g_to_grid = gwh(-n.links_t.p1[v2g])                    # arrives in the grid (after losses)
    e = n.stores_t.e[stores].sum(axis=1)
    delta_stored = (e.iloc[-1] - e.iloc[0]) / 1e3

    rows[label] = {
        "Charger efficiency": n.links.loc[chargers, "efficiency"].mean(),
        "V2G efficiency": n.links.loc[v2g, "efficiency"].mean(),
        "Store standing loss": n.stores.loc[stores, "standing_loss"].mean(),
        "Grid charging, drawn from grid [GWh]": grid_drawn,
        "Grid charging, arriving in cars [GWh]": into_bus_from_grid,
        "VIPV into cars [GWh]": vipv_in,
        "TOTAL INTO CARS [GWh]": into_bus_from_grid + vipv_in,
        "Driving [GWh]": driving,
        "V2G, taken out of cars [GWh]": v2g_from_bus,
        "V2G, arriving in grid [GWh]": v2g_to_grid,
        "TOTAL OUT OF CARS [GWh]": driving + v2g_from_bus,
        "Change in stored energy [GWh]": delta_stored,
        "Mismatch (should be ~0) [GWh]": into_bus_from_grid + vipv_in - driving - v2g_from_bus - delta_stored,
        "Average stored energy [GWh]": e.mean() / 1e3,
    }

table = pd.DataFrame(rows)
table["Difference"] = table["VIPV"] - table["Baseline"]
print(table.round(3).to_string())