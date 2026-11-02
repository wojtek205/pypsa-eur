"""
Check (and plot) the EV energy balance for the baseline and VIPV runs.

Save as:  pypsa-eur/analysis/check_ev_energy_balance.py
Run with: pixi run python analysis/check_ev_energy_balance.py

Everything entering the EV batteries must equal everything leaving them
(plus the change in stored energy). If the 'mismatch' is ~0, the results are consistent.

Plots are saved to: results/comparison_vipv/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

RUNS = {"Baseline": "dk-base-2030-jun", "VIPV": "dk-vipv-2030-jun-d100-e100"}
OUT_DIR = Path("results") / f"comparison_{RUNS['Baseline']}_vs_{RUNS['VIPV']}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "grid": "tab:green",
    "vipv": "darkorange",
    "driving": "dimgrey",
    "v2g": "tab:red",
    "loss": "lightgrey",
}

# ---------------------------------------------------------------------------
# STEP 1 - Calculate the energy balance for each run
# ---------------------------------------------------------------------------
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

    grid_drawn = gwh(n.links_t.p0[chargers])  # taken from the grid
    into_bus_from_grid = gwh(-n.links_t.p1[chargers])  # arrives in the cars (after losses)
    vipv_in = gwh(n.generators_t.p[vipv]) if len(vipv) else 0.0
    driving = gwh(n.loads_t.p[loads])
    v2g_from_bus = gwh(n.links_t.p0[v2g])  # taken out of the cars
    v2g_to_grid = gwh(-n.links_t.p1[v2g])  # arrives in the grid (after losses)
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
        "Charging losses [GWh]": grid_drawn - into_bus_from_grid,
        "V2G losses [GWh]": v2g_from_bus - v2g_to_grid,
    }

table = pd.DataFrame(rows)
if {"Baseline", "VIPV"} <= set(table.columns):
    table["Difference"] = table["VIPV"] - table["Baseline"]
print(table.round(3).to_string())
table.to_csv(OUT_DIR / "ev_energy_balance_check.csv")

labels = list(RUNS)
t = table[labels]

# ---------------------------------------------------------------------------
# PLOT 15 - Energy IN vs energy OUT of the EV batteries (both scenarios)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x_pos, x_labels = [], []
for i, label in enumerate(labels):
    x_in, x_out = i * 3, i * 3 + 1
    grid_in = t.loc["Grid charging, arriving in cars [GWh]", label]
    vipv_in = t.loc["VIPV into cars [GWh]", label]
    drv = t.loc["Driving [GWh]", label]
    v2g_out = t.loc["V2G, taken out of cars [GWh]", label]

    ax.bar(x_in, grid_in, color=COLORS["grid"], label="From grid (after charger loss)" if i == 0 else None)
    ax.bar(x_in, vipv_in, bottom=grid_in, color=COLORS["vipv"], label="From VIPV" if i == 0 else None)
    ax.bar(x_out, drv, color=COLORS["driving"], label="Driving" if i == 0 else None)
    ax.bar(x_out, v2g_out, bottom=drv, color=COLORS["v2g"], label="V2G (taken out of cars)" if i == 0 else None)
    ax.text(x_in, grid_in + vipv_in + 3, f"{grid_in + vipv_in:.1f}", ha="center", fontsize=9)
    ax.text(x_out, drv + v2g_out + 3, f"{drv + v2g_out:.1f}", ha="center", fontsize=9)
    x_pos += [x_in, x_out]
    x_labels += [f"{label}\nIN", f"{label}\nOUT"]

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylabel("Energy [GWh]")
ax.set_title("EV batteries: energy in = energy out (June)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "15_ev_energy_in_vs_out.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# PLOT 16 - Waterfall: where does the VIPV energy go?
# ---------------------------------------------------------------------------
if "Difference" in table.columns:
    d = table["Difference"]
    steps = [
        ("VIPV\ninto cars", d["VIPV into cars [GWh]"], COLORS["vipv"]),
        ("Less grid\ncharging\n(in cars)", d["Grid charging, arriving in cars [GWh]"], COLORS["grid"]),
        ("= Extra\nenergy\nin cars", None, "black"),
        ("More V2G\n(out of cars)", -d["V2G, taken out of cars [GWh]"], COLORS["v2g"]),
        ("Change in\ndriving", -d["Driving [GWh]"], COLORS["driving"]),
        ("Remaining", None, "black"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    level = 0.0
    for i, (name, value, color) in enumerate(steps):
        if value is None:  # subtotal bar
            ax.bar(i, level, color=color, alpha=0.7)
            ax.text(i, level + 0.6, f"{level:+.1f}", ha="center", fontsize=9)
            continue
        bottom = level if value >= 0 else level + value
        ax.bar(i, abs(value), bottom=bottom, color=color)
        ax.text(i, max(level, level + value) + 0.6, f"{value:+.1f}", ha="center", fontsize=9)
        level += value
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=8)
    ax.set_ylabel("Change vs baseline [GWh]")
    ax.set_title("Where does the VIPV energy go? (VIPV minus baseline, June)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "16_vipv_energy_waterfall.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# PLOT 17 - Losses in the EV system (charging and V2G)
# ---------------------------------------------------------------------------
loss = t.loc[["Charging losses [GWh]", "V2G losses [GWh]"]]
loss.index = ["Charging losses", "V2G losses"]
fig, ax = plt.subplots(figsize=(7, 4))
loss.T.plot.bar(stacked=True, ax=ax, color=[COLORS["grid"], COLORS["v2g"]], rot=0, alpha=0.6)
for i, label in enumerate(labels):
    ax.text(i, loss[label].sum() + 0.5, f"{loss[label].sum():.1f}", ha="center", fontsize=9)
ax.set_ylabel("Energy lost [GWh]")
ax.set_title("EV conversion losses (June)")
fig.tight_layout()
fig.savefig(OUT_DIR / "17_ev_losses.png", dpi=150)
plt.close(fig)

print(f"\nPlots saved to: {OUT_DIR}")