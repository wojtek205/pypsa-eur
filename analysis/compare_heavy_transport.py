"""
Compare runs fleet by fleet: cars, buses and trucks (with or without VIPV).

Save as:  pypsa-eur/analysis/compare_heavy_transport.py
Run with: pixi run python analysis/compare_heavy_transport.py

Works for any number of runs (2 or more). Fleets that don't exist in a run are skipped.
Plots and tables go to: results/comparison_<run1>_vs_<run2>.../
All times in Danish local time.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

# ---------------------------------------------------------------------------
# SETTINGS - change only RUNS
# ---------------------------------------------------------------------------
RUNS = {
    "Baseline": "dk-base-2030-jun",
    "Heavy transport": "dk-hd-base-2030-jun",
    # "HD + VIPV": "dk-hd-vipv-2030-jun-d075-e095",   # add when finished
}
OUT_DIR = Path("results") / ("comparison_" + "_vs_".join(RUNS.values()))
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIMEZONE = "Europe/Copenhagen"

# fleet: carrier names used in the model
FLEETS = {
    "car":   {"charger": "BEV charger",   "store": "EV battery",       "load": "land transport EV",       "v2g": "V2G", "vipv": "VIPV"},
    "van":   {"charger": "van charger",   "store": "van EV battery",   "load": "land transport van EV",   "v2g": "van V2G", "vipv": "VIPV van"},
    "bus":   {"charger": "bus charger",   "store": "bus EV battery",   "load": "land transport bus EV",   "v2g": None,  "vipv": "VIPV bus"},
    "truck": {"charger": "truck charger", "store": "truck EV battery", "load": "land transport truck EV", "v2g": None,  "vipv": "VIPV truck"},
}
FLEET_COLORS = {"car": "tab:blue", "van": "tab:cyan", "bus": "tab:green", "truck": "tab:purple"}
RE_CARRIERS = ["onwind", "offwind-ac", "offwind-dc", "offwind-float", "solar", "solar-hsat", "solar rooftop"]


def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def to_local(obj):
    obj = obj.copy()
    obj.index = pd.DatetimeIndex(obj.index).tz_localize("UTC").tz_convert(TIMEZONE).tz_localize(None)
    return obj


def cols(df, index):
    return [c for c in index if c in df.columns]


def load_network(run):
    folder = Path("results") / run / "networks"
    files = sorted(folder.glob("solved_*.nc")) or sorted(folder.glob("*.nc"))
    if not files:
        print(f"WARNING: no solved network in {folder} - skipped.")
        return None
    print(f"Loading {run}: {files[-1]}")
    return pypsa.Network(files[-1])


# ---------------------------------------------------------------------------
# STEP 1 - Extract numbers per run and fleet
# ---------------------------------------------------------------------------
summary, series, capacities, prices = {}, {}, {}, {}
for label, run in RUNS.items():
    n = load_network(run)
    if n is None:
        continue
    w = n.snapshot_weightings.generators
    gwh = lambda s: (s * w).sum() / 1e3

    rows, ts = {}, {}
    for fleet, c in FLEETS.items():
        chargers = n.links.index[n.links.carrier == c["charger"]]
        if chargers.empty:
            continue
        stores = n.stores.index[n.stores.carrier == c["store"]]
        loads = n.loads.index[n.loads.carrier == c["load"]]
        v2g = n.links.index[n.links.carrier == c["v2g"]] if c["v2g"] else []
        vipv = n.generators.index[n.generators.carrier == c["vipv"]]

        charging = n.links_t.p0[cols(n.links_t.p0, chargers)].sum(axis=1)
        driving = n.loads_t.p[cols(n.loads_t.p, loads)].sum(axis=1)
        v2g_out = -n.links_t.p1[cols(n.links_t.p1, v2g)].sum(axis=1) if len(v2g) else charging * 0
        vipv_used = n.generators_t.p[cols(n.generators_t.p, vipv)].sum(axis=1) if len(vipv) else charging * 0
        e_nom = n.stores.loc[stores, "e_nom_opt"].sum()
        soc = n.stores_t.e[cols(n.stores_t.e, stores)].sum(axis=1) / e_nom * 100 if e_nom > 0 else charging * 0

        days = w.sum() / 24
        rows[fleet] = {
            "Charger capacity [MW]": n.links.loc[chargers, "p_nom_opt"].sum(),
            "Battery capacity [MWh]": e_nom,
            "Driving demand [GWh]": gwh(driving),
            "Charging from grid [GWh]": gwh(charging),
            "V2G to grid [GWh]": gwh(v2g_out),
            "VIPV used [GWh]": gwh(vipv_used),
            "Peak charging [MW]": charging.max(),
            "SoC min [%]": soc.min(),
            "SoC mean [%]": soc.mean(),
        }
        ts[fleet] = to_local(pd.DataFrame({"charging": charging, "driving": driving,
                                           "v2g": v2g_out, "vipv": vipv_used, "soc": soc}))

    table = pd.DataFrame(rows)
    table["TOTAL"] = table.sum(axis=1)
    table.loc[["SoC min [%]", "SoC mean [%]", "Peak charging [MW]"], "TOTAL"] = float("nan")
    summary[label], series[label] = table, ts

    re_gens = n.generators[n.generators.carrier.isin(RE_CARRIERS)]
    capacities[label] = re_gens.groupby("carrier").p_nom_opt.sum() / 1e3
    ac = n.buses.index[n.buses.carrier == "AC"]
    prices[label] = to_local(n.buses_t.marginal_price[ac].mean(axis=1))
    summary[label].attrs["cost"] = n.objective / 1e6 if n.objective else float("nan")

labels = list(summary)

# ---------------------------------------------------------------------------
# STEP 2 - Print and save tables
# ---------------------------------------------------------------------------
for label in labels:
    print(f"\n=== {label} ({RUNS[label]}) - per fleet, modelled period ===")
    print(summary[label].round(1).to_string())
    summary[label].to_csv(OUT_DIR / f"fleets_{RUNS[label]}.csv")

system = pd.DataFrame({
    label: {
        "Total system cost [M EUR]": summary[label].attrs["cost"],
        "Total EV driving demand [GWh]": summary[label].loc["Driving demand [GWh]", "TOTAL"],
        "Total EV grid charging [GWh]": summary[label].loc["Charging from grid [GWh]", "TOTAL"],
        "Total V2G to grid [GWh]": summary[label].loc["V2G to grid [GWh]", "TOTAL"],
        "Average electricity price [EUR/MWh]": prices[label].mean(),
        "Solar capacity [GW]": capacities[label].reindex(["solar", "solar-hsat", "solar rooftop"]).sum(),
        "Wind capacity [GW]": capacities[label].reindex(["onwind", "offwind-ac", "offwind-dc", "offwind-float"]).sum(),
    } for label in labels
})
print("\n=== System comparison ===")
print(system.round(2).to_string())
print("\nCheck: 'Total EV driving demand' should be (almost) equal in all runs.")
system.to_csv(OUT_DIR / "system_comparison.csv")

print("\nMaking plots...")

# ---------------------------------------------------------------------------
# PLOT 1 - Driving demand split by fleet (stacked)
# ---------------------------------------------------------------------------
demand = pd.DataFrame({l: summary[l].loc["Driving demand [GWh]"].drop("TOTAL") for l in labels}).T.fillna(0)
fig, ax = plt.subplots(figsize=(7, 4))
demand.plot.bar(stacked=True, ax=ax, rot=0, color=[FLEET_COLORS[f] for f in demand.columns])
ax.set_ylabel("Driving demand [GWh]")
ax.set_title("EV driving demand by fleet (total should be equal)")
save(fig, "01_driving_demand_by_fleet")

# ---------------------------------------------------------------------------
# PLOT 2 - Average day: grid charging per fleet, one panel per run
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 4), sharey=True)
axes = [axes] if len(labels) == 1 else axes
for ax, label in zip(axes, labels):
    for fleet, df in series[label].items():
        daily = df.groupby(df.index.hour).charging.mean()
        ax.plot(daily.index, daily, color=FLEET_COLORS[fleet], label=f"{fleet} charging")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (local time)")
    ax.set_title(label)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Average grid charging [MW]")
save(fig, "02_charging_average_day_by_fleet")

# ---------------------------------------------------------------------------
# PLOT 3 - Average day: driving demand vs charging, per fleet (last run)
# ---------------------------------------------------------------------------
last = labels[-1]
fleets = list(series[last])
fig, axes = plt.subplots(1, len(fleets), figsize=(5 * len(fleets), 4))
axes = [axes] if len(fleets) == 1 else axes
for ax, fleet in zip(axes, fleets):
    d = series[last][fleet].groupby(series[last][fleet].index.hour).mean()
    ax.plot(d.index, d.charging, color=FLEET_COLORS[fleet], label="charging from grid")
    ax.plot(d.index, d.driving, color="black", lw=1, label="driving demand")
    if d.v2g.sum() > 0:
        ax.plot(d.index, -d.v2g, color="tab:red", label="V2G (negative)")
    if d.vipv.sum() > 0:
        ax.plot(d.index, d.vipv, color="darkorange", ls=":", label="VIPV used")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (local time)")
    ax.set_title(f"{fleet} - {last}")
    ax.legend(fontsize=7)
axes[0].set_ylabel("Average power [MW]")
save(fig, "03_fleet_profiles_average_day")

# ---------------------------------------------------------------------------
# PLOT 4 - State of charge per fleet, first week (last run)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4))
for fleet, df in series[last].items():
    d = df.iloc[: 24 * 7]
    ax.plot(d.index, d.soc, color=FLEET_COLORS[fleet], label=fleet)
ax.set_ylim(0, 100)
ax.set_ylabel("State of charge [%]")
ax.set_title(f"Battery state of charge per fleet, first week - {last}")
ax.legend()
save(fig, "04_soc_per_fleet_week")

# ---------------------------------------------------------------------------
# PLOT 5 - Electricity price, average day, all runs
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
for label in labels:
    p = prices[label]
    ax.plot(p.groupby(p.index.hour).mean(), label=label)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day (local time)")
ax.set_ylabel("Electricity price [EUR/MWh]")
ax.set_title("Average electricity price by hour")
ax.legend()
save(fig, "05_price_average_day")

# ---------------------------------------------------------------------------
# PLOT 6 - Wind and solar capacities, all runs
# ---------------------------------------------------------------------------
cap = pd.DataFrame(capacities).fillna(0)
cap = cap[(cap > 1e-3).any(axis=1)]
fig, ax = plt.subplots(figsize=(8, 4))
cap.plot.barh(ax=ax)
ax.set_xlabel("Installed capacity [GW]")
ax.set_title("Optimised wind and solar capacity")
save(fig, "06_capacities")

print(f"\nDone! Open: {OUT_DIR}")