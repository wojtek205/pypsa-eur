"""
What does VIPV change in the Danish energy system?
Compares the heavy-transport baseline (no VIPV) with the same model + VIPV.

Save as:  pypsa-eur/analysis/compare_vipv_impact.py
Run with: pixi run python analysis/compare_vipv_impact.py

Output: results/vipv_impact_<base>_vs_<vipv>/  (tables as CSV + plots as PNG)
All times in Danish local time.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

# ---------------------------------------------------------------------------
# SETTINGS - change only these two run names
# ---------------------------------------------------------------------------
BASE = "dk-hd-base-2030-jun"  # new baseline: cars, vans, buses, trucks, no VIPV
VIPV = "dk-hd-vipv-2030-jun-d075-e100"  # same + VIPV
OUT_DIR = Path("results") / f"vipv_impact_{BASE}_vs_{VIPV}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIMEZONE = "Europe/Copenhagen"
WEEK_START = "2013-06-03"  # a Monday

FLEETS = {
    "car": {"charger": "BEV charger", "store": "EV battery", "load": "land transport EV", "v2g": "V2G", "vipv": "VIPV"},
    "van": {"charger": "van charger", "store": "van EV battery", "load": "land transport van EV", "v2g": "van V2G", "vipv": "VIPV van"},
    "bus": {"charger": "bus charger", "store": "bus EV battery", "load": "land transport bus EV", "v2g": None, "vipv": "VIPV bus"},
    "truck": {"charger": "truck charger", "store": "truck EV battery", "load": "land transport truck EV", "v2g": None, "vipv": "VIPV truck"},
}
FLEET_COLORS = {"car": "tab:blue", "van": "tab:cyan", "bus": "tab:green", "truck": "tab:purple"}
RUN_COLORS = {"No VIPV": "tab:blue", "VIPV": "darkorange"}
VIPV_CARRIERS = [c["vipv"] for c in FLEETS.values()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def load(run):
    folder = Path("results") / run / "networks"
    files = sorted(folder.glob("solved_*.nc")) or sorted(folder.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No solved network in {folder} - check the run name.")
    print(f"Loading {run}: {files[-1]}")
    return pypsa.Network(files[-1])


def extract(n):
    """All numbers we need from one solved network."""
    w = n.snapshot_weightings.generators
    r = {"w": w, "fleets": {}}

    # --- per fleet time series (MW) ---
    for fleet, c in FLEETS.items():
        chargers = n.links.index[n.links.carrier == c["charger"]]
        if chargers.empty:
            continue
        stores = n.stores.index[n.stores.carrier == c["store"]]
        loads = n.loads.index[n.loads.carrier == c["load"]]
        v2g = n.links.index[n.links.carrier == c["v2g"]] if c["v2g"] else []
        vipv = n.generators.index[n.generators.carrier == c["vipv"]]
        avail = n.get_switchable_as_dense("Generator", "p_max_pu")
        e_nom = n.stores.loc[stores, "e_nom_opt"].sum()
        zero = pd.Series(0.0, index=n.snapshots)
        r["fleets"][fleet] = pd.DataFrame({
            "charging": n.links_t.p0[cols(n.links_t.p0, chargers)].sum(axis=1),
            "v2g": -n.links_t.p1[cols(n.links_t.p1, v2g)].sum(axis=1) if len(v2g) else zero,
            "driving": n.loads_t.p[cols(n.loads_t.p, loads)].sum(axis=1),
            "vipv_used": n.generators_t.p[cols(n.generators_t.p, vipv)].sum(axis=1) if len(vipv) else zero,
            "vipv_avail": avail[vipv].mul(n.generators.loc[vipv, "p_nom_opt"]).sum(axis=1) if len(vipv) else zero,
            "soc": n.stores_t.e[cols(n.stores_t.e, stores)].sum(axis=1) / e_nom * 100 if e_nom > 0 else zero,
        })
        r["fleets"][fleet].attrs["vipv_mw"] = n.generators.loc[vipv, "p_nom_opt"].sum() if len(vipv) else 0.0

    # --- system ---
    r["cost"] = n.objective / 1e6
    ac = n.buses.index[n.buses.carrier == "AC"]
    r["price"] = n.buses_t.marginal_price[ac].mean(axis=1)

    # capacities of everything the optimiser can build (GW / GWh)
    gens = n.generators[~n.generators.carrier.isin(VIPV_CARRIERS)]
    cap = {
        **(gens.groupby("carrier").p_nom_opt.sum() / 1e3).add_prefix("Generator: ").to_dict(),
        **(n.links[n.links.p_nom_extendable].groupby("carrier").p_nom_opt.sum() / 1e3).add_prefix("Link: ").to_dict(),
        **(n.stores[n.stores.e_nom_extendable].groupby("carrier").e_nom_opt.sum() / 1e3).add_prefix("Store [GWh]: ").to_dict(),
    }
    r["capacity"] = pd.Series(cap)

    # generation per carrier (GWh) and curtailment
    p = n.generators_t.p
    gen = p.mul(w, axis=0).sum().groupby(n.generators.carrier).sum() / 1e3
    r["generation"] = gen
    avail_all = n.get_switchable_as_dense("Generator", "p_max_pu").mul(n.generators.p_nom_opt)
    curt = (avail_all - p.reindex(columns=avail_all.columns, fill_value=0)).clip(lower=0)
    r["curtailment"] = curt.mul(w, axis=0).sum().groupby(n.generators.carrier).sum() / 1e3

    # distribution grid: flow from high to low voltage (the "grid" seen by EV chargers)
    dist = n.links.index[n.links.carrier == "electricity distribution grid"]
    r["dist_flow"] = n.links_t.p0[cols(n.links_t.p0, dist)].sum(axis=1)
    r["dist_cap"] = n.links.loc[dist, "p_nom_opt"].sum()
    return r


# ---------------------------------------------------------------------------
# STEP 1 - Load both runs
# ---------------------------------------------------------------------------
res = {"No VIPV": extract(load(BASE)), "VIPV": extract(load(VIPV))}
b, v = res["No VIPV"], res["VIPV"]
gwh = lambda s, r: (s * r["w"]).sum() / 1e3

# ---------------------------------------------------------------------------
# STEP 2 - Table 1: VIPV per fleet
# ---------------------------------------------------------------------------
rows = {}
for fleet in v["fleets"]:
    fb, fv = b["fleets"][fleet], v["fleets"][fleet]
    avail, used = gwh(fv.vipv_avail, v), gwh(fv.vipv_used, v)
    drive = gwh(fv.driving, v)
    rows[fleet] = {
        "VIPV capacity [MW]": fv.attrs["vipv_mw"],
        "VIPV available [GWh]": avail,
        "VIPV used [GWh]": used,
        "VIPV utilisation [%]": 100 * used / avail if avail > 0 else float("nan"),
        "Driving covered by VIPV [%]": 100 * used / drive if drive > 0 else float("nan"),
        "Grid charging, no VIPV [GWh]": gwh(fb.charging, b),
        "Grid charging, VIPV [GWh]": gwh(fv.charging, v),
        "Change in grid charging [GWh]": gwh(fv.charging, v) - gwh(fb.charging, b),
        "V2G, no VIPV [GWh]": gwh(fb.v2g, b),
        "V2G, VIPV [GWh]": gwh(fv.v2g, v),
        "Change in V2G [GWh]": gwh(fv.v2g, v) - gwh(fb.v2g, b),
        "SoC mean, no VIPV [%]": fb.soc.mean(),
        "SoC mean, VIPV [%]": fv.soc.mean(),
    }
fleet_table = pd.DataFrame(rows)
fleet_table["TOTAL"] = fleet_table.sum(axis=1)
# percentages can't be summed - recalculate them for the TOTAL column
total_used = fleet_table.loc["VIPV used [GWh]", "TOTAL"]
total_avail = fleet_table.loc["VIPV available [GWh]", "TOTAL"]
total_driving = sum(gwh(v["fleets"][f].driving, v) for f in v["fleets"])
fleet_table.loc["VIPV utilisation [%]", "TOTAL"] = 100 * total_used / total_avail
fleet_table.loc["Driving covered by VIPV [%]", "TOTAL"] = 100 * total_used / total_driving
fleet_table.loc[["SoC mean, no VIPV [%]", "SoC mean, VIPV [%]"], "TOTAL"] = float("nan")
print("\n=== Table 1: VIPV per fleet (modelled period) ===")
print(fleet_table.round(2).to_string())
fleet_table.to_csv(OUT_DIR / "table1_vipv_per_fleet.csv")

# ---------------------------------------------------------------------------
# STEP 3 - Table 2: the energy system
# ---------------------------------------------------------------------------
vipv_used_total = fleet_table.loc["VIPV used [GWh]", "TOTAL"]
system = pd.DataFrame({
    label: {
        "Total system cost [M EUR]": r["cost"],
        "Average electricity price [EUR/MWh]": (r["price"] * r["w"]).sum() / r["w"].sum(),
        "Total EV grid charging [GWh]": sum(gwh(f.charging, r) for f in r["fleets"].values()),
        "Total V2G to grid [GWh]": sum(gwh(f.v2g, r) for f in r["fleets"].values()),
        "Distribution grid capacity [GW]": r["dist_cap"] / 1e3,
        "Distribution grid peak flow [GW]": r["dist_flow"].max() / 1e3,
        "Distribution grid energy [GWh]": gwh(r["dist_flow"], r),
        "Wind + solar curtailed (excl. VIPV) [GWh]": r["curtailment"].drop(VIPV_CARRIERS, errors="ignore").sum(),
    }
    for label, r in res.items()
})
system["Change"] = system["VIPV"] - system["No VIPV"]
saving = -system.loc["Total system cost [M EUR]", "Change"]
print("\n=== Table 2: energy system ===")
print(system.round(2).to_string())
print(f"\nValue of VIPV energy: {saving * 1e6 / (vipv_used_total * 1e3):.1f} EUR per MWh used")
system.to_csv(OUT_DIR / "table2_system.csv")

# ---------------------------------------------------------------------------
# STEP 4 - Table 3: what the optimiser builds differently
# ---------------------------------------------------------------------------
cap = pd.DataFrame({label: r["capacity"] for label, r in res.items()}).fillna(0)
cap["Change"] = cap["VIPV"] - cap["No VIPV"]
cap_changed = cap[cap["Change"].abs() > 0.005].sort_values("Change")  # > 5 MW
print("\n=== Table 3: capacities that change (GW, stores in GWh) ===")
print(cap_changed.round(3).to_string() if not cap_changed.empty else "No capacity changes above 5 MW.")
cap.to_csv(OUT_DIR / "table3_capacities_all.csv")

gen = pd.DataFrame({label: r["generation"] for label, r in res.items()}).fillna(0)
gen["Change"] = gen["VIPV"] - gen["No VIPV"]
gen.to_csv(OUT_DIR / "table4_generation_all.csv")

print("\nMaking plots...")

# ---------------------------------------------------------------------------
# PLOT 1 - VIPV per fleet: available vs used (GWh)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
fleets = list(fleet_table.columns.drop("TOTAL"))
x = range(len(fleets))
ax.bar([i - 0.2 for i in x], fleet_table.loc["VIPV available [GWh]", fleets], 0.4, color="navajowhite", label="available")
ax.bar([i + 0.2 for i in x], fleet_table.loc["VIPV used [GWh]", fleets], 0.4, color="darkorange", label="used")
ax.set_xticks(list(x))
ax.set_xticklabels(fleets)
ax.set_ylabel("Energy [GWh]")
ax.set_title("VIPV generation per fleet")
ax.legend()
save(fig, "01_vipv_per_fleet")

# ---------------------------------------------------------------------------
# PLOT 2 - Change in grid charging and V2G per fleet, average day
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for fleet in fleets:
    db, dv = to_local(b["fleets"][fleet]), to_local(v["fleets"][fleet])
    diff = (dv - db).groupby(dv.index.hour).mean()
    ax1.plot(diff.index, diff.charging, color=FLEET_COLORS[fleet], label=fleet)
    if FLEETS[fleet]["v2g"]:
        ax2.plot(diff.index, diff.v2g, color=FLEET_COLORS[fleet], label=fleet)
for ax, title in [(ax1, "Change in grid charging"), (ax2, "Change in V2G to grid")]:
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (local time)")
    ax.set_title(f"{title}: VIPV minus no VIPV (average day)")
    ax.legend(fontsize=8)
ax1.set_ylabel("Change [MW]")
save(fig, "02_change_charging_v2g_average_day")

# ---------------------------------------------------------------------------
# PLOT 3 - Electricity price: both runs + difference, average day
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
for label, r in res.items():
    p = to_local(r["price"])
    ax1.plot(p.groupby(p.index.hour).mean(), color=RUN_COLORS[label], label=label)
diff = to_local(v["price"] - b["price"])
ax2.bar(range(24), diff.groupby(diff.index.hour).mean(), color="grey")
ax2.axhline(0, color="black", lw=0.5)
for ax in (ax1, ax2):
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (local time)")
ax1.set_ylabel("Price [EUR/MWh]")
ax1.set_title("Average electricity price")
ax1.legend()
ax2.set_ylabel("Change [EUR/MWh]")
ax2.set_title("Price change: VIPV minus no VIPV")
save(fig, "03_price_average_day")

# ---------------------------------------------------------------------------
# PLOT 4 - Capacity changes (what is built more / less)
# ---------------------------------------------------------------------------
if not cap_changed.empty:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(cap_changed))))
    colors = ["tab:green" if c > 0 else "tab:red" for c in cap_changed["Change"]]
    ax.barh(cap_changed.index, cap_changed["Change"] * 1e3, color=colors)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Change [MW]  (stores: MWh)")
    ax.set_title("Capacity built with VIPV minus without VIPV")
    save(fig, "04_capacity_changes")

# ---------------------------------------------------------------------------
# PLOT 5 - One week: total EV grid charging, VIPV generation and price
# ---------------------------------------------------------------------------
week = slice(pd.Timestamp(WEEK_START), pd.Timestamp(WEEK_START) + pd.Timedelta(days=7))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
for label, r in res.items():
    charging = to_local(sum(f.charging for f in r["fleets"].values())).loc[week]
    ax1.plot(charging.index, charging, color=RUN_COLORS[label], label=f"EV grid charging - {label}")
    p = to_local(r["price"]).loc[week]
    ax2.plot(p.index, p, color=RUN_COLORS[label], label=label)
vipv_gen = to_local(sum(f.vipv_used for f in v["fleets"].values())).loc[week]
ax1.fill_between(vipv_gen.index, vipv_gen, color="gold", alpha=0.5, label="VIPV used (all fleets)")
ax1.set_ylabel("Power [MW]")
ax1.set_title("One week: EV grid charging and VIPV generation (all fleets)")
ax1.legend(fontsize=8)
ax2.set_ylabel("Price [EUR/MWh]")
ax2.set_title("Electricity price")
ax2.legend(fontsize=8)
ax2.set_xlabel("Date (Danish local time)")
save(fig, "05_week_charging_vipv_price")

# ---------------------------------------------------------------------------
# PLOT 6 - Distribution grid load, average day (does VIPV relieve the grid?)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
for label, r in res.items():
    f = to_local(r["dist_flow"])
    ax.plot(f.groupby(f.index.hour).mean() / 1e3, color=RUN_COLORS[label], label=label)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day (local time)")
ax.set_ylabel("Flow to low voltage [GW]")
ax.set_title("Distribution grid load (average day)")
ax.legend()
save(fig, "06_distribution_grid_average_day")

print(f"\nDone! Open: {OUT_DIR}")