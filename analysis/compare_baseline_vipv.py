"""
Compare the dk-baseline and dk-vipv PyPSA-Eur runs.

Save this file as:   pypsa-eur/analysis/compare_baseline_vipv.py
Run it from the pypsa-eur folder with:
    pixi run python analysis/compare_baseline_vipv.py

Plots (PNG) and tables (CSV) are saved to: results/comparison_vipv/
All times are shown in Danish local time (the model itself uses UTC).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

# ---------------------------------------------------------------------------
# SETTINGS - change these if needed
# ---------------------------------------------------------------------------
RUNS = {"Baseline": "dk-baseline", "VIPV": "dk-vipv"}  # label: run name
OUT_DIR = Path("results") / "comparison_vipv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEZONE = "Europe/Copenhagen"
ZOOM_DAYS = 7  # length of the zoomed-in plots (first week)
COLORS = {"Baseline": "tab:blue", "VIPV": "tab:orange"}
RE_CARRIERS = [
    "onwind", "offwind-ac", "offwind-dc", "offwind-float",
    "solar", "solar-hsat", "solar rooftop", "VIPV",
]
SOLAR_CARRIERS = ["solar", "solar-hsat", "solar rooftop", "VIPV"]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def save(fig, name):
    """Save a figure as PNG and close it."""
    path = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def to_local(obj):
    """Convert a UTC time index to Danish local time."""
    obj = obj.copy()
    obj.index = (
        pd.DatetimeIndex(obj.index)
        .tz_localize("UTC")
        .tz_convert(TIMEZONE)
        .tz_localize(None)
    )
    return obj


def sum_cols(df, cols):
    """Sum the given columns of a time-series table (0 if none exist)."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1)


def format_dates(ax):
    """Readable date labels, no overlap."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("Date (Danish local time)")


def load_network(run_name):
    """Load the solved network of a run, or return None if it does not exist."""
    folder = Path("results") / run_name / "networks"
    files = sorted(folder.glob("solved_*.nc")) or sorted(folder.glob("*.nc"))
    if not files:
        print(f"WARNING: no solved network found in {folder} - skipping this run.")
        return None
    print(f"Loading {run_name}: {files[-1]}")
    return pypsa.Network(files[-1])


# ---------------------------------------------------------------------------
# STEP 1 - Extract the numbers we need from each network
# ---------------------------------------------------------------------------
def extract(n):
    r = {}
    gens = n.generators
    re_gens = gens[gens.carrier.isin(RE_CARRIERS)]

    # Capacities (GW)
    r["capacity_GW"] = re_gens.groupby("carrier").p_nom_opt.sum() / 1e3

    # Generation used and generation available (MW), per carrier
    p = n.generators_t.p.reindex(columns=gens.index, fill_value=0.0)
    avail = n.get_switchable_as_dense("Generator", "p_max_pu").mul(gens.p_nom_opt, axis=1)
    r["gen_MW"] = to_local(p[re_gens.index].T.groupby(re_gens.carrier).sum().T)
    r["avail_MW"] = to_local(avail[re_gens.index].T.groupby(re_gens.carrier).sum().T)

    # EV system (MW or MWh)
    links, stores, loads = n.links, n.stores, n.loads
    chargers = links.index[links.carrier == "BEV charger"]
    v2g = links.index[links.carrier == "V2G"]
    ev_stores = stores.index[stores.carrier == "EV battery"]
    ev_loads = loads.index[loads.carrier == "land transport EV"]
    vipv = gens.index[gens.carrier == "VIPV"]

    ev = pd.DataFrame(index=n.snapshots)
    ev["charging_MW"] = sum_cols(n.links_t.p0, chargers)  # taken from grid
    ev["V2G_to_grid_MW"] = -sum_cols(n.links_t.p1, v2g)  # delivered to grid
    ev["driving_MW"] = sum_cols(n.loads_t.p, ev_loads)
    ev["VIPV_used_MW"] = sum_cols(p, vipv)
    ev["VIPV_available_MW"] = sum_cols(avail, vipv)
    e_nom = stores.loc[ev_stores, "e_nom_opt"].sum()
    ev["battery_MWh"] = sum_cols(n.stores_t.e, ev_stores)
    ev["SoC_percent"] = ev["battery_MWh"] / e_nom * 100 if e_nom > 0 else float("nan")
    r["ev"] = to_local(ev)

    # Average electricity price at the high-voltage (AC) nodes (EUR/MWh)
    ac = n.buses.index[n.buses.carrier == "AC"]
    r["price"] = to_local(n.buses_t.marginal_price.reindex(columns=ac).mean(axis=1))

    # Hours per snapshot (1 for hourly resolution)
    r["hours"] = to_local(n.snapshot_weightings.generators)

    # Total system cost (million EUR)
    cost = getattr(n, "objective", None)
    if cost is None or pd.isna(cost):
        try:
            cost = n.statistics.capex().sum() + n.statistics.opex().sum()
        except Exception:
            cost = float("nan")
    r["system_cost_MEUR"] = cost / 1e6

    return r


def summary(r):
    """Key numbers for one scenario (energy in GWh)."""
    h = r["hours"]
    ev = r["ev"]
    gwh = lambda s: (s * h).sum() / 1e3
    curtail = (r["avail_MW"] - r["gen_MW"]).clip(lower=0)

    grid_in = gwh(ev.charging_MW)
    vipv_used = gwh(ev.VIPV_used_MW)
    vipv_avail = gwh(ev.VIPV_available_MW)
    driving = gwh(ev.driving_MW)
    v2g = gwh(ev.V2G_to_grid_MW)

    return pd.Series(
        {
            "Total system cost [M EUR]": r["system_cost_MEUR"],
            "Solar capacity incl. VIPV [GW]": r["capacity_GW"].reindex(SOLAR_CARRIERS).sum(),
            "Wind capacity [GW]": r["capacity_GW"].drop(SOLAR_CARRIERS, errors="ignore").sum(),
            "VIPV capacity [MW]": r["capacity_GW"].get("VIPV", 0.0) * 1e3,
            "VIPV available [GWh]": vipv_avail,
            "VIPV used [GWh]": vipv_used,
            "VIPV curtailed [GWh]": vipv_avail - vipv_used,
            "VIPV utilisation [%]": 100 * vipv_used / vipv_avail if vipv_avail > 0 else float("nan"),
            "EV charging from grid [GWh]": grid_in,
            "EV driving demand [GWh]": driving,
            "V2G to grid [GWh]": v2g,
            "EV losses [GWh]": grid_in + vipv_used - driving - v2g,
            "Driving covered by VIPV [%]": 100 * vipv_used / driving if driving > 0 else float("nan"),
            "Solar + wind curtailed (excl. VIPV) [GWh]": gwh(curtail.drop(columns="VIPV", errors="ignore").sum(axis=1)),
            "SoC min [%]": ev.SoC_percent.min(),
            "SoC mean [%]": ev.SoC_percent.mean(),
            "Average electricity price [EUR/MWh]": (r["price"] * h).sum() / h.sum(),
        }
    )


# ---------------------------------------------------------------------------
# STEP 2 - Load both runs
# ---------------------------------------------------------------------------
results = {}
for label, run in RUNS.items():
    n = load_network(run)
    if n is not None:
        results[label] = extract(n)

if not results:
    raise SystemExit("No results found. Check the run names in RUNS.")

# ---------------------------------------------------------------------------
# STEP 3 - Summary table (printed and saved as CSV)
# ---------------------------------------------------------------------------
table = pd.DataFrame({label: summary(r) for label, r in results.items()})
if {"Baseline", "VIPV"} <= set(table.columns):
    table["Difference"] = table["VIPV"] - table["Baseline"]
table.to_csv(OUT_DIR / "summary_baseline_vs_vipv.csv")
print("\n=== Summary: baseline vs VIPV (June) ===")
print(table.round(2).to_string())

print("\nMaking plots...")

# ---------------------------------------------------------------------------
# PLOT 01 - Wind and solar capacities
# ---------------------------------------------------------------------------
cap = pd.DataFrame({label: r["capacity_GW"] for label, r in results.items()}).fillna(0.0)
cap = cap[(cap > 1e-3).any(axis=1)]  # hide technologies with zero capacity
fig, ax = plt.subplots(figsize=(8, 4))
cap.sort_values(cap.columns[0]).plot.barh(ax=ax, color=[COLORS[c] for c in cap.columns])
ax.set_xlabel("Installed capacity [GW]")
ax.set_ylabel("")
ax.set_title("Optimised wind and solar capacity")
save(fig, "01_capacities")

# ---------------------------------------------------------------------------
# PLOT 02 - EV power flows, first week (one panel per scenario)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.5 * len(results)), sharex=True, sharey=True)
axes = [axes] if len(results) == 1 else axes
for ax, (label, r) in zip(axes, results.items()):
    d = r["ev"].iloc[: 24 * ZOOM_DAYS]
    ax.plot(d.index, d.charging_MW, color="tab:green", label="Charging from grid")
    ax.plot(d.index, -d.V2G_to_grid_MW, color="tab:red", label="V2G to grid (negative)")
    ax.plot(d.index, d.driving_MW, color="black", lw=1, label="Driving demand")
    if d.VIPV_used_MW.sum() > 0:
        ax.plot(d.index, d.VIPV_used_MW, color="darkorange", label="VIPV used")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Power [MW]")
    ax.set_title(f"{label}: EV power flows (first week)")
    ax.legend(loc="upper right", fontsize=8)
format_dates(axes[-1])
save(fig, "02_ev_power_week")

# ---------------------------------------------------------------------------
# PLOT 03 - State of charge (month and week), both scenarios on one plot
# ---------------------------------------------------------------------------
for period, n_hours in [("month", None), ("week", 24 * ZOOM_DAYS)]:
    fig, ax = plt.subplots(figsize=(12, 4))
    for label, r in results.items():
        d = r["ev"] if n_hours is None else r["ev"].iloc[:n_hours]
        ax.plot(d.index, d.SoC_percent, color=COLORS[label], label=label)
    ax.set_ylim(0, 100)
    ax.set_ylabel("State of charge [%]")
    ax.set_title(f"EV fleet battery state of charge ({period})")
    ax.legend()
    format_dates(ax)
    save(fig, f"03_ev_soc_{period}")

# ---------------------------------------------------------------------------
# PLOT 04 - Average day: charging and V2G, both scenarios
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for label, r in results.items():
    daily = r["ev"].groupby(r["ev"].index.hour).mean()
    ax1.plot(daily.index, daily.charging_MW, color=COLORS[label], label=f"Charging from grid - {label}")
    ax2.plot(daily.index, daily.V2G_to_grid_MW, color=COLORS[label], label=f"V2G to grid - {label}")
    if daily.VIPV_used_MW.sum() > 0:
        ax1.plot(daily.index, daily.VIPV_used_MW, color="darkorange", ls=":", label="VIPV used")
first = next(iter(results.values()))["ev"]
driving_daily = first.groupby(first.index.hour).driving_MW.mean()
ax1.plot(driving_daily.index, driving_daily, color="black", lw=1, label="Driving demand")
for ax, title in [(ax1, "Charging"), (ax2, "V2G")]:
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (Danish local time)")
    ax.set_title(f"Average day: {title}")
    ax.legend(fontsize=8)
ax1.set_ylabel("Average power [MW]")
save(fig, "04_ev_average_day")

# ---------------------------------------------------------------------------
# PLOT 05 - Wind and solar generation, first week (one panel per scenario)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.5 * len(results)), sharex=True, sharey=True)
axes = [axes] if len(results) == 1 else axes
for ax, (label, r) in zip(axes, results.items()):
    g = r["gen_MW"].iloc[: 24 * ZOOM_DAYS] / 1e3
    g = g.loc[:, g.sum() > 0]  # hide empty technologies
    g.plot.area(ax=ax, lw=0)
    ax.set_ylabel("Generation [GW]")
    ax.set_title(f"{label}: wind and solar generation (first week)")
    ax.legend(loc="upper right", fontsize=8)
format_dates(axes[-1])
save(fig, "05_wind_solar_generation_week")

# ---------------------------------------------------------------------------
# VIPV-SPECIFIC PLOTS (only if the VIPV run exists and contains VIPV)
# ---------------------------------------------------------------------------
if "VIPV" in results and results["VIPV"]["ev"].VIPV_available_MW.sum() > 0:
    ev = results["VIPV"]["ev"]

    # PLOT 06 - VIPV available vs used, first week
    d = ev.iloc[: 24 * ZOOM_DAYS]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(d.index, d.VIPV_available_MW, color="navajowhite", label="VIPV available")
    ax.plot(d.index, d.VIPV_used_MW, color="darkorange", label="VIPV used")
    ax.set_ylabel("Power [MW]")
    ax.set_title("VIPV: available vs used generation (first week) - gap = curtailed")
    ax.legend(loc="upper right")
    format_dates(ax)
    save(fig, "06_vipv_available_vs_used_week")

    # PLOT 07 - VIPV average day
    daily = ev.groupby(ev.index.hour).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(daily.index, daily.VIPV_available_MW, color="navajowhite", label="VIPV available")
    ax.plot(daily.index, daily.VIPV_used_MW, color="darkorange", label="VIPV used")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (Danish local time)")
    ax.set_ylabel("Average power [MW]")
    ax.set_title("VIPV: average day")
    ax.legend()
    save(fig, "07_vipv_average_day")

    # PLOT 08 - Change in grid charging caused by VIPV (average day)
    if "Baseline" in results:
        base = results["Baseline"]["ev"]
        diff = (ev.charging_MW - base.charging_MW).groupby(ev.index.hour).mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(diff.index, diff.values, color=["tab:green" if v > 0 else "tab:red" for v in diff.values])
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour of day (Danish local time)")
        ax.set_ylabel("Change in grid charging [MW]")
        ax.set_title("Grid charging: VIPV minus baseline (average day)\nnegative = VIPV reduces grid charging")
        save(fig, "08_change_in_grid_charging")

# ---------------------------------------------------------------------------
# PLOT 09 - EV energy balance for the month (GWh), both scenarios
# ---------------------------------------------------------------------------
rows = [
    "EV charging from grid [GWh]", "VIPV used [GWh]", "EV driving demand [GWh]",
    "V2G to grid [GWh]", "EV losses [GWh]",
]
bal = table.loc[rows, [c for c in results]]
bal.index = [i.replace(" [GWh]", "") for i in bal.index]
fig, ax = plt.subplots(figsize=(9, 4))
bal.plot.bar(ax=ax, color=[COLORS[c] for c in bal.columns], rot=0)
ax.set_ylabel("Energy [GWh]")
ax.set_title("EV energy balance for the modelled month")
ax.tick_params(axis="x", labelsize=8)
save(fig, "09_ev_energy_balance")

# ---------------------------------------------------------------------------
# PLOT 10 - Curtailment by technology (GWh), both scenarios
# ---------------------------------------------------------------------------
curt = {}
for label, r in results.items():
    c = (r["avail_MW"] - r["gen_MW"]).clip(lower=0).mul(r["hours"], axis=0).sum() / 1e3
    curt[label] = c
curt = pd.DataFrame(curt).fillna(0.0)
curt = curt[(curt > 1e-3).any(axis=1)]
if not curt.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    curt.plot.bar(ax=ax, color=[COLORS[c] for c in curt.columns], rot=0)
    ax.set_ylabel("Curtailed energy [GWh]")
    ax.set_title("Curtailed (thrown away) wind and solar energy")
    save(fig, "10_curtailment")

# ---------------------------------------------------------------------------
# PLOT 11 - Does EV charging follow the sun? (average day)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
base_label = next(iter(results))
solar = results[base_label]["gen_MW"].reindex(columns=SOLAR_CARRIERS, fill_value=0.0).sum(axis=1)
solar_daily = solar.groupby(solar.index.hour).mean() / 1e3
ax.fill_between(solar_daily.index, solar_daily, color="gold", alpha=0.5, label=f"Solar generation ({base_label})")
ax.set_ylabel("Solar generation [GW]")
ax2 = ax.twinx()
for label, r in results.items():
    daily = r["ev"].groupby(r["ev"].index.hour).charging_MW.mean()
    ax2.plot(daily.index, daily, color=COLORS[label], label=f"EV charging - {label}")
ax2.set_ylabel("EV charging from grid [MW]")
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day (Danish local time)")
ax.set_title("EV charging vs solar generation (average day)")
lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax.legend(lines, labels, loc="upper left", fontsize=8)
save(fig, "11_charging_vs_solar")

# ---------------------------------------------------------------------------
# PLOT 12 - Average electricity price (average day)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
for label, r in results.items():
    daily = r["price"].groupby(r["price"].index.hour).mean()
    ax.plot(daily.index, daily, color=COLORS[label], label=label)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day (Danish local time)")
ax.set_ylabel("Electricity price [EUR/MWh]")
ax.set_title("Average electricity price by hour")
ax.legend()
save(fig, "12_electricity_price_average_day")

print(f"\nDone! Open the folder: {OUT_DIR}")