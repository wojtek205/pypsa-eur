"""
Plot the results of the dk-baseline PyPSA-Eur run.

Save this file as:   pypsa-eur/analysis/plot_dk_results.py
Run it from the pypsa-eur folder with:
    pixi run python analysis/plot_dk_results.py

Plots and CSV tables are saved to: results/dk-baseline/my_plots/
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

# ---------------------------------------------------------------------------
# SETTINGS - change these if needed
# ---------------------------------------------------------------------------
RUN_NAME = "dk-baseline"
NETWORK_DIR = Path("results") / RUN_NAME / "networks"
OUT_DIR = Path("results") / RUN_NAME / "my_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZOOM_DAYS = 7  # length of the "zoomed in" plots (first week)


def save(fig, name):
    """Save a figure as PNG and close it."""
    path = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def by_carrier(df, words):
    """Return rows whose 'carrier' contains any of the given words."""
    pattern = "|".join(words)
    return df[df.carrier.str.contains(pattern, case=False, na=False)]


# ---------------------------------------------------------------------------
# STEP 1 - Load the solved network
# ---------------------------------------------------------------------------
files = sorted(NETWORK_DIR.glob("*.nc"))
if not files:
    raise FileNotFoundError(f"No .nc file found in {NETWORK_DIR}. Did the solve finish?")
network_file = files[-1]
print(f"\nLoading solved network: {network_file}")
n = pypsa.Network(network_file)
print(f"Snapshots: {n.snapshots[0]} to {n.snapshots[-1]} ({len(n.snapshots)} hours)")

# ---------------------------------------------------------------------------
# STEP 2 - Overview: what is inside the model?
# (Read this printout once - it shows the names of all technologies.)
# ---------------------------------------------------------------------------
print("\n=== Electricity nodes (clusters) ===")
print(list(n.buses.index[n.buses.carrier == "AC"]))

components = {
    "Generators": n.generators,
    "Links": n.links,
    "Stores": n.stores,
    "StorageUnits": n.storage_units,
    "Loads": n.loads,
}
print("\n=== Technologies (carriers) per component type ===")
for name, df in components.items():
    carriers = sorted(df.carrier.dropna().unique())
    print(f"{name}: {carriers}")

# ---------------------------------------------------------------------------
# STEP 3 - Installed capacities of wind and solar
# p_nom_opt = capacity chosen by the optimisation (MW)
# ---------------------------------------------------------------------------
print("\n=== Wind and solar capacities ===")
re_gen = by_carrier(n.generators, ["solar", "wind"])
re_gen = re_gen[~re_gen.carrier.str.contains("thermal", case=False)]  # skip solar thermal
capacities = re_gen.groupby("carrier")["p_nom_opt"].sum() / 1e3  # MW -> GW
print(capacities.round(2).to_string())
capacities.to_csv(OUT_DIR / "capacities_wind_solar_GW.csv")

fig, ax = plt.subplots(figsize=(8, 4))
capacities.sort_values().plot.barh(ax=ax, color="tab:blue")
ax.set_xlabel("Installed capacity [GW]")
ax.set_title("Optimised wind and solar capacity - Denmark 2030")
save(fig, "01_capacities_wind_solar")

# ---------------------------------------------------------------------------
# STEP 4 - Find the EV components
# ---------------------------------------------------------------------------
ev_charger = by_carrier(n.links, ["BEV charger"])
ev_v2g = by_carrier(n.links, ["V2G"])
ev_store = by_carrier(n.stores, ["EV battery"])
ev_load = by_carrier(n.loads, ["land transport EV"])

print("\n=== EV components found ===")
print(f"Chargers (grid -> car):  {list(ev_charger.index)}")
print(f"V2G links (car -> grid): {list(ev_v2g.index)}")
print(f"EV battery stores:       {list(ev_store.index)}")
print(f"EV driving demand loads: {list(ev_load.index)}")

if ev_store.empty:
    print("\nWARNING: No EV battery found. Check the carrier names printed in STEP 2")
    print("and adjust the words in STEP 4. Skipping EV plots.")
else:
    # --- Fleet data (the values from your config) ---
    fleet_energy_MWh = ev_store.e_nom_opt.sum()
    charger_power_MW = ev_charger.p_nom_opt.sum()
    print(f"\nTotal flexible EV battery capacity: {fleet_energy_MWh:,.0f} MWh")
    print(f"Total charger power:                {charger_power_MW:,.0f} MW")

    # --- Time series (MW) ---
    charging = n.links_t.p0[ev_charger.index].sum(axis=1)  # power taken from grid
    if not ev_v2g.empty:
        v2g_to_grid = -n.links_t.p1[ev_v2g.index].sum(axis=1)  # power fed to grid
    else:
        v2g_to_grid = pd.Series(0.0, index=n.snapshots)
    driving = n.loads_t.p[ev_load.index].sum(axis=1) if not ev_load.empty else None
    energy = n.stores_t.e[ev_store.index].sum(axis=1)  # MWh in the batteries
    soc = energy / fleet_energy_MWh * 100  # %

    ev_table = pd.DataFrame(
        {
            "charging_MW": charging,
            "V2G_to_grid_MW": v2g_to_grid,
            "driving_demand_MW": driving if driving is not None else float("nan"),
            "battery_energy_MWh": energy,
            "SoC_percent": soc,
        }
    )
    ev_table.to_csv(OUT_DIR / "ev_timeseries.csv")
    print("\n=== EV summary for the whole period ===")
    print(f"Energy charged:       {charging.sum() / 1e3:,.1f} GWh")
    print(f"Energy returned (V2G): {v2g_to_grid.sum() / 1e3:,.1f} GWh")
    if driving is not None:
        print(f"Energy for driving:   {driving.sum() / 1e3:,.1f} GWh")
    print(f"SoC min / mean / max: {soc.min():.0f}% / {soc.mean():.0f}% / {soc.max():.0f}%")

    for label, data in [("month", ev_table), ("week", ev_table.iloc[: 24 * ZOOM_DAYS])]:
        # --- Plot: charging, V2G and driving demand ---
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(data.index, data.charging_MW, label="Charging (grid -> EVs)", color="tab:green")
        ax.plot(data.index, -data.V2G_to_grid_MW, label="V2G (EVs -> grid), negative", color="tab:red")
        if driving is not None:
            ax.plot(data.index, data.driving_demand_MW, label="Driving demand", color="black", lw=1)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("Power [MW]")
        ax.set_title(f"EV charging, V2G and driving demand ({label})")
        ax.legend(loc="upper right")
        save(fig, f"02_ev_power_{label}")

        # --- Plot: state of charge ---
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(data.index, data.SoC_percent, color="tab:purple")
        ax.set_ylim(0, 100)
        ax.set_ylabel("State of charge [%]")
        ax.set_title(f"EV fleet battery state of charge ({label})")
        save(fig, f"03_ev_soc_{label}")

    # --- Plot: average day (shows the daily pattern clearly) ---
    daily = ev_table.groupby(ev_table.index.hour).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(daily.index, daily.charging_MW, label="Charging", color="tab:green")
    ax.plot(daily.index, -daily.V2G_to_grid_MW, label="V2G (negative)", color="tab:red")
    if driving is not None:
        ax.plot(daily.index, daily.driving_demand_MW, label="Driving demand", color="black")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average power [MW]")
    ax.set_title("Average daily EV profile")
    ax.legend()
    save(fig, "04_ev_average_day")

# ---------------------------------------------------------------------------
# STEP 5 - Solar and wind generation over time (useful for comparing with EV charging)
# ---------------------------------------------------------------------------
gen_by_carrier = n.generators_t.p[re_gen.index].T.groupby(re_gen.carrier).sum().T
fig, ax = plt.subplots(figsize=(12, 4))
(gen_by_carrier.iloc[: 24 * ZOOM_DAYS] / 1e3).plot.area(ax=ax, lw=0)
ax.set_ylabel("Generation [GW]")
ax.set_title("Wind and solar generation (first week)")
ax.legend(loc="upper right", fontsize=8)
save(fig, "05_wind_solar_generation_week")

print(f"\nDone! Open the folder: {OUT_DIR}")