"""
Separate road fleets: vans, buses and trucks (in addition to PyPSA-Eur's passenger cars).

Save this file as:  pypsa-eur/scripts/heavy_transport.py   (replaces the old version)

WHAT IT DOES
------------
PyPSA-Eur puts ALL road transport energy (cars, vans, buses, trucks, ...) on the
passenger-car EV battery. This function moves the van, bus and truck shares to their own fleets:

    GRID --car charger--> [car EV battery]   --> cars driving        (V2G, car profiles)
         --van charger--> [van EV battery]   --> vans driving        (V2G, car profiles)
         --bus charger--> [bus EV battery]   --> buses driving       (no V2G, depot charging)
       --truck charger--> [truck EV battery] --> trucks driving      (no V2G, depot charging)

For each node and each fleet in the config:
  1. demand share   = fleet energy / (total road + non-electric rail)   [JRC-IDEES, per node]
  2. the car "land transport EV" demand is reduced by these shares (total demand unchanged)
  3. a new bus, charger, battery store and demand are added (+ V2G link if v2g: true)
  4. electric vehicles = national number of vehicles x electric share x node weight

Two charging types (config: charging):
  "car"   -> same plug-in availability and morning SoC rule as PyPSA-Eur's cars (for vans)
  "depot" -> plugged in mostly at night (depot_hours), for buses and trucks

Config (sector: heavy_transport:), see config.dk-hd-base-2030-jun.yaml
"""

import logging

import pandas as pd

from scripts._helpers import get

logger = logging.getLogger(__name__)

# JRC-IDEES columns in pop_weighted_energy_totals.csv (TWh/a per node)
DEMAND_COLUMNS = {
    "van": "total light duty road freight",  # light goods vehicles <= 3.5 t
    "bus": "total other road passenger",  # buses and coaches
    "truck": "total heavy duty road freight",  # heavy goods vehicles > 3.5 t
}


def depot_availability(snapshots, hours, avail_depot, avail_road, timezone):
    """Share of the fleet plugged in: high during depot hours (local time), low otherwise."""
    local_hour = (
        pd.DatetimeIndex(snapshots).tz_localize("UTC").tz_convert(timezone).hour
    )
    start, end = hours
    if start > end:  # e.g. 19 -> 6 (over midnight)
        at_depot = (local_hour >= start) | (local_hour < end)
    else:
        at_depot = (local_hour >= start) & (local_hour < end)
    return pd.Series(
        [avail_depot if d else avail_road for d in at_depot], index=snapshots
    )


def add_heavy_transport(n, inputs, sector_options, current_horizon):
    cfg = sector_options.get("heavy_transport", {}) or {}
    if not cfg.get("enable", False):
        logger.info("Heavy transport disabled - all road transport stays in the car demand.")
        return

    electric_share = get(sector_options["land_transport_electric_share"], current_horizon)
    if electric_share <= 0:
        logger.info("Electric share is 0 - no electric fleets added.")
        return

    car_loads = n.loads.index[n.loads.carrier == "land transport EV"]
    car_chargers = n.links[n.links.carrier == "BEV charger"]
    car_stores = n.stores[n.stores.carrier == "EV battery"]
    if car_loads.empty or car_chargers.empty:
        logger.warning("No car EV components found - fleets not added.")
        return

    # --- demand shares per node (from JRC-IDEES, already prepared by PyPSA-Eur) ---
    energy = pd.read_csv(inputs["pop_weighted_energy_totals"], index_col=0)
    base = energy["total road"] + energy["total rail"] - energy["electricity rail"]

    # node -> names of the car components at that node
    location = n.buses.location
    car_load_at = pd.Series(car_loads, index=n.loads.loc[car_loads, "bus"].map(location))
    charger_at = pd.Series(car_chargers.index, index=location[car_chargers.bus1].values)
    store_at = pd.Series(car_stores.index, index=location[car_stores.bus].values)
    node_weight = car_chargers.p_nom.groupby(location[car_chargers.bus1].values).sum()
    node_weight = node_weight / node_weight.sum()

    # car time series, reused for fleets with charging: "car"
    car_avail = n.get_switchable_as_dense("Link", "p_max_pu")[car_chargers.index]
    car_min_soc = n.get_switchable_as_dense("Store", "e_min_pu")[car_stores.index]

    if len({loc[:2] for loc in node_weight.index}) > 1:
        logger.warning(
            "More than one country: national vehicle numbers in the config are "
            "distributed over ALL nodes. Only correct for single-country models."
        )

    total_share = pd.Series(0.0, index=node_weight.index)

    for fleet, column in DEMAND_COLUMNS.items():
        if fleet not in cfg:
            continue  # fleet not in config -> stays in the car demand
        fcfg = cfg[fleet]
        share = (energy[column] / base).reindex(node_weight.index).fillna(0.0)
        if share.sum() == 0:
            raise ValueError(
                f"No {fleet} demand found for nodes {list(node_weight.index)} in "
                f"pop_weighted_energy_totals.csv - check that node names match."
            )
        total_share += share

        n_vehicles = fcfg["number_vehicles"] * electric_share * node_weight
        charging = fcfg.get("charging", "depot")
        flex_share = fcfg.get("dsm_availability", 1.0)  # share of fleet with smart charging/V2G
        if charging == "depot":
            depot = depot_availability(
                n.snapshots,
                fcfg.get("depot_hours", [19, 6]),
                fcfg.get("availability_depot", 0.9),
                fcfg.get("availability_road", 0.2),
                cfg.get("timezone", "Europe/Copenhagen"),
            )

        carriers = {
            "bus": f"{fleet} EV battery",
            "charger": f"{fleet} charger",
            "load": f"land transport {fleet} EV",
            "v2g": f"{fleet} V2G",
        }
        for c in carriers.values():
            if c not in n.carriers.index:
                n.add("Carrier", c)

        for node in node_weight.index:
            car_load = car_load_at[node]
            fleet_demand = n.loads_t.p_set[car_load] * share[node]  # MW, time series
            ev_bus = f"{node} {fleet} EV battery"
            p_charger = n_vehicles[node] * fcfg["charger_kw"] / 1e3  # kW -> MW

            if charging == "car":  # same plug-in profile and morning SoC rule as cars
                avail = car_avail[charger_at[node]]
                min_soc = car_min_soc[store_at[node]] if node in store_at.index else 0.0
            else:  # depot charging
                avail = depot
                min_soc = fcfg.get("min_soc", 0.0)

            n.add("Bus", ev_bus, location=node, carrier=carriers["bus"], unit="MWh_el")
            n.add(
                "Load",
                f"{node} land transport {fleet} EV",
                bus=ev_bus,
                carrier=carriers["load"],
                p_set=fleet_demand,
            )
            n.add(
                "Link",
                f"{node} {fleet} charger",
                bus0=car_chargers.at[charger_at[node], "bus0"],  # same LV bus as cars
                bus1=ev_bus,
                carrier=carriers["charger"],
                p_nom=p_charger,
                p_max_pu=avail,
                efficiency=sector_options["bev_charge_efficiency"],
                lifetime=1,
            )
            n.add(
                "Store",
                f"{node} {fleet} EV battery",
                bus=ev_bus,
                carrier=carriers["bus"],
                e_nom=n_vehicles[node] * fcfg["battery_kwh"] / 1e3 * flex_share,  # kWh -> MWh
                e_cyclic=True,
                e_min_pu=min_soc,
            )
            if fcfg.get("v2g", False):  # same set-up as PyPSA-Eur's car V2G
                n.add(
                    "Link",
                    f"{node} {fleet} V2G",
                    bus0=ev_bus,
                    bus1=car_chargers.at[charger_at[node], "bus0"],
                    carrier=carriers["v2g"],
                    p_nom=p_charger * flex_share,
                    p_max_pu=avail,
                    efficiency=sector_options["bev_charge_efficiency"],
                    lifetime=1,
                )

        energy_period = (
            n.loads_t.p_set[[f"{nd} land transport {fleet} EV" for nd in node_weight.index]]
            .mul(n.snapshot_weightings.generators, axis=0)
            .sum()
            .sum()
        )
        logger.info(
            f"Fleet {fleet}: {n_vehicles.sum():,.0f} EVs, "
            f"{100 * share.mean():.1f}% of land transport demand, "
            f"charging={charging}, V2G={fcfg.get('v2g', False)}, "
            f"{n_vehicles.sum() * fcfg['charger_kw'] / 1e3:,.0f} MW chargers, "
            f"{n_vehicles.sum() * fcfg['battery_kwh'] / 1e3 * flex_share:,.0f} MWh batteries, "
            f"{energy_period / 1e3:,.1f} GWh demand in modelled period"
        )

    # --- remove the moved shares from the car demand (total stays the same) ---
    for node in node_weight.index:
        n.loads_t.p_set[car_load_at[node]] *= 1 - total_share[node]
    logger.info(f"Car EV demand reduced by {100 * total_share.mean():.1f}% (moved to other fleets)")