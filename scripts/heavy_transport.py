"""
Heavy-duty road transport (buses and trucks) as separate electric fleets.

Save this file as:  pypsa-eur/scripts/heavy_transport.py

WHAT IT DOES
------------
PyPSA-Eur puts ALL road transport energy (cars, vans, buses, trucks, ...) on the
passenger-car EV battery. This function moves the bus and truck share to their own fleets:

    GRID (low voltage) --charger--> [bus EV battery bus]  --> bus driving demand
                                     + bus battery store  <-- VIPV bus (added in vipv.py)
    (no V2G for buses and trucks)

For each node and each fleet (bus, truck):
  1. demand share   = fleet energy / (total road + non-electric rail)   [JRC-IDEES, per node]
  2. the car "land transport EV" demand is reduced by these shares (total demand unchanged)
  3. a new bus, charger (depot availability profile), battery store and demand are added
  4. electric vehicles = national number of vehicles x electric share x node weight
     (node weight = same population-based split as the car chargers)

Config (sector: heavy_transport:), see config.dk-hd-base-2030-jun.yaml
"""

import logging

import pandas as pd

from scripts._helpers import get

logger = logging.getLogger(__name__)

# JRC-IDEES columns in pop_weighted_energy_totals.csv (TWh/a per node)
DEMAND_COLUMNS = {
    "bus": "total other road passenger",  # buses and coaches
    "truck": "total heavy duty road freight",  # heavy goods vehicles
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
        logger.info("Heavy transport disabled - buses and trucks stay in the car demand.")
        return

    electric_share = get(sector_options["land_transport_electric_share"], current_horizon)
    if electric_share <= 0:
        logger.info("Electric share is 0 - no electric heavy transport added.")
        return

    car_loads = n.loads.index[n.loads.carrier == "land transport EV"]
    car_chargers = n.links[n.links.carrier == "BEV charger"]
    if car_loads.empty or car_chargers.empty:
        logger.warning("No car EV components found - heavy transport not added.")
        return

    # --- demand shares per node (from JRC-IDEES, already prepared by PyPSA-Eur) ---
    energy = pd.read_csv(inputs["pop_weighted_energy_totals"], index_col=0)
    base = energy["total road"] + energy["total rail"] - energy["electricity rail"]

    # node -> names of the car components at that node
    car_load_at = pd.Series(car_loads, index=n.loads.loc[car_loads, "bus"].map(n.buses.location))
    charger_at = pd.Series(car_chargers.index, index=n.buses.location[car_chargers.bus1].values)
    node_weight = car_chargers.p_nom.groupby(n.buses.location[car_chargers.bus1].values).sum()
    node_weight = node_weight / node_weight.sum()

    if len({loc[:2] for loc in node_weight.index}) > 1:
        logger.warning(
            "More than one country: national vehicle numbers in the config are "
            "distributed over ALL nodes. Only correct for single-country models."
        )

    total_share = pd.Series(0.0, index=node_weight.index)

    for fleet, column in DEMAND_COLUMNS.items():
        fcfg = cfg[fleet]
        share = (energy[column] / base).reindex(node_weight.index).fillna(0.0)
        if share.sum() == 0:
            raise ValueError(
                f"No {fleet} demand found for nodes {list(node_weight.index)} in "
                f"pop_weighted_energy_totals.csv - check that node names match."
            )
        total_share += share

        n_vehicles = fcfg["number_vehicles"] * electric_share * node_weight
        avail = depot_availability(
            n.snapshots,
            fcfg.get("depot_hours", [19, 6]),
            fcfg.get("availability_depot", 0.9),
            fcfg.get("availability_road", 0.2),
            cfg.get("timezone", "Europe/Copenhagen"),
        )

        carrier_bus = f"{fleet} EV battery"
        carrier_charger = f"{fleet} charger"
        carrier_load = f"land transport {fleet} EV"
        for c in [carrier_bus, carrier_charger, carrier_load]:
            if c not in n.carriers.index:
                n.add("Carrier", c)

        for node in node_weight.index:
            car_load = car_load_at[node]
            fleet_demand = n.loads_t.p_set[car_load] * share[node]  # MW, time series
            ev_bus = f"{node} {fleet} EV battery"
            p_charger = n_vehicles[node] * fcfg["charger_kw"] / 1e3  # kW -> MW

            n.add("Bus", ev_bus, location=node, carrier=carrier_bus, unit="MWh_el")
            n.add(
                "Load",
                f"{node} land transport {fleet} EV",
                bus=ev_bus,
                carrier=carrier_load,
                p_set=fleet_demand,
            )
            n.add(
                "Link",
                f"{node} {fleet} charger",
                bus0=car_chargers.at[charger_at[node], "bus0"],  # same LV bus as cars
                bus1=ev_bus,
                carrier=carrier_charger,
                p_nom=p_charger,
                p_max_pu=avail,
                efficiency=sector_options["bev_charge_efficiency"],
                lifetime=1,
            )
            n.add(
                "Store",
                f"{node} {fleet} EV battery",
                bus=ev_bus,
                carrier=carrier_bus,
                e_nom=n_vehicles[node] * fcfg["battery_kwh"] / 1e3,  # kWh -> MWh
                e_cyclic=True,
            )

        energy_month = (
            n.loads_t.p_set[[f"{nd} land transport {fleet} EV" for nd in node_weight.index]]
            .mul(n.snapshot_weightings.generators, axis=0)
            .sum()
            .sum()
        )
        logger.info(
            f"Heavy transport {fleet}: {n_vehicles.sum():,.0f} EVs, "
            f"{100 * share.mean():.1f}% of land transport demand, "
            f"{n_vehicles.sum() * fcfg['charger_kw'] / 1e3:,.0f} MW chargers, "
            f"{n_vehicles.sum() * fcfg['battery_kwh'] / 1e3:,.0f} MWh batteries, "
            f"{energy_month / 1e3:,.1f} GWh demand in modelled period"
        )

    # --- remove the bus + truck share from the car demand (total stays the same) ---
    for node in node_weight.index:
        n.loads_t.p_set[car_load_at[node]] *= 1 - total_share[node]
    logger.info(f"Car EV demand reduced by {100 * total_share.mean():.1f}% (moved to buses/trucks)")