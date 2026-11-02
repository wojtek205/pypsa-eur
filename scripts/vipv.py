"""
Vehicle-integrated PV (VIPV) for PyPSA-Eur - cars, buses and trucks.

Save this file as:  pypsa-eur/scripts/vipv.py   (replaces the old version)

Adds one VIPV generator per node and fleet, connected to that fleet's EV battery bus,
so VIPV electricity charges the vehicles directly (it never feeds the grid directly).

Capacity per node  = number of EVs * capacity per vehicle
Number of EVs      = charger capacity / charger power per vehicle
Hourly profile     = rooftop solar profile of the same node * derating_factor * conversion_efficiency

Fleets:
  cars    -> carrier "VIPV"        (charger "BEV charger",   power = bev_charge_rate)
  buses   -> carrier "VIPV bus"    (charger "bus charger",   power = heavy_transport: bus: charger_kw)
  trucks  -> carrier "VIPV truck"  (charger "truck charger", power = heavy_transport: truck: charger_kw)

Config (sector: vipv:):
    enable: true
    capacity_per_car_kw: 0.4
    capacity_per_bus_kw: 4.0
    capacity_per_truck_kw: 5.0
    derating_factor: 0.75
    conversion_efficiency: 0.95
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_vipv(n, sector_options):
    """Add vehicle-integrated PV (VIPV) to the EV battery bus of every fleet at every node."""
    cfg = sector_options.get("vipv", {}) or {}
    if not cfg.get("enable", False):
        logger.info("VIPV disabled - skipping.")
        return

    derating = cfg.get("derating_factor", 1.0)
    efficiency = cfg.get("conversion_efficiency", 1.0)  # DC-DC converter (MPPT) losses
    logger.info(f"VIPV settings: derating={derating}, conversion_efficiency={efficiency}")

    heavy = sector_options.get("heavy_transport", {}) or {}
    fleets = {
        # name: (charger carrier, MW charger per vehicle, kW VIPV per vehicle, VIPV carrier)
        "car": ("BEV charger", sector_options["bev_charge_rate"],
                cfg.get("capacity_per_car_kw", 0.4), "VIPV"),
    }
    if heavy.get("enable", False):
        fleets["bus"] = ("bus charger", heavy["bus"]["charger_kw"] / 1e3,
                         cfg.get("capacity_per_bus_kw", 0.0), "VIPV bus")
        fleets["truck"] = ("truck charger", heavy["truck"]["charger_kw"] / 1e3,
                           cfg.get("capacity_per_truck_kw", 0.0), "VIPV truck")

    rooftop = n.generators[n.generators.carrier == "solar rooftop"]
    if rooftop.empty:
        logger.warning("No rooftop solar found - VIPV not added.")
        return
    rooftop_by_location = pd.Series(
        rooftop.index, index=n.buses.location.reindex(rooftop.bus).values
    )
    p_max_pu_all = n.get_switchable_as_dense("Generator", "p_max_pu")

    for fleet, (charger_carrier, rate, kw_per_vehicle, vipv_carrier) in fleets.items():
        chargers = n.links[n.links.carrier == charger_carrier]
        if chargers.empty or kw_per_vehicle <= 0:
            logger.info(f"VIPV {fleet}: no chargers or 0 kW per vehicle - skipped.")
            continue
        if vipv_carrier not in n.carriers.index:
            n.add("Carrier", vipv_carrier)

        total_mw, total_ev = 0.0, 0.0
        for _, charger in chargers.iterrows():
            location = n.buses.at[charger.bus1, "location"]
            if location not in rooftop_by_location.index:
                logger.warning(f"No rooftop solar at {location} - no VIPV {fleet} there.")
                continue
            n_evs = charger.p_nom / rate
            p_nom = n_evs * kw_per_vehicle / 1e3  # kW -> MW
            profile = p_max_pu_all[rooftop_by_location[location]] * derating * efficiency

            n.add(
                "Generator",
                f"{location} {vipv_carrier}",
                bus=charger.bus1,  # the fleet's EV battery bus
                carrier=vipv_carrier,
                p_nom=p_nom,
                p_nom_extendable=False,  # fixed capacity from the formula
                marginal_cost=0.0,
                p_max_pu=profile,  # rooftop PV shape, reduced by losses
            )
            total_mw += p_nom
            total_ev += n_evs

        logger.info(
            f"VIPV {fleet}: {total_ev:,.0f} EVs x {kw_per_vehicle} kW -> {total_mw:,.1f} MW"
        )