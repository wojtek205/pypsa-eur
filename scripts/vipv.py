"""
Vehicle-integrated PV (VIPV) for PyPSA-Eur.

Save this file as:  pypsa-eur/scripts/vipv.py

Adds one VIPV generator per node, connected to the EV battery bus, so that
VIPV electricity charges the cars directly (it never feeds the grid directly).

Capacity per node  = number of EVs * capacity_per_car_kw
Number of EVs      = BEV charger capacity / bev_charge_rate
Hourly profile     = rooftop solar profile of the same node * derating_factor

Config (in config.denmark_vipv.yaml):
    sector:
      vipv:
        enable: true
        capacity_per_car_kw: 0.4
        derating_factor: 1.0
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_vipv(n, sector_options):
    """Add vehicle-integrated PV (VIPV) to the EV battery bus at every node."""
    cfg = sector_options.get("vipv", {}) or {}
    if not cfg.get("enable", False):
        logger.info("VIPV disabled - skipping.")
        return

    kw_per_car = cfg.get("capacity_per_car_kw", 0.4)
    derating = cfg.get("derating_factor", 1.0)
    charge_rate = sector_options["bev_charge_rate"]  # MW charger per car

    chargers = n.links[n.links.carrier == "BEV charger"]
    rooftop = n.generators[n.generators.carrier == "solar rooftop"]
    if chargers.empty or rooftop.empty:
        logger.warning("No BEV chargers or rooftop solar found - VIPV not added.")
        return

    rooftop_by_location = pd.Series(
        rooftop.index, index=n.buses.location.reindex(rooftop.bus).values
    )
    p_max_pu_all = n.get_switchable_as_dense("Generator", "p_max_pu")

    if "VIPV" not in n.carriers.index:
        n.add("Carrier", "VIPV")

    total_mw = 0.0
    for charger_name, charger in chargers.iterrows():
        location = n.buses.at[charger.bus1, "location"]
        if location not in rooftop_by_location.index:
            logger.warning(f"No rooftop solar at {location} - no VIPV added there.")
            continue

        n_evs = charger.p_nom / charge_rate
        p_nom = n_evs * kw_per_car / 1e3  # kW -> MW
        profile = p_max_pu_all[rooftop_by_location[location]] * derating

        n.add(
            "Generator",
            f"{location} VIPV",
            bus=charger.bus1,  # the EV battery bus
            carrier="VIPV",
            p_nom=p_nom,
            p_nom_extendable=False,  # fixed capacity from the formula
            marginal_cost=0.0,
            p_max_pu=profile,  # same shape as rooftop solar
        )
        total_mw += p_nom
        logger.info(f"VIPV at {location}: {n_evs:,.0f} EVs -> {p_nom:,.1f} MW")

    logger.info(f"VIPV total capacity: {total_mw:,.1f} MW")