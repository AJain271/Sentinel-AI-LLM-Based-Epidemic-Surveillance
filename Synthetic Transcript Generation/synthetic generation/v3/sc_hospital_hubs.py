"""
South Carolina hospital hub definitions and geo-randomization.

Each hub represents a major medical centre.  When generating a synthetic
transcript the pipeline assigns the patient to a hub (or picks one at
random) and jitters the lat/lng within a configurable radius so the map
shows a realistic scatter pattern around the hospital rather than every
patient sitting on the same pin.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

# ─── Hub definitions ─────────────────────────────────────────────────────────

SC_HOSPITAL_HUBS: List[Dict] = [
    {"hub_id": "H1", "name": "AnMed Health Medical Center",   "city": "Anderson",     "state": "SC", "lat": 34.526, "lng": -82.646},
    {"hub_id": "H2", "name": "Prisma Health Greenville",      "city": "Greenville",   "state": "SC", "lat": 34.819, "lng": -82.416},
    {"hub_id": "H3", "name": "Prisma Health Richland",        "city": "Columbia",     "state": "SC", "lat": 34.032, "lng": -81.033},
    {"hub_id": "H4", "name": "Aiken Regional Medical Center", "city": "Aiken",        "state": "SC", "lat": 33.565, "lng": -81.763},
    {"hub_id": "H5", "name": "MUSC Health",                   "city": "Charleston",   "state": "SC", "lat": 32.785, "lng": -79.947, "coastal": True},
    {"hub_id": "H6", "name": "McLeod Regional Medical Center","city": "Florence",     "state": "SC", "lat": 34.192, "lng": -79.765},
    {"hub_id": "H7", "name": "Grand Strand Medical Center",   "city": "Myrtle Beach", "state": "SC", "lat": 33.748, "lng": -78.847, "coastal": True},
]

_HUB_INDEX: Dict[str, Dict] = {h["hub_id"]: h for h in SC_HOSPITAL_HUBS}


# ─── Geo helpers ─────────────────────────────────────────────────────────────

# SC coastline approximation: a series of line segments from north to south.
# If a point is east (right) of this polyline, it's in the water.
# Format: list of (lat, lng) running roughly NE → SW along the coast.
_SC_COASTLINE = [
    (33.85, -78.55),   # north of Myrtle Beach (NC border at coast)
    (33.65, -78.90),   # south of Myrtle Beach
    (33.40, -79.10),   # Georgetown area
    (33.00, -79.45),   # McClellanville
    (32.80, -79.85),   # Charleston harbor
    (32.65, -80.05),   # south of Charleston
    (32.35, -80.45),   # Beaufort / Hilton Head area
    (32.05, -80.85),   # Savannah River mouth (SC/GA border)
]


def _is_in_water(lat: float, lng: float) -> bool:
    """Return True if (lat, lng) is east of the SC coastline polyline."""
    # Find the two coastline vertices that bracket this latitude
    for i in range(len(_SC_COASTLINE) - 1):
        lat_a, lng_a = _SC_COASTLINE[i]
        lat_b, lng_b = _SC_COASTLINE[i + 1]
        if min(lat_a, lat_b) <= lat <= max(lat_a, lat_b):
            # Interpolate the coastline longitude at this latitude
            t = (lat - lat_b) / (lat_a - lat_b) if lat_a != lat_b else 0.5
            coast_lng = lng_b + t * (lng_a - lng_b)
            return lng > coast_lng
    # Outside the coastline latitude range — use nearest endpoint
    if lat >= _SC_COASTLINE[0][0]:
        return lng > _SC_COASTLINE[0][1]
    return lng > _SC_COASTLINE[-1][1]


def _jitter_location(
    rng: random.Random,
    lat: float,
    lng: float,
    radius_km: float = 25.0,
    coastal: bool = False,
) -> Dict[str, float]:
    """Return a uniformly-distributed random point within *radius_km* of
    (lat, lng).  Uses a simple equirectangular approximation (good enough
    for ≤50 km offsets at SC latitudes).

    For coastal hubs, resamples until the point falls on land."""
    km_per_deg_lat = 111.0
    km_per_deg_lng = 111.0 * math.cos(math.radians(lat))

    for _ in range(50):  # resample up to 50 times
        angle = rng.uniform(0, 2 * math.pi)
        r = radius_km * math.sqrt(rng.random())

        d_lat = (r * math.sin(angle)) / km_per_deg_lat
        d_lng = (r * math.cos(angle)) / km_per_deg_lng

        new_lat = lat + d_lat
        new_lng = lng + d_lng

        if not coastal or not _is_in_water(new_lat, new_lng):
            return {
                "lat": round(new_lat, 5),
                "lng": round(new_lng, 5),
            }

    # Fallback: place at hub center
    return {"lat": round(lat, 5), "lng": round(lng, 5)}


def get_hub(hub_id: str) -> Dict:
    """Return the hub dict for a given hub_id, or raise ValueError."""
    hub = _HUB_INDEX.get(hub_id)
    if hub is None:
        valid = ", ".join(sorted(_HUB_INDEX))
        raise ValueError(f"Unknown hub_id {hub_id!r}. Valid ids: {valid}")
    return hub


def random_hub(rng: random.Random) -> Dict:
    """Pick a hub at random."""
    return rng.choice(SC_HOSPITAL_HUBS)


def assign_location(
    rng: random.Random,
    hub_id: Optional[str] = None,
    radius_km: float = 25.0,
) -> Dict:
    """Pick (or use) a hub and return a location dict with jittered coords.

    Returns
    -------
    dict with keys: hub_id, hospital, city, state, lat, lng
    """
    hub = get_hub(hub_id) if hub_id else random_hub(rng)
    jittered = _jitter_location(
        rng, hub["lat"], hub["lng"], radius_km,
        coastal=hub.get("coastal", False),
    )
    return {
        "hub_id":   hub["hub_id"],
        "hospital": hub["name"],
        "city":     hub["city"],
        "state":    hub["state"],
        "lat":      jittered["lat"],
        "lng":      jittered["lng"],
    }


# ─── CLI for quick testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json  # noqa: E401

    parser = argparse.ArgumentParser(description="SC hospital hub geo-randomization test")
    parser.add_argument("--hub", default=None, help="Hub id (e.g. H3). Default: random")
    parser.add_argument("--n", type=int, default=5, help="Number of points to generate")
    parser.add_argument("--radius", type=float, default=25.0, help="Jitter radius in km")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for i in range(args.n):
        loc = assign_location(rng, hub_id=args.hub, radius_km=args.radius)
        print(json.dumps(loc, indent=2))
