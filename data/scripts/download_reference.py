#!/usr/bin/env python3
"""
Download small reference datasets for local development.

Currently downloads:
  - OurAirports airports.csv (~10MB) — airport coordinates, codes, metadata
"""

import csv
import io
import urllib.request
import urllib.error
from pathlib import Path

# OurAirports data — public, no authentication, small CSV
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def download_reference_data(output_dir: Path) -> list[Path]:
    """Download reference datasets to output_dir.

    Returns list of paths to downloaded files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    # --- OurAirports ---
    airports_path = output_dir / "ourairports_airports.csv"
    print("  Downloading OurAirports airports.csv...")
    try:
        req = urllib.request.Request(
            OURAIRPORTS_URL,
            headers={"User-Agent": "flight-telemetry-bootstrap/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()

        airports_path.write_bytes(data)
        downloaded.append(airports_path)

        # Quick sanity check — should be CSV with expected headers
        header_line = data.decode("utf-8", errors="replace").split("\n")[0]
        if "ident" in header_line and "latitude_deg" in header_line:
            # Count rows for reporting
            row_count = data.count(b"\n") - 1
            print(f"  ✓ OurAirports: {row_count:,} airports")
        else:
            print("  ⚠ OurAirports: unexpected header format")

    except (urllib.error.URLError, OSError) as e:
        print(f"  ⚠ OurAirports download failed: {e}")
        print("  → Generating minimal fallback reference data")
        _generate_fallback_airports(airports_path)
        downloaded.append(airports_path)

    return downloaded


def _generate_fallback_airports(path: Path) -> None:
    """Generate a tiny fallback airports file for offline development.

    Contains a handful of major European airports that match the
    sample data's bounding box.
    """
    airports = [
        {
            "ident": "EHAM",
            "type": "large_airport",
            "name": "Amsterdam Airport Schiphol",
            "latitude_deg": 52.3086,
            "longitude_deg": 4.7639,
            "elevation_ft": -11,
            "iso_country": "NL",
            "iso_region": "NL-NH",
            "municipality": "Amsterdam",
            "iata_code": "AMS",
        },
        {
            "ident": "EDDF",
            "type": "large_airport",
            "name": "Frankfurt am Main Airport",
            "latitude_deg": 50.0333,
            "longitude_deg": 8.5706,
            "elevation_ft": 364,
            "iso_country": "DE",
            "iso_region": "DE-HE",
            "municipality": "Frankfurt",
            "iata_code": "FRA",
        },
        {
            "ident": "EGLL",
            "type": "large_airport",
            "name": "Heathrow Airport",
            "latitude_deg": 51.4706,
            "longitude_deg": -0.4619,
            "elevation_ft": 83,
            "iso_country": "GB",
            "iso_region": "GB-ENG",
            "municipality": "London",
            "iata_code": "LHR",
        },
        {
            "ident": "LFPG",
            "type": "large_airport",
            "name": "Charles de Gaulle International Airport",
            "latitude_deg": 49.0097,
            "longitude_deg": 2.5479,
            "elevation_ft": 392,
            "iso_country": "FR",
            "iso_region": "FR-IDF",
            "municipality": "Paris",
            "iata_code": "CDG",
        },
        {
            "ident": "LEBL",
            "type": "large_airport",
            "name": "Josep Tarradellas Barcelona-El Prat Airport",
            "latitude_deg": 41.2971,
            "longitude_deg": 2.0785,
            "elevation_ft": 12,
            "iso_country": "ES",
            "iso_region": "ES-CT",
            "municipality": "Barcelona",
            "iata_code": "BCN",
        },
        {
            "ident": "KJFK",
            "type": "large_airport",
            "name": "John F Kennedy International Airport",
            "latitude_deg": 40.6398,
            "longitude_deg": -73.7789,
            "elevation_ft": 13,
            "iso_country": "US",
            "iso_region": "US-NY",
            "municipality": "New York",
            "iata_code": "JFK",
        },
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=airports[0].keys())
        writer.writeheader()
        writer.writerows(airports)

    print(f"  ✓ Fallback airports: {len(airports)} airports")
