"""Freeze and validate the official geography used by the study."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.geo.meshes import materialize_municipality_mesh
from brepi.io.cache import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    path, report = materialize_municipality_mesh(
        year=args.year, refresh=args.refresh
    )
    manifest = write_manifest(
        f"geography_{args.year}",
        [f"ibge/meshes/{args.year}/BR_Municipios_{args.year}.zip"],
    )
    print(
        json.dumps(
            {
                "geometry": str(path),
                "manifest": str(manifest),
                "report": asdict(report),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
