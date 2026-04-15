#!/usr/bin/env python3
import argparse
from pathlib import Path

from baseline.cosa_adapter import export_cosa_inputs_from_architecture


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CoSA arch/mapspace YAMLs from MIREDO architecture definitions."
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="ZigzagAcc",
        help="Architecture module name under 112/Architecture, e.g. ZigzagAcc",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model tag used by adapter workspace naming",
    )
    parser.add_argument(
        "--arch-out",
        type=str,
        default="Evaluation/CoSA/cosa/src/cosa/configs/arch/zigzagacc_miredo.yaml",
        help="Output path for CoSA arch yaml",
    )
    parser.add_argument(
        "--mapspace-out",
        type=str,
        default="Evaluation/CoSA/cosa/src/cosa/configs/mapspace/mapspace_zigzagacc_miredo.yaml",
        help="Output path for CoSA mapspace yaml",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Adapter temporary workspace root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    args = parser.parse_args()

    arch_path, mapspace_path = export_cosa_inputs_from_architecture(
        architecture=args.architecture,
        arch_out=Path(args.arch_out),
        mapspace_out=Path(args.mapspace_out),
        model=args.model,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )

    print(f"arch={arch_path}")
    print(f"mapspace={mapspace_path}")


if __name__ == "__main__":
    main()
