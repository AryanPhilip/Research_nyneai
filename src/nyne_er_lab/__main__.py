"""CLI entrypoint for generating Entity Resolution Lab artifacts."""

from __future__ import annotations

import argparse

from nyne_er_lab.demo import build_demo_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Entity Resolution Lab demo artifacts.")
    parser.add_argument("--output-dir", default="reports", help="Directory for generated reports.")
    args = parser.parse_args()
    build_demo_artifacts(args.output_dir)


if __name__ == "__main__":
    main()
