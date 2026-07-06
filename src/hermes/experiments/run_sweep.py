"""Run a HERMES experiment sweep."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Sweep runner is scaffolded but not implemented yet: {args.config}")


if __name__ == "__main__":
    main()

