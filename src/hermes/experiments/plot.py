"""Plot HERMES experiment outputs."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"Plotting is scaffolded but not implemented yet: {args.results}")


if __name__ == "__main__":
    main()

