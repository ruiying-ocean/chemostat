from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from chemostat import Chemostat


@dataclass(frozen=True)
class Scenario:
    name: str
    temperature: float
    nutrient: float
    config_file: Path


DEFAULT_SCENARIOS = (
    Scenario("baseline", 10.0, 30.0, Path("model_config.toml")),
    Scenario("low_dilution", 10.0, 10.0, Path("model_config_lowdilution.toml")),
    Scenario("low_nutrient", 10.0, 5.0, Path("model_config_lowdilution.toml")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chemostat model scenarios.")
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        choices=[scenario.name for scenario in DEFAULT_SCENARIOS],
        nargs="*",
        help="Scenario(s) to run; defaults to all.",
    )
    parser.add_argument(
        "--sum-pft",
        action="store_true",
        default=False,
        help="Aggregate biomass by functional type in plots (default plots each PFT).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to resolve configuration files from (default: current working directory).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save figures to this directory as <scenario>.png instead of showing them.",
    )
    return parser.parse_args()


def resolve_scenarios(args: argparse.Namespace) -> tuple[Scenario, ...]:
    if args.scenarios:
        selected = tuple(s for s in DEFAULT_SCENARIOS if s.name in args.scenarios)
    else:
        selected = DEFAULT_SCENARIOS
    return tuple(
        Scenario(s.name, s.temperature, s.nutrient, args.config_dir / s.config_file)
        for s in selected
    )


def run_scenario(scenario: Scenario, *, sum_pft: bool, save_dir: Path | None) -> None:
    print(f"=== Running scenario: {scenario.name} ===")
    if not scenario.config_file.exists():
        print(
            f"Skipping {scenario.name}: missing config file {scenario.config_file}",
            file=sys.stderr,
        )
        return

    model = Chemostat(scenario.temperature, scenario.nutrient)
    model.load_ecoconfig(file=str(scenario.config_file))
    model.run()

    save_path = save_dir / f"{scenario.name}.png" if save_dir is not None else None
    model.plot(sum_PFT=sum_pft, save_path=save_path)
    if save_path is not None:
        print(f"saved figure: {save_path}")


def main() -> None:
    args = parse_args()
    scenarios = resolve_scenarios(args)

    if not scenarios:
        print("No scenarios selected.", file=sys.stderr)
        raise SystemExit(1)

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        run_scenario(scenario, sum_pft=args.sum_pft, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
