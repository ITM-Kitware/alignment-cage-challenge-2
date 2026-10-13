#!/usr/bin/env bash
set -euo pipefail
CAGE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CAGE_SCRIPT_DIR/.."

read -r -a CAGE_SCENARIO_LIST <<< "${CAGE_SCENARIOS:-base hosts_2 hosts_3 hosts_4}"
for scenario in "${CAGE_SCENARIO_LIST[@]}"; do
    python test_experiment.py --config="experiment_configs/${scenario}.yaml"
    python test_experiment.py --config="experiment_configs/cage_agent_config/${scenario}.yaml"
    for attr in C I A; do
        python test_experiment.py --config="experiment_configs/bline_agent_config/${attr}/${scenario}.yaml"
    done
    for red in cia cage bline_A bline_C bline_I; do
        python test_experiment.py --config="experiment_configs/adaptive_alpha_config/${scenario}/${red}.yaml"
    done
done
