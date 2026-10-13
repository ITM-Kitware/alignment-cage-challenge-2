#!/usr/bin/env bash
set -euo pipefail
CAGE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CAGE_SCRIPT_DIR/.."

for scenario in base hosts_2 hosts_3 hosts_4; do #hosts_3

    python alignment_analysis.py --config=experiment_configs/${scenario}.yaml

    python alignment_analysis.py --config=experiment_configs/cage_agent_config/${scenario}.yaml

    for CIA in C I A; do
        echo "=== Running ${scenario} + ${CIA} ==="

        python alignment_analysis.py \
            --config=experiment_configs/bline_agent_config/${CIA}/${scenario}.yaml
    done

done


for scenario in base hosts_2 hosts_3 hosts_4; do  #
    python alignment_analysis.py \
        --config=experiment_configs/adaptive_alpha_config/${scenario}/cage.yaml
    python alignment_analysis.py \
        --config=experiment_configs/adaptive_alpha_config/${scenario}/cia.yaml

    for CIA in C I A; do
        echo "=== Running Adaptive ${scenario} + ${CIA} ==="

        python alignment_analysis.py \
            --config=experiment_configs/adaptive_alpha_config/${scenario}/bline_${CIA}.yaml
    done

done

for scenario in base hosts_2 hosts_3 hosts_4; do #hosts_3

    python alignment_analysis.py --config=experiment_configs/training_time_config/${scenario}.yaml
    python alignment_analysis.py --config=experiment_configs/training_time_config/${scenario}_cage.yaml

    for CIA in C I A; do
        echo "=== Running TT ${scenario} + ${CIA} ==="

        python alignment_analysis.py \
            --config=experiment_configs/training_time_config/bline_agent_config/${scenario}_${CIA}.yaml
    done

done