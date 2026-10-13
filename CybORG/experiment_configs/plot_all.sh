#!/usr/bin/env bash
set -euo pipefail
CAGE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CAGE_SCRIPT_DIR/.."

# CIA red agent (RedMeanderAgent_Combination_CIA)
for scenario in base hosts_2 hosts_3 hosts_4; do
    python plot_drops_from_csv.py --config=experiment_configs/${scenario}.yaml
done

# CAGE red agent
for scenario in base hosts_2 hosts_3 hosts_4; do
    python plot_drops_from_csv.py --config=experiment_configs/cage_agent_config/${scenario}.yaml
done

# B-line red agents (A, C, I)
for CIA in A C I; do
    for scenario in base hosts_2 hosts_3 hosts_4; do
        python plot_drops_from_csv.py --config=experiment_configs/bline_agent_config/${CIA}/${scenario}.yaml
    done
done

# Adaptive alpha
for scenario in base hosts_2 hosts_3 hosts_4; do
    for red in cia cage bline_A bline_C bline_I; do
        python plot_drops_from_csv.py --config=experiment_configs/adaptive_alpha_config/${scenario}/${red}.yaml
    done
done

# Training-time configs (CIA)
for scenario in base hosts_2 hosts_3 hosts_4; do
    python plot_drops_from_csv.py --config=experiment_configs/training_time_config/${scenario}.yaml
done

# Training-time configs (CAGE)
for scenario in base hosts_2 hosts_3 hosts_4; do
    python plot_drops_from_csv.py --config=experiment_configs/training_time_config/${scenario}_cage.yaml
done

# Training-time B-line agents (A, C, I)
for CIA in A C I; do
    for scenario in base hosts_2 hosts_3 hosts_4; do
        python plot_drops_from_csv.py --config=experiment_configs/training_time_config/bline_agent_config/${scenario}_${CIA}.yaml
    done
done

# Alpha comparison plots ---------------------------------------------------

# CIA red agent
for scenario in base hosts_2 hosts_3 hosts_4; do
    python plot_alpha_comparison.py --config=experiment_configs/${scenario}.yaml
done

# CAGE red agent
for scenario in base hosts_2 hosts_3 hosts_4; do
    python plot_alpha_comparison.py --config=experiment_configs/cage_agent_config/${scenario}.yaml
done

# B-line red agents (A, C, I)
for CIA in A C I; do
    for scenario in base hosts_2 hosts_3 hosts_4; do
        python plot_alpha_comparison.py --config=experiment_configs/bline_agent_config/${CIA}/${scenario}.yaml
    done
done
