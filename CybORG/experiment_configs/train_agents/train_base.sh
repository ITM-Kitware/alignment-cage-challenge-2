#!/usr/bin/env bash
set -euo pipefail
CAGE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CAGE_SCRIPT_DIR/../.."

read -r -a CAGE_SCENARIO_LIST <<< "${CAGE_SCENARIOS:-base hosts_2 hosts_3 hosts_4}"
read -r -a CAGE_SEED_LIST <<< "${CAGE_TRAIN_SEEDS:-0 1 2 3 4}"
CAGE_LOG_ARGS=()
if [[ "${CAGE_WANDB:-0}" == 1 ]]; then CAGE_LOG_ARGS+=(--wandb); fi

for scenario in "${CAGE_SCENARIO_LIST[@]}"; do
    case "$scenario" in
        base) scale=1 ;;
        hosts_2) scale=2 ;;
        hosts_3) scale=3 ;;
        hosts_4) scale=4 ;;
        *) echo "Unknown paper scenario: $scenario" >&2; exit 1 ;;
    esac
    training_iter=$((500000 * scale))
    steps=$((100 + 100 * scale))
    for seed in "${CAGE_SEED_LIST[@]}"; do
        python train_agents.py \
            --scenario="scalability_experiments/$scenario" \
            --agent_export="checkpoints/PPO_${scenario}_0_seed${seed}" \
            --training_iter="$training_iter" \
            --red_agent=RedMeanderAgent_Combination \
            --agent_type=PPOAgent --alpha=0 --seed="$seed" \
            --game_steps="$steps" "${CAGE_LOG_ARGS[@]}"
    done
done
