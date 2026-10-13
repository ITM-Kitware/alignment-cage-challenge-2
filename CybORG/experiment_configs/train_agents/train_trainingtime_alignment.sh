#!/usr/bin/env bash
set -euo pipefail
CAGE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CAGE_SCRIPT_DIR/../.."

read -r -a CAGE_ATTRIBUTE_LIST <<< "${CAGE_ATTRIBUTES:-C I A}"
read -r -a CAGE_ALPHA_LIST <<< "${CAGE_ALPHAS:-0.5 1.0 1.5 2.0 2.5 3.0}"
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
    for attr in "${CAGE_ATTRIBUTE_LIST[@]}"; do
        for alpha in "${CAGE_ALPHA_LIST[@]}"; do
            for seed in "${CAGE_SEED_LIST[@]}"; do
                python train_agents.py \
                    --scenario="scalability_experiments/$scenario" \
                    --agent_export="checkpoints/PPO_${scenario}_${alpha}_${attr}_seed${seed}" \
                    --training_iter="$training_iter" \
                    --red_agent=RedMeanderAgent_Combination \
                    --agent_type=AlignedPPOAgent --alpha="$alpha" \
                    --alignment_attribute="$attr" --seed="$seed" \
                    --game_steps="$steps" "${CAGE_LOG_ARGS[@]}"
            done
        done
    done
done
