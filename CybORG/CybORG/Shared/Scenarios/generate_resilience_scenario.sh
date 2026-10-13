#!/usr/bin/env bash
set -euo pipefail
CAGE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CAGE_SCRIPT_DIR/."

python scenario_generation.py --export_path='scalability_experiments/base' --subnets=3 --operational_n=4 --resilience_n=1 --enterprise_n=3 --user_n=5 --blue_reward='AlignedReward'

python scenario_generation.py --export_path='scalability_experiments/hosts_2' --subnets=3 --operational_n=8 --resilience_n=2 --enterprise_n=6 --user_n=10 --blue_reward='AlignedReward'

python scenario_generation.py --export_path='scalability_experiments/hosts_3' --subnets=3 --operational_n=12 --resilience_n=3 --enterprise_n=9 --user_n=15 --blue_reward='AlignedReward'

python scenario_generation.py --export_path='scalability_experiments/hosts_4' --subnets=3 --operational_n=16 --resilience_n=4 --enterprise_n=12 --user_n=20 --blue_reward='AlignedReward'

python scenario_generation.py --export_path='scalability_experiments/hosts_5' --subnets=3 --operational_n=20 --resilience_n=5 --enterprise_n=16 --user_n=25 --blue_reward='AlignedReward'
