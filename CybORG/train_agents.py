from cage_experiment import CAGEExperiment
from CybORG.Agents.SimpleAgents.PPOAgent import PPOAgent, PPOAgentReactDecoys, AlignedPPOAgent
from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric
from CybORG.Agents.SimpleAgents.Meander_Resilience import RedMeanderAgent_Resilience, RedMeanderAgent_RandomImpact, RedMeanderAgent_Combination
import argparse
from pathlib import Path


def initialize_blue_agent(args, red_agent):
    if args.agent_type == "PPOAgent":
        return PPOAgent(
        scenario_config=args.scenario,
        red_agent=red_agent,
        game_steps=args.game_steps,
        seed=args.seed,
        learning_rate=args.learning_rate)

    elif args.agent_type == "PPOAgentReactDecoys":
        return PPOAgentReactDecoys(scenario_config=args.scenario,
        red_agent=red_agent,
        game_steps=args.game_steps,
        seed=args.seed,
        learning_rate=args.learning_rate)

    elif args.agent_type == "AlignedPPOAgent":
        return AlignedPPOAgent(
        base_agent=args.base_agent,
        red_agent=red_agent,
        game_steps=args.game_steps,
        scenario_config=args.scenario,
        alpha=args.alpha,
        align_attribute=args.alignment_attribute,
        seed=args.seed, 
        learning_rate=args.learning_rate)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--game_steps", type=int, default=250)

    parser.add_argument("--agent_type", type=str, default="AlignedPPOAgent")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--base_agent", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="Scenario2_resilience_debug")
    parser.add_argument("--agent_export", type=str, default="checkpoints/ppo_adaptive_3M-a05-C")
    parser.add_argument("--training_iter", type=int, default=3e6)
    parser.add_argument("--alignment_attribute", type=str, default="C")
    parser.add_argument("--red_agent", type=str, default="RedMeanderAgent_Resilience")
    parser.add_argument("--wandb", action="store_true", help="Enable optional Weights & Biases logging")
    args = parser.parse_args()
    Path(args.agent_export).parent.mkdir(parents=True, exist_ok=True)
    
    #TODO: Configure a resilience reward function
    # that trains with resilience instead of alignment
    # and allows changing of CIA weights in resilience
    metric = ResilienceMetric()
    red_agent = globals()[args.red_agent]
    agent = initialize_blue_agent(args, red_agent)
    
    cage = CAGEExperiment(agent,
            red_agent=red_agent,
            scenario=args.scenario,
            seed=args.seed,
            metric=metric,
            average_metric=True,
            use_wrapper=True,
            alpha=args.alpha,
            align_attribute=args.alignment_attribute)
    
    cage.train(args.agent_export, wandb=args.wandb, training_iter=args.training_iter)
    results = cage.run_experiment(steps=100, episodes=10)
    print(results)
