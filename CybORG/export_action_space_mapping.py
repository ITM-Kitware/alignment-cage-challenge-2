###
# Exports mapping of vector action spaces to text 
# for all scenario files
#
# Used in alignment_analysis, if we want to ignore certain 
# actions in post (e.g., Remove action, which can fail)
###
from cage_experiment import CAGEExperiment
from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric
from CybORG.Agents.SimpleAgents.BlueMonitorAgent import BlueMonitorAgent
from CybORG.Agents.SimpleAgents.Meander_Resilience import RedMeanderAgent_Combination_CIA
import json
from pathlib import Path

if __name__ == "__main__":
    scenario_path = "scalability_eval_scenarios/"
    export_path = Path(__file__).resolve().parent / "CybORG/Shared/Scenarios/scalability_eval_scenarios/action_spaces"
    export_path.mkdir(parents=True, exist_ok=True)
    
    for scenario in ["base", "hosts_2", "hosts_3", "hosts_4"]:
        #initialize CAGE envrionment
        agent1 = BlueMonitorAgent()
        cage = CAGEExperiment(
            agent1,
            red_agent=RedMeanderAgent_Combination_CIA,
            scenario=scenario_path + scenario,
            seed=1,
            metric=ResilienceMetric(),
            experiment_export_dir="results/actions",
            average_metric=False,
            use_wrapper=True,
        )
        
        #get action mapping
        actionlist = []
        for action in cage.env.get_attr('possible_actions'):
            actionlist.append(
                {
                    "hostname": action.hostname if hasattr(action, "hostname") else None,
                    "action": action.__class__.__name__
                }
            )
            
        #export/save
        with open(export_path / f"{scenario}.json", "w") as f:
            json.dump({"action_space": actionlist}, f, indent=2)
