#####
# Takes a folder of trajectories and plots them
# 
# Example: python trajectory_viewer.py -t "Foldername_with_jsons" -a 1 -r 0 90
# where 0 - 90 are the steps to plot (if left out, will plot all the steps)
#####


import argparse
from agent_trajectory import AgentTrajectory
import os

def plot_game(file_path, export_path):
    print("Plotting", file_path)
    trajectory = AgentTrajectory.load(file_path)
    
    start, end = args.step_range
    
    trajectory.view(export_path, agent_type="red",
                    show_actions=args.show_actions,
                    limit_steps_to_range = [start, end]
                    )
    trajectory.view(export_path, agent_type="blue",
                    show_actions=args.show_actions,
                    limit_steps_to_range = [start, end])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_folder", type=str, default="trajectories/")
    parser.add_argument("-t", "--trajectory_name", type=str, required=True)
    parser.add_argument("-a", "--show_actions", type=int, default=1)
    parser.add_argument("-r", "--step_range", type=int, nargs=2, metavar=('START', 'END'), default=[None, None])
    parser.add_argument("--export_path", type=str, default="results/trajectories/")
    args = parser.parse_args()
    
    export_path = args.export_path
    trajectory_name = args.trajectory_name
    trajectory_folder = args.trajectory_folder

    #loop through all the trajectories for that experiment and plot
    print(trajectory_folder)
    for root, dirs, files in os.walk(trajectory_folder + trajectory_name):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                plot_game(file_path, export_path)
