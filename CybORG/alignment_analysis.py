# analyze alignment score and choices made by each agent

import argparse
import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import ast
import json
import yaml
from adjustText import adjust_text
from tqdm import tqdm


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(config_path: str):
    cfg = load_yaml(config_path)
    experiment_dir = cfg["paths"]["experiment_dir"]
    attributes = cfg["alignment"]["attributes_to_run"]
    alpha_range = cfg["alignment"].get("alpha_range", {
        "start": 0,
        "stop": 1,
        "num": 1
    })
    alphas = np.linspace(alpha_range["start"], alpha_range["stop"], alpha_range["num"])
    analysis = cfg.get("analysis", {})
    metric_to_study = analysis.get("metric", "Reward")
    action_space = analysis.get("action_space_file", {})
    if bool(action_space):
        with open(action_space) as f:
            action_space = json.load(f)
    training_time = analysis.get("training_time", False)
    recalculate = analysis.get("recalculate", True)
    blue_agent_base_results = analysis.get("blue_agent_base_results", None) 
    calculate_kneepoint = analysis.get("calculate_kneepoint", True)
    alpha_sampling_range = analysis.get("alpha_sampling_range", alpha_range["num"])
    alphas = alphas[np.linspace(0, len(alphas) - 1, alpha_sampling_range, dtype=int)]
    red_agent = cfg.get("agents", {}).get("red_agent", "")
    return experiment_dir, attributes, alphas, metric_to_study, training_time, recalculate, calculate_kneepoint, blue_agent_base_results, action_space, red_agent

def distance_to_line(p, p1, p2):
    d = p2 - p1
    return np.abs(d[0] * (p1[1] - p[1]) - d[1] * (p1[0] - p[0])) / np.linalg.norm(d)

def find_kneepoint(df, x_col, y_col, experiment_dir):
    sub = df.sort_values(x_col, ascending=False).reset_index(drop=True)

    x = sub[x_col].values
    y = sub[y_col].values
    # Endpoints of the Pareto line
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    distances = np.array([
        distance_to_line(np.array([xi, yi]), p1, p2)
        for xi, yi in zip(x, y)
    ])

    knee_idx = int(np.argmax(distances))
            
    alpha = sub.loc[knee_idx]
    # Return full row for convenience
    knee = {
        "alpha": alpha["alpha_label"].split("=")[-1],
        "knee_index": knee_idx,
        "knee_x": x[knee_idx],
        "knee_y": y[knee_idx]
    }
    other_data = {
        "p1": p1,
        "p2": p2,
        "distances": list(distances)
    }
        
    return knee, other_data, alpha
    
def calculate_alignment_score(experiment_dir, attributes, alphas, training_time, blue_agent_base_results, action_space):

    agent_list = ["PPO agent"]
        
    for alpha in alphas:
        alpha = round(alpha, 3)
        if training_time:
            agent_list += ["Aligned RL "+str(alpha)+" " + attr + " agent" for attr in attributes]
        else:
            agent_list += ["RL-a"+str(alpha)+"-" + attr.upper() + "1 agent" for attr in attributes]

    agent_results = [["Agent", "Metric", "Alignment Score", "Std. Dev."]]
    
    choice_idx_mapping = {"C": 0, "I": 1, "A": 2}

    #filter
    action_list = action_space.get("action_space", [])

    valid_action_mask = (
        None
        if not action_list
        else np.array([a["action"] != "Remove" for a in action_list], dtype=bool)
    )

    for agent in tqdm(agent_list):
        base_dir = experiment_dir

        seed_files = sorted(glob.glob(f"{base_dir}/scenarios/scenario_choices_{agent}_seed*.npz"))
        if seed_files:
            all_choices = []
            all_selected = []
            for sf in seed_files:
                d = np.load(sf)
                all_choices.append(d['choices'])
                sel = d['selected'].copy()
                if training_time:
                    sel = np.insert(sel, 0, 176)
                all_selected.append(sel[:len(d['choices'])])
            choices_array = np.concatenate(all_choices, axis=0)
            selected_raw = np.concatenate(all_selected, axis=0)
        else:
            # fall back to unsuffixed file (single-seed or legacy runs)
            data = np.load(f"{base_dir}/scenarios/scenario_choices_{agent}.npz")
            choices_array = data['choices']
            selected_raw = data['selected'].copy()
            if training_time:
                selected_raw = np.insert(selected_raw, 0, 176)
            selected_raw = selected_raw[:len(choices_array)]

        for i, attr in enumerate(attributes):
            attr_choices = choices_array[:, :, choice_idx_mapping[attr]].astype(float)  # C, I, or A slice

            #filter out remove actions if prompted
            if valid_action_mask is not None:
                attr_choices = np.where(valid_action_mask, attr_choices, -np.inf)

            valid_mask = (attr_choices != -100) #drop defender as action
            masked_choices = np.where(valid_mask, attr_choices, -np.inf)
            max_per_row = masked_choices.max(axis=1)

            selected_score = attr_choices[np.arange(len(selected_raw)), selected_raw]
            is_in_max = (selected_score == max_per_row)

            invalid_selected = (
                np.zeros(len(selected_raw), dtype=bool)
                if valid_action_mask is None
                else ~valid_action_mask[selected_raw]
            )

            is_global_tie = (max_per_row < 0)
            # Exclude steps where multiple actions tie at the max — an agent can
            # land on one by chance, inflating scores for non-aligned agents like PPO.
            n_at_max = (masked_choices == max_per_row[:, None]).sum(axis=1)
            has_unique_best = (n_at_max == 1)
            alignment = np.where(
                invalid_selected | is_global_tie | ~has_unique_best,
                np.nan,
                is_in_max.astype(float),
            )
            agent_clean = agent.replace("agent", "").strip()

            # Determine which attribute this agent was trained on
            if training_time:
                # For "Aligned RL 0.5 C", extract the last token as the trained attribute
                agent_trained_attr = agent_clean.split()[-1]
            else:
                # for "RL-a0.5-C1", extract the attribute before the digit
                if agent_clean[-1].isdigit():
                    agent_trained_attr = agent_clean[-2]
                else:
                    agent_trained_attr = None
            
            # Create the display name
            if training_time and agent_clean != "PPO":
                #Aligned RL 0.5 C to "RL-α0.5"
                parts = agent_clean.split()
                alpha_val = parts[2]
                display_name = f"RL-\u03b1{alpha_val}"
            else:
                display_name = agent_clean.replace("-a", "-\u03b1")
                if display_name[-1].isdigit():
                    display_name = display_name[:-1]

            results = [display_name, attr, round(np.nanmean(alignment), 4), round(np.nanstd(alignment), 4)]
            
            # Only append if this metric matches the attribute the agent was trained on
            #or if it's the PPO baseline (alpha=0.0)
            if agent_trained_attr == attr or agent_trained_attr is None:
                agent_results.append(results)
            
    df = pd.DataFrame(agent_results[1:], columns=agent_results[0])
    df.to_csv(f"{experiment_dir}/all_alignment_scores.csv", index=False)
    return df

def plot_alignment(cfg_metric_to_study,
                   reward_stats_filtered,
                   agents,
                   experiment_dir,
                   alignment_df,
                   training_time=False,
                   scenario_name=""
                   ):

    num_to_show = 5
    # show no more than 5 points
    if len(agents) > num_to_show:
        idx = np.linspace(0, len(agents) - 1, num_to_show).astype(int)
        agents = agents[idx]
    

    # Filter and sort reward stats to match the selected agents
    reward_stats_filtered = reward_stats_filtered[reward_stats_filtered["Agent"].isin(agents)].copy()
    reward_stats_filtered["Agent"] = pd.Categorical(
        reward_stats_filtered["Agent"], categories=agents, ordered=True)
    reward_stats_filtered = reward_stats_filtered.sort_values("Agent")

    alignment_df = alignment_df[alignment_df["Agent"].isin(agents)].copy()
    alignment_df["Agent"] = pd.Categorical(alignment_df["Agent"], categories=agents, ordered=True)
    agents_reward = reward_stats_filtered["Agent"].astype(str).to_list()
    x_positions = np.arange(len(agents_reward))

    # Scoped to this figure only, so it doesn't leak into the Pareto front plot below.
    with sns.axes_style("whitegrid"), sns.plotting_context("paper"):
        fig, ax = plt.subplots(figsize=(6, 4))

        # Same CIA/reward colors as plot_from_csv.py's METRICS palette
        palette = {"C": "#1f6f9e", "I": "#2a9d8f", "A": "#52a675"}
        # Seaborn barplot using the axis
        sns.barplot(
            x='Agent', y='Alignment Score', hue="Metric", data=alignment_df,
            palette=palette, ax=ax, order=agents, width=0.9,
            edgecolor="black", linewidth=0.5,
        )
        ax.set_axisbelow(True)
        # Same dashed grid as plot_combined_comparison.py's style_axes
        ax.grid(True, axis="both", linestyle="--", linewidth=0.7, color="#bbbbbb", alpha=0.6, zorder=0)

        # Customize axis

        reward_color = "#e53935"
        ax2 = ax.twinx()
        ax2.set_zorder(10)
        ax2.patch.set_visible(False)
        # Avoid a second grid on top of the bars; ax's grid (axisbelow) is the only one shown
        ax2.grid(False)

        ax2.set_ylabel(cfg_metric_to_study, color=reward_color)
        ax2.tick_params(axis='y', labelcolor=reward_color)
        # Scaled/scientific notation (e.g. 1e4) instead of raw values like 10000
        ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

        ax2.plot(
            x_positions,
            reward_stats_filtered[cfg_metric_to_study].to_numpy(),
            color=reward_color,
            marker='o',
            zorder=25,
            linewidth=2,
            label=cfg_metric_to_study
        )

        # Plot line + error bars
        ax2.errorbar(
            x_positions,
            reward_stats_filtered[cfg_metric_to_study],
            yerr=reward_stats_filtered[cfg_metric_to_study + '_std'],
            fmt="none",
            color=reward_color,
            capsize=3,          # add little caps at ends of error bars
            linewidth=1,
            alpha=0.65,
            zorder=20,
            label=cfg_metric_to_study + '\u00b1 std'
        )

        # plots labels on reward values - uncomment to show, but overlaps with text/looks messy
        # for x, y in zip(x_positions, reward_stats_filtered[cfg_metric_to_study]):
        #     if y < -3000:
        #         offset = (6, 12)
        #         va = 'bottom'
        #     else:
        #         offset = (6, -12)
        #         va = 'top'

        #     ax2.annotate(
        #         f"{y:.0f}",
        #         xy=(x, y),
        #         xytext=offset,
        #         textcoords='offset points',
        #         ha='left',
        #         va=va,
        #         color=reward_color
        #     )


        for p in ax.patches:
            if not isinstance(p, Rectangle):
                continue

            h = p.get_height()
            if h < 0:
                continue

            x = p.get_x() + p.get_width() / 2.

            text_kwargs = dict(
                xy=(x, h),
                xycoords=ax.transData,
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="semibold",
                zorder=50,  # very high so it's always on top
                clip_on=False,  # allow drawing even if near edges
            )

            if h < 0.10:
                # bar too short for the label to fit inside -> plain black label above
                ax2.annotate(
                    f"{h:.2f}", xytext=(0, 3), va="bottom", color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    **text_kwargs,
                )
            else:
                # inside the bar -> plain white label
                ax2.annotate(
                    f"{h:.2f}", xytext=(0, -6), va="top", color="white",
                    **text_kwargs,
                )
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#222222")

        for spine in ax2.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#222222")
        ax.tick_params(axis='both', width=1.5, length=6)
        ax2.tick_params(axis='y', width=1.5, length=6)

        ax.set_xlabel('RL Agent')
        ax.set_ylabel('Alignment Score')
        ax.set_ylim(0, 1.1)

        if training_time:
            tick_labels = [t.get_text().replace('\u03b1', '\u03bb') for t in ax.get_xticklabels()]
            ax.set_xticklabels(tick_labels)

        symbol = '\u03bb' if training_time else '\u03b1'
        scenario_display = scenario_name.replace("_", " ") if scenario_name else ""
        title = (
            f"Alignment Score and Reward per {symbol}-Aligned RL Agent for {scenario_display.capitalize()} Scenario"
            if scenario_display
            else f"Alignment Score and Reward per {symbol}-Aligned RL Agent"
        )
        ax.set_title(title, fontweight="bold", pad=12)

        ax.legend(loc='center left', bbox_to_anchor=(0.025, 0.5))
        ax.margins(x=0.01)
        ax.set_xlim(-0.5, len(agents) - 0.5)
        fig.tight_layout()
        fig.savefig(f'{experiment_dir}/alignment_barplot_with_reward_over.pdf', bbox_inches="tight")
        plt.close(fig)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (same format as test_experiment.py)",
    )
    parser.add_argument(
        "--no-recalculate",
        action="store_true",
        default=False,
        help="Skip recomputing alignment scores; load from cached all_alignment_scores.csv",
    )
    args = parser.parse_args()

    experiment_dir, cfg_attributes, cfg_interpolation_alphas, cfg_metric_to_study, cfg_training_time, recalculate_alignment_score, calculate_kneepoint, blue_agent_base_results, action_space, red_agent_name = load_config(args.config)

    if args.no_recalculate:
        recalculate_alignment_score = False

    if recalculate_alignment_score:
        df = calculate_alignment_score(experiment_dir, cfg_attributes, cfg_interpolation_alphas, cfg_training_time, blue_agent_base_results, action_space)
        
    df = pd.read_csv(f"{experiment_dir}/all_alignment_scores.csv")
    df.loc[df["Agent"] == "PPO", "Agent"] = "RL-\u03b10.0"
    df["Agent"] = df["Agent"].str.replace("-C", "").str.replace("-I", "").str.replace("-A", "")
    
    
    df_reward_resilience = pd.read_csv(f"{experiment_dir}/test_results.csv")
    df_reward_resilience.loc[df_reward_resilience['Agent'] == 'RL', 'Agent'] = "RL-\u03b10.0"
    df_reward_resilience = df_reward_resilience[df_reward_resilience['Agent'] != "Blue Monitor"] #remove blue monitor
    df_reward_resilience["Metric"] = df_reward_resilience.apply(lambda x: x["Agent"].split("-")[-1], axis=1)
    df_reward_resilience["Agent"] = df_reward_resilience.apply(lambda x: x["Agent"].replace("-a", "-\u03b1").replace("-C", "").replace("-I", "").replace("-A", ""), axis=1)

    reward_scores = df_reward_resilience.groupby('Agent', as_index=False)[cfg_metric_to_study].mean()
    reward_std = df_reward_resilience.groupby('Agent', as_index=False)[cfg_metric_to_study].std()
    reward_stats = reward_scores.merge(reward_std, on='Agent', suffixes=('', '_std'))

    agents = df['Agent'].unique()
    reward_stats_filtered = reward_stats[reward_stats['Agent'].isin(agents)]
    agent_centers = {agent.strip(): i for i, agent in enumerate(reward_stats_filtered['Agent'])}

    scenario_name = os.path.splitext(os.path.basename(args.config))[0]

    plot_alignment(cfg_metric_to_study,
                   reward_stats_filtered,
                   agents,
                   experiment_dir,
                   df,
                   training_time=cfg_training_time,
                   scenario_name=scenario_name)

    # generate pareto front
    # Remove rows with missing values
    df = df.dropna()
    alpha0 = df_reward_resilience[df_reward_resilience["Agent"]=="RL-\u03b10.0"].iloc[0]
    expanded_alpha0 = pd.DataFrame({
        "Agent": ["RL-\u03b10.0"]*len(cfg_attributes),
        "Metric": list(cfg_attributes),
        cfg_metric_to_study: [alpha0[cfg_metric_to_study]]*len(cfg_attributes)
    })
    df_reward_resilience = df_reward_resilience[df_reward_resilience["Agent"]!="RL-\u03b10.0"]
    df_reward_resilience = pd.concat([df_reward_resilience, expanded_alpha0], ignore_index=True)

    baseline = df_reward_resilience[df_reward_resilience["Agent"] == "RL-\u03b10.0"].iloc[0]
    ppo_rows = pd.DataFrame({
        "Agent": ["RL-\u03b10.0-X"] * len(cfg_attributes),
        "Metric": list(cfg_attributes),
        cfg_metric_to_study: [baseline[cfg_metric_to_study]] * len(cfg_attributes),
    })
    df_reward_resilience = pd.concat([df_reward_resilience, ppo_rows], ignore_index=True)

    #hack to merge
    df["Agent"] = df["Agent"].str.replace("\u03b1", "a")
    df_reward_resilience["Agent"] = df_reward_resilience["Agent"].str.replace("\u03b1", "a")

    def clean(s):
        return (
            s.astype(str).str.strip().str.replace("\u00a0", "").str.replace("\u200b", "")
        )

    df["Agent"] = clean(df["Agent"])
    df["Metric"] = clean(df["Metric"])
    df_reward_resilience["Agent"] = clean(df_reward_resilience["Agent"])
    df_reward_resilience["Metric"] = clean(df_reward_resilience["Metric"])

    merged = df.merge(
        df_reward_resilience,
        on=["Agent","Metric"],
        how="left"
    )
    merged = merged[[
        "Agent", "Metric", "Alignment Score", cfg_metric_to_study
    ]]

    merged["alpha_label"] = merged["Agent"].apply(lambda x: f"\u03b1={x.split('a')[1]}")

    colors = sns.color_palette("tab10", n_colors=len(cfg_attributes))
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']  # Extend if more than 8 metrics

    fig, ax = plt.subplots(figsize=(6, 5))

    # Build display name for pareto title from red agent config name
    _RED_DISPLAY = {
        "RedMeanderAgent_Combination_CIA": "RedM - CIA",
        "RedMeanderAgent_Combination_CAGE": "RedM - CAGE",
        "B_lineAgent_Resilience_C": "Bline - C",
        "B_lineAgent_Resilience_I": "Bline - I",
        "B_lineAgent_Resilience_A": "Bline - A",
        "B_lineAgent_Resilience": "Bline",
    }
    red_display = _RED_DISPLAY.get(red_agent_name, red_agent_name)
    pareto_title = f"Pareto Front — {red_display}" if red_display else "Pareto Front for All Attributes"

    all_texts = []
    knee_points = {} #per-attribute
    for i, metric in enumerate(cfg_attributes):
        sub = merged[merged["Metric"] == metric].sort_values("Alignment Score")
        if sub.empty:
            continue
        ax.plot(sub["Alignment Score"], sub[cfg_metric_to_study],
                color=colors[i], marker=markers[i % len(markers)], label=metric, linestyle='-')

        first_row = sub.iloc[0]
        last_row  = sub.iloc[-1]

        for row in [first_row, last_row]:
            txt = ax.text(
                row["Alignment Score"],
                row[cfg_metric_to_study],
                row["alpha_label"],
                fontsize=12
            )
            all_texts.append(txt)
            
        #knee-point selection
        if calculate_kneepoint:
            knee, otherdata, knee_label = find_kneepoint(sub, "Alignment Score", cfg_metric_to_study, experiment_dir)
            
            knee_points[metric] = knee
            kneepoint = {**knee, **otherdata}
            # knee point annotation
            knee_x = kneepoint["knee_x"]
            knee_y = kneepoint["knee_y"]

            if len(knee_label) > 0:
                txt = ax.text(
                    knee_x + 0.03,
                    knee_y + 1,
                    knee_label["alpha_label"],
                    fontsize=12,
                    color=colors[i],
                    weight="bold",
                    ha="left",
                    va="bottom"
                )
                all_texts.append(txt)

            p1 = kneepoint["p1"]
            p2 = kneepoint["p2"]

            #knee point
            knee_x = kneepoint["knee_x"]
            knee_y = kneepoint["knee_y"]
            
            ideal_x = sub["Alignment Score"].max()
            ideal_y = sub[cfg_metric_to_study].max()

            #Z ideal point
            ax.scatter(ideal_x, ideal_y, s=150, color=colors[i], marker="*",zorder=6)

            # Line from first endpoint to ideal point
            ax.plot([p1[0], ideal_x], [p1[1], ideal_y], color="gray", linestyle=":", linewidth=1.2)

            # Line from second endpoint to ideal point
            ax.plot([p2[0], ideal_x], [p2[1], ideal_y], color="gray", linestyle=":", linewidth=1.2)

            #end points
            ax.scatter(p1[0], p1[1], s=50, color="black", marker="s", zorder=5)
            ax.scatter(p2[0], p2[1], s=50, color="black", marker="s", zorder=5)
            ax.plot([ideal_x, knee_x], [ideal_y, knee_y], color="lightgrey", linestyle="--", linewidth=1.5)

            ax.scatter(knee_x, knee_y, s=100, color=colors[i], edgecolor="grey", marker=markers[i % len(markers)], zorder=10, label=f"{metric} Optimal Point")


    if calculate_kneepoint:
        #export as json
        with open(experiment_dir + "/knee_point.json", "w") as f:
            json.dump(knee_points, f, indent=2)
    adjust_text(all_texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    ax.set_title(pareto_title, fontsize=16)
    ax.set_xlabel("Alignment Score", fontsize=14)
    ax.set_ylabel(cfg_metric_to_study, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12, title_fontsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(1.85)

    ax.tick_params(axis='both', width=1.5, length=6)
    fig.tight_layout()
    plt.savefig(f"{experiment_dir}/pareto_front.png", dpi=300)
