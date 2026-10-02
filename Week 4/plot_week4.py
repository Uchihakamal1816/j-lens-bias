import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    with open("week4_ablation_results.json", "r") as f:
        data = json.load(f)

    # Filter for Late Layer (The Spike Zone) where the interesting stuff happened
    late_data = [d for d in data if d["layer_band"] == "Late"]

    pressures = ["Turn1_Subtle", "Turn2_Pressure", "Turn3_Explicit"]
    ablations = ["None", "Single_Token_She", "Single_Token_He", "Subspace_Diff", "Random_Control"]
    
    # We want to plot P(she) vs P(he) for each ablation type under explicit pressure
    explicit_data = [d for d in late_data if d["prompt_tier"] == "Turn3_Explicit"]
    
    # Add baseline from early/none if missing in late/none (Baseline is early/none)
    baseline = next((d for d in data if d["layer_band"] == "Early" and d["ablation_type"] == "None" and d["prompt_tier"] == "Turn3_Explicit"), None)
    
    plot_labels = ["Baseline (No Ablation)", "Ablate 'she'", "Ablate 'he'", "Subspace ('she'-'he')", "Random Control"]
    p_she = []
    p_he = []
    
    # Gather data in order
    p_she.append(baseline["P_she"] if baseline else 0.4466)
    p_he.append(baseline["P_he"] if baseline else 0.0156)
    
    for abl in ["Single_Token_She", "Single_Token_He", "Subspace_Diff", "Random_Control"]:
        item = next((d for d in explicit_data if d["ablation_type"] == abl), None)
        p_she.append(item["P_she"] if item else 0)
        p_he.append(item["P_he"] if item else 0)

    x = np.arange(len(plot_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, p_she, width, label='P(she) [Stereotype]', color='#ff69b4')
    rects2 = ax.bar(x + width/2, p_he, width, label='P(he) [Counter-Stereotype]', color='#4169e1')

    ax.set_ylabel('Output Probability')
    ax.set_title('Late-Layer Interventions under Explicit Stereotyping Pressure (Turn 3)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=15, ha='right')
    ax.legend()

    # Label bars
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    fig.tight_layout()
    plt.savefig("week4_ablation_visualization.png", dpi=300)
    print("Saved plot to week4_ablation_visualization.png")

if __name__ == '__main__':
    main()
