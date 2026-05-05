import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_figures(plot_data_cache: dict, sparsity_ratio: float, output_dir: str = "results", pred_len: int = 24):
    """
    Generate publication‑quality figures (PDF, PNG, SVG) for the three models.
    plot_data_cache should contain entries with keys matching the model names.
    """
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['svg.fonttype'] = 'none'

    COLORS = {
        "proposed": {"line": "#337ab7", "fill": "#337ab7"},
        "targetonly": {"line": "#5cb85c", "fill": "#5cb85c"},
        "lstm": {"line": "#f0ad4e", "fill": "#f0ad4e"},
    }

    FIGURE_LIST = [
        {"save_name": "a_proposed_model", "cache_key": "DLinear (Ours)_MSDA",
         "label": "Proposed MSDA-DLinear", "color": COLORS["proposed"]},
        {"save_name": "b_target_only", "cache_key": "DLinear_Target-Only",
         "label": "Target-Only Baseline", "color": COLORS["targetonly"]},
        {"save_name": "c_lstm_msda", "cache_key": "LSTM_MSDA",
         "label": "LSTM-MSDA", "color": COLORS["lstm"]},
    ]

    os.makedirs(output_dir, exist_ok=True)

    for cfg in FIGURE_LIST:
        # Find matching key
        matched_key = None
        for key in plot_data_cache.keys():
            if cfg["cache_key"] in key:
                matched_key = key
                break
        if matched_key is None:
            print(f"Skipping {cfg['cache_key']}: not found in plot data.")
            continue

        data = plot_data_cache[matched_key]
        tru_plot = data["true"].squeeze()[:pred_len]
        med_plot = data["median"].squeeze()[:pred_len]
        low_plot = data["lower"].squeeze()[:pred_len]
        upp_plot = data["upper"].squeeze()[:pred_len]
        x_axis = np.arange(len(tru_plot))

        plt.figure(figsize=(8, 3.2), dpi=300)
        plt.plot(x_axis, tru_plot, color="black", linewidth=1.6, label="Actual Load")
        plt.plot(x_axis, med_plot, color=cfg["color"]["line"], linewidth=1.9, label=f"{cfg['label']} Median")
        plt.fill_between(x_axis, low_plot, upp_plot, color=cfg["color"]["fill"], alpha=0.3, label="80% Prediction Interval")

        plt.xlabel("Hours", fontsize=11)
        plt.ylabel("Load (MW)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="lower right", fontsize=9, framealpha=0.9)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(np.arange(0, pred_len, 4))
        plt.ylim(top=850)
        plt.tight_layout()

        base_path = os.path.join(output_dir, f"{cfg['save_name']}_fs{sparsity_ratio:.3f}")
        for fmt in ["pdf", "png", "svg"]:
            plt.savefig(f"{base_path}.{fmt}", bbox_inches="tight")
        plt.close()
        print(f"Saved {base_path} (.pdf + .png + .svg)")