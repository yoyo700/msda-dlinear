import argparse
import os
import sys
import pickle
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_utils import set_seed, LoadDataset, build_and_cache_data, load_data_cache
from losses import PinballLoss
from models import DLinear_MSDA, Holistic_DLinear_MSDA, LSTM_MSDA
from train_eval import train_model, evaluate
from plot_utils import draw_figures


def main():
    parser = argparse.ArgumentParser(description="MSDA-DLinear: Probabilistic Load Forecasting under Data Scarcity")
    parser.add_argument("--scenario", type=int, default=2, choices=[1, 2, 3],
                        help="Predefined source/target scenario (1:Intra-state, 2:Climate-aligned, 3:Urban-to-Rural)")
    parser.add_argument("--few_shot_ratio", type=float, default=0.06,
                        help="Fraction of historical target data used for training (e.g., 0.06 = 6%%)")
    parser.add_argument("--data_path", type=str, default="dataset/gefcom2017_clean.csv",
                        help="Path to the GEFCom2017 cleaned CSV file")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Directory to store/load data and model caches")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results, figures, and CSV")
    parser.add_argument("--rebuild_cache", action="store_true", default=False,
                        help="Force rebuild of data cache and retrain models")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    # Hyperparameters
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # Deterministic seed
    worker_seed_fn = set_seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    # Scenario configuration
    SCENARIO_CONFIG = {
        1: {"source": ["SEMASS", "NEMASSBOST"], "target": "WCMASS", "name": "Intra-state"},
        2: {"source": ["CT", "RI", "NH"], "target": "VT", "name": "Climate-aligned"},
        3: {"source": ["SEMASS", "WCMASS", "NEMASSBOST", "CT"], "target": "ME", "name": "Urban-to-Rural"},
    }
    cfg = SCENARIO_CONFIG[args.scenario]
    source_zones = cfg["source"]
    target_zone = cfg["target"]
    scenario_name = cfg["name"]

    # File names
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    cache_file = os.path.join(args.cache_dir, f"data_sc{args.scenario}_fs{args.few_shot_ratio:.3f}.pkl")
    model_cache_file = os.path.join(args.cache_dir, f"models_fs{args.few_shot_ratio:.3f}.pt")
    plot_data_cache_file = os.path.join(args.output_dir, f"plot_data_fs{args.few_shot_ratio:.3f}.pkl")
    csv_file = os.path.join(args.output_dir, f"results_sc{args.scenario}_fs{args.few_shot_ratio:.3f}.csv")

    # ---------- 1. Data ----------
    if args.rebuild_cache or not os.path.exists(cache_file):
        data = build_and_cache_data(
            data_path=args.data_path,
            source_zones=source_zones,
            target_zone=target_zone,
            test_hours=8760,
            max_source_hours=8760 * 3,
            few_shot_ratio=args.few_shot_ratio,
            cache_file=cache_file
        )
    else:
        data = load_data_cache(cache_file)

    mu_vt = data["scalers"][target_zone]["mu"]
    sigma_vt = data["scalers"][target_zone]["sigma"]
    feature_dim = len(data["FEATURE_COLS"])
    quantiles = [0.1, 0.5, 0.9]

    # DataLoaders
    src_dataset = LoadDataset(data["source_data"], args.seq_len, args.pred_len)
    tgt_train_dataset = LoadDataset(data["target_train"], args.seq_len, args.pred_len)
    tgt_test_dataset = LoadDataset(data["target_test"], args.seq_len, args.pred_len)

    src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            generator=generator, worker_init_fn=worker_seed_fn)
    tgt_train_loader = DataLoader(tgt_train_dataset,
                                  batch_size=min(len(data["target_train"]) - 50, args.batch_size),
                                  shuffle=True, generator=generator, worker_init_fn=worker_seed_fn)
    tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=256, shuffle=False,
                                 worker_init_fn=worker_seed_fn)

    # ---------- 2. Models ----------
    configs = [
        {"name": "DLinear (Ours)", "strat": "MSDA", "use_msda": True, "model_cls": DLinear_MSDA},
        {"name": "DLinear (Ablation)", "strat": "Holistic-MSDA", "use_msda": True, "model_cls": Holistic_DLinear_MSDA},
        {"name": "DLinear", "strat": "Target-Only", "use_msda": False, "model_cls": DLinear_MSDA},
        {"name": "LSTM", "strat": "MSDA", "use_msda": True, "model_cls": LSTM_MSDA},
    ]

    if args.rebuild_cache or not os.path.exists(model_cache_file):
        print("Training models from scratch...")
        trained_models = {}
        for cfg_ in configs:
            print(f"\n>>> Training {cfg_['name']} ({cfg_['strat']})")
            model = cfg_["model_cls"](args.seq_len, args.pred_len, feature_dim, quantiles).to(args.device)
            s_loader = src_loader if cfg_["strat"] != "Target-Only" else None
            model = train_model(
                model, tgt_train_loader, s_loader,
                use_msda=cfg_["use_msda"],
                device=args.device,
                lr=args.lr,
                epochs=args.epochs,
                quantiles=quantiles
            )
            trained_models[cfg_["name"]] = model
        # Save state dicts
        state_dicts = {name: m.state_dict() for name, m in trained_models.items()}
        torch.save(state_dicts, model_cache_file)
        print("Models saved.")
    else:
        print("Loading models from cache...")
        state_dicts = torch.load(model_cache_file, map_location=args.device)
        trained_models = {}
        for cfg_ in configs:
            model = cfg_["model_cls"](args.seq_len, args.pred_len, feature_dim, quantiles).to(args.device)
            model.load_state_dict(state_dicts[cfg_["name"]])
            trained_models[cfg_["name"]] = model

    # ---------- 3. Evaluation ----------
    results = []
    plot_data_cache = {}
    for cfg_ in configs:
        model = trained_models[cfg_["name"]]
        pl, picp, pinaw, true, median, lower, upper = evaluate(
            model, tgt_test_loader, mu_vt, sigma_vt,
            device=args.device, quantiles=quantiles, pred_len=args.pred_len
        )
        results.append([cfg_["name"], cfg_["strat"], pl, picp, pinaw])
        plot_data_cache[f"{cfg_['name']}_{cfg_['strat']}"] = {
            "true": true, "median": median, "lower": lower, "upper": upper
        }

    # Print table
    print("\n" + "=" * 90)
    print(f"{'Model':<25}{'Strategy':<20}{'Pinball':<10}{'PICP(%)':<10}{'PINAW':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r[0]:<25}{r[1]:<20}{r[2]:<10}{r[3]:<10}{r[4]:<10}")

    # Save plot data
    with open(plot_data_cache_file, "wb") as f:
        pickle.dump(plot_data_cache, f)

    # Save CSV
    scenario_cn_map = {1: "Intra-state", 2: "Climate-aligned", 3: "Urban-to-Rural"}
    source_zones_str = ", ".join(source_zones)
    run_time_str = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")

    csv_data = []
    for i, cfg_ in enumerate(configs):
        csv_data.append({
            "Model": cfg_["name"],
            "Strategy": cfg_["strat"],
            "Pinball Loss": results[i][2],
            "PICP (%)": results[i][3],
            "PINAW": results[i][4],
            "Scenario": scenario_cn_map[args.scenario],
            "Source Zones": source_zones_str,
            "Target Zone": target_zone,
            "Few-Shot Ratio": f"{args.few_shot_ratio:.3f}",
            "Seq Len": args.seq_len,
            "Pred Len": args.pred_len,
            "Epochs": args.epochs,
            "Learning Rate": args.lr,
            "Run Time": run_time_str
        })
    pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"Results saved to {csv_file}")

    # ---------- 4. Plots ----------
    print("\nGenerating paper figures...")
    draw_figures(plot_data_cache, args.few_shot_ratio, output_dir=args.output_dir, pred_len=args.pred_len)

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()