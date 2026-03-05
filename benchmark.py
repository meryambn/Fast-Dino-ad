"""
Ablation & Benchmarking Suite for Fast-DINO-AD.
Evaluates Exact FlatL2 vs HNSW vs IVF-PQ on:
- Latency (FPS)
- Memory Compression Ratio (MB)
- Localization Accuracy (Pixel-AUROC & PRO Score)
"""

import time
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from src.config import PipelineConfig, MemoryBankConfig
from src.pipeline.detector import FastDinoDetector
from src.utils.dataset import SyntheticIndustrialDataset
from src.utils.metrics import compute_image_auroc, compute_pixel_auroc, compute_pro_score


def run_benchmark():
    console = Console()
    console.print("\n[bold cyan]=====================================================[/bold cyan]")
    console.print("[bold green]  Fast-DINO-AD: Pareto-Frontier Ablation Benchmark  [/bold green]")
    console.print("[bold cyan]=====================================================[/bold cyan]\n")

    # 1. Create Datasets
    console.print("[yellow]Generating procedural industrial inspection dataset...[/yellow]")
    train_dataset = SyntheticIndustrialDataset(num_samples=40, anomaly_ratio=0.0, seed=101)
    test_dataset = SyntheticIndustrialDataset(num_samples=30, anomaly_ratio=0.5, seed=202)

    train_tensors = torch.stack([train_dataset[i]["image"] for i in range(len(train_dataset))])
    test_samples = [test_dataset[i] for i in range(len(test_dataset))]

    index_types = ["exact", "hnsw", "ivf_pq"]
    results = []

    for idx_type in index_types:
        console.print(f"\n[bold magenta]>>> Evaluating Index Strategy: {idx_type.upper()}[/bold magenta]")

        cfg = PipelineConfig()
        cfg.memory.index_type = idx_type
        detector = FastDinoDetector(cfg)

        # Fit memory bank
        t_fit_start = time.perf_counter()
        detector.fit_from_tensors(train_tensors, batch_size=10)
        fit_sec = time.perf_counter() - t_fit_start

        mem_mb = detector.memory_bank.get_memory_size_mb()
        total_vectors = detector.memory_bank.total_patches

        # Inference loop
        y_true = []
        y_scores = []
        gt_masks = []
        pred_maps = []
        latencies = []

        for sample in test_samples:
            res = detector.predict(sample["image"])
            y_true.append(sample["label"])
            y_scores.append(res.anomaly_score)
            gt_masks.append(sample["mask"])
            pred_maps.append(res.anomaly_map)
            latencies.append(res.total_latency_ms)

        img_auroc = compute_image_auroc(np.array(y_true), np.array(y_scores))
        pix_auroc = compute_pixel_auroc(gt_masks, pred_maps)
        pro_score = compute_pro_score(gt_masks, pred_maps)
        mean_latency = float(np.mean(latencies))
        fps = 1000.0 / mean_latency

        results.append({
            "Strategy": idx_type.upper(),
            "Total Vectors": total_vectors,
            "Memory (MB)": f"{mem_mb:.2f}",
            "Latency (ms)": f"{mean_latency:.1f}",
            "FPS": f"{fps:.1f}",
            "Img-AUROC": f"{img_auroc:.4f}",
            "Pix-AUROC": f"{pix_auroc:.4f}",
            "PRO Score": f"{pro_score:.4f}",
        })

    # Display Rich Table
    table = Table(title="Fast-DINO-AD Index Ablation Results", header_style="bold cyan")
    columns = ["Strategy", "Total Vectors", "Memory (MB)", "Latency (ms)", "FPS", "Img-AUROC", "Pix-AUROC", "PRO Score"]
    for col in columns:
        table.add_column(col, justify="center")

    for row in results:
        table.add_row(*[str(row[c]) for c in columns])

    console.print("\n")
    console.print(table)
    console.print("\n[bold green]Benchmark execution completed successfully![/bold green]\n")


if __name__ == "__main__":
    run_benchmark()
