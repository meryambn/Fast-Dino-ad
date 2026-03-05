"""
Fast-DINO-AD: Streamlit Industrial Visual Inspection Dashboard.
Interactive edge anomaly detection, real-time heatmap visualization, and performance profiling.
"""

import time
import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.config import PipelineConfig
from src.pipeline.detector import FastDinoDetector
from src.utils.dataset import SyntheticIndustrialDataset
from src.utils.visualization import create_heatmap_overlay, normalize_heatmap

st.set_page_config(
    page_title="Fast-DINO-AD | Industrial Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for industrial inspection station look
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #334155;
    }
    .badge-pass {
        background-color: #064e3b;
        color: #34d399;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-fail {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading DINOv2 Feature Extractor & Calibrating Memory Bank...")
def get_cached_detector(index_type: str, top_k: int, sigma: float):
    config = PipelineConfig()
    config.memory.index_type = index_type
    config.detector.top_k = top_k
    config.detector.gaussian_sigma = sigma

    detector = FastDinoDetector(config)

    # Train on 25 nominal industrial reference samples
    train_dataset = SyntheticIndustrialDataset(num_samples=25, anomaly_ratio=0.0, seed=42)
    train_tensors = torch.stack([train_dataset[i]["image"] for i in range(len(train_dataset))])
    detector.fit_from_tensors(train_tensors, batch_size=10)

    return detector


st.title("🏭 Fast-DINO-AD: High-Throughput Edge Inspection")
st.caption("Self-Supervised ViT Feature Pyramids + Quantized Vector Memory Bank (FAISS IVF-PQ)")

# Sidebar Settings
st.sidebar.header("⚙️ Pipeline Configuration")
index_type = st.sidebar.selectbox(
    "Memory Index Strategy",
    options=["ivf_pq", "hnsw", "exact"],
    format_func=lambda x: {
        "ivf_pq": "IVF-PQ (Product Quantization - 85% Compressed)",
        "hnsw": "HNSW (Graph ANN - Sub-millisecond)",
        "exact": "Exact FlatL2 (Baseline Benchmark)"
    }[x]
)

threshold = st.sidebar.slider("Anomaly Decision Threshold", min_value=0.1, max_value=1.5, value=0.48, step=0.02)
top_k = st.sidebar.slider("Top-K Nearest Neighbors", min_value=1, max_value=10, value=3)
gaussian_sigma = st.sidebar.slider("Heatmap Smoothing Sigma", min_value=1.0, max_value=8.0, value=3.5, step=0.5)

detector = get_cached_detector(index_type, top_k, gaussian_sigma)

# Memory stats
mem_mb = detector.memory_bank.get_memory_size_mb()
st.sidebar.markdown("---")
st.sidebar.metric("Memory Footprint", f"{mem_mb:.2f} MB")
st.sidebar.metric("Nominal Patch Vectors", f"{detector.memory_bank.total_patches:,}")

# Sample selection
st.subheader("1. Select Inspection Sample")
dataset = SyntheticIndustrialDataset(num_samples=8, anomaly_ratio=0.6, seed=777)

col_s1, col_s2 = st.columns([1, 2])
sample_options = [
    "Sample 0: Nominal Metal Surface (PASS)",
    "Sample 1: Scratch Flaw (DEFECT)",
    "Sample 2: Oxidation Stain (DEFECT)",
    "Sample 3: Burn Void Defect (DEFECT)",
    "Sample 4: Nominal Brushed Finish (PASS)",
    "Sample 5: Micro-crack Defect (DEFECT)",
]
selected_option = col_s1.selectbox("Preset Test Specimens", sample_options)
sample_idx = int(selected_option.split(":")[0].split()[-1])
sample_data = dataset[sample_idx]

uploaded_file = col_s2.file_uploader("Or upload custom industrial inspection image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    test_image = Image.open(uploaded_file).convert("RGB")
    gt_mask = None
else:
    test_image = Image.fromarray(sample_data["raw_image"])
    gt_mask = sample_data["mask"]

# Run Prediction
result = detector.predict(test_image, threshold=threshold)

st.markdown("---")
st.subheader("2. Real-Time Inspection Telemetry")

m1, m2, m3, m4 = st.columns(4)
with m1:
    if result.is_anomaly:
        st.markdown('<div class="badge-fail">CRITICAL DEFECT DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-pass">NOMINAL SPECIMEN (PASS)</div>', unsafe_allow_html=True)

with m2:
    st.metric("Anomaly Score", f"{result.anomaly_score:.3f}", delta=f"{result.anomaly_score - threshold:.3f}", delta_color="inverse")

with m3:
    st.metric("Inference Latency", f"{result.total_latency_ms:.1f} ms", f"{result.fps:.1f} FPS")

with m4:
    st.metric("Search Time (FAISS)", f"{result.search_latency_ms:.2f} ms")

st.markdown("---")
st.subheader("3. Spatial Anomaly Localization")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.caption("Input Sample")
    st.image(test_image, use_container_width=True)

with c2:
    st.caption("DINOv2 Thermal Map")
    norm_heat = normalize_heatmap(result.anomaly_map)
    fig_heat, ax = plt.subplots(figsize=(4, 4), facecolor="none")
    ax.imshow(norm_heat, cmap="jet")
    ax.axis("off")
    st.pyplot(fig_heat, use_container_width=True)
    plt.close(fig_heat)

with c3:
    st.caption("Translucent Defect Overlay")
    overlay = create_heatmap_overlay(test_image, result.anomaly_map, alpha=0.45)
    st.image(overlay, use_container_width=True)

with c4:
    st.caption("Segmented Defect Mask")
    st.image((result.binary_mask.astype(np.uint8) * 255), use_container_width=True)
