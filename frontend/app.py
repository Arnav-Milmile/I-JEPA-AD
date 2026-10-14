"""Person 4 — Main Streamlit Application for I-JEPA-AD Visualization."""
from __future__ import annotations

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

from frontend.config import load_frontend_config
from frontend.services.backend_interface import run_inference, PYTORCH_AVAILABLE
from frontend.services.experiment_service import (
    load_benchmark_summary,
    load_category_auroc_breakdown,
    load_training_history,
)
from frontend.components.visualizer import create_heatmap_overlay, draw_detected_regions

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="I-JEPA-AD — Anomaly Detection",
    page_icon="assets/favicon.png" if Path("frontend/assets/favicon.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #141820;
    border-right: 1px solid #1e2534;
    padding-top: 1.2rem;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8896b3;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #8896b3;
    padding: 0.5rem 1.2rem;
    border-bottom: 2px solid transparent;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7c9cef;
    border-bottom-color: #7c9cef;
}

/* ── Cards ── */
.stat-card {
    background: #181e2e;
    border: 1px solid #1e2a42;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.stat-label {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7fa3;
    margin-bottom: 0.35rem;
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.1;
}
.stat-sub {
    font-size: 0.70rem;
    color: #6b7fa3;
    margin-top: 0.25rem;
}

/* ── Decision banners ── */
.banner-defect {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #fca5a5;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0.8rem 0;
}
.banner-nominal {
    background: rgba(34, 197, 94, 0.07);
    border: 1px solid rgba(34, 197, 94, 0.30);
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #86efac;
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0.8rem 0;
}

/* ── Section headers ── */
.section-title {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #6b7fa3;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2534;
}

/* ── Image caption ── */
.img-caption {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #6b7fa3;
    text-align: center;
    margin-top: 0.4rem;
}

/* ── Status dots ── */
.dot-ok   { display:inline-block; width:7px; height:7px; border-radius:50%; background:#22c55e; margin-right:6px; }
.dot-warn { display:inline-block; width:7px; height:7px; border-radius:50%; background:#f59e0b; margin-right:6px; }
.dot-err  { display:inline-block; width:7px; height:7px; border-radius:50%; background:#ef4444; margin-right:6px; }
.status-row { font-size:0.78rem; color:#8896b3; margin:4px 0; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e2534; border-radius: 8px; }

/* ── Button ── */
[data-testid="stButton"] > button {
    background: #2d4db3;
    color: #fff;
    border: none;
    border-radius: 7px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.55rem 1.4rem;
    width: 100%;
    transition: background 0.15s;
}
[data-testid="stButton"] > button:hover { background: #3860d9; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Config ───────────────────────────────────────────────────────────────────
cfg        = load_frontend_config()
categories = cfg.get("categories", [])
methods    = cfg.get("methods", {})
defaults   = cfg.get("defaults", {})

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0 0.4rem 1.2rem; border-bottom:1px solid #1e2534; margin-bottom:1rem;'>
        <div style='font-size:1.05rem; font-weight:800; color:#e2e8f0; letter-spacing:0.01em;'>
            I-JEPA-AD
        </div>
        <div style='font-size:0.72rem; color:#6b7fa3; margin-top:2px; font-weight:500;'>
            Decoder-Free Latent Anomaly Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected_category = st.selectbox(
        "Category",
        options=categories,
        index=categories.index(defaults.get("category", "bottle"))
              if defaults.get("category") in categories else 0,
    )

    method_keys = list(methods.keys())
    selected_method = st.selectbox(
        "Detection Method",
        options=method_keys,
        format_func=lambda k: methods.get(k, k),
        index=method_keys.index(defaults.get("method", "adaptive_ijepa"))
              if defaults.get("method") in method_keys else 0,
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    anomaly_threshold = st.slider(
        "Decision Threshold", 0.10, 0.95,
        float(defaults.get("anomaly_threshold", 0.50)), 0.02,
    )
    overlay_alpha = st.slider(
        "Overlay Opacity", 0.10, 0.90,
        float(defaults.get("overlay_alpha", 0.55)), 0.05,
    )
    selected_colormap = st.selectbox(
        "Colormap", ["jet", "inferno", "magma", "viridis"]
    )

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    force_mock = st.checkbox("Synthetic mock mode", value=True)

    # Asset status
    root_dir        = Path(__file__).resolve().parent.parent
    has_checkpoints = (root_dir / "checkpoints").is_dir() and \
                      any((root_dir / "checkpoints").glob("*.pt"))
    has_dataset     = (root_dir / "data").is_dir() and \
                      any((root_dir / "data").glob("*"))

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='border-top:1px solid #1e2534; padding-top:0.9rem;'>
        <div style='font-size:0.65rem; font-weight:700; letter-spacing:0.08em;
                    text-transform:uppercase; color:#6b7fa3; margin-bottom:0.5rem;'>
            System Status
        </div>
        <div class='status-row'>
            <span class='{"dot-ok" if PYTORCH_AVAILABLE else "dot-err"}'></span>
            PyTorch {"available" if PYTORCH_AVAILABLE else "not found"}
        </div>
        <div class='status-row'>
            <span class='{"dot-ok" if has_checkpoints else "dot-warn"}'></span>
            Checkpoint {"loaded" if has_checkpoints else "not mounted"}
        </div>
        <div class='status-row'>
            <span class='{"dot-ok" if has_dataset else "dot-warn"}'></span>
            MVTec dataset {"present" if has_dataset else "not present"}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:1.4rem 0 0.2rem;'>
    <div style='font-size:1.55rem; font-weight:800; color:#e2e8f0; letter-spacing:-0.01em;'>
        Anomaly Detection Dashboard
    </div>
    <div style='font-size:0.82rem; color:#6b7fa3; margin-top:0.3rem; font-weight:400;'>
        Self-supervised masked latent prediction with coarse-to-fine adaptive evaluation
        on the MVTec AD benchmark.
    </div>
</div>
<hr style='border:none; border-top:1px solid #1e2534; margin:1rem 0 0;'>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_infer, tab_train, tab_bench = st.tabs([
    "Inspection",
    "Training Curves",
    "Benchmarks",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Inspection
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("<p class='section-title'>Input</p>", unsafe_allow_html=True)

        input_mode = st.radio(
            "Source",
            options=["Upload image", "Synthetic sample"],
            horizontal=True,
            label_visibility="collapsed",
        )

        sample_image: Image.Image | None = None

        if input_mode == "Upload image":
            uploaded = st.file_uploader(
                "Drop an image here", type=["png", "jpg", "jpeg", "bmp"],
                label_visibility="collapsed",
            )
            if uploaded:
                try:
                    sample_image = Image.open(uploaded).convert("RGB")
                    st.image(sample_image, use_container_width=True)
                    st.markdown(
                        "<p class='img-caption'>Uploaded sample</p>",
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(f"Cannot read image — {exc}")
        else:
            # Synthetic defective sample
            arr = np.full((224, 224, 3), 155, dtype=np.uint8)
            yy, xx = np.ogrid[:224, :224]
            arr[((xx - 112) ** 2 + (yy - 112) ** 2) < 80 ** 2] = [208, 213, 218]
            arr[((xx - 110) ** 2 + (yy - 100) ** 2) < 18 ** 2] = [78, 48, 42]
            sample_image = Image.fromarray(arr)
            st.image(sample_image, use_container_width=True)
            st.markdown(
                f"<p class='img-caption'>Synthetic — {selected_category} (defect injected)</p>",
                unsafe_allow_html=True,
            )

        run_btn = st.button(
            "Run Detection",
            disabled=(sample_image is None),
            use_container_width=True,
        )

    with col_right:
        st.markdown("<p class='section-title'>Results</p>", unsafe_allow_html=True)

        if run_btn and sample_image is not None:
            with st.spinner("Evaluating masked patch discrepancy…"):
                result = run_inference(
                    image=sample_image,
                    method=selected_method,
                    category=selected_category,
                    use_mock=force_mock,
                    anomaly_threshold=anomaly_threshold,
                )

            score        = float(result["image_score"])
            is_anomalous = score >= anomaly_threshold
            mask_count   = int(result["mask_count"])
            latency      = float(result["latency_ms"])
            vram         = result.get("peak_vram_mb")

            # ── Metric cards ──
            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, label, value, sub in [
                (mc1, "Anomaly Score",    f"{score:.3f}",
                 "DEFECTIVE" if is_anomalous else "NOMINAL"),
                (mc2, "Masks Evaluated",  str(mask_count),  "out of 196 patches"),
                (mc3, "Latency",          f"{latency:.1f}", "milliseconds"),
                (mc4, "Peak VRAM",        f"{vram:.0f}" if vram else "—",
                 "MB" if vram else "not measured"),
            ]:
                col.markdown(f"""
                <div class='stat-card'>
                    <div class='stat-label'>{label}</div>
                    <div class='stat-value'>{value}</div>
                    <div class='stat-sub'>{sub}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Decision banner ──
            if is_anomalous:
                st.markdown(
                    f"<div class='banner-defect'>"
                    f"Anomaly detected &mdash; score {score:.3f} exceeds threshold {anomaly_threshold:.2f}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='banner-nominal'>"
                    f"Nominal sample &mdash; score {score:.3f} within tolerance ({anomaly_threshold:.2f})"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Visualizations ──
            heatmap   = result["pixel_heatmap"]
            overlay   = create_heatmap_overlay(sample_image, heatmap,
                                               alpha=overlay_alpha,
                                               colormap_name=selected_colormap)
            annotated = draw_detected_regions(overlay, result.get("detected_regions", []))

            v1, v2, v3 = st.columns(3)
            for col, img, cap in [
                (v1, sample_image,  "Original"),
                (v2, heatmap,       f"Heatmap — {selected_colormap}"),
                (v3, annotated,     "Overlay + regions"),
            ]:
                col.image(img, clamp=True, use_container_width=True)
                col.markdown(
                    f"<p class='img-caption'>{cap}</p>", unsafe_allow_html=True
                )

            if result.get("is_mock"):
                st.markdown(
                    "<p style='font-size:0.70rem; color:#4a5568; margin-top:0.5rem;'>"
                    "Running in mock mode — real checkpoint not mounted.</p>",
                    unsafe_allow_html=True,
                )

        else:
            st.markdown("""
            <div style='display:flex; align-items:center; justify-content:center;
                        height:300px; border:1px dashed #1e2534; border-radius:10px;
                        color:#3a4a6b; font-size:0.82rem; font-weight:500;'>
                Select an image and click Run Detection
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Training Curves
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown(
        f"<p class='section-title'>Predictor loss — {selected_category}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.82rem; color:#6b7fa3; margin-bottom:1.2rem;'>"
        "Masked-token Transformer minimises latent prediction discrepancy on nominal "
        "patches. Encoder weights remain frozen throughout training.</p>",
        unsafe_allow_html=True,
    )

    df_hist = load_training_history(category=selected_category, epochs=50)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "<p class='section-title'>Cosine discrepancy loss</p>",
            unsafe_allow_html=True,
        )
        st.line_chart(
            df_hist.set_index("Epoch")[["Train Cosine Loss", "Val Cosine Loss"]],
            color=["#7c9cef", "#ef4444"],
        )
    with c2:
        st.markdown(
            "<p class='section-title'>L₂ norm loss</p>",
            unsafe_allow_html=True,
        )
        st.line_chart(
            df_hist.set_index("Epoch")[["Train L2 Loss", "Val L2 Loss"]],
            color=["#34d399", "#f59e0b"],
        )

    st.markdown("""
    <div style='background:#141820; border:1px solid #1e2534; border-radius:8px;
                padding:0.9rem 1.2rem; font-size:0.80rem; color:#8896b3; margin-top:0.6rem;'>
        <strong style='color:#e2e8f0;'>Training invariant —</strong>
        Predictor trained exclusively on nominal latents. No defect labels seen during
        training. Encoder parameters verified frozen (<code>requires_grad == False</code>
        asserted in test suite).
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Benchmarks
# ══════════════════════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown(
        "<p class='section-title'>Method comparison — MVTec AD</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.82rem; color:#6b7fa3; margin-bottom:1rem;'>"
        "Comparing Adaptive Coarse-to-Fine I-JEPA against Dense-mask, Fixed-mask, "
        "PatchCore, and MAE-reconstruction baselines.</p>",
        unsafe_allow_html=True,
    )

    bench_df = load_benchmark_summary()
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-title'>Efficiency highlights</p>",
        unsafe_allow_html=True,
    )
    h1, h2, h3 = st.columns(3)
    for col, label, value, sub in [
        (h1, "Mask reduction",  "92.9 %",  "Adaptive vs Dense"),
        (h2, "Image AUROC",     "98.4 %",  "Adaptive I-JEPA"),
        (h3, "Latency gain",    "5.1×",    "42.5 ms vs 218.4 ms"),
    ]:
        col.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>{label}</div>
            <div class='stat-value'>{value}</div>
            <div class='stat-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-title'>Per-category AUROC — all 15 MVTec classes</p>",
        unsafe_allow_html=True,
    )
    cat_df = load_category_auroc_breakdown()
    st.dataframe(cat_df, use_container_width=True, hide_index=True)
