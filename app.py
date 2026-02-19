"""
app.py — Voice Timbre Transfer · Streamlit Frontend
====================================================
Upload a vocal track, select a target voice model, and convert.
"""

import os
import sys
import time
import tempfile
import logging
from pathlib import Path

import streamlit as st

# ─── Setup Logging ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")
logger = logging.getLogger(__name__)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Timbre Transfer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-top: 4px;
        font-weight: 300;
    }

    /* ── Cards ── */
    .glass-card {
        background: rgba(30, 30, 46, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(102, 126, 234, 0.5);
    }

    /* ── Status badges ── */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .badge-ready {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-processing {
        background: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }
    .badge-empty {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ── Metric cards ── */
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 10px 0;
    }
    .metric-item {
        flex: 1;
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 10px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.95);
        border-right: 1px solid rgba(102, 126, 234, 0.15);
    }
    .sidebar-header {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #667eea;
        font-weight: 600;
        margin-bottom: 6px;
        margin-top: 16px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35) !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #22c55e, #16a34a) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        width: 100%;
    }

    /* ── Divider ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Imports (engine) ─────────────────────────────────────────────────────────
from engine import (
    discover_models,
    ensure_assets,
    get_audio_info as _get_audio_info_engine,
    VoiceConverter,
    VOICES_DIR,
    OUTPUT_DIR,
)

# Cache audio info to avoid re-running expensive librosa load
@st.cache_data
def get_cached_audio_info(file_path: str) -> dict:
    return _get_audio_info_engine(file_path)

# ─── Session State Init ──────────────────────────────────────────────────────
if "converter" not in st.session_state:
    st.session_state.converter = None
if "converted_path" not in st.session_state:
    st.session_state.converted_path = None
if "assets_ready" not in st.session_state:
    st.session_state.assets_ready = False
if "last_upload_name" not in st.session_state:
    st.session_state.last_upload_name = None
if "tmp_input_path" not in st.session_state:
    st.session_state.tmp_input_path = None


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🎙️ Voice Timbre Transfer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    'Transform vocal timbre with RVC-powered neural voice conversion'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─── Asset Check ──────────────────────────────────────────────────────────────
if not st.session_state.assets_ready:
    with st.status("🔍 Checking required model weights...", expanded=True) as status:
        try:
            st.write("Looking for `hubert_base.pt` and `rmvpe.pt`...")
            ensure_assets(
                progress_callback=lambda p, d: st.write(f"⬇️  Downloading {d}… {p*100:.0f}%")
            )
            st.session_state.assets_ready = True
            status.update(label="✅ All backbone weights ready", state="complete")
        except Exception as e:
            status.update(label="❌ Asset download failed", state="error")
            st.error(f"Could not download required weights: {e}")
            st.stop()


# ─── Discover Voice Models ───────────────────────────────────────────────────
models = discover_models()
model_names = [m["name"] for m in models]


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Conversion Settings")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Voice Model ──
    st.markdown('<p class="sidebar-header">🗣️ Target Voice</p>', unsafe_allow_html=True)
    if model_names:
        selected_model = st.selectbox(
            "Voice Model",
            model_names,
            index=0,
            help="Auto-discovered from the `voices/` folder. Drop a .pth file there to add more.",
            label_visibility="collapsed",
        )
        model_info = next(m for m in models if m["name"] == selected_model)
        st.caption(f"📦 {model_info['size_mb']} MB")
    else:
        st.warning("No voice models found!")
        st.caption(f"Place `.pth` files in:\n`{VOICES_DIR}`")
        selected_model = None

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Pitch ──
    st.markdown('<p class="sidebar-header">🎵 Pitch Transpose</p>', unsafe_allow_html=True)
    pitch = st.slider(
        "Semitones",
        min_value=-24,
        max_value=24,
        value=12,
        step=1,
        help=(
            "+12 = one octave up (typical Male→Female). "
            "0 = timbre only, no pitch shift. "
            "-12 = one octave down."
        ),
        label_visibility="collapsed",
    )
    pitch_label = f"{pitch:+d} semitones"
    if pitch == 12:
        pitch_label += "  ·  M → F"
    elif pitch == -12:
        pitch_label += "  ·  F → M"
    elif pitch == 0:
        pitch_label += "  ·  Timbre only"
    st.caption(pitch_label)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Advanced ──
    with st.expander("🔬 Advanced Parameters", expanded=False):
        f0_method = st.selectbox(
            "f₀ Extraction Method",
            ["rmvpe", "crepe", "harvest", "pm"],
            index=0,
            help="rmvpe gives the cleanest results for vocal tracks.",
        )
        index_rate = st.slider(
            "Index Rate",
            0.0, 1.0, 0.75, 0.05,
            help="Feature search ratio. Higher = more target voice character.",
        )
        filter_radius = st.slider(
            "Filter Radius",
            0, 7, 3,
            help="Median filter radius for f0 smoothing.",
        )
        rms_mix_rate = st.slider(
            "RMS Mix Rate",
            0.0, 1.0, 0.25, 0.05,
            help="Volume envelope blend. 0 = output, 1 = input.",
        )
        protect = st.slider(
            "Consonant Protection",
            0.0, 0.5, 0.33, 0.01,
            help="Protects voiceless consonants and breath sounds.",
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Device ──
    st.markdown('<p class="sidebar-header">💻 Device</p>', unsafe_allow_html=True)

    # Detect GPU availability without importing torch in this process
    # (avoids circular import issues with some torch versions)
    if "gpu_info" not in st.session_state:
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "import torch; "
                 "print('cuda' if torch.cuda.is_available() else '', end=','); "
                 "print('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else '', end='')"],
                capture_output=True, text=True, timeout=15
            )
            parts = result.stdout.strip().split(",")
            st.session_state.gpu_info = {
                "cuda": len(parts) > 0 and parts[0] == "cuda",
                "mps": len(parts) > 1 and parts[1] == "mps",
            }
        except Exception:
            st.session_state.gpu_info = {"cuda": False, "mps": False}

    has_cuda = st.session_state.gpu_info["cuda"]
    has_mps = st.session_state.gpu_info["mps"]

    device_options = ["cpu"]
    if has_cuda:
        device_options.insert(0, "cuda:0")
    if has_mps:
        device_options.insert(0, "mps")

    device = st.selectbox(
        "Compute Device",
        device_options,
        index=0,
        label_visibility="collapsed",
    )
    if device == "cpu":
        st.caption("⚡ GPU not detected — using CPU (slower)")
    else:
        st.caption(f"🚀 Using {device.upper()}")

    # ── Resource Management ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-header">🧠 Memory</p>', unsafe_allow_html=True)
    
    if st.button("🧹 Unload Model & Clear RAM", help="Free up resources when idle"):
        if st.session_state.converter:
            st.session_state.converter.unload_model()
            st.rerun()

    # ── Model count ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-badge badge-ready">{len(models)} model(s) available</span>',
        unsafe_allow_html=True,
    )


# ─── Main Content ─────────────────────────────────────────────────────────────
col_upload, col_result = st.columns(2, gap="large")

with col_upload:
    st.markdown("### 📤 Upload Vocal Track")
    uploaded_file = st.file_uploader(
        "Drop a .wav or .mp3 file",
        type=["wav", "mp3"],
        label_visibility="collapsed",
        help="Supported formats: WAV, MP3",
    )

    if uploaded_file:
        # Check if file changed since last run to avoid re-writing temp
        file_id = f"{uploaded_file.name}-{uploaded_file.size}"
        
        if st.session_state.last_upload_name != file_id or not st.session_state.tmp_input_path or not os.path.exists(st.session_state.tmp_input_path):
            # Save to temp file
            suffix = Path(uploaded_file.name).suffix
            # Use fixed name or delete old one if needed, but namedtemp with delete=False implies we manage it
            tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_input.write(uploaded_file.getvalue())
            tmp_input.flush()
            tmp_input.close()
            
            st.session_state.tmp_input_path = tmp_input.name
            st.session_state.last_upload_name = file_id
            # Reset conversion on new file
            st.session_state.converted_path = None
        
        tmp_input_path = st.session_state.tmp_input_path

        # Audio info (Cached)
        info = get_cached_audio_info(tmp_input_path)

        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:600; color:#e2e8f0;">🎵 {uploaded_file.name}</span>
                <span class="status-badge badge-ready">Ready</span>
            </div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-label">Duration</div>
                    <div class="metric-value">{info['duration_s']:.1f}s</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Sample Rate</div>
                    <div class="metric-value">{info['sample_rate']:,} Hz</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Format</div>
                    <div class="metric-value">{info['format']}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Channels</div>
                    <div class="metric-value">{info['channels']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Original Audio**")
        st.audio(uploaded_file, format=f"audio/{info['format'].lower()}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Convert Button ──
        can_convert = selected_model is not None
        if not can_convert:
            st.warning("⚠️ No voice model selected. Add a .pth file to the `voices/` folder.")

        if st.button(
            f"🔄  Convert to {selected_model or '...'}" if can_convert else "No model available",
            disabled=not can_convert,
            use_container_width=True,
        ):
            st.session_state.converted_path = None

            # Prepare output path
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_name = f"{Path(uploaded_file.name).stem}_{selected_model}_p{pitch:+d}.wav"
            out_path = str(OUTPUT_DIR / out_name)

            with st.spinner(f"🎛️ Converting with **{selected_model}** (pitch {pitch:+d})…"):
                try:
                    start = time.time()

                    # Init / reuse converter
                    if (
                        st.session_state.converter is None
                        or st.session_state.converter.device != device
                    ):
                        st.session_state.converter = VoiceConverter(
                            device=device,
                            voices_dir=VOICES_DIR,
                        )

                    converter = st.session_state.converter
                    converter.load_model(selected_model)
                    result_path = converter.convert(
                        input_path=tmp_input_path,
                        output_path=out_path,
                        pitch=pitch,
                        method=f0_method,
                        index_rate=index_rate,
                        filter_radius=filter_radius,
                        rms_mix_rate=rms_mix_rate,
                        protect=protect,
                    )

                    elapsed = time.time() - start
                    st.session_state.converted_path = result_path
                    st.session_state.conversion_time = round(elapsed, 1)
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Conversion failed: {e}")
                    logger.exception("Conversion error")

    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem 1.5rem;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">🎙️</div>
            <div style="color:#9ca3af; font-size:1rem;">
                Drag & drop a <strong>.wav</strong> or <strong>.mp3</strong> vocal file<br>
                <span style="font-size:0.85rem;">to begin timbre transfer</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Result Column ────────────────────────────────────────────────────────────
with col_result:
    st.markdown("### 🎧 Converted Output")

    if st.session_state.converted_path and Path(st.session_state.converted_path).exists():
        result_path = st.session_state.converted_path

        out_info = _get_audio_info_engine(result_path)
        conversion_time = st.session_state.get("conversion_time", "?")

        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:600; color:#e2e8f0;">🎧 {Path(result_path).name}</span>
                <span class="status-badge badge-ready">Complete</span>
            </div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-label">Duration</div>
                    <div class="metric-value">{out_info['duration_s']:.1f}s</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Processing</div>
                    <div class="metric-value">{conversion_time}s</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Pitch</div>
                    <div class="metric-value">{pitch:+d}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Converted Audio**")
        with open(result_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Download button
        st.download_button(
            label=f"⬇️  Download {Path(result_path).name}",
            data=audio_bytes,
            file_name=Path(result_path).name,
            mime="audio/wav",
            use_container_width=True,
        )

    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem 1.5rem;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">🎧</div>
            <div style="color:#9ca3af; font-size:1rem;">
                Converted output will appear here<br>
                <span style="font-size:0.85rem;">after processing</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#6b7280; font-size:0.8rem; padding:1rem 0;">'
    'Voice Timbre Transfer · Powered by '
    '<span style="color:#667eea;">RVC</span> + '
    '<span style="color:#764ba2;">librosa</span> + '
    '<span style="color:#f093fb;">Streamlit</span>'
    '</div>',
    unsafe_allow_html=True,
)
