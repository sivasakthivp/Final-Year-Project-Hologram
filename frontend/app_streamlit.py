"""
AI Hologram Medical Visualization System - Streamlit Frontend
"""

import streamlit as st
import requests
import os
import time
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import io

# ── Configuration ──
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "hologram-medical-key-2024")

st.set_page_config(
    page_title="AI Hologram Medical Visualization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling ──
st.markdown('''
<style>
    :root {
        --cyan: #00d4ff;
        --purple: #7b2fff;
        --bg-dark: #020b18;
        --text: #c8e6f5;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(123, 47, 255, 0.1));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .pipeline-step {
        display: inline-block;
        background: linear-gradient(135deg, rgba(123, 47, 255, 0.2), rgba(0, 212, 255, 0.2));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 12px 20px;
        margin: 5px;
        font-weight: 500;
        color: #c8e6f5;
    }
    .pipeline-step.active {
        background: linear-gradient(135deg, #00d4ff, #7b2fff);
        color: white;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    .pipeline-step.completed {
        background: linear-gradient(135deg, #00d97e, #00b85f);
        color: white;
        box-shadow: 0 0 15px rgba(0, 217, 126, 0.3);
    }
    .status-container {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(123, 47, 255, 0.1));
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #00d4ff;
    }
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #00d4ff;
        margin: 10px 0;
    }
    .stat-label {
        font-size: 12px;
        color: #8ab8d8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-item {
        flex: 1;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(123, 47, 255, 0.1));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
''', unsafe_allow_html=True)

# ── Session State ──
if "current_session" not in st.session_state:
    st.session_state.current_session = None
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "gpu_enabled" not in st.session_state:
    st.session_state.gpu_enabled = False


# ── API Client Functions ──

@st.cache_resource
def get_api_session():
    """Cached requests session with connection pooling."""
    session = requests.Session()
    from requests.adapters import HTTPAdapter
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@st.cache_data(ttl=60)
def health_check():
    """Check if backend is running."""
    try:
        resp = get_api_session().get(f"{API_BASE_URL}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.gpu_enabled = data.get("gpu_enabled", False)
            return True
        return False
    except Exception:
        return False


def upload_dicom(file_bytes, filename, patient_id=None):
    """Upload DICOM file to backend."""
    try:
        files = {"file": (filename, file_bytes, "application/dicom")}
        data = {"patient_id": patient_id} if patient_id else {}
        headers = {"Authorization": f"Bearer {API_KEY}"}
        resp = get_api_session().post(
            f"{API_BASE_URL}/api/upload/dicom",
            files=files, data=data, headers=headers, timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


@st.cache_data(ttl=2)
def get_session_status(session_id):
    """Get processing session status."""
    try:
        resp = get_api_session().get(
            f"{API_BASE_URL}/api/session/{session_id}/status", timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=60)
def get_holographic_view(session_id):
    """Fetch 3D holographic visualization."""
    try:
        resp = get_api_session().get(
            f"{API_BASE_URL}/api/session/{session_id}/hologram", timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        traces = data.get("data", [])
        if not traces:
            return None
        return go.Figure(data=traces, layout=data.get("layout", {}))
    except Exception as e:
        st.warning(f"Could not fetch holographic view: {e}")
        return None


@st.cache_data(ttl=3600)
def get_segmentation_mask(session_id):
    """Fetch segmentation mask image."""
    try:
        resp = get_api_session().get(
            f"{API_BASE_URL}/api/session/{session_id}/segmentation", timeout=30
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        if img.size[0] > 1024 or img.size[1] > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_depth_map(session_id):
    """Fetch depth map image."""
    try:
        resp = get_api_session().get(
            f"{API_BASE_URL}/api/session/{session_id}/depth", timeout=30
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        if img.size[0] > 1024 or img.size[1] > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_super_resolution(session_id):
    """Fetch super-resolution enhanced image."""
    try:
        resp = get_api_session().get(
            f"{API_BASE_URL}/api/session/{session_id}/super-resolution", timeout=30
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        if img.size[0] > 1024 or img.size[1] > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


# ── Header ──
st.markdown("# 🏥 AI Hologram Medical Visualization System")
st.markdown("*Advanced Deep Learning for Medical Imaging & 3D Holographic Rendering*")

col1, col2, col3 = st.columns(3)
with col1:
    if health_check():
        st.markdown('<div class="metric-card"><div class="stat-label">Backend</div><div class="stat-value" style="color:#00d97e">Online</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><div class="stat-label">Backend</div><div class="stat-value" style="color:#ff5252">Offline</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="stat-label">Sessions</div><div class="stat-value">{len(st.session_state.session_history)}</div></div>', unsafe_allow_html=True)
with col3:
    gpu_label = "GPU" if st.session_state.gpu_enabled else "CPU"
    st.markdown(f'<div class="metric-card"><div class="stat-label">Device</div><div class="stat-value">{gpu_label}</div></div>', unsafe_allow_html=True)

st.divider()

# ── Main Tabs ──
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Results", "📈 History", "⚙️ Settings"])

# ── Tab 1: Upload & Process ──
with tab1:
    st.subheader("📤 Upload DICOM File")

    col_upload, col_meta = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drag and drop DICOM file or click to browse",
            type=["dcm", "dicom"],
            help="DICOM medical imaging files (CT, MRI, X-Ray)"
        )

    with col_meta:
        st.markdown("### Patient Info")
        patient_id = st.text_input("Patient ID", placeholder="e.g., PAT-001")
        modality = st.selectbox("Modality", ["CT", "MRI", "X-Ray", "Other"])

    if uploaded_file is not None:
        st.success(f"File: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.2f} MB)")

        if st.button("🚀 Upload & Process", type="primary"):
            with st.spinner("Uploading..."):
                result = upload_dicom(uploaded_file.read(), uploaded_file.name, patient_id)

                if result:
                    session_id = result.get("session_id")
                    st.session_state.current_session = session_id
                    st.session_state.session_history.append(session_id)
                    st.success("Upload successful!")

                    with st.container(border=True):
                        st.markdown("### 🔄 Processing Pipeline")

                        pipeline_stages = [
                            ("extracting_dicom", "📥", "DICOM Extract", 15),
                            ("preprocessing", "🖼️", "Preprocess", 25),
                            ("segmenting", "🎯", "Segmentation", 40),
                            ("super_resolution", "📈", "Super-Res", 55),
                            ("volumetric_features", "🧬", "Features", 65),
                            ("depth_estimation", "📐", "Depth", 80),
                            ("pointcloud", "☁️", "Point Cloud", 90),
                            ("hologram_generation", "🌀", "Hologram", 95),
                            ("completed", "✅", "Complete", 100),
                        ]
                        progress_info = {s[0]: s[3] for s in pipeline_stages}

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        pipeline_display = st.empty()

                        start_time = time.time()
                        completed = False

                        while not completed and (time.time() - start_time) < 600:
                            status = get_session_status(session_id)
                            if status:
                                current = status.get("status", "pending")
                                status_text.markdown(f"**Stage:** {current.replace('_', ' ').title()}")
                                pct = progress_info.get(current, 50)
                                progress_bar.progress(pct)

                                html = '<div style="display:flex;flex-wrap:wrap;gap:8px;">'
                                for key, icon, label, _ in pipeline_stages:
                                    if key == current:
                                        html += f'<div class="pipeline-step active">{icon} {label}</div>'
                                    elif pct > progress_info.get(key, 0):
                                        html += f'<div class="pipeline-step completed">{icon} {label}</div>'
                                    else:
                                        html += f'<div class="pipeline-step">{icon} {label}</div>'
                                html += "</div>"
                                pipeline_display.markdown(html, unsafe_allow_html=True)

                                if current == "completed":
                                    completed = True
                                    elapsed = time.time() - start_time
                                    st.balloons()
                                    st.success(f"Processing complete in {elapsed:.1f}s! View results in the Results tab.")
                                elif current == "error":
                                    st.error(f"Error: {status.get('error', 'Unknown')}")
                                    break

                            if not completed:
                                time.sleep(2)
    else:
        st.markdown('''
        <div class="status-container">
            <h3>📋 Get Started</h3>
            <p>1. Select a DICOM medical imaging file</p>
            <p>2. Click "Upload & Process" to start the AI pipeline</p>
            <p>3. View 3D holographic results in the Results tab</p>
        </div>
        ''', unsafe_allow_html=True)

# ── Tab 2: Results ──
with tab2:
    st.subheader("📊 Results & Visualizations")

    if st.session_state.current_session:
        session_id = st.session_state.current_session
        status = get_session_status(session_id)

        if status and status.get("status") == "completed":

            # 3D Holographic View - Main Feature (FULL WIDTH)
            holo_header_col, holo_btn_col = st.columns([8, 1])
            with holo_header_col:
                st.markdown("### 🌀 3D Holographic View")
            with holo_btn_col:
                if st.button("🔄 Refresh", key="refresh_hologram", use_container_width=True):
                    get_holographic_view.clear()
                    st.rerun()
            with st.container(border=True):
                st.markdown("**Interactive 3D anatomical visualization** - Solid mesh organ rendering with holographic effect")
                hologram_fig = get_holographic_view(session_id)
                if hologram_fig:
                    st.plotly_chart(hologram_fig, use_container_width=True, config={
                        "displayModeBar": True,
                        "toImageButtonOptions": {"format": "png", "filename": f"hologram_{session_id[:8]}"}
                    })
                else:
                    st.info("Holographic view generating... Please refresh.")

            st.divider()

            # Visualization Grid - ALL FRAMES DISPLAYED SIMULTANEOUSLY
            st.markdown("### 🎨 Visualization Results - All Frames at Once")

            # Row 1: Segmentation + Depth Map
            col1, col2 = st.columns(2)

            with col1:
                with st.container(border=True):
                    st.markdown("#### 🎯 Segmentation Mask")
                    st.markdown("Organ Detection")
                    seg_img = get_segmentation_mask(session_id)
                    if seg_img:
                        st.image(seg_img, caption="Segmented organs and structures", width=550)
                    else:
                        st.info("Segmentation unavailable")

            with col2:
                with st.container(border=True):
                    st.markdown("#### 📐 Depth Map")
                    st.markdown("3D Surface Estimation")
                    depth_img = get_depth_map(session_id)
                    if depth_img:
                        st.image(depth_img, caption="Depth information for 3D reconstruction", width=550)
                    else:
                        st.info("Depth map unavailable")

            # Row 2: Super-Resolution + Processing Info
            col3, col4 = st.columns(2)

            with col3:
                with st.container(border=True):
                    st.markdown("#### 🔍 Super-Resolution")
                    st.markdown("4x Enhanced Quality")
                    sr_img = get_super_resolution(session_id)
                    if sr_img:
                        st.image(sr_img, caption="4x upscaled enhanced image", width=550)
                    else:
                        st.info("Super-resolution unavailable")

            with col4:
                with st.container(border=True):
                    st.markdown("#### 📋 Processing Information")
                    st.markdown("Session Details & Pipeline")
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.markdown("**Session**")
                        st.markdown(f"```\n{session_id[:20]}\n```")
                        st.markdown(f"**Status:** {status.get('status').upper()}")
                        st.markdown("**Device:** GPU" if st.session_state.gpu_enabled else "**Device:** CPU")
                    with col_info2:
                        st.markdown("**Models Used**")
                        st.markdown("🧠 U-Net Segmentation")
                        st.markdown("📏 MiDaS Depth Est.")
                        st.markdown("⚡ GAN Super-Res (4x)")
                        st.markdown("🔬 3D-CNN (512-dim)")

            st.divider()

            # EXPANDED VIEWING OPTIONS
            st.markdown("### 🖼️ Click to View Larger Versions (800px)")
            
            expand_col1, expand_col2 = st.columns(2)
            
            with expand_col1:
                if st.button("🎯 View Segmentation Full Size", use_container_width=True):
                    with st.expander("🎯 Segmentation Mask (Full Size - 800px)", expanded=True):
                        seg_img = get_segmentation_mask(session_id)
                        if seg_img:
                            st.image(seg_img, caption="Segmentation Mask - Organ Detection", width=800)
                            st.success("Segmentation shows all detected and segmented anatomical structures")
                        else:
                            st.warning("Segmentation unavailable")
                            
            with expand_col2:
                if st.button("📐 View Depth Map Full Size", use_container_width=True):
                    with st.expander("📐 Depth Map (Full Size - 800px)", expanded=True):
                        depth_img = get_depth_map(session_id)
                        if depth_img:
                            st.image(depth_img, caption="Depth Map - 3D Surface Geometry", width=800)
                            st.success("Depth map represents surface geometry for 3D reconstruction")
                        else:
                            st.warning("Depth map unavailable")
            
            expand_col3, expand_col4 = st.columns(2)
            
            with expand_col3:
                if st.button("🔍 View Super-Resolution Full Size", use_container_width=True):
                    with st.expander("🔍 Super-Resolution (Full Size - 800px)", expanded=True):
                        sr_img = get_super_resolution(session_id)
                        if sr_img:
                            st.image(sr_img, caption="Super-Resolution Image - 4x Enhanced Quality", width=800)
                            st.success("Image upscaled 4x using GAN-based super-resolution")
                        else:
                            st.warning("Super-resolution unavailable")
                            
            with expand_col4:
                if st.button("📊 View Metrics & Details", use_container_width=True):
                    with st.expander("📈 Complete Processing Metrics", expanded=True):
                        st.markdown("**Analysis Results**")
                        st.json({
                            "session_id": session_id[:32],
                            "status": status.get("status"),
                            "models_used": ["U-Net Segmentation", "MiDaS Depth", "GAN Super-Resolution", "3D-CNN Features"],
                            "gpu_enabled": st.session_state.gpu_enabled,
                            "device": "GPU" if st.session_state.gpu_enabled else "CPU"
                        })

        elif status and status.get("status") == "error":
            st.error(f"Processing failed: {status.get('error', 'Unknown')}")
        else:
            st.info("⏳ Processing in progress... Please wait and refresh.")
    else:
        st.markdown('''
        <div class="status-container">
            <h3>📤 No Active Session</h3>
            <p>Upload a DICOM file in the Upload tab to see results here.</p>
        </div>
        ''', unsafe_allow_html=True)

# ── Tab 3: History ──
with tab3:
    st.subheader("📈 Session History")

    if st.session_state.session_history:
        for i, sid in enumerate(reversed(st.session_state.session_history)):
            with st.container(border=True):
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    s = get_session_status(sid)
                    badge = "✅ Complete" if s and s.get("status") == "completed" else "⏳ Processing"
                    st.markdown(f"**Session {len(st.session_state.session_history) - i}** - {badge}")
                    st.caption(f"{sid[:32]}...")
                with c2:
                    if st.button("📊 View", key=f"v_{sid}", use_container_width=True):
                        st.session_state.current_session = sid
                        st.rerun()
                with c3:
                    if st.button("🗑️", key=f"d_{sid}", use_container_width=True):
                        st.session_state.session_history.remove(sid)
                        if st.session_state.current_session == sid:
                            st.session_state.current_session = None
                        st.rerun()
    else:
        st.info("No sessions yet. Upload a DICOM file to get started.")

# ── Tab 4: Settings ──
with tab4:
    st.subheader("⚙️ Settings")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### API Configuration")
        with st.container(border=True):
            st.text_input("Backend URL", value=API_BASE_URL, disabled=True)
            if st.button("🔗 Test Connection", use_container_width=True):
                if health_check():
                    st.success("Connected!")
                else:
                    st.error("Cannot reach backend")

    with c2:
        st.markdown("### System Info")
        with st.container(border=True):
            gpu_status = "GPU" if st.session_state.gpu_enabled else "CPU"
            st.markdown(f"**Device:** {gpu_status}")
            st.markdown("**Version:** 1.0.0")
            st.markdown("**Framework:** Streamlit + FastAPI")
            st.markdown(f"**API Docs:** {API_BASE_URL}/docs")
