import streamlit as st
import os
import sys
import time
import tempfile
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import soundfile as sf
import librosa
import torch
from aasist.predict import predict as detect_ai_voice
from dnn.predict import predict as detect_replayed_voice
from pathlib import Path

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(__file__))

# Configure page
st.set_page_config(
    page_title="Voice Spoofing Detection",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bonafide-card {
        background: linear-gradient(45deg, #4A9782, #004030);
        border-left: 5px solid #28a745;
    }
    
    .spoofed-card {
        background: linear-gradient(45deg, #FF894F, #FE7743);
        border-left: 5px solid #dc3545;
    }
    .super-spoofed-card {
        background: linear-gradient(45deg, #FF5733, #dc3545);
        border-left: 5px solid #DC3C22;
    }
    
    .neutral-card {
        background: linear-gradient(45deg, #fff3cd, #ffeaa7);
        border-left: 5px solid #ffc107;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_models():
    """Cache models to avoid reloading"""
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = True
        # Models will be loaded when needed during prediction
    return True


def create_confidence_gauge(confidence, title):
    """Create a gauge chart for confidence score"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 80], "color": "yellow"},
                    {"range": [80, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def visualize_audio(audio_path):
    """Create audio waveform and spectrogram visualizations"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Create time axis
        time = np.linspace(0, len(audio) / sr, len(audio))

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Waveform", "Spectrogram"),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        )

        # Add waveform
        fig.add_trace(
            go.Scatter(
                x=time,
                y=audio,
                mode="lines",
                name="Waveform",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )

        # Add spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        fig.add_trace(
            go.Heatmap(z=D, colorscale="Viridis", name="Spectrogram"), row=2, col=1
        )

        # Update layout
        fig.update_layout(height=600, title_text="Audio Analysis", showlegend=False)

        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_xaxes(title_text="Time Frames", row=2, col=1)
        fig.update_yaxes(title_text="Frequency Bins", row=2, col=1)

        return fig
    except Exception as e:
        st.error(f"Error visualizing audio: {str(e)}")
        return None


def predict_spoofing(audio_path):
    """Run both spoofing detection models"""
    results = {}
    errors = []

    with st.spinner("Running AI voice detection..."):
        try:
            # AI voice detection (AASIST)
            model_path = (
                Path(__file__).parent / "aasist" / "models" / "weights" / "AASIST.pth"
            )
            config_path = Path(__file__).parent / "aasist" / "config" / "AASIST.conf"
            ai_result = detect_ai_voice(audio_path, model_path, config_path)
            results["ai_detection"] = ai_result
        except Exception as e:
            errors.append(f"AI detection error: {str(e)}")
            results["ai_detection"] = None

    with st.spinner("Running replay voice detection..."):
        try:
            # Replay voice detection (CNN-GRU)
            model_path = (
                Path(__file__).parent / "dnn" / "models" / "cnn_gru" / "model.pt"
            )
            config_path = (
                Path(__file__).parent / "dnn" / "models" / "cnn_gru" / "cfg.yaml"
            )
            replay_result = detect_replayed_voice(audio_path, model_path, config_path)
            results["replay_detection"] = replay_result
        except Exception as e:
            errors.append(f"Replay detection error: {str(e)}")
            results["replay_detection"] = None

    return results, errors


def display_results(results, errors):
    """Display prediction results in an organized manner"""
    st.markdown(
        '<div class="section-header">🔍 Detection Results</div>', unsafe_allow_html=True
    )

    if errors:
        st.error("Some models encountered errors:")
        for error in errors:
            st.error(f"• {error}")
        st.info("Check that model files exist in the correct locations.")

    col1, col2 = st.columns(2)

    # AI Voice Detection Results
    with col1:
        st.subheader("🤖 AI Voice Detection (with AASIST)")
        if results.get("ai_detection"):
            ai_result = results["ai_detection"]
            prediction = ai_result["prediction"]
            confidence = ai_result["confidence"]

            # Determine result type for styling
            if prediction == 1:  # Bonafide
                card_class = "bonafide-card"
                result_text = "✅ **REAL VOICE**"
                interpretation = "This appears to be a genuine human voice."
            else:  # Spoofed
                card_class = "spoofed-card"
                result_text = "⚠️ **AI GENERATED**"
                interpretation = "This appears to be AI-generated voice."

            st.markdown(
                f"""
            <div class="result-card {card_class}">
                <h3>{result_text}</h3>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><em>{interpretation}</em></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Confidence gauge
            fig_ai = create_confidence_gauge(confidence, "AI Detection Confidence")
            st.plotly_chart(fig_ai, use_container_width=True)

        else:
            st.error("AI detection model failed to run")

    # Replay Voice Detection Results
    with col2:
        st.subheader("🔁 Replay Attack Detection (with CNN-GRU)")
        if results.get("replay_detection"):
            replay_result = results["replay_detection"]
            prediction = replay_result["prediction"]
            confidence = replay_result["confidence"]

            # Determine result type for styling
            if prediction == 1:  # Bonafide
                card_class = "bonafide-card"
                result_text = "✅ **LIVE VOICE**"
                interpretation = "This appears to be a live recording."
            else:  # Spoofed
                card_class = "spoofed-card"
                result_text = "⚠️ **REPLAYED**"
                interpretation = "This appears to be a replayed recording."

            st.markdown(
                f"""
            <div class="result-card {card_class}">
                <h3>{result_text}</h3>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><em>{interpretation}</em></p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Confidence gauge
            fig_replay = create_confidence_gauge(
                confidence, "Replay Detection Confidence"
            )
            st.plotly_chart(fig_replay, use_container_width=True)

        else:
            st.error("Replay detection model failed to run")

    # Overall Assessment
    if results.get("ai_detection") and results.get("replay_detection"):
        st.markdown(
            '<div class="section-header">📊 Overall Assessment</div>',
            unsafe_allow_html=True,
        )

        ai_pred = results["ai_detection"]["prediction"]
        replay_pred = results["replay_detection"]["prediction"]

        if ai_pred == 1 and replay_pred == 1:
            overall_class = "bonafide-card"
            overall_result = "✅ **AUTHENTIC VOICE**"
            overall_desc = "Both models indicate this is a genuine, live human voice."
        elif ai_pred == 0 or replay_pred == 0:
            overall_class = "super-spoofed-card"
            overall_result = "⚠️ **SPOOFED VOICE DETECTED**"
            if ai_pred == 0 and replay_pred == 0:
                overall_desc = "Both models detected spoofing - possibly AI-generated AND/OR replayed."
            elif ai_pred == 0:
                overall_desc = "AI-generated voice detected (possibly)."
            else:
                overall_desc = "Replay attack detected (possibly)."
        else:
            overall_class = "neutral-card"
            overall_result = "❓ **UNCERTAIN**"
            overall_desc = "Mixed results from detection models."

        st.markdown(
            f"""
        <div class="result-card {overall_class}">
            <h2>{overall_result}</h2>
            <p><em>{overall_desc}</em></p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    # Header
    st.markdown(
        '<div class="main-header">🎤 Voice Spoofing Detection System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    This application uses two state-of-the-art models to detect voice spoofing:
    - **AASIST**: Detects AI-generated voices
    - **CNN-GRU**: Detects replay attacks
    """
    )

    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown(
            """
        1. Upload an audio file or record one
        2. Click 'Analyze Audio' to run detection
        3. View results and confidence scores
        4. Check overall assessment
        
        **Supported formats:** WAV, MP3, FLAC
        """
        )

        st.header("⚙️ Model Information")
        st.markdown(
            """
        **AASIST Model:**
        - Architecture: Attention-based
        - Purpose: AI voice detection
        
        **CNN-GRU Model:**
        - Architecture: Convolutional + RNN
        - Purpose: Replay attack detection
        """
        )

    # File upload
    st.markdown(
        '<div class="section-header">📁 Audio Input</div>', unsafe_allow_html=True
    )

    upload_tab, record_tab, sample_tab = st.tabs(
        ["Upload File", "Record Audio", "Use Sample"]
    )

    audio_file = None

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "m4a"],
            help="Upload an audio file to analyze for voice spoofing",
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_file = tmp_file.name

            st.success(f"File uploaded: {uploaded_file.name}")
            st.audio(uploaded_file.getvalue())

    with record_tab:
        st.info(
            "Audio recording feature would require additional setup with microphone access."
        )
        st.markdown("For now, please use the upload or sample options.")

    with sample_tab:
        sample_files = []
        sample_dir = "data"
        if os.path.exists(sample_dir):
            sample_files = [
                f
                for f in os.listdir(sample_dir)
                if f.endswith((".wav", ".mp3", ".flac"))
            ]

        if sample_files:
            selected_sample = st.selectbox("Choose a sample file:", sample_files)
            if selected_sample:
                audio_file = os.path.join(sample_dir, selected_sample)
                st.success(f"Selected sample: {selected_sample}")

                # Display audio player
                with open(audio_file, "rb") as f:
                    st.audio(f.read())
        else:
            st.info("No sample files found in the data directory.")

    # Analysis section
    if audio_file:
        st.markdown(
            '<div class="section-header">🔍 Analysis</div>', unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 3])

        with col1:
            analyze_button = st.button(
                "🚀 Analyze Audio", type="primary", use_container_width=True
            )

        with col2:
            show_visualization = st.checkbox("Show audio visualization", value=True)

        if analyze_button:
            # Show audio visualization if requested
            if show_visualization:
                with st.spinner("Creating audio visualization..."):
                    fig = visualize_audio(audio_file)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            # Run predictions
            results, errors = predict_spoofing(audio_file)

            # Display results
            display_results(results, errors)

            # Clean up temporary file if it was uploaded
            if uploaded_file is not None:
                try:
                    os.unlink(audio_file)
                except:
                    pass

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        Voice Spoofing Detection System | Built with Streamlit | CS117 Project
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    # Load models (cached)
    load_models()
    main()
