# Voice Spoofing Detection

A modern web-based user interface for detecting voice spoofing using two powerful frameworks:

- **AASIST**: Advanced audio spoofing detection for AI-generated voices
- **CNN-GRU**: Convolutional neural network with GRU for replay attack detection
## Features

**Audio Input Options**
- Upload audio files (WAV, MP3, FLAC)
- Use sample files from the data directory
- Future: Live recording capability

**Comprehensive Analysis**
- Real-time voice spoofing detection
- Confidence scores with visual gauges
- Audio waveform and spectrogram visualization
- Overall authentication assessment

**Modern Interface**
- Clean, intuitive design
- Real-time feedback and progress indicators
- Responsive layout for different screen sizes
- Color-coded results for easy interpretation

## Quick Start
### Prerequisite: please install [`astral/uv`](https://github.com/astral-sh/uv)
1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Run the UI:**
   ```bash
   uv run main.py
   ```

3. **Open your browser** to `http://localhost:8501`

## How to Use

1. **Upload an audio file** or select a sample from the available options
2. **Click "Analyze Audio"** to run both detection models
3. **View the results:**
   - Individual model predictions with confidence scores
   - Visual confidence gauges
   - Overall assessment combining both models
4. **Optional:** Enable audio visualization to see waveform and spectrogram

## Model Information

### AASIST (AI Voice Detection)
- **Purpose**: Detects AI-generated synthetic voices
- **Architecture**: Attention-based deep learning model
- **Output**: Binary classification (Real/AI-generated) with confidence score

### CNN-GRU (Replay Attack Detection)
- **Purpose**: Detects replay attacks (recorded voice played back)
- **Architecture**: Convolutional Neural Network + Gated Recurrent Unit
- **Output**: Binary classification (Live/Replayed) with confidence score

## Result Interpretation

- **✅ AUTHENTIC VOICE**: Both models indicate genuine, live human speech
- **⚠️ SPOOFED VOICE**: One or both models detected spoofing

## File Structure

```
src/
├── app.py              # Main Streamlit application
├── main.py             # Original CLI script
├── aasist/             # AASIST model files
│   ├── predict.py      # AI voice detection
│   ├── models/         # Model weights
│   └── config/         # Configuration files
└── dnn/                # CNN-GRU model files
    ├── predict.py      # Replay detection
    ├── models/         # Model weights
    └── data/           # Training data
```

## Technical Details

- **Frontend**: Streamlit web framework
- **Visualization**: Plotly for interactive charts and gauges
- **Audio Processing**: librosa, soundfile
- **Deep Learning**: PyTorch models
- **Supported Python**: 3.10+

## References
[1] *Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu, & Nicholas Evans. (2021). AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.*

   - Github: https://github.com/clovaai/aasist 
   - Paper:  ***arXiv:2110.01200 [eess.AS]*** (https://arxiv.org/abs/2110.01200)

[2]  *Jee-weon Jung, Hye-jin Shim, Hee-Soo Heo, & Ha-Jin Yu. (2019). Replay attack detection with complementary high-resolution information using end-to-end DNN for the ASVspoof 2019 Challenge.*

   - Github: https://github.com/Jungjee/ASVspoof_PA
   - Paper: ***arXiv:1904.10134 [eess.AS]*** (https://arxiv.org/abs/1904.10134)


