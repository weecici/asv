import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import numpy as np
import json
from importlib import import_module


def predict(audio_path, model_path, config_path, device="cuda"):
    """
    Predict whether an audio input is bonafide or spoofed.

    Args:
        audio_path (str): Path to the audio file
        model_path (str): Path to the trained model weights
        config_path (str): Path to the model configuration file
        device (str): Device to run inference on

    Returns:
        dict: Contains prediction result, confidence score, and raw output
    """

    # Load configuration
    with open(config_path, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    if model_path == "":
        model_path = config["model_path"]

    # Load and prepare audio
    audio_data = load_audio(audio_path)

    # Get model
    model = get_model(model_config, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare input tensor
    audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        _, output = model(audio_tensor)

        # Get probabilities
        probabilities = F.softmax(output, dim=1)
        confidence = torch.max(probabilities, dim=1)[0].item()

        # Get prediction (0 = spoofed, 1 = bonafide)
        prediction = torch.argmax(output, dim=1).item()

    # Format result
    result = {
        "prediction": prediction,
        "confidence": confidence,
    }

    return result


def load_audio(audio_path, target_sr=16000):
    """
    Load audio with soundfile and resample with librosa if needed.
    Best of both worlds approach.
    """

    # Load with soundfile (faster)
    audio, sr = sf.read(audio_path, dtype="float32")

    # Handle stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample with librosa if needed (higher quality)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Pad or truncate
    target_length = 64600
    audio_len = audio.shape[0]
    if audio_len >= target_length:
        return audio[:target_length]
    num_repeats = int(target_length / audio_len) + 1
    padded_x = np.tile(audio, (1, num_repeats))[:, :target_length][0]

    return padded_x


def get_model(model_config, device):
    """
    Get model instance based on configuration.
    """
    module = import_module("aasist.models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    return model
