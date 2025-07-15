import yaml
import numpy as np
import librosa

import torch

from dnn.cnn_gru import spec_CNN_GRU


def extract_spectrogram_features(
    audio_file, n_fft=2048, hop_length=800, win_length=480, sr=16000
):
    """
    Extract spectrogram features from audio file
    Returns features in the same format as training data
    """
    # Load audio
    y, sr_orig = librosa.load(audio_file, sr=sr)

    # Extract magnitude spectrogram
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude = np.abs(stft)

    # Add channel dimension to match training format (1, time, freq)
    magnitude = magnitude.T[np.newaxis, :, :]  # Shape: (1, time, freq)

    return magnitude


def load_trained_model(model_path, config_path, device):
    """
    Load a trained model for prediction

    Args:
        model_path: Path to the saved model (.pt file)
        config_path: Path to the model configuration (.yaml file)
        device: PyTorch device

    Returns:
        model: Loaded model
        parser: Configuration dictionary
    """
    # Load configuration
    with open(config_path, "r") as f:
        parser = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize model
    model = spec_CNN_GRU(parser["model"], device=device, do_pretrn=True).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, parser


def predict(
    audio_path,
    model_path,
    config_path,
    device="cuda",
):
    """
    Predict whether an audio file is bonafide or spoofed

    Args:
        model: Trained model
        audio_file: Path to audio file
        parser: Configuration dictionary
        device: PyTorch device

    Returns:
        prediction: 'bonafide' or 'spoofed'
        confidence: Confidence score (0-1)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, parser = load_trained_model(model_path, config_path, device)

    model.eval()

    # Extract features
    features = extract_spectrogram_features(audio_path)

    # Handle time dimension according to training settings
    nb_time = features.shape[1]
    target_time = parser["nb_time"]

    if nb_time > target_time:
        # Take center segment for prediction (more stable than random)
        start_idx = (nb_time - target_time) // 2
        features = features[:, start_idx : start_idx + target_time, :]
    elif nb_time < target_time:
        # Repeat to match target length
        nb_dup = int(target_time / nb_time) + 1
        features = np.tile(features, (1, nb_dup, 1))[:, :target_time, :]

    # Convert to tensor and add batch dimension
    features_tensor = (
        torch.FloatTensor(features).unsqueeze(0).to(device)
    )  # (1, 1, time, freq)

    with torch.no_grad():
        _, output = model(features_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=1)

        # Get prediction (0 = spoofed, 1 = bonafide in most implementations)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities, dim=1)[0].item()

    result = {
        "prediction": prediction,
        "confidence": confidence,
    }

    return result
