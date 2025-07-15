from aasist.predict import predict as detect_ai_voice
from dnn.predict import predict as detect_replayed_voice

audio_path = "../data/test.wav"

ai_voice_detection = detect_ai_voice(
    audio_path, "aasist/models/weights/AASIST.pth", "aasist/config/AASIST.conf"
)
replayed_voice_detection = detect_replayed_voice(
    audio_path,
    "dnn/models/cnn_gru/model.pt",
    "dnn/models/cnn_gru/cfg.yaml",
)
