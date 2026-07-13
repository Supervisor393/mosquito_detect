import os
import argparse
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np

import config
from model import get_model

SEGMENT_DURATION = 3.0
SEGMENT_SAMPLES = int(SEGMENT_DURATION * config.SAMPLE_RATE)
OVERLAP_RATIO = 0.5


class MelNormalize:
    def __init__(self, mean=-20, std=15):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


class AudioProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.FMIN,
            f_max=config.FMAX,
        )
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=80)
        self.normalize = MelNormalize()

    def load_audio(self, file_path):
        import soundfile as sf
        waveform, sample_rate = sf.read(file_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        return waveform

    def extract_features(self, waveform):
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        log_mel_spec = self.normalize(log_mel_spec)
        return log_mel_spec

    def split_into_segments(self, waveform, segment_samples=SEGMENT_SAMPLES, overlap_ratio=OVERLAP_RATIO):
        total_samples = waveform.size(1)
        segments = []

        if total_samples <= segment_samples:
            pad_len = segment_samples - total_samples
            padded = torch.nn.functional.pad(waveform, (0, pad_len), mode="constant", value=0)
            segments.append(padded)
            return segments

        step_size = int(segment_samples * (1 - overlap_ratio))
        current = 0

        while current + segment_samples <= total_samples:
            segments.append(waveform[:, current : current + segment_samples])
            current += step_size

        last_segment_start = max(0, total_samples - segment_samples)
        if last_segment_start != current - step_size:
            segments.append(waveform[:, last_segment_start : last_segment_start + segment_samples])

        return segments


def predict_audio_segment(model, audio_processor, waveform, device):
    model.eval()
    with torch.no_grad():
        features = audio_processor.extract_features(waveform)
        features = features.unsqueeze(0).to(device)
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        return prediction, probabilities[0].cpu().numpy()


def predict_audio(model, audio_processor, file_path, device, aggregation="majority"):
    waveform = audio_processor.load_audio(file_path)
    segments = audio_processor.split_into_segments(waveform)

    predictions = []
    probabilities_list = []

    for segment in segments:
        pred, probs = predict_audio_segment(model, audio_processor, segment, device)
        predictions.append(pred)
        probabilities_list.append(probs)

    if aggregation == "majority":
        final_prediction = np.bincount(predictions).argmax()
        avg_prob = np.mean(probabilities_list, axis=0)
        confidence = avg_prob[final_prediction]
    elif aggregation == "probability":
        avg_prob = np.mean(probabilities_list, axis=0)
        final_prediction = np.argmax(avg_prob)
        confidence = avg_prob[final_prediction]
    else:
        final_prediction = predictions[0]
        confidence = probabilities_list[0][final_prediction]

    return final_prediction, confidence, len(segments)


def main():
    parser = argparse.ArgumentParser(description="Run inference on audio files")
    parser.add_argument("--audio_path", required=True, help="Path to audio file or directory")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--model_type", default="efficient", choices=["efficient", "simple"], help="Model type")
    parser.add_argument("--aggregation", default="probability", choices=["majority", "probability"], help="Aggregation method for segments")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for inference")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = get_model(args.model_type, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")

    audio_processor = AudioProcessor()

    if os.path.isfile(args.audio_path):
        files = [args.audio_path]
    elif os.path.isdir(args.audio_path):
        files = [os.path.join(args.audio_path, f) for f in os.listdir(args.audio_path) if f.endswith(".wav")]
        files.sort()
    else:
        print(f"Error: {args.audio_path} is not a valid file or directory")
        return

    print(f"\nProcessing {len(files)} audio files...")
    print(f"Aggregation method: {args.aggregation}")

    for file_path in files:
        prediction, confidence, num_segments = predict_audio(model, audio_processor, file_path, device, args.aggregation)
        label = "MOSQUITO" if prediction == 1 else "NO MOSQUITO"
        print(f"{os.path.basename(file_path)}: {label} (confidence: {confidence:.4f}, segments: {num_segments})")


if __name__ == "__main__":
    main()