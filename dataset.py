import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np
import config

SEGMENT_DURATION = 3.0
SEGMENT_SAMPLES = int(SEGMENT_DURATION * config.SAMPLE_RATE)


class MelNormalize:
    def __init__(self, mean=-20, std=15):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std


class MosquitoDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_sample_rate=16000, segment_duration=3.0, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * target_sample_rate)
        self.is_train = is_train
        self.samples = []
        self._load_samples()

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.FMIN,
            f_max=config.FMAX,
        )
        self.amplitude_to_db = AmplitudeToDB(stype="power", top_db=80)
        self.normalize = MelNormalize()

    def _load_samples(self):
        for label in ["mosquito", "no_mosquito"]:
            label_dir = os.path.join(self.data_dir, label)
            if not os.path.exists(label_dir):
                continue
            label_int = 1 if label == "mosquito" else 0
            for filename in os.listdir(label_dir):
                if filename.endswith(".wav"):
                    file_path = os.path.join(label_dir, filename)
                    self.samples.append((file_path, label_int))

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, file_path):
        import soundfile as sf
        waveform, sample_rate = sf.read(file_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)

        return waveform

    def _extract_features(self, waveform):
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)
        log_mel_spec = self.normalize(log_mel_spec)
        return log_mel_spec

    def _get_segment(self, waveform):
        total_samples = waveform.size(1)

        if total_samples <= self.segment_samples:
            pad_len = self.segment_samples - total_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_len), mode="constant", value=0)
            return waveform

        if self.is_train:
            start_idx = np.random.randint(0, total_samples - self.segment_samples)
        else:
            start_idx = (total_samples - self.segment_samples) // 2

        return waveform[:, start_idx : start_idx + self.segment_samples]

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        waveform = self._load_audio(file_path)
        segment = self._get_segment(waveform)
        features = self._extract_features(segment)

        if self.transform:
            features = self.transform(features)

        return features, torch.tensor(label)


class AudioAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=40)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)

    def __call__(self, x):
        if np.random.random() < self.p:
            x = self.time_masking(x)
        if np.random.random() < self.p:
            x = self.freq_masking(x)
        return x


class BackgroundNoiseAugmentation:
    def __init__(self, noise_dir=None, max_noise_level=0.05):
        self.max_noise_level = max_noise_level
        self.noise_files = []
        if noise_dir and os.path.exists(noise_dir):
            for f in os.listdir(noise_dir):
                if f.endswith(".wav"):
                    self.noise_files.append(os.path.join(noise_dir, f))

    def __call__(self, waveform):
        if not self.noise_files or np.random.random() > 0.3:
            return waveform

        noise_file = np.random.choice(self.noise_files)
        noise_waveform, _ = torchaudio.load(noise_file)
        noise_waveform = noise_waveform.mean(dim=0, keepdim=True)

        if noise_waveform.size(1) > waveform.size(1):
            start = np.random.randint(0, noise_waveform.size(1) - waveform.size(1))
            noise_waveform = noise_waveform[:, start : start + waveform.size(1)]
        else:
            pad_len = waveform.size(1) - noise_waveform.size(1)
            noise_waveform = torch.nn.functional.pad(noise_waveform, (0, pad_len), mode="constant", value=0)

        noise_level = np.random.uniform(0.01, self.max_noise_level)
        noise_waveform = noise_waveform * noise_level

        return waveform + noise_waveform


def collate_fn(batch):
    features = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return torch.stack(features), labels


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_dataloader(data_dir, batch_size=32, shuffle=True, transform=None, is_train=True, num_workers=None):
    if num_workers is None:
        num_workers = 2 if torch.cuda.is_available() else 1
    
    dataset = MosquitoDataset(data_dir, transform=transform, is_train=is_train)

    if is_train:
        class_counts = [0, 0]
        for _, label in dataset.samples:
            class_counts[label] += 1
        weights = [1 / class_counts[item[1]] for item in dataset.samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=num_workers)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)