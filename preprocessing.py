import random
from pathlib import Path

def audio_features(path, training=False):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    wav, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if not len(wav) or not np.isfinite(wav).all():
        raise ValueError(f'Audio vide ou non fini: {path}')
    wav = torch.from_numpy(wav.mean(axis=1)).float()
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    # Jabra adaptation: one second of real audio, with the original V6
    # feature padding. For complete recordings use analyze_jabra.py.
    if len(wav) > 16000:
        start = random.randint(0, len(wav)-16000) if training else (len(wav)-16000)//2
        wav = wav[start:start+16000]
    if len(wav) < 16000:
        wav = torch.nn.functional.pad(wav, (0, 16000-len(wav)))
    if len(wav) < 400:
        wav = torch.nn.functional.pad(wav, (0, 400-len(wav)))
    wav = wav - wav.mean()
    if training:
        wav = wav * random.uniform(.7, 1.3)
    mel = torchaudio.compliance.kaldi.fbank(wav.unsqueeze(0), htk_compat=True,
        sample_frequency=16000, use_energy=False, window_type='hanning',
        num_mel_bins=128, dither=0.0, frame_shift=10)
    mel = torch.nn.functional.pad(mel[:1024], (0, 0, 0, max(0, 1024-len(mel))))
    mel = (mel + 4.268) / (4.569 * 2)
    return mel.unsqueeze(0)
