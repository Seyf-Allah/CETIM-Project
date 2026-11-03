"""Pipeline fenêtre glissante autour du modèle SSLAM V6.

wav brut -> (filtrage moteur, voir denoise.py) -> fenêtres de 0,5 s / 1 s -> features
(fbank identiques à l'entraînement) -> modèle -> probabilités -> décision (silence,
lissage, seuil de confiance) -> tableau.

À placer dans le dossier SSLAM_V6 (à côté de predict.py et preprocessing.py).
"""
from __future__ import annotations

import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 16000
TARGET_FRAMES = 1024          # longueur d'entrée utilisée à l'entraînement (10,24 s)
SILENCE = "silence"
UNCERTAIN = "incertain"

# --------------------------------------------------------------------------- audio


def load_audio(src) -> tuple[torch.Tensor, int]:
    """src : chemin, bytes ou objet fichier. Retourne (wav mono 16 kHz float32, sr d'origine)."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    data, sr = sf.read(src, dtype="float32", always_2d=True)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("Audio vide ou contenant des valeurs non finies.")
    wav = torch.from_numpy(data.mean(axis=1)).float()
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav, sr


def wav_to_bytes(wav: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav.clamp(-1, 1).numpy(), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def rms_dbfs(x: torch.Tensor) -> float:
    return float(20 * torch.log10(torch.sqrt(torch.mean(x ** 2)) + 1e-10))


# --------------------------------------------------------------------------- fenêtrage


def window_starts(n_samples: int, win_s: float, hop_s: float) -> list[int]:
    """Débuts (en échantillons) des fenêtres. La dernière fenêtre est calée sur la fin du
    fichier pour que la queue de l'enregistrement soit couverte."""
    W, H = int(round(win_s * SR)), int(round(hop_s * SR))
    if n_samples <= W:
        return [0]
    starts = list(range(0, n_samples - W + 1, H))
    if starts[-1] != n_samples - W:
        starts.append(n_samples - W)
    return starts


# --------------------------------------------------------------------------- features


def window_features(win: torch.Tensor, pad_mode: str = "train") -> torch.Tensor:
    """Mêmes fbank / normalisation que preprocessing.audio_features, appliqués à une fenêtre.

    pad_mode (comment compléter une fenêtre courte, le modèle ayant été entraîné sur 1024 frames) :
      'train' : zéros jusqu'à 1024 frames, exactement comme à l'entraînement (référence).
      'tile'  : la fenêtre est répétée jusqu'à 1024 frames.
      'crop'  : aucune extension, séquence courte (multiple de 16 frames) -> ~15x plus rapide,
                mais plus éloigné de ce que le modèle a vu.
    """
    win = win - win.mean()
    if len(win) < 400:
        win = F.pad(win, (0, 400 - len(win)))
    mel = torchaudio.compliance.kaldi.fbank(
        win.unsqueeze(0), htk_compat=True, sample_frequency=SR, use_energy=False,
        window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
    n = mel.shape[0]
    if pad_mode == "train":
        mel = F.pad(mel[:TARGET_FRAMES], (0, 0, 0, max(0, TARGET_FRAMES - n)))
    elif pad_mode == "tile":
        mel = mel.repeat(math.ceil(TARGET_FRAMES / n), 1)[:TARGET_FRAMES]
    elif pad_mode == "crop":
        mel = F.pad(mel, (0, 0, 0, math.ceil(n / 16) * 16 - n))
    else:
        raise ValueError(f"pad_mode inconnu: {pad_mode}")
    mel = (mel + 4.268) / (4.569 * 2)
    return mel.unsqueeze(0)  # (1, T, 128)


# --------------------------------------------------------------------------- modèle


class Streamer:
    """Charge SSLAM V6 une fois et prédit sur une liste de fenêtres."""

    def __init__(self, device: str = "auto", n_threads: int | None = None):
        from predict import load_model  # fourni avec le modèle
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if n_threads:
            torch.set_num_threads(n_threads)
        self.model, self.labels = load_model(self.device)

    @torch.inference_mode()
    def predict(self, windows: list[torch.Tensor], pad_mode: str = "train",
                batch_size: int = 8, progress=None) -> np.ndarray:
        feats = [window_features(w, pad_mode) for w in windows]
        out = []
        for i in range(0, len(feats), batch_size):
            x = torch.stack(feats[i:i + batch_size]).to(self.device)  # (B,1,T,128)
            out.append(self.model(x).softmax(-1).cpu().numpy())
            if progress:
                progress(min(i + batch_size, len(feats)), len(feats))
        return np.concatenate(out, axis=0)


def compute_window_probs(wav: torch.Tensor, streamer: Streamer, win_s: float = 0.5,
                         hop_s: float = 0.25, pad_mode: str = "train",
                         batch_size: int = 8, progress=None) -> dict:
    """Étape coûteuse : découpe + inférence. Retourne starts/ends (s), probs, rms_db, temps."""
    starts = window_starts(len(wav), win_s, hop_s)
    W = int(round(win_s * SR))
    wins = [wav[s:s + W] for s in starts]
    t0 = time.time()
    probs = streamer.predict(wins, pad_mode, batch_size, progress)
    return dict(
        t_start=np.array(starts) / SR,
        t_end=np.minimum(np.array(starts) + W, len(wav)) / SR,
        probs=probs,
        rms_db=np.array([rms_dbfs(w) for w in wins]),
        labels=list(streamer.labels),
        win_s=win_s, hop_s=hop_s, pad_mode=pad_mode,
        compute_s=time.time() - t0,
    )


# --------------------------------------------------------------------------- décision


def decide(res: dict, silence_db: float | None = -55.0, smooth_k: int = 1,
           min_conf: float = 0.0):
    """Étape bon marché (recalculable à volonté) : probabilités -> étiquette par fenêtre.

    silence_db : fenêtres dont le niveau RMS < seuil (dBFS) étiquetées 'silence' (None = désactivé)
    smooth_k   : moyenne glissante CAUSALE des probabilités sur k fenêtres (1 = aucun lissage)
    min_conf   : si la proba max < min_conf -> 'incertain'
    """
    P, labels = res["probs"], res["labels"]
    if smooth_k > 1:
        Ps = np.stack([P[max(0, i - smooth_k + 1):i + 1].mean(axis=0) for i in range(len(P))])
    else:
        Ps = P
    idx = Ps.argmax(axis=1)
    conf = Ps.max(axis=1)
    cls = np.array([labels[i] for i in idx], dtype=object)
    cls[conf < min_conf] = UNCERTAIN
    conf_out = conf.astype(float)
    if silence_db is not None:
        silent = res["rms_db"] < silence_db
        cls[silent] = SILENCE
        conf_out[silent] = np.nan

    import pandas as pd
    df = pd.DataFrame({
        "t_debut_s": res["t_start"].round(3),
        "t_fin_s": res["t_end"].round(3),
        "classe": cls,
        "confiance": conf_out.round(3),
        "niveau_dBFS": res["rms_db"].round(1),
    })
    for j, name in enumerate(labels):
        df[f"p_{name}"] = Ps[:, j].round(3)
    return df


def to_segments(df):
    """Fusionne les fenêtres consécutives de même classe en segments."""
    import pandas as pd
    rows, cur = [], None
    for r in df.itertuples():
        if cur is not None and r.classe == cur["classe"]:
            cur["t_fin_s"] = r.t_fin_s
            cur["n_fenetres"] += 1
            cur["_conf"].append(r.confiance)
        else:
            if cur is not None:
                rows.append(cur)
            cur = dict(classe=r.classe, t_debut_s=r.t_debut_s, t_fin_s=r.t_fin_s,
                       n_fenetres=1, _conf=[r.confiance])
    if cur is not None:
        rows.append(cur)
    for c in rows:
        v = [x for x in c.pop("_conf") if x == x]
        c["confiance_moy"] = round(float(np.mean(v)), 3) if v else float("nan")
    return pd.DataFrame(rows)[["t_debut_s", "t_fin_s", "classe", "n_fenetres", "confiance_moy"]]
