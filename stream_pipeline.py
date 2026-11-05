"""Pipeline autour de SSLAM V7 : fenêtre de contexte 1 s, pas 0,5 s, prédiction sur la
DERNIÈRE demi-seconde de chaque fenêtre (première sortie à 1 s, 10 s -> 19 sorties).

wav brut -> (filtre moteur, voir denoise.py) -> fenêtres de 1 s -> predict.fbank + modèle V7
-> probabilités -> décision (silence, lissage, seuil de confiance) -> tableau.

À placer dans le dossier SSLAM_V7 (à côté de predict.py). Les étapes de prétraitement et le
modèle sont ceux de predict.py : rien n'est réimplémenté.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 16000
WIN = SR          # contexte de 1 s (imposé par le modèle V7)
HOP = SR // 2     # pas de 0,5 s
SILENCE = "silence"
UNCERTAIN = "incertain"


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


def rms_dbfs(x) -> float:
    x = np.asarray(x, dtype="float32")
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10))


class Streamer:
    """Charge SSLAM V7 une fois (via predict.load_model) et prédit par lots de fenêtres."""

    def __init__(self, device: str = "cpu"):
        import predict
        self._p = predict
        self.device = torch.device(device)
        self.model, self.labels = predict.load_model(self.device)

    @torch.inference_mode()
    def predict_windows(self, windows, batch_size: int = 8, progress=None) -> np.ndarray:
        n, k = len(windows), len(self.labels)
        probs = np.zeros((n, k), dtype=np.float32)
        neg = self.labels.index("negatif")
        todo = []
        for i, w in enumerate(windows):          # silence numérique exact -> négatif (comme predict.py)
            if np.max(np.abs(w)) == 0:
                probs[i, neg] = 1.0
            else:
                todo.append(i)
        done = n - len(todo)
        for b in range(0, len(todo), batch_size):
            idx = todo[b:b + batch_size]
            x = torch.stack([self._p.fbank(windows[i]) for i in idx]).to(self.device)  # (B,1,T,128)
            probs[idx] = self.model(x).softmax(-1).cpu().numpy()
            if progress:
                progress(done + b + len(idx), n)
        return probs


def compute_window_probs(wav, streamer: Streamer, batch_size: int = 8, progress=None) -> dict:
    """Étape coûteuse : découpe en fenêtres de 1 s (pas 0,5 s) + inférence."""
    w = wav.numpy() if torch.is_tensor(wav) else np.asarray(wav, dtype="float32")
    if len(w) < WIN:
        raise ValueError("Audio trop court : minimum 1 s.")
    ends = list(range(WIN, len(w) + 1, HOP))
    wins = [w[e - WIN:e] for e in ends]
    t0 = time.time()
    probs = streamer.predict_windows(wins, batch_size, progress)
    e = np.array(ends) / SR
    return dict(ctx_start=e - 1.0, ctx_end=e, t_start=e - 0.5, t_end=e, probs=probs,
                rms_db=np.array([rms_dbfs(x) for x in wins]), labels=list(streamer.labels),
                compute_s=time.time() - t0)


def decide(res: dict, silence_db: float | None = -55.0, smooth_k: int = 1, min_conf: float = 0.0):
    """Étape bon marché (recalculable à volonté) : probabilités -> étiquette par sortie.

    t_debut_s / t_fin_s : demi-seconde cible de la prédiction (la dernière de la fenêtre).
    silence_db : niveau RMS de la fenêtre de 1 s sous lequel on étiquette 'silence' (None = off)
    smooth_k   : moyenne glissante CAUSALE des probabilités sur k sorties (1 = aucun lissage)
    min_conf   : si la proba max < min_conf -> 'incertain'
    """
    import pandas as pd
    P, labels = res["probs"], res["labels"]
    Ps = np.stack([P[max(0, i - smooth_k + 1):i + 1].mean(axis=0) for i in range(len(P))]) if smooth_k > 1 else P
    idx, conf = Ps.argmax(axis=1), Ps.max(axis=1).astype(float)
    cls = np.array([labels[i] for i in idx], dtype=object)
    cls[conf < min_conf] = UNCERTAIN
    if silence_db is not None:
        silent = res["rms_db"] < silence_db
        cls[silent] = SILENCE
        conf[silent] = np.nan
    df = pd.DataFrame({
        "t_debut_s": res["t_start"].round(3), "t_fin_s": res["t_end"].round(3),
        "classe": cls, "confiance": conf.round(3), "niveau_dBFS": res["rms_db"].round(1),
        "contexte_debut_s": res["ctx_start"].round(3), "contexte_fin_s": res["ctx_end"].round(3)})
    for j, name in enumerate(labels):
        df[f"p_{name}"] = Ps[:, j].round(3)
    return df


def to_segments(df):
    """Fusionne les sorties consécutives de même classe en segments."""
    import pandas as pd
    rows, cur = [], None
    for r in df.itertuples():
        if cur is not None and r.classe == cur["classe"]:
            cur["t_fin_s"] = r.t_fin_s
            cur["n_sorties"] += 1
            cur["_c"].append(r.confiance)
        else:
            if cur is not None:
                rows.append(cur)
            cur = dict(classe=r.classe, t_debut_s=r.t_debut_s, t_fin_s=r.t_fin_s, n_sorties=1, _c=[r.confiance])
    if cur is not None:
        rows.append(cur)
    for c in rows:
        v = [x for x in c.pop("_c") if x == x]
        c["confiance_moy"] = round(float(np.mean(v)), 3) if v else float("nan")
    return pd.DataFrame(rows)[["t_debut_s", "t_fin_s", "classe", "n_sorties", "confiance_moy"]]
