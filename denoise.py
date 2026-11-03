"""Filtrage du bruit du moteur PORTEUR (celui sur lequel le micro est fixé).

Principe : soustraction spectrale avec un profil de bruit "moteur porteur".
Le profil est une puissance par bande de fréquence (un spectre moyen). On retire
uniquement CE niveau-là, bande par bande. Conséquence importante :

  - le moteur porteur (proche du micro, très fort, constant) est atténué ;
  - un 2e engin (plus loin, donc plus faible) AJOUTE de l'énergie par-dessus le
    plancher du moteur porteur : cette énergie en plus n'est pas retirée.

Deux façons d'obtenir le profil :

  1. "ref"  (recommandé) : un court extrait où SEUL le moteur porteur tourne
     (2-3 s suffisent, enregistré au début de la journée par exemple).
  2. "auto" : estimé dans le fichier lui-même = plancher bas (percentile) de
     l'énergie par bande dans le temps. Suppose que le moteur porteur tourne
     pendant tout l'enregistrement et qu'aucun autre bruit continu n'est présent
     du début à la fin. Si un 2e engin tourne pendant TOUT l'extrait, il sera
     pris pour le moteur porteur et filtré aussi -> utiliser "ref".
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SR = 16000
N_FFT = 1024
HOP = 256


def _stft(x: torch.Tensor) -> torch.Tensor:
    return torch.stft(x, N_FFT, HOP, N_FFT, window=torch.hann_window(N_FFT),
                      center=True, return_complex=True)  # (freq, frames)


def _rms_db(x: torch.Tensor) -> float:
    return float(20 * torch.log10(torch.sqrt(torch.mean(x ** 2)) + 1e-10))


def profile_from_reference(ref: torch.Tensor) -> torch.Tensor:
    """Profil = puissance moyenne par bande d'un extrait 'moteur porteur seul'."""
    if len(ref) < N_FFT * 2:
        raise ValueError("Extrait de référence trop court (< 0,15 s).")
    P = _stft(ref).abs() ** 2
    return P.mean(dim=1)


def profile_auto(wav: torch.Tensor, percentile: float = 20.0, smooth_s: float = 0.3) -> torch.Tensor:
    """Profil = plancher bas (percentile) de la puissance lissée dans le temps."""
    P = _stft(wav).abs() ** 2
    k = int(smooth_s * SR / HOP) | 1  # noyau impair
    k = min(k, P.shape[1] if P.shape[1] % 2 == 1 else P.shape[1] - 1)
    k = max(k, 1)
    Ps = F.avg_pool1d(P.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2,
                      count_include_pad=False)[0, :, :P.shape[1]]
    return torch.quantile(Ps, percentile / 100.0, dim=1)


def suppress(wav: torch.Tensor, profile: torch.Tensor, alpha: float = 1.5,
             max_atten_db: float = 18.0, smooth_freq: int = 3, smooth_time: int = 3):
    """Soustraction spectrale.

    alpha        : sur-soustraction (1.0 = niveau estimé, >1 = plus agressif)
    max_atten_db : atténuation maximale par bande (limite les artefacts "musicaux")
    smooth_*     : lissage du masque de gain (réduit les artefacts)
    """
    Y = _stft(wav)
    P = Y.abs() ** 2 + 1e-12
    Pn = profile.unsqueeze(1)
    G = torch.sqrt(torch.clamp(1.0 - alpha * Pn / P, min=0.0))
    G = torch.clamp(G, min=10 ** (-max_atten_db / 20.0), max=1.0)
    if smooth_freq > 1 or smooth_time > 1:
        G = F.avg_pool2d(G[None, None], kernel_size=(smooth_freq, smooth_time), stride=1,
                         padding=(smooth_freq // 2, smooth_time // 2))[0, 0]
    out = torch.istft(Y * G, N_FFT, HOP, N_FFT, window=torch.hann_window(N_FFT),
                      center=True, length=len(wav))
    info = dict(rms_avant_db=_rms_db(wav), rms_apres_db=_rms_db(out))
    info["attenuation_db"] = info["rms_avant_db"] - info["rms_apres_db"]
    return out, info


def remove_ego_engine(wav: torch.Tensor, method: str = "auto", ref: torch.Tensor | None = None,
                      alpha: float = 1.5, max_atten_db: float = 18.0, percentile: float = 20.0):
    """Point d'entrée. method = 'auto' | 'ref'. Retourne (wav_filtré, info)."""
    if method == "ref":
        if ref is None:
            raise ValueError("method='ref' demande un extrait de référence.")
        profile = profile_from_reference(ref)
    elif method == "auto":
        profile = profile_auto(wav, percentile=percentile)
    else:
        raise ValueError(f"method inconnue: {method}")
    out, info = suppress(wav, profile, alpha=alpha, max_atten_db=max_atten_db)
    info["methode"] = method
    info["profile_db"] = (10 * torch.log10(profile + 1e-12)).numpy()
    return out, info
