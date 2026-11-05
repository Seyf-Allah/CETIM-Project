"""Vérifications (nécessite modele.pt) : python selftest.py
1. fenêtrage V7 : 10 s -> 19 sorties, cibles 0,5-1,0 ... 9,5-10,0
2. mon pipeline == predict.analyze (code officiel V7), probabilités identiques
3. filtre moteur : atténue le moteur porteur, garde un 2e engin plus faible
Écrit aussi demo_synthetique.wav (sons synthétiques : sert à tester l'interface, pas la précision).
"""
import numpy as np
import soundfile as sf
import torch

import predict
import stream_pipeline as sp
from denoise import remove_ego_engine

SR = sp.SR


def engine(n, f0, level, seed):
    g = np.random.default_rng(seed)
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f0 * k * t + g.uniform(0, 6.28)) / k for k in range(1, 25))
    return level * (x / 6 + 0.4 * np.convolve(g.standard_normal(n), np.ones(20) / 20, mode="same"))


def make_demo(dur=10.0):
    n, g = int(dur * SR), np.random.default_rng(0)
    t = np.arange(n) / SR
    ego = engine(n, 38.0, 0.25, 1)
    other = engine(n, 71.0, 0.05, 2) * ((t > 5.5) & (t < 8.5))
    click = np.zeros(n)
    i = int(3 * SR)
    click[i:i + 400] = g.standard_normal(400) * np.exp(-np.arange(400) / 60) * 0.8
    return (ego + other + click).astype(np.float32), ego.astype(np.float32)


def test_windows():
    w = np.zeros(10 * SR, dtype="float32") + 1e-3
    class Fake:
        labels = ["a", "negatif"]
        def predict_windows(self, wins, bs=8, progress=None):
            return np.full((len(wins), 2), .5, dtype="float32")
    r = sp.compute_window_probs(w, Fake())
    assert len(r["t_start"]) == 19 and r["t_start"][0] == .5 and r["t_end"][-1] == 10.0
    print("[OK] fenêtrage : 19 sorties sur 10 s, cibles 0,5-1,0 ... 9,5-10,0")


def test_matches_official(wav):
    st = sp.Streamer("cpu")
    r = sp.compute_window_probs(torch.from_numpy(wav), st, batch_size=4)
    rows = predict.analyze(st.model, st.labels, wav, SR, "cpu")
    ref = np.array([[row[f"p_{n}"] for n in st.labels] for row in rows])
    err = np.abs(ref - r["probs"]).max()
    assert err < 1e-4, err
    print(f"[OK] pipeline == predict.analyze sur {len(rows)} sorties (écart max {err:.1e})")


def band_db(x, lo, hi):
    X = torch.stft(torch.as_tensor(x), 1024, 256, 1024, window=torch.hann_window(1024), return_complex=True).abs() ** 2
    f = np.linspace(0, SR / 2, X.shape[0]); m = (f >= lo) & (f <= hi)
    return float(10 * torch.log10(X[m].mean() + 1e-12))


def test_denoise(wav, ego):
    t = np.arange(len(wav)) / SR; seg2 = (t > 5.5) & (t < 8.5)
    for method, ref in (("auto", None), ("ref", torch.from_numpy(ego[:3 * SR]))):
        out, _ = remove_ego_engine(torch.from_numpy(wav), method, ref)
        o = out.numpy()
        d = band_db(wav[:3 * SR], 0, 1500) - band_db(o[:3 * SR], 0, 1500)
        keep = band_db(o[seg2], 0, 1500) - band_db(o[:3 * SR], 0, 1500)
        raw = band_db(wav[seg2], 0, 1500) - band_db(wav[:3 * SR], 0, 1500)
        print(f"[{method:4s}] moteur porteur atténué de {d:.1f} dB ; 2e engin : {raw:+.1f} dB au-dessus du plancher "
              f"(brut) -> {keep:+.1f} dB (filtré)")
        assert d > 8 and keep > 0.5, "le 2e engin doit rester au-dessus du plancher filtré"


if __name__ == "__main__":
    wav, ego = make_demo()
    sf.write("demo_synthetique.wav", wav, SR, subtype="PCM_16")
    test_windows()
    test_matches_official(wav)
    test_denoise(wav, ego)
