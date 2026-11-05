"""Analyse d'un wav en ligne de commande (V7 : fenêtre 1 s, pas 0,5 s), avec filtre moteur optionnel.

  python run_cli.py parking.wav --out resultats.csv
  python run_cli.py parking.wav --denoise ref --ref reference_moteur_4s.wav --compare
(Pour la CLI officielle sans filtre : python predict.py enregistrement.wav --output predictions.csv)
"""
import argparse
from pathlib import Path

import pandas as pd

import stream_pipeline as sp
from denoise import remove_ego_engine


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--denoise", choices=["off", "auto", "ref"], default="off")
    ap.add_argument("--ref", type=Path, help="extrait 'moteur porteur seul' (pour --denoise ref)")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--max-atten-db", type=float, default=18.0)
    ap.add_argument("--silence-db", type=float, default=-55.0, help="seuil silence (dBFS), 'nan' = off")
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--compare", action="store_true", help="analyse aussi le brut")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-filtered", type=Path, default=None)
    a = ap.parse_args()

    wav, _ = sp.load_audio(a.audio)
    variants = {"brut": wav}
    if a.denoise != "off":
        ref = sp.load_audio(a.ref)[0] if a.ref else None
        wav_f, info = remove_ego_engine(wav, a.denoise, ref, a.alpha, a.max_atten_db)
        print(f"Filtre moteur ({a.denoise}) : {info['attenuation_db']:.1f} dB d'atténuation")
        if a.save_filtered:
            a.save_filtered.write_bytes(sp.wav_to_bytes(wav_f))
        variants = {"filtre": wav_f, **({"brut": wav} if a.compare else {})}

    streamer = sp.Streamer("cpu")
    sil = None if a.silence_db != a.silence_db else a.silence_db
    dfs = {}
    for name, w in variants.items():
        res = sp.compute_window_probs(w, streamer)
        dfs[name] = sp.decide(res, sil, a.smooth, a.min_conf)
        print(f"\n== {name} ({len(dfs[name])} sorties, {res['compute_s']:.1f} s)")
        print(dfs[name].drop(columns=["contexte_debut_s", "contexte_fin_s"]).to_string(index=False))
        print("\nSegments :\n" + sp.to_segments(dfs[name]).to_string(index=False))
    if a.out:
        out = pd.concat(dfs, names=["variante"]).reset_index(level=0) if len(dfs) > 1 else next(iter(dfs.values()))
        out.to_csv(a.out, index=False, encoding="utf-8-sig")
        print(f"\nCSV écrit : {a.out}")


if __name__ == "__main__":
    main()
