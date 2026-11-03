"""Interface Streamlit : dépose un wav -> filtrage moteur -> fenêtre glissante -> tableau.

Lancer :  streamlit run app.py
"""
import hashlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

import stream_pipeline as sp
from denoise import remove_ego_engine

st.set_page_config(page_title="SSLAM V6 – analyse chantier", layout="wide")

COLORS = {
    "moteur_engin": "#e08a1e", "presence_humaine": "#2f7fd1", "choc": "#c0392b",
    "cri_humain": "#8e44ad", "negatif": "#b8bfc6", sp.SILENCE: "#f1f3f5", sp.UNCERTAIN: "#f4d03f",
}

# --------------------------------------------------------------------------- ressources


@st.cache_resource(show_spinner="Chargement du modèle SSLAM V6 (360 Mo)…")
def get_streamer(device: str):
    return sp.Streamer(device)


def md5(arr) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


# --------------------------------------------------------------------------- tracé


def spectrogram_db(wav: np.ndarray):
    t = torch.from_numpy(wav).float()
    S = torch.stft(t, 512, 128, 512, window=torch.hann_window(512), return_complex=True).abs() ** 2
    return 10 * torch.log10(S + 1e-12).numpy()


def plot_analysis(wav: torch.Tensor, df: pd.DataFrame, labels: list, title: str, vmax: float):
    w = wav.numpy()
    dur = len(w) / sp.SR
    hop = float(df.t_debut_s.diff().median()) if len(df) > 1 else dur
    mid = (df.t_debut_s + df.t_fin_s).to_numpy() / 2
    lo, hi = mid - hop / 2, mid + hop / 2
    lo[0], hi[-1] = 0, dur  # la 1re/dernière décision couvre les bords

    fig, ax = plt.subplots(4, 1, figsize=(11, 8.2), sharex=True,
                           gridspec_kw={"height_ratios": [0.5, 1, 1.5, 1.1], "hspace": 0.12})
    # 0 : bande de décision
    for a, b, c in zip(lo, hi, df.classe):
        ax[0].axvspan(max(a, 0), min(b, dur), color=COLORS.get(c, "#999"), lw=0)
    ax[0].set_yticks([]); ax[0].set_title(title, loc="left", fontsize=11)
    ax[0].set_ylabel("décision", rotation=0, ha="right", va="center", fontsize=8)
    # 1 : forme d'onde
    ax[1].plot(np.arange(len(w)) / sp.SR, w, lw=0.4, color="#333")
    ax[1].set_ylabel("amplitude")
    # 2 : spectrogramme
    S = spectrogram_db(w)
    ax[2].imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 90, vmax=vmax,
                 extent=[0, dur, 0, sp.SR / 2 / 1000])
    ax[2].set_ylabel("kHz")
    # 3 : probabilités
    bottom = np.zeros(len(df))
    for name in labels:
        p = df[f"p_{name}"].to_numpy()
        ax[3].bar(mid, p, width=hop, bottom=bottom, color=COLORS[name], label=name, lw=0)
        bottom += p
    ax[3].set_ylim(0, 1); ax[3].set_ylabel("probabilités"); ax[3].set_xlabel("temps (s)")
    ax[3].legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax[3].set_xlim(0, dur)
    return fig


def style_table(df: pd.DataFrame, cols=("classe",)):
    def color(v):
        c = COLORS.get(v)
        return f"background-color:{c}55" if c else ""
    sty = df.style
    fn = getattr(sty, "map", None) or sty.applymap
    return fn(color, subset=list(cols))


# --------------------------------------------------------------------------- barre latérale

with st.sidebar:
    st.header("Fenêtre glissante")
    win_s = st.select_slider("Taille de fenêtre (s)", options=[0.5, 1.0, 1.5, 2.0], value=0.5)
    hop_s = st.slider("Pas entre fenêtres (s)", 0.1, float(win_s), float(win_s) / 2, 0.05,
                      key=f"hop_{win_s}")
    pad_mode = st.radio(
        "Fenêtre courte → entrée du modèle", ["train", "tile", "crop"], index=0,
        format_func={"train": "train : zéros jusqu'à 10 s (comme à l'entraînement)",
                     "tile": "tile : fenêtre répétée jusqu'à 10 s",
                     "crop": "crop : séquence courte, ~15× plus rapide"}.get,
        help="Le modèle a été entraîné sur 10 s. Une fenêtre de 0,5 s est hors distribution : "
             "teste les 3 modes sur tes enregistrements et garde le plus fiable.")

    st.header("Filtre moteur porteur")
    use_filter = st.checkbox("Activer le filtre", value=True)
    method = st.radio("Profil du moteur", ["auto", "ref"], disabled=not use_filter,
                      format_func={"auto": "Auto (plancher du fichier)",
                                   "ref": "Référence (extrait moteur seul)"}.get,
                      help="Référence = plus fiable : n'enlève QUE le moteur porteur, jamais un 2e engin.")
    alpha = st.slider("Agressivité", 1.0, 3.0, 1.5, 0.1, disabled=not use_filter)
    max_att = st.slider("Atténuation max par bande (dB)", 6, 30, 18, disabled=not use_filter)
    compare = st.checkbox("Comparer avec le brut", value=True, disabled=not use_filter)

    st.header("Décision")
    silence_on = st.checkbox("Détecter le silence (niveau)", value=True)
    silence_db = st.slider("Seuil silence (dBFS)", -90, -20, -55, disabled=not silence_on)
    smooth_k = st.slider("Lissage causal (nb de fenêtres)", 1, 8, 1)
    min_conf = st.slider("Confiance minimale", 0.0, 0.95, 0.0, 0.05,
                         help="En dessous : « incertain ». Les softmax ne sont pas calibrés.")

    st.header("Calcul")
    device = st.selectbox("Device", ["auto", "cpu", "cuda"])

# --------------------------------------------------------------------------- page

st.title("SSLAM V6 — analyse audio chantier")
st.caption("Dépose un enregistrement : filtrage du moteur porteur → fenêtres glissantes → prédictions.")

up = st.file_uploader("Dépose ton fichier audio ici (wav / flac / ogg)", type=["wav", "flac", "ogg"])
ref_up = None
if use_filter and method == "ref":
    ref_up = st.file_uploader("Extrait de référence : moteur porteur SEUL (2–3 s suffisent)",
                              type=["wav", "flac", "ogg"], key="ref")

if up is None:
    st.info("En attente d'un fichier. Les paramètres sont dans la barre de gauche.")
    st.stop()
if use_filter and method == "ref" and ref_up is None:
    st.warning("Méthode « Référence » : dépose aussi l'extrait du moteur porteur seul "
               "(ou repasse en « Auto » dans la barre latérale).")
    st.stop()

try:
    wav_raw, sr0 = sp.load_audio(up.getvalue())
    ref_wav = sp.load_audio(ref_up.getvalue())[0] if ref_up is not None else None
except Exception as e:
    st.error(f"Lecture impossible : {e}")
    st.stop()

dur = len(wav_raw) / sp.SR
if dur > 60:
    st.warning(f"Fichier long ({dur:.0f} s) : le calcul peut être lent sur CPU.")
# --- filtrage
variants = {}
info = None
if use_filter:
    try:
        wav_f, info = remove_ego_engine(wav_raw, method, ref_wav, alpha, float(max_att))
    except Exception as e:
        st.error(f"Filtrage impossible : {e}")
        st.stop()
    variants["filtré"] = wav_f
    if compare:
        variants["brut"] = wav_raw
else:
    variants["brut"] = wav_raw

# --- inférence (mise en cache par fichier + paramètres coûteux)
streamer = get_streamer(device)
cache = st.session_state.setdefault("probs_cache", {})
results = {}
for name, w in variants.items():
    key = (name, md5(w.numpy()), win_s, round(hop_s, 3), pad_mode, str(streamer.device))
    if key not in cache:
        bar = st.progress(0.0, text=f"Inférence ({name}) sur {streamer.device}…")
        cache[key] = sp.compute_window_probs(
            w, streamer, win_s, hop_s, pad_mode,
            progress=lambda i, n, bar=bar, name=name: bar.progress(i / n, text=f"Inférence ({name}) : {i}/{n} fenêtres"))
        bar.empty()
    results[name] = cache[key]

dfs = {n: sp.decide(r, silence_db if silence_on else None, smooth_k, min_conf) for n, r in results.items()}
main_name = "filtré" if "filtré" in dfs else "brut"
main_df = dfs[main_name]
labels = results[main_name]["labels"]

# --- résumé
c1, c2, c3, c4 = st.columns(4)
c1.metric("Durée", f"{dur:.1f} s")
c2.metric("Fenêtres", len(main_df))
c3.metric("Temps de calcul", f"{results[main_name]['compute_s']:.1f} s")
if info:
    c4.metric("Atténuation moteur", f"{info['attenuation_db']:.1f} dB")

st.subheader("Écoute")
a1, a2 = st.columns(2)
a1.caption("Original")
a1.audio(sp.wav_to_bytes(wav_raw), format="audio/wav")
if use_filter:
    a2.caption("Filtré (moteur porteur retiré)")
    a2.audio(sp.wav_to_bytes(variants["filtré"]), format="audio/wav")

# --- onglets
tab_names = [f"Résultats ({main_name})"]
if "brut" in dfs and main_name != "brut":
    tab_names += ["Résultats (brut)", "Brut vs filtré"]
tabs = st.tabs(tab_names)
vmax = float(spectrogram_db(wav_raw.numpy()).max())


def show_result(tab, name):
    df = dfs[name]
    with tab:
        fig = plot_analysis(variants[name], df, labels, f"Analyse – {name}", vmax)
        st.pyplot(fig)
        plt.close(fig)
        share = (df.classe.value_counts(normalize=True) * 100).round(1).rename("% du temps").to_frame()
        l, r = st.columns([1, 2])
        l.markdown("**Répartition**")
        l.dataframe(share)
        r.markdown("**Segments**")
        r.dataframe(style_table(sp.to_segments(df)))
        st.markdown("**Prédiction par fenêtre**")
        st.dataframe(style_table(df), height=380)
        st.download_button(f"Télécharger le CSV ({name})", df.to_csv(index=False).encode(),
                           file_name=f"predictions_{name}.csv", mime="text/csv")


show_result(tabs[0], main_name)
if len(tabs) > 1:
    show_result(tabs[1], "brut")
    with tabs[2]:
        b, f = dfs["brut"], dfs["filtré"]
        cmp = pd.DataFrame({"t_debut_s": b.t_debut_s, "t_fin_s": b.t_fin_s,
                            "brut": b.classe, "filtré": f.classe})
        cmp["accord"] = cmp["brut"] == cmp["filtré"]
        st.metric("Accord brut / filtré", f"{cmp.accord.mean():.0%}")
        st.caption("Si le filtre change beaucoup de décisions, vérifie à l'oreille (lecteur « Filtré ») "
                   "qu'il n'a pas mangé ou déformé les événements.")
        st.dataframe(style_table(cmp, ("brut", "filtré")), height=420)
        if info is not None:
            fig, ax = plt.subplots(figsize=(8, 2.6))
            ax.plot(np.linspace(0, sp.SR / 2 / 1000, len(info["profile_db"])), info["profile_db"], color="#e08a1e")
            ax.set_xlabel("kHz"); ax.set_ylabel("dB"); ax.set_title("Profil du moteur porteur retiré", fontsize=10)
            st.pyplot(fig); plt.close(fig)

st.divider()
st.caption("Rappel : le modèle choisit UNE classe par fenêtre (pas de multilabel), les softmax ne sont pas "
           "calibrés, et il n'a pas été validé sur des enregistrements micro-sur-engin ni sur des fenêtres de 0,5 s.")
