"""Interface Streamlit SSLAM V7 : dépose un wav -> filtre moteur (option) -> fenêtres 1 s / pas 0,5 s -> tableau.

Lancer en local :  streamlit run app.py
Hébergé : si modele.pt est absent, il est téléchargé depuis Hugging Face (secrets HF_REPO, HF_TOKEN).
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

import stream_pipeline as sp
from denoise import remove_ego_engine

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="SSLAM V7 – analyse chantier", layout="wide")

COLORS = {
    "moteur_engin": "#e08a1e", "presence_humaine": "#2f7fd1", "choc": "#c0392b",
    "cri_humain": "#8e44ad", "negatif": "#b8bfc6", sp.SILENCE: "#f1f3f5", sp.UNCERTAIN: "#f4d03f",
}


# --------------------------------------------------------------------------- modèle


def ensure_model():
    """modele.pt local, sinon téléchargement depuis Hugging Face (version hébergée)."""
    if (ROOT / "modele.pt").exists():
        return
    try:
        repo, token = st.secrets["HF_REPO"], st.secrets["HF_TOKEN"]
    except Exception:
        st.error("modele.pt est introuvable à côté de app.py, et les secrets HF_REPO / HF_TOKEN "
                 "ne sont pas définis pour le télécharger.")
        st.stop()
    from huggingface_hub import hf_hub_download
    with st.spinner("Téléchargement du modèle (360 Mo)…"):
        hf_hub_download(repo_id=repo, filename="modele.pt", token=token, local_dir=ROOT)


@st.cache_resource(show_spinner="Chargement du modèle SSLAM V7…")
def get_streamer():
    h = hashlib.sha256()
    with open(ROOT / "modele.pt", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    expected = None
    try:
        expected = json.loads((ROOT / "verification.json").read_text())["merged_model_sha256"]
    except Exception:
        pass
    return sp.Streamer("cpu"), h.hexdigest(), expected


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
    lo, hi = df.t_debut_s.to_numpy(), df.t_fin_s.to_numpy()   # demi-seconde cible de chaque sortie
    mid = (lo + hi) / 2
    fig, ax = plt.subplots(4, 1, figsize=(11, 8.2), sharex=True,
                           gridspec_kw={"height_ratios": [0.5, 1, 1.5, 1.1], "hspace": 0.12})
    for a, b, c in zip(lo, hi, df.classe):
        ax[0].axvspan(a, b, color=COLORS.get(c, "#999"), lw=0)
    ax[0].set_yticks([]); ax[0].set_title(title, loc="left", fontsize=11)
    ax[0].set_ylabel("décision", rotation=0, ha="right", va="center", fontsize=8)
    ax[1].plot(np.arange(len(w)) / sp.SR, w, lw=0.4, color="#333"); ax[1].set_ylabel("amplitude")
    ax[2].imshow(spectrogram_db(w), origin="lower", aspect="auto", cmap="magma", vmin=vmax - 90, vmax=vmax,
                 extent=[0, dur, 0, sp.SR / 2 / 1000])
    ax[2].set_ylabel("kHz")
    bottom = np.zeros(len(df))
    for name in labels:
        p = df[f"p_{name}"].to_numpy()
        ax[3].bar(mid, p, width=hi - lo, bottom=bottom, color=COLORS[name], label=name, lw=0)
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
    return (getattr(sty, "map", None) or sty.applymap)(color, subset=list(cols))


# --------------------------------------------------------------------------- barre latérale

with st.sidebar:
    st.header("Modèle V7")
    st.caption("Fenêtre de 1 s, pas de 0,5 s. Chaque prédiction porte sur la dernière demi-seconde "
               "de la fenêtre : première sortie à 1 s, 10 s d’audio donnent 19 sorties.")

    st.header("Filtre moteur porteur")
    use_filter = st.checkbox("Activer le filtre", value=False,
                             help="Optionnel et non validé avec V7 : compare toujours avec le brut.")
    method = st.radio("Profil du moteur", ["auto", "ref"], disabled=not use_filter,
                      format_func={"auto": "Auto (plancher du fichier)",
                                   "ref": "Référence (extrait moteur seul)"}.get,
                      help="Référence = plus fiable : n'enlève que le niveau du moteur porteur mesuré.")
    alpha = st.slider("Agressivité", 1.0, 3.0, 1.5, 0.1, disabled=not use_filter)
    max_att = st.slider("Atténuation max par bande (dB)", 6, 30, 18, disabled=not use_filter)
    compare = st.checkbox("Comparer avec le brut", value=True, disabled=not use_filter)

    st.header("Décision")
    silence_on = st.checkbox("Détecter le silence (niveau)", value=True)
    silence_db = st.slider("Seuil silence (dBFS)", -90, -20, -55, disabled=not silence_on)
    smooth_k = st.slider("Lissage causal (nb de sorties)", 1, 8, 1)
    min_conf = st.slider("Confiance minimale", 0.0, 0.95, 0.0, 0.05,
                         help="En dessous : « incertain ». Les softmax ne sont pas calibrés.")

# --------------------------------------------------------------------------- page

st.title("SSLAM V7 — analyse audio chantier")
st.caption("Dépose un enregistrement : (filtre moteur) → fenêtres glissantes → prédictions.")

up = st.file_uploader("Dépose ton fichier audio ici (wav / flac / ogg, 1 s minimum)", type=["wav", "flac", "ogg"])
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
    if len(wav_raw) < sp.WIN:
        raise ValueError("Audio trop court : minimum 1 s.")
except Exception as e:
    st.error(f"Lecture impossible : {e}")
    st.stop()

dur = len(wav_raw) / sp.SR
if dur > 120:
    st.warning(f"Fichier long ({dur:.0f} s) : le calcul peut prendre du temps.")

variants, info = {}, None
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

ensure_model()
streamer, sha, expected = get_streamer()
if expected and sha != expected:
    st.sidebar.warning("modele.pt ne correspond pas à la somme de contrôle de verification.json.")
else:
    st.sidebar.caption(f"modele.pt vérifié (sha256 {sha[:12]}…)" if expected else f"sha256 {sha[:12]}…")

cache = st.session_state.setdefault("probs_cache", {})
results = {}
for name, w in variants.items():
    key = (name, md5(w.numpy()))
    if key not in cache:
        bar = st.progress(0.0, text=f"Inférence ({name})…")
        cache[key] = sp.compute_window_probs(
            w, streamer,
            progress=lambda i, n, bar=bar, name=name: bar.progress(i / n, text=f"Inférence ({name}) : {i}/{n} fenêtres"))
        bar.empty()
    results[name] = cache[key]

dfs = {n: sp.decide(r, silence_db if silence_on else None, smooth_k, min_conf) for n, r in results.items()}
main_name = "filtré" if "filtré" in dfs else "brut"
main_df, labels = dfs[main_name], results[main_name]["labels"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Durée", f"{dur:.1f} s")
c2.metric("Sorties", len(main_df))
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
        share = (df.classe.value_counts(normalize=True) * 100).round(1).rename("% des sorties").to_frame()
        l, r = st.columns([1, 2])
        l.markdown("**Répartition**")
        l.dataframe(share)
        r.markdown("**Segments**")
        r.dataframe(style_table(sp.to_segments(df)))
        st.markdown("**Prédiction par sortie** (t_debut_s à t_fin_s = demi-seconde cible)")
        st.dataframe(style_table(df), height=380)
        st.download_button(f"Télécharger le CSV ({name})", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"predictions_{name}.csv", mime="text/csv")


show_result(tabs[0], main_name)
if len(tabs) > 1:
    show_result(tabs[1], "brut")
    with tabs[2]:
        b, f = dfs["brut"], dfs["filtré"]
        cmp = pd.DataFrame({"t_debut_s": b.t_debut_s, "t_fin_s": b.t_fin_s, "brut": b.classe, "filtré": f.classe})
        cmp["accord"] = cmp["brut"] == cmp["filtré"]
        st.metric("Accord brut / filtré", f"{cmp.accord.mean():.0%}")
        st.caption("Si le filtre change beaucoup de décisions, écoute le lecteur « Filtré » : "
                   "il peut avoir déformé les événements.")
        st.dataframe(style_table(cmp, ("brut", "filtré")), height=420)

st.divider()
st.caption("Rappel : une seule classe par sortie (pas de multilabel), softmax non calibrés, et aucune mesure "
           "de précision indépendante n'est certifiée pour V7. Le filtre moteur est une option expérimentale.")
