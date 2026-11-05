# Interface Streamlit SSLAM V7

Dossier = modèle V7 (modele.pt, backbone/, compat/, licenses/, predict.py, verification.json) + ces fichiers.
Fenêtre de 1 s, pas de 0,5 s, prédiction sur la dernière demi-seconde ; 10 s d'audio = 19 sorties.

## En local (Windows, Python 3.12)
Double-clic sur `Lancer_Interface.bat` (premier lancement : crée .venv et installe, plusieurs minutes, Internet requis).
Ou à la main :
    py -3.12 -m venv .venv
    .venv\Scripts\activate
    python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -r requirements.txt -r requirements_app.txt
    python selftest.py
    python -m streamlit run app.py
Dépose ensuite un wav dans la page (exemples/ contient deux fichiers de test).

## Hébergé (Streamlit Community Cloud)
Dépôt GitHub = tout SAUF modele.pt (le modèle est téléchargé depuis Hugging Face). À la racine du dépôt,
`requirements_cloud.txt` est renommé en `requirements.txt`. Secrets de l'app : HF_REPO, HF_TOKEN
(et au besoin HF_HUB_DISABLE_XET = "1"). Python 3.12.

## En ligne de commande
    python run_cli.py enregistrement.wav --out resultats.csv
    python predict.py enregistrement.wav --output predictions.csv     (CLI officielle V7, sans filtre)

## À savoir
- Aucune mesure de précision indépendante n'est certifiée pour V7 (voir README.md).
- Le filtre moteur est une option expérimentale, non validée avec V7 : comparer avec le brut.
- Ne jamais mettre de jeton (HF_TOKEN) dans le code ou le dépôt.
