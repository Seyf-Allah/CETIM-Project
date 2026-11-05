# SSLAM V7 officielle — minimal complet

`modele.pt` contient tout le modèle V7 : backbone et poids fine-tunés fusionnés. Aucun modèle V6 ni patch séparé à récupérer. Poids strictement équivalents à la V7 officielle, sans réduction de précision ni nouvel entraînement.

Python 3.12, CPU. Installation :
```
py -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
py -m pip install -r requirements.txt
```
Utilisation :
```
py predict.py enregistrement.wav --output predictions.csv
```
L'installation nécessite Internet ; l'analyse fonctionne ensuite hors ligne. Le CSV choisi est remplacé s'il existe déjà.

Fenêtre réelle 1 s, pas 0,5 s, cible dernière demi-seconde. Première sortie à 1 s. 10 s donnent 19 sorties. Audio moyenné en mono et rééchantillonné à 16 kHz. Cinq classes : moteur_engin, presence_humaine, choc, cri_humain, negatif. Silence numérique exactement nul : negatif.

Tests : tous les tenseurs du modèle fusionné sont identiques à ceux du modèle officiel chargé avec son patch ; prédictions identiques sur un signal synthétique de 3 s. Ce contrôle technique ne constitue pas une mesure de précision. Aucun test indépendant de généralisation n'est certifié.

Conserver les dossiers backbone, compat et licenses avec le modèle. Les licences d'origine sont incluses. Les rapports, datasets, anciens modèles et scripts d'entraînement sont exclus.
