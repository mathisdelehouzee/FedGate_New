# Data Folder

Ce dossier regroupe les fichiers dataset legers et utiles pour le papier.

## Contenu

- `cbms_dataset.csv`
  Table principale utilisee dans les configs `fedgate_full/configs/*.yaml` pour les features tabulaires et les splits.

- `cbms_dataset.stats.json`
  Resume statistique associe a `cbms_dataset.csv`.

- `AD_CN_clinical_data.csv`
  Table clinique de reference.

- `AD_CN_df.csv`
  Table de reference AD/CN avec informations MRI et metadata utiles a la traceabilite.

- `MixedDB/`
  Documentation sur la structure du dataset complet :
  - `DATASETS.md`
  - `explain_dataset.md`
  - `cbms.pdf`

## Ce qui n'est PAS inclus

Ce dossier ne contient pas :
- les volumes MRI bruts de `MixedDB/`
- les caches `artifacts/*/mri_cache/`

## Interpretation

Ce `data/` est suffisant pour :
- documenter le dataset utilise dans le papier ;
- conserver les CSV utiles a l'analyse ;
- partager la partie legere et lisible du dataset.

Ce `data/` n'est pas suffisant pour :
- relancer l'ensemble du pipeline MRI from scratch ;
- regenerer les caches MRI sans les donnees brutes.
