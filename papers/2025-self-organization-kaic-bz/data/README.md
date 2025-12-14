# Data folder (local)

This paper expects raw and processed data locally. Do not commit raw datasets to git.

## Layout
- data/raw/       (downloaded source files)
- data/processed/ (derived intermediate files)

## Suggested filenames
- KaiC (eLife 2017): place the relevant source data files under data/raw/kaiC/
- BZ (SOTON/D0363): place the relevant source data files under data/raw/bz/

Once the fetch/preprocess scripts are wired, they will populate data/processed/ automatically.
