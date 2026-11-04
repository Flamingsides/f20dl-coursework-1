# f20dl-coursework-1

## Setup

1. Place the `CUB_200_2011/` dataset folder at the repo root (it is git-ignored).
2. Create the virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Start Jupyter and run `notebooks/dataPrep.ipynb` to generate the feature files in `data/`:
   ```bash
   jupyter lab
   ```

See [`tabularDataAnalysis.md`](tabularDataAnalysis.md) for the data analysis and preparation decisions.
