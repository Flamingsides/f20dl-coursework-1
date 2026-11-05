# F20DL Coursework 1: Birds' Species Classification (Group 2)

Classifying **200 bird species** from the CUB-200-2011 dataset, using
(1) 312 binary visual attributes (tabular) and (2) the bird photographs (images).

**Team:**

- Suhaib (chair, Logistic Regression)
- Ryan Goh (Decision Trees / Random Forest)
- Tengku Adam Syafiq (k-NN)
- Muhd Faizal (R2 data preparation, PIC)
- David Ling (R2 data preparation, image pipeline)

## For model builders: start here

Do the **Setup** and **Running the pipeline** steps below once. Then, in your model notebook in `notebooks/`:

**Tabular models** (k-NN, Decision Tree / Random Forest, Logistic Regression)

```python
import sys, json; sys.path.insert(0, "../scripts")
import r2_utils as u

df = u.load_attributes(clean=True)                 # cleaned table, 11,769 rows
train, test = df[df.is_train == 1], df[df.is_train == 0]

ATTR = u.attribute_columns(df)                     # all 312 attributes, or a smaller set:
sets = json.load(open("../data/processed/feature_sets.json"))["sets"]
# ATTR = sets["ranked top 75"]                     # other sets: see notebook 03

X_train, y_train = train[ATTR], train["class_id"]
X_test,  y_test  = test[ATTR],  test["class_id"]
```

**Image models** (CNNs; first run `python scripts/prepare_images.py` and `pip install torch torchvision`)

```python
import sys; sys.path.insert(0, "../scripts") # find augmentation.py
from augmentation import make_loaders # loads function
train_loader, val_loader, test_loader = make_loaders("../data/processed", batch_size=32) # Split into 3 sets, feed 32 images at a time
# train_loader is already augmented (training only); val/test are not, only resized.
```

**Rules everyone follows**

- Use the official split (`is_train`). Never tune on the test set. Use cross-validation, or `val_loader`, on the training data.
- Report **macro-F1** alongside accuracy (test-set sizes differ per species).
- Labels are `class_id` 1–200. The image loaders use 0–199.

## Data

|        | Dataset 1: tabular                                               | Dataset 2: images          |
| ------ | ---------------------------------------------------------------- | -------------------------- |
| Source | CUB-200-2011 attribute annotations                               | CUB-200-2011 photographs   |
| Rows   | 11,788 images, 200 species                                       | 11,788 images, 200 species |
| Inputs | 312 binary attributes (bill shape, wing colour, …)               | RGB photo + bounding box   |
| Split  | Official train/test split (`is_train`): 5,994 train / 5,794 test | Same split                 |

Data is **not stored in git** (~1.1 GB). Everyone recreates it with the pipeline below.

> Wah, C., Branson, S., Welinder, P., Perona, P. and Belongie, S. (2011). _The Caltech-UCSD Birds-200-2011 Dataset._
> Technical Report CNS-TR-2011-001, California Institute of Technology.
> Non-commercial research and educational use only.

## Setup

Requires [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html). From the repository root:

```bash
conda env create -f environment.yml      # first time only
conda activate f20dl-birds
```

> **Windows + VS Code:** open VS Code from the Anaconda Prompt _after_ `conda activate f20dl-birds`
> (`cd` to the repo, then `code .`).

## Running the pipeline

Run these in order, from the repository root:

| Step            | Command                                    | Output (in `data/`)                                             |
| --------------- | ------------------------------------------ | --------------------------------------------------------------- |
| 1. Download     | `python scripts/download_data.py`          | `raw/CUB_200_2011/`                                             |
| 2. Build tables | `python scripts/prepare_data.py`           | `processed/cub_attributes.csv`, `processed/cub_images.csv`      |
| 3. Clean        | run `notebooks/01_data_cleaning.ipynb`     | `processed/cub_attributes_clean.csv`                            |
| 4. Feature sets | run `notebooks/03_feature_selection.ipynb` | `processed/feature_sets.json`                                   |
| 5. Crop images  | `python scripts/prepare_images.py`         | `processed/images_cropped/`, `processed/cub_images_cropped.csv` |

Notebooks 02 (EDA), 04 (clustering) and 05 (image checks) are analysis only and can run any time after the steps they depend on.

## Repository structure

```
├── environment.yml           one conda environment for the whole project (pinned versions)
├── scripts/
│   ├── download_data.py      downloads CUB-200-2011 from Caltech
│   ├── prepare_data.py       raw files -> tabular + image tables; fixes malformed lines, clips boxes
│   ├── prepare_images.py     RGB conversion, crop to bounding box (+10%), resize to 256 px
│   ├── augmentation.py       train/val/test loaders; train-only augmentation for image models
│   └── r2_utils.py           shared paths, plot style and loaders
├── notebooks/
│   ├── 01_data_cleaning.ipynb       cleaning decisions 1–8, with evidence
│   ├── 02_eda.ipynb                 exploratory data analysis
│   ├── 03_feature_selection.ipynb   ranked feature sets (150 / 75 / 30 / PCA-75) for R3
│   ├── 04_clustering.ipynb          K-Means and GMM on images and species profiles
│   └── 05_image_preparation.ipynb   image pipeline checks, augmentation preview (Decision 9)
├── data/                     (git-ignored) raw/ and processed/
└── documentation/            weekly meeting logs and saved figures
```

## Key data preparation decisions (summary)

1. No missing values or duplicated ids. 606 malformed raw lines repaired, so all images are kept.
2. Classes are balanced (41–60 images per species), so all 200 species are kept and **macro-F1** is reported.
3. "Not visible" attributes are kept as raw 0/1 (best in 5-fold CV).
4. 19 conflicting duplicates (same attributes, different species) are removed from **train only**.
5. Images are converted to RGB, cropped to the bird and resized. Augmentation is applied to **training images only**.
6. The official test split is never used for any preparation decision.
