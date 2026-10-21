# Tabular Data Analysis — CUB-200-2011 (Decision Tree / Random Forest)

**Author:** Ryan Goh · **Date:** 21 October 2026
**Goal:** Classify bird images into one of 200 fine-grained species using only the tabular attribute data (no pixels), with a Decision Tree and a Random Forest.

---

## 1. Dataset layout

All files are space-separated, have no header row, and use 1-based integer IDs. They all join on `image_id` (11,788 images in total).

| File | Rows | Format | Use for our model? |
|---|---|---|---|
| `classes.txt` | 200 | `class_id class_name` (e.g. `1 001.Black_footed_Albatross`) | **Yes**, maps label IDs to names |
| `images.txt` | 11,788 | `image_id relative_path` | Only for joining and sanity checks |
| `image_class_labels.txt` | 11,788 | `image_id class_id` | **Yes**, this is the target `y` |
| `train_test_split.txt` | 11,788 | `image_id is_train` (1 = train, 0 = test) | **Yes**, official split |
| `attributes/attributes.txt` | 312 | `attr_id group::value` (e.g. `1 has_bill_shape::curved_(up_or_down)`) | **Yes**, feature names |
| `attributes/certainties.txt` | 4 | 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely | **Yes**, for cleaning / weighting |
| `attributes/image_attribute_labels.txt` | 3,677,856 | `image_id attr_id is_present certainty_id time` (long format) | **No**, use the `_clean` version (see §3.1) |
| `attributes/image_attribute_labels_clean.txt` | 3,677,856 | same as above | **Yes**, this is the main feature source |
| `attributes/class_attribute_labels_continuous.txt` | 200 × 312 | % of images in each class where the attribute was marked present | **No** as a feature (see §3.6) |
| `bounding_boxes.txt` | 11,788 | `image_id x y w h` (pixels) | No, comes from the image |
| `parts/parts.txt`, `part_locs.txt`, `part_click_locs.txt` | 15 / 176,820 / 794,258 | part pixel coordinates + visibility | **No**, excluded (decision §6.4) |

Integrity checks (all passed):
- The same set of 11,788 image IDs appears in `images.txt`, `image_class_labels.txt`, `train_test_split.txt` and the attribute file.
- Every image has exactly one row per attribute: 11,788 × 312 = 3,677,856 rows, with no duplicate pairs.
- The class prefix in each image's folder path (`001.`, `002.`, …) matches its label in `image_class_labels.txt` for every image.
- There are no NaN or null values in any file.

## 2. What the attribute data looks like

### 2.1 Attributes are one-hot encoded categorical groups
The 312 binary attributes are really **28 categorical "questions"**, each split into one-hot columns (`group::value`):

- **Colour groups (15 values each):** back, belly, bill, breast, crown, forehead, leg, nape, primary, throat, under_tail, underparts, upper_tail, upperparts, wing. Eye colour has 14 values.
- **Pattern groups (4 values: solid / spotted / striped / multi-coloured):** back, belly, breast, tail, wing.
- **Other groups:** bill_shape (9), bill_length (3), head_pattern (11), shape (14), size (5), tail_shape (6), wing_shape (5).

**Important:** colour groups are multi-label, so one image can have up to 4 colours ticked for the same body part. For example, 57% of images have more than one wing colour. Pattern, shape and size groups are effectively single-choice (at most 1 value ticked). This means the columns **cannot be collapsed into 28 single categorical columns** without losing information. Keeping the 312 binary columns is the natural fit for trees.

### 2.2 Labels come from one MTurk worker per image, so they are noisy
- The `time` value is identical within each attribute group of an image (28 distinct times per image). This shows that **one worker annotated each image**, answering one question per body part. The labels are therefore one person's opinion, not a consensus.
- About 10% of all entries are `is_present = 1`. On average an image has **31.5 attributes present** (min 3, max 72).
- If we average the image-level labels per class, the result correlates 0.987 with the provided class-level file. The labels are consistent overall, but individual images deviate from their class average by about 0.11 on average (on a 0–1 scale). **Expect a noisy, hard problem** with 200 classes and roughly 30 training images per class.

### 2.3 Certainty and the "not visible" problem
| certainty | rows | is_present = 1 |
|---|---|---|
| 1 not visible | 394,906 (10.7%) | **0, always** |
| 2 guessing | 228,021 | 27,306 |
| 3 probably | 1,021,155 | 121,630 |
| 4 definitely | 2,033,774 | 222,111 |

"Not visible" is **always** recorded as `is_present = 0`. A 0 can therefore mean either "the bird doesn't have this" or "this body part couldn't be seen". **This is effectively missing data disguised as a 0.** On average 10.7% of an image's attributes are not visible, and 353 images have more than 50% not visible. A tree cannot tell the two meanings apart unless we encode them differently.

### 2.4 Attribute prevalence (sparse, skewed)
- The median attribute is present in only 4.1% of images.
- 48 attributes are present in less than 1% of images, and 27 in less than 0.5%. Most of these are rare colours such as green or purple eyes, legs and bills.
- No attribute is entirely constant (all appear at least once). The most common is `has_eye_color::black` at 84%.

### 2.5 Class balance
- 200 classes. Most have 60 images, the smallest has **41** (`006.Least_Auklet`), and the mean is 58.9 (std 2.9). **Imbalance is mild.**
- The official split is 5,994 train / 5,794 test (about 51/49), with **29–30 training images in every class**. The official split is therefore already almost perfectly balanced for training.

### 2.6 Duplicates
- 55 images share their exact feature vector with at least one other image (31 extra copies beyond the first). In 16 of these duplicate groups, the identical vectors carry **different species labels**. No model can separate those cases, so they are irreducible error. The number is too small to worry about. Keep them and just note them in the report.

## 3. Recommended cleaning and preparation steps

### 3.1 Use `image_attribute_labels_clean.txt`, not the raw file
The raw `image_attribute_labels.txt` has **606 malformed rows** (images 2275 and 9364) with a stray extra `0` column (6 fields instead of 5). This breaks `pd.read_csv(sep=' ')`. The `_clean` version removes the extra field. Note that it uses **Windows line endings (CRLF)**, so read it with `sep=r'\s+'` (or `delim_whitespace`) to strip the `\r`.

```python
ial = pd.read_csv('CUB_200_2011/attributes/image_attribute_labels_clean.txt',
                  sep=r'\s+', header=None,
                  names=['image_id','attr_id','is_present','certainty','time'])
```

### 3.2 Pivot from long to wide
```python
X = ial.pivot(index='image_id', columns='attr_id', values='is_present')  # 11788 x 312
C = ial.pivot(index='image_id', columns='attr_id', values='certainty')
```
Rename the columns using `attributes.txt` so the tree's splits are readable (`has_wing_color::blue`, etc.). Join `y` from `image_class_labels.txt` and the split from `train_test_split.txt` on `image_id`.

### 3.3 Handle "not visible" (decided: compare all three, §6.2)
Options, roughly from simplest to richest:
- **A. Ignore it.** Use the raw 0/1 values. This is the simplest baseline.
- **B. Three-state encoding.** Present = 1, absent = 0, not visible = −1. Trees handle this natively without needing imputation.
- **C. Add 28 "group not visible" flags,** one per body part. This tells the tree explicitly when a part was hidden.

I suggest running A as the baseline and B or C as the improvement, and reporting both. That gives us a justified, evidence-based decision for the report.

### 3.4 Optionally use certainty
For example, treat `guessing` positives (27k) as 0, or keep only `probably` or `definitely` positives. Doing so would lower the mean number of positives per image from 31.5 to 29.2. Try this as an experiment rather than a fixed rule.

### 3.5 Feature selection and dimensionality
- Trees do not need scaling or normalisation, and the features are already binary, so **no scaling step is required**. (KNN and Logistic Regression teammates *will* need to think about this.)
- Rare attributes (<0.5% prevalence) could be dropped, but trees naturally ignore useless features. Random Forest feature importances (or `max_features`) will tell us which ones matter, so it's better to measure first than to drop blindly.
- `time` should **not** be used as a feature. It describes the annotator, not the bird.

### 3.6 Leakage to avoid
- `class_attribute_labels_continuous.txt` is computed **from the labels of all images, including test images, grouped by class**. Using it as a feature, or deriving class-mean features from it, would leak the target. Keep it for EDA only.
- Any feature selection or certainty threshold must be tuned on the training split only, using cross-validation, and never on the test set.

### 3.7 Train / test split and class balance
- The official split already gives 29–30 training images per class, so further subsampling to the smallest class (41) is not needed when using that split.
- Use `StratifiedKFold` on the training set to tune hyperparameters (`max_depth`, `min_samples_leaf`, `n_estimators`, `max_features`, `criterion`), then evaluate once on the test split.
- Report accuracy plus macro-F1, and look at which species get confused (likely the many similar sparrows, warblers and gulls).

## 4. Proposed pipeline (for the shared data-prep notebook)
1. Load the 5 needed files (§1), joined on `image_id`.
2. Pivot the attributes long → wide (312 columns) and attach readable names.
3. Apply the not-visible encoding (variant A / B / C).
4. Attach `y` (class_id 1–200) and `is_train`.
5. Save the result to `data/` as a single tidy table (e.g. `cub_tabular.parquet` or `.csv`). The KNN and LogReg notebooks can then load the same prepared data.
6. Model notebooks: DT baseline → tuned DT → Random Forest → compare.

## 5. Environment notes
- `pandas` / `numpy` are installed (Python 3.12). **`scikit-learn` is not installed yet:** `pip install scikit-learn`.
- `CUB_200_2011/` is already excluded by `.gitignore`. Each teammate needs their own local copy of the dataset at the repo root.

## 6. Decisions (21 Oct 2026)
1. **Split:** use the **official `train_test_split.txt`**. Training classes are already balanced at 29–30 each, so the lecturer's subsample-to-smallest-class step isn't needed for training. Note in the report that test classes range from 11 to 30 images, which is why macro-F1 is reported alongside accuracy.
2. **"Not visible":** run variant **A (raw 0/1) as the baseline**, then compare it with **B (−1/0/1)** and **C (+28 per-part flags)**. Report all three and justify the final choice with CV results.
3. **`_clean` file:** this came with the Kaggle download, so it was likely added by whoever uploaded it there. It isn't in the original Caltech release. It is identical to the raw file except for (a) CRLF line endings and (b) the 606 malformed rows (images 2275 and 9364), which have the stray extra field removed. State this in the report. It can be reproduced from the raw file with `awk 'NF==6{print $1,$2,$3,$4,$6; next} {print}'`. I checked this: the output is byte-identical to the Kaggle `_clean` file once its CRLFs are stripped.

4. **Part locations excluded:** `parts/` (part_locs, part_click_locs) and `bounding_boxes.txt` are not used. They are pixel-level image annotations, not bird attributes, and we're classifying from tabular attributes only.

## 7. Data-prep notebook
Implemented in `notebooks/dataPrep.ipynb`. It loads and validates the data, pivots it to wide format, and builds feature variants A, B and C (plus an optional "drop guessing positives" switch). It writes these to `data/`:

| File | Contents |
|---|---|
| `data/cub_features_A.csv` | `image_id`, 312 attribute columns (0/1), `class_id`, `class_name`, `is_train` |
| `data/cub_features_B.csv` | same, with not-visible encoded as −1 |
| `data/cub_features_C.csv` | variant A + 28 `notvis::<group>` flags |

Model notebooks should load one of these files and split on `is_train`.

## 8. Open questions
1. **Shared prep:** Faizal and David own EDA/cleaning (R2). These notes should be shared with them so the data-prep notebook uses the same loading steps (§3.1–3.2).

## 9. Decision tree results (`notebooks/decisionTree.ipynb`)
Protocol: official split, stratified 5-fold CV on the training set for all tuning, and a single evaluation on the test set. `random_state = 42`.

| Metric | Value |
|---|---|
| Chance accuracy | 0.5% |
| CV accuracy (best config) | 24.7% ± 0.5% |
| Test accuracy | 24.7% |
| Test macro-F1 | 0.245 |
| Train accuracy | 99.9% |
| Tree size | depth 46, 3,429 leaves |

- **Tuning didn't help.** The grid search covered 480 combinations of criterion, max_depth, min_samples_leaf, min_samples_split and ccp_alpha. The best one is effectively the unpruned default tree. Every kind of regularisation lowers CV accuracy: with 200 classes and about 30 images each, leaves have to be tiny to tell species apart.
- **Very high variance** (99.9% train vs 24.7% CV). This is the main motivation for the Random Forest.
- **Feature variants:** A 0.247, C 0.243, B 0.243 (CV). The differences are within fold noise, so we keep A (simplest) and should re-test B/C with the Random Forest.
- **Errors are between near-identical species:** crows, waterthrushes, hummingbirds, Blue Grosbeak vs Indigo Bunting. Colour attributes dominate the feature importances. The root split is `has_throat_color::yellow`.

## 10. Random forest results (`notebooks/randomForest.ipynb`)
Same protocol and CV folds as the decision tree. One-at-a-time validation curves use the out-of-bag (OOB) score.

| Metric | Decision tree | Random forest |
|---|---|---|
| CV accuracy (5-fold, train) | 24.7% | 44.5% |
| Test accuracy | 24.7% | **45.6%** |
| Test macro-F1 | 0.245 | **0.435** |
| Test top-5 accuracy | 26.4% | **72.1%** |

- **Selected configuration:** variant C, gini, `max_features='sqrt'`, `min_samples_leaf=2`, `max_depth=30`, 750 trees (where the OOB curve plateaus).
- **The forest is significantly better:** McNemar's exact test gives p ≈ 1e-221, and 183 of the 200 species improve.
- **Gini beats entropy** by about 6 points. The top grid configurations are within about 0.5 points of each other, so the forest is robust to tuning.
- **"Not visible" encoding:** C ≈ A (0.445 vs 0.444, well within the fold std of about 1.5 points) > B (0.435). There is no measurable benefit from encoding visibility, which is consistent with the decision tree results.
- **Most important attribute groups (group permutation on test):** throat colour, bill shape, bill length, forehead, crown and bill colour. Tail and wing shape and pattern contribute about nothing.
