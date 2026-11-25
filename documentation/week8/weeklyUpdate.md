### 📅 Log Entry: [Week 8] — [5th November 2026]
**Type:**
- [X] Internal Team Meeting
- [ ] Lecturer Q&A
- [ ] Hybrid (Team Meeting + Lecturer Check-in)

**Attendees:** Suhaib, Ryan Goh, Tengku Adam Syafiq, Muhd Faizal, David Ling\
**Chair:** Suhaib\
**Note-taker:** David Ling  

---

#### A. Meeting Agenda
1. Discuss attribute model results from the first round of implementation
2. Discuss issues or challenges phased during implementation
3. Select and delegate neural networks

#### B. Internal Progress & Technical Discussion
* **Results of Attribute Models:**
  * **Decision Tree (Ryan)**
    * Limiting the tree (max_depth etc.) did not help; larger trees performed better.
    * Best settings: criterion = gini, max_depth = None, min_samples_leaf = 1, min_samples_split = 2.
    * Accuracy: 24%. In the ABC feature comparison, feature set A performed best (24%).
    * Error analysis: frequent mix-ups between similar species, and some species were never predicted correctly. This was improved by moving to Random Forest.
  * **Random Forest (Ryan)**
    * Baseline with 100 trees: accuracy 42%, F1 40% (lower because classes are imbalanced).
    * More trees helped but plateaued at 750–1000, so we stopped at 750.
    * Other hyperparameters left at defaults as with the DT; criterion = gini. Feature set A again performed best.
    * Test F1: 43%. Top-5 accuracy: 72%.
    * Error analysis showed similar mix-ups to the DT. Overall, RF > DT.
  * **K-Nearest Neighbours (Tengku)**
    * Used Ryan's data preparation for consistency.
    * Compared values of k; k = 21 performed best. Accuracy: 44%.
  * **Logistic Regression (Suhaib)**
    * 50/50 train/test split. Accuracy: 47%, improved slightly with an L2 (Euclidean) penalty.
  * **MLP on attributes (Suhaib)**
    * Accuracy did not improve over LR (44%).
* **Optimisation:**
  * Best F1 so far: 49% (RF + LR).
  * Different preprocessing steps gave noticeably different results.
  * Regional aggregation: grouped fine-grained part attributes into larger regions (e.g. crown and forehead → head = yellow), reducing to 73 attributes.
  * Cross-validation to be used for validation and hyperparameter tuning.
* **Issues & Challenges:**
  * Class imbalance lowers F1 compared to accuracy.
  * Visually similar species are commonly confused across all models.
* **Implementation decisions:**
  * Team to use Python 3.12 for a consistent environment.
  * Next round: implement neural networks (MLP and CNN). Keras preferred as the framework.

#### C. Lecturer Q&A & Feedback Log

#### D. Action Items & Next Steps
* Share findings and results of neural networks
