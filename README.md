# Employee Attrition Analysis — IBM HR Analytics
**Covers:** Veracity + judgment · classification, feature analysis, storytelling

## The Question
What actually predicts attrition here — and which "obvious" factors turn out not to matter?

## Data
- Source: [IBM HR Analytics Attrition, Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1,470 employees, no missing values
- Target: Attrition (Yes/No) — imbalanced, 83.9% No / 16.1% Yes (~5:1)

## Tools
Python/Pandas + scikit-learn (logistic regression, class-weighted) → matplotlib/seaborn for visualization

## Process

### 1. Cleaning & judgment calls
- Dropped `EmployeeCount`, `Over18`, `StandardHours` — truly constant across all 1,470 rows, zero predictive information
- Dropped `EmployeeNumber` — a pure row identifier (1,470 unique values / 1,470 rows), same logic as excluding a primary key from any model
- Noted class imbalance (5:1) upfront — accuracy alone would be misleading here (predicting "No" for everyone scores 84% while catching zero at-risk employees), so precision/recall/ROC-AUC are used instead, and the model uses class weighting rather than resampling

### 2. Exploratory analysis
**Overtime — confirmed as a strong, consistent driver.** Employees who work overtime have a 30.5% attrition rate vs. 10.4% for those who don't — nearly 3x higher. This holds in every department, not just one:

| Department | No overtime | Overtime |
|---|---|---|
| Sales | 13.8% | 37.5% |
| Human Resources | 15.2% | 29.4% |
| R&D | 8.6% | 27.3% |

**Job satisfaction — real, but weaker and non-linear.** Level 1 (lowest) shows 22.8% attrition, but levels 2 and 3 are nearly identical (16.4% vs 16.5%), and level 4 drops to 11.3%. Real signal, but far weaker than overtime.

**Monthly income — mostly a seniority proxy, not an independent driver.** The raw gap looked strong (leavers earned a median of $3,202 vs. $5,204 for stayers), but once controlled for job level, the gap mostly disappears — at Levels 1, 2, 3, and 5 income is nearly identical between leavers and stayers. This is the project's clearest example of an assumption that didn't survive scrutiny.

### 3. Classifier
Logistic regression with class weighting (accounts for the 5:1 imbalance), using OverTime, JobSatisfaction, JobLevel, Age, TotalWorkingYears, YearsAtCompany, MonthlyIncome, DistanceFromHome, and WorkLifeBalance as features.

**Performance (test set, 368 employees):**
- Recall on "Left" = 0.68 — catches 68% of employees who actually leave
- Precision on "Left" = 0.32 — about 1 in 3 flagged as high-risk actually leaves
- ROC-AUC = 0.750
- Accuracy (0.72) deliberately not the headline metric, given the imbalance

This is a deliberate trade-off: class weighting favors catching more true leavers (recall) at the cost of more false alarms (precision) — the right direction for an HR retention use case, where missing an at-risk employee is costlier than double-checking someone who wasn't going to leave.

**Feature importance (coefficients):**

| Feature | Coefficient | Direction |
|---|---|---|
| OverTime | +0.59 | Strongest predictor — confirms EDA |
| JobLevel | +0.22 | See follow-up below — not a simple linear effect |
| DistanceFromHome | +0.21 | New finding, not part of original hypotheses |
| Age | -0.35 | Strongest negative predictor — younger employees leave more |
| MonthlyIncome | -0.33 | Independent effect once job-level confound is accounted for |
| JobSatisfaction | -0.29 | Confirms EDA — matters, but not dominant |
| YearsAtCompany | -0.22 | More tenure → less attrition |
| WorkLifeBalance | -0.21 | Better balance → less attrition |
| TotalWorkingYears | -0.17 | More career experience → less attrition |

**Follow-up on JobLevel:** raw attrition by level shows a non-linear pattern — Level 1 (entry-level) is the clear outlier at 26.3% attrition, dropping sharply and staying low through the other levels (4.7–14.7%). The real story is "entry-level is the risk category," not a smooth level-by-level increase.

**Follow-up on DistanceFromHome:** a genuinely new finding — attrition climbs roughly linearly from 13.8% (under 5 miles) to 22.1% (20-30 miles), confirming the model's coefficient direction.

## What I found
Overtime is the strongest and most consistent attrition driver in this dataset, holding across every department. Age is the strongest independent predictor overall, with younger/early-career employees at meaningfully higher risk — and this risk is concentrated specifically at entry-level (Level 1), not spread evenly across seniority. Job satisfaction and work-life balance both matter but less than overtime. The commonly assumed "pay drives attrition" story is mostly a seniority artifact — controlling for job level nearly erases the raw income gap. Distance from home is a new, real finding that wasn't part of the original hypothesis set.

## So what
Three things HR could act on:
1. **Overtime policy** — review load in Sales specifically, where both baseline attrition and the overtime gap are worst (13.8% → 37.5%)
2. **Entry-level retention** — targeted onboarding/mentorship for Level 1 employees, since risk is concentrated there rather than spread evenly by seniority or pay
3. **Commute flexibility** — worth investigating remote/hybrid options for employees with long commutes, though this is a new lead rather than a confirmed causal driver and deserves its own follow-up

**What didn't hold up:** the raw income-attrition relationship was largely a seniority proxy, not an independent effect — a caution against treating a bivariate correlation as a standalone driver without checking for confounds.

## Repo structure
```
/notebooks
  01_eda.ipynb   — cleaning, EDA, classifier, feature importance, follow-ups
/data
  WA_Fn-UseC_-HR-Employee-Attrition.csv
README.md
```