# 📊 Marketing Campaign Analysis — Multi-Model Statistical & ML Pipeline

> **End-to-end analysis of a fast-food chain's A/B promotion campaign across 548 store-week observations.**  
> Combines classical econometrics (OLS, Logit, Probit) with Machine Learning (Random Forest) to identify what drives sales — and what doesn't.

---

## 🧠 Business Problem

A fast-food brand tested **3 different promotions** across stores of varying **market sizes** and **ages**. The goal: find which promotion drives the most sales, and build a model that can predict sales performance for any new store configuration.

---

## 📁 Dataset

| Field | Description |
|---|---|
| `MarketID` | Market identifier |
| `MarketSize` | Small / Medium / Large |
| `LocationID` | Store identifier |
| `AgeOfStore` | Years since opening |
| `Promotion` | 1, 2, or 3 (A/B/C test) |
| `week` | Week number (1–4) |
| `SalesInThousands` | Weekly sales in £000s |

**548 observations** across 137 unique store-week combinations.

---

## 🔬 Analysis Pipeline

```
Raw Data
   ↓
EDA (distributions, trends, heatmaps)
   ↓
ANOVA + Tukey HSD (is there a real difference between promotions?)
   ↓
OLS Regression         → How much does each variable change Sales?
Random Forest          → Which variables matter most? (non-linear)
   ↓
Logit Classification   → What drives probability of High Sales? + Odds Ratios
Probit Classification  → Same question, different statistical curve
   ↓
Unified Prediction Tool (input any store → get predictions from all 4 models)
```

---

## 📸 Visual Outputs

> **Save and upload each of the following screenshots from your notebook into a folder called `/images` in this repo. File names are listed below each placeholder.**

---

### 1. Sales Distribution
> *Histogram of SalesInThousands with mean line*  
> 📌 **File:** `images/01_sales_distribution.png`

---

### 2. Sales by Promotion (Boxplot)
> *Boxplot comparing Sales across Promotion 1, 2, 3*  
> 📌 **File:** `images/02_sales_by_promotion.png`

---

### 3. Sales by Market Size (Boxplot)
> *Boxplot comparing Sales across Small, Medium, Large markets*  
> 📌 **File:** `images/03_sales_by_marketsize.png`

---

### 4. Weekly Sales Trend by Promotion
> *Line chart showing sales trend across 4 weeks for each promotion*  
> 📌 **File:** `images/04_weekly_trend.png`

---

### 5. Heatmap — Market Size vs Promotion
> *Average Sales grid: rows = MarketSize, columns = Promotion*  
> 📌 **File:** `images/05_heatmap_marketsize_promotion.png`

---

### 6. OLS Marginal Effects Plot
> *Horizontal bar chart with 95% CI — Blue = positive, Red = negative*  
> 📌 **File:** `images/06_ols_marginal_effects.png`

---

### 7. Random Forest Feature Importances
> *Horizontal bar chart with importance scores labelled*  
> 📌 **File:** `images/07_rf_feature_importances.png`

---

### 8. Logit Marginal Effects Plot
> *Horizontal bar chart — change in P(HighSales) per variable*  
> 📌 **File:** `images/08_logit_marginal_effects.png`

---

### 9. Logit Odds Ratios Plot
> *Horizontal bar chart — Green = increases odds, Red = decreases odds*  
> 📌 **File:** `images/09_logit_odds_ratios.png`

---

### 10. Probit Marginal Effects Plot
> *Horizontal bar chart — same format as Logit for direct comparison*  
> 📌 **File:** `images/10_probit_marginal_effects.png`

---

### 11. Confusion Matrix — Logit
> *2x2 heatmap (Blues)*  
> 📌 **File:** `images/11_confusion_matrix_logit.png`

---

### 12. Confusion Matrix — Probit
> *2x2 heatmap (Purples)*  
> 📌 **File:** `images/12_confusion_matrix_probit.png`

---

## 📊 Key Results

### OLS Linear Regression (R² = 0.57)

| Variable | Effect on Sales | Significant? |
|---|---|---|
| MarketSize_Medium vs Large | **−£25,900** | ✅ Yes |
| MarketSize_Small vs Large | **−£14,300** | ✅ Yes |
| Promotion_2 vs Promo 1 | **−£11,300** | ✅ Yes |
| Promotion_3 vs Promo 1 | −£1,300 | ❌ No |
| AgeOfStore (+1 year) | +£151 | ❌ Borderline (p=0.06) |
| week | +£27 | ❌ No |

---

### Random Forest Feature Importances

| Variable | Importance |
|---|---|
| MarketSize_Medium | **47.4%** |
| AgeOfStore | **24.6%** |
| Promotion_2 | 10.4% |
| week | 8.8% |
| MarketSize_Small | 6.5% |
| Promotion_3 | 2.4% |

---

### Logit — Marginal Effects & Odds Ratios

| Variable | Δ P(HighSales) | Odds Ratio | Significant? |
|---|---|---|---|
| MarketSize_Medium vs Large | **−48.6%** | **0.038** | ✅ Yes |
| Promotion_2 vs Promo 1 | **−33.7%** | **0.104** | ✅ Yes |
| AgeOfStore (+1 year) | +0.52% | 1.036 | ✅ Yes |
| MarketSize_Small vs Large | +0.87% | 1.060 | ❌ No |
| Promotion_3 vs Promo 1 | −6.4% | 0.651 | ❌ No |
| week | −1.2% | 0.920 | ❌ No |

---

### Probit — Marginal Effects (Median Threshold, Accuracy = 84%)

| Variable | Δ P(HighSales) | Significant? |
|---|---|---|
| MarketSize_Medium vs Large | **−48.2%** | ✅ Yes |
| Promotion_2 vs Promo 1 | **−31.9%** | ✅ Yes |
| AgeOfStore (+1 year) | +0.51% | ✅ Yes |
| Others | near zero | ❌ No |

---

## 🔑 Business Insights

**1. Market Size dominates everything.**  
Medium markets underperform Large markets by ~£26K/week — more than Small markets. Worth investigating whether Medium market stores are being assigned the wrong promotion strategy.

**2. Promotion 2 consistently underperforms — across every single model.**  
OLS: −£11,300. Logit OR: 0.10 (only 10% of the odds of High Sales vs Promo 1). Probit: −31.9% probability. This promotion should be re-evaluated or retired.

**3. Promotion 3 ≈ Promotion 1 statistically.**  
If they have different costs, Promo 1 is the better ROI choice.

**4. Store Age has non-linear effects.**  
OLS misses it (borderline), but Random Forest (25% importance) and both classification models find it significant. Older stores likely have established customer bases — worth using as a targeting variable.

**5. Week number is irrelevant.**  
Sales are stable across the 4-week period regardless of promotion. No ramp-up or decay effect detected.

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scipy.stats` | ANOVA |
| `statsmodels` | OLS, Logit, Probit, Marginal Effects |
| `sklearn` | Random Forest, train/test split, metrics |

---

## 📂 Repository Structure

```
📦 marketing-campaign-analysis
 ┣ 📓 Project_A_B.ipynb        ← Full notebook
 ┣ 📄 README.md                ← This file
 ┣ 📁 images/                  ← All chart outputs (see list above)
 ┃ ┣ 01_sales_distribution.png
 ┃ ┣ 02_sales_by_promotion.png
 ┃ ┣ 03_sales_by_marketsize.png
 ┃ ┣ 04_weekly_trend.png
 ┃ ┣ 05_heatmap_marketsize_promotion.png
 ┃ ┣ 06_ols_marginal_effects.png
 ┃ ┣ 07_rf_feature_importances.png
 ┃ ┣ 08_logit_marginal_effects.png
 ┃ ┣ 09_logit_odds_ratios.png
 ┃ ┣ 10_probit_marginal_effects.png
 ┃ ┣ 11_confusion_matrix_logit.png
 ┃ ┗ 12_confusion_matrix_probit.png
 ┗ 📄 WA_Marketing-Campaign.csv  ← Dataset (if shareable)
```

---

## ▶️ How to Run

1. Clone the repo
```bash
git clone https://github.com/abdul-aziz-12/marketing-campaign-analysis
```

2. Open the notebook in Google Colab or Jupyter

3. Update the CSV path in Cell 1:
```python
df = pd.read_csv('WA_Marketing-Campaign.csv')
```

4. Run all cells in order

5. Use the prediction tool at the end — input any store config to get Sales predictions + High/Low classification from all 4 models simultaneously

---

## 👤 Author

**Abdul Aziz** — MSc Business Intelligence & Analytics, Clermont School of Business  
Previously @ S&P Global | Product Manager & BI Professional  
[LinkedIn](https://linkedin.com/in/your-link) · [GitHub](https://github.com/abdul-aziz-12)
