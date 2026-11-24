# E-Commerce Intelligence Dashboard
### Global Sales Analytics and Machine Learning Prediction System
**Introduction to Data Science (IDS) Academic Project**

---

## 📌 1. Project Overview

The **E-Commerce Intelligence Dashboard** is an enterprise-grade analytics and predictive machine learning web application built on top of the academic project:
> *"Global E-commerce Sales Analysis and Total Sales Prediction Using Random Forest Regression"*

The application translates the exploratory data analysis, data cleaning, financial field recovery, outlier handling, feature engineering, and Random Forest regression model from the original Jupyter Notebook (`IDS PROJECT FINAL (2).ipynb`) and dataset (`Global_Ecommerce_Sales.csv`) into a fully interactive web prototype designed for college faculty demonstrations.

---

## 🏗️ 2. Project Architecture & Structure

```text
ids pro/
│
├── app.py                      # Main Streamlit web application & multi-page UI
├── requirements.txt            # Project Python dependencies
├── README.md                   # Comprehensive documentation and faculty presentation guide
│
├── data/
│   └── Global_Ecommerce_Sales.csv  # Clean transactional dataset copy (2,001 raw rows)
│
├── models/
│   ├── model_pipeline.py       # End-to-end model training, evaluation, & inference script
│   ├── rf_model.joblib         # Serialized Random Forest Regressor (200 trees, random_state=42)
│   ├── model_metadata.json     # Hyperparameters, R2 score (0.8808), RMSE ($261.70), feature list
│   └── test_predictions.json   # First 15 test predictions for actual vs predicted audit
│
├── utils/
│   ├── data_loader.py          # Data ingestion, tri-field financial recovery, & quality diagnostics
│   └── visualizations.py       # Publication-quality Plotly interactive charts
│
├── Global_Ecommerce_Sales.csv  # Root original dataset (preserved)
└── IDS PROJECT FINAL (2).ipynb # Original academic Jupyter notebook (preserved)
```

---

## 🔄 3. Data Science Workflow & Notebook Reconciliation

The application preserves the exact data processing and machine learning workflow developed in the notebook:

1. **Summary Row Purging:** The raw dataset contains a trailing summary row (`Order ID = 'Total'`) which is removed.
2. **Type Parsing:** Commas are stripped from `Total Sales`, and fields are cast to numeric `float64` and `datetime64[ns]`.
3. **Tri-Field Financial Cross-Recovery:**
   - If `Total Sales` is null & `Quantity` and `Unit Price` exist: $\text{Total Sales} = \text{Quantity} \times \text{Unit Price}$.
   - If `Unit Price` is null & `Total Sales` and `Quantity` exist: $\text{Unit Price} = \text{Total Sales} / \text{Quantity}$.
   - If `Quantity` is null & `Total Sales` and `Unit Price` exist: $\text{Quantity} = \text{Total Sales} / \text{Unit Price}$.
4. **Categorical Imputation & Deduplication:** Missing countries are marked `'Unknown'`, missing categories marked `'Unspecified'`, and duplicates on `Order ID` are removed.
5. **IQR Outlier Capping:** Outliers in `Unit Price` are capped using $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$ (ceiling at $\$367.37$).
6. **Feature Engineering:**
   - Temporal features: `Year`, `Month`, `DayOfWeek`.
   - Interaction feature: $\text{Qty\_x\_Price} = \text{Quantity Sold} \times \text{Unit Price}$ (with clipped price).
   - One-hot encoding with `drop_first=True` yielding 31 dummy feature dimensions.
7. **Random Forest Regression:**
   - Algorithm: `RandomForestRegressor(n_estimators=200, random_state=42)`
   - Train/Test Split: 80% / 20% ($1,406$ training records, $352$ testing records).
   - **Evaluation Metrics (Exact Notebook Match):**
     - $R^2 \text{ Score} = \mathbf{0.8808}$
     - $\text{RMSE} = \mathbf{\$261.7020}$
     - $\text{MAE} = \mathbf{\$71.5517}$

---

## 💡 4. Academic Disclosure: Target Leakage & Outlier Dynamics

In e-commerce transactions, $\text{Total Sales} = \text{Quantity Sold} \times \text{Unit Price}$ by mathematical definition.

- **Why is $R^2 = 0.8808$ instead of $1.0$?**
  During data cleaning in the notebook, `Unit Price` was capped at $\$367.37$ via IQR, but the ground-truth target `Total Sales` was left unclipped. For premium products (such as laptops or audio equipment retailing at $\$700 - \$1,000+$), `Qty_x_Price` uses the capped price, requiring the Random Forest model to learn non-linear corrections based on country, category, and seasonal indicators.
- **Academic Transparency:**
  The dashboard explicitly discloses that `Qty_x_Price` acts as a strong mathematical proxy and displays both the machine learning estimate and the direct unclipped base calculation for faculty verification.

---

## 🚀 5. Quickstart & Execution Instructions

### Prerequisites
Make sure Python 3.10+ is installed on your machine.

### Installation

1. Navigate to the project directory:
   ```bash
   cd "C:\Users\ANJALI\OneDrive\Desktop\ids pro"
   ```

2. (Optional) Activate your Python environment or Anaconda:
   ```bash
   conda activate base
   ```

3. Install required packages (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
   Or if using Anaconda Python directly:
   ```bash
   C:\Users\ANJALI\anaconda3\python.exe -m streamlit run app.py
   ```

5. The application will open in your web browser at:
   `http://localhost:8501`

---

## 🖥️ 6. Dashboard Features Walkthrough

| Page | Features & Capabilities |
| :--- | :--- |
| **🏠 Home Dashboard** | 6 executive KPI cards (Total Revenue, Orders, Quantity, AOV, Countries, Categories), monthly revenue trend, market leaders, category performance, and recent transactions. |
| **🔍 Data Explorer** | Complete dataset view, real-time multi-dimensional filters (Country, Category, Status, Date Range), instant keyword search, data quality diagnostic audit, and filtered CSV export. |
| **📊 Sales Analysis** | 8 interactive Plotly charts covering geographical revenue, product category sales, monthly/yearly timelines, quantity volumes, and order fulfillment status, with educational insights below each chart. |
| **🤖 Machine Learning Model** | Complete explanation of the 200-tree Random Forest Regressor, performance metrics ($R^2=0.8808$, $\text{RMSE}=\$261.70$), actual vs predicted scatter plot, top 10 feature importances, and academic target proxy disclosure. |
| **🔮 New Sales Prediction** | User-friendly input form with automated IQR bounds checking, real-time feature alignment, dual calculation comparison, result explanations, and form reset. |
| **📜 Prediction History** | Session log capturing each generated prediction, showing inputs, estimated sales, baseline values, variances, and outlier flags, with CSV export. |
| **ℹ️ About Project** | Academic problem statement, objectives, 8-stage data science workflow, tech stack summary, and faculty presentation viva script. |

---

## 🎓 7. Faculty Presentation Demonstration Script

When presenting this project to your college faculty:

1. **Introduction:** Open the **Home Dashboard** and point out that the dashboard reflects real business metrics computed on $1,758$ cleaned records from 12 countries with $\$1.09\text{M}$ in total sales.
2. **Data Science Rigor:** Navigate to **Data Explorer** and expand the *Dataset Schema & Quality Diagnostics* box. Show how the 3-way financial recovery formula resolved missing values.
3. **Machine Learning Transparency:** Go to the **Machine Learning Model** page. Point out that the model achieves $R^2 = 0.8808$, matching the notebook to the 4th decimal place. Explain the IQR clipping effect to show deep mastery over the data science life cycle.
4. **Live Interactive Prediction:**
   - Go to **New Sales Prediction**.
   - **Scenario 1 (Standard):** Select *United States*, *Apparel & Accessories*, *Completed*, Quantity *4*, Unit Price *$50.00*. Click **Predict Sales**. Note that the prediction is approximately $\$200.00$.
   - **Scenario 2 (High-Value Outlier):** Select *China*, *Electronics*, Quantity *4*, Unit Price *$995.00*. Notice the warning banner explaining that the unit price was capped by the IQR ceiling at $\$367.37$, illustrating the model's learned behavior!
5. **Prediction History:** Navigate to **Prediction History** to show how all live tests were recorded in session memory and can be exported as a report.

---

## ⚖️ 8. License & Academic Attribution
Developed for the Introduction to Data Science (IDS) Coursework.
Designed and implemented with Streamlit, Scikit-learn, Pandas, and Plotly.

