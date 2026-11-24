import os
import sys
import pandas as pd
import joblib
import json

from utils.data_loader import load_and_process_data, compute_kpis
from models.model_pipeline import predict_single_order
from utils.visualizations import (
    plot_sales_by_country, plot_sales_by_category, plot_monthly_sales_trend,
    plot_yearly_sales, plot_quantity_by_category, plot_order_status_distribution,
    plot_actual_vs_predicted, plot_feature_importance
)

print("1. Testing data loader...")
df_raw, df_clean, stats = load_and_process_data("data/Global_Ecommerce_Sales.csv")
print(f"Raw shape: {df_raw.shape}, Clean shape: {df_clean.shape}")

print("2. Testing KPI calculation...")
kpis = compute_kpis(df_clean)
print(f"Total Revenue: ${kpis['total_revenue']:,.2f}, Total Orders: {kpis['total_orders']}")

print("3. Testing model loading...")
model = joblib.load("models/rf_model.joblib")
with open("models/model_metadata.json") as f:
    meta = json.load(f)
print(f"Model loaded: {meta['algorithm']}, R2: {meta['r2_score']:.4f}, RMSE: {meta['rmse']:.2f}")

print("4. Testing prediction with normal inputs...")
test_order_1 = {
    "Country": "United States",
    "Product Category": "Apparel & Accessories",
    "Order Status": "Completed",
    "Quantity Sold": 4,
    "Unit Price": 50.0,
    "Order Date": "2026-05-15"
}
res1 = predict_single_order(model, meta, test_order_1)
print("Test 1 Prediction:", res1)

print("5. Testing prediction with outlier price...")
test_order_2 = {
    "Country": "China",
    "Product Category": "Electronics",
    "Order Status": "Completed",
    "Quantity Sold": 2,
    "Unit Price": 850.0,
    "Order Date": "2026-07-20"
}
res2 = predict_single_order(model, meta, test_order_2)
print("Test 2 Prediction:", res2)

print("6. Testing visualization functions...")
fig1 = plot_sales_by_country(df_clean)
fig2 = plot_sales_by_category(df_clean)
fig3 = plot_monthly_sales_trend(df_clean)
fig4 = plot_yearly_sales(df_clean)
fig5 = plot_quantity_by_category(df_clean)
fig6 = plot_order_status_distribution(df_clean)
fig7 = plot_feature_importance(meta["top_features"])
print("ALL TESTS PASSED SUCCESSFULLY!")

