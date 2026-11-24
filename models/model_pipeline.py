"""
Machine Learning Pipeline for Global E-commerce Sales Prediction
Implements the exact model architecture and preprocessing logic from the IDS project notebook.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def build_and_train_model(data_path="data/Global_Ecommerce_Sales.csv", save_dir="models"):
    """
    Trains the Random Forest Regressor using the exact preprocessing and feature engineering
    from the IDS project notebook.
    """
    if not os.path.exists(data_path):
        if os.path.exists("Global_Ecommerce_Sales.csv"):
            data_path = "Global_Ecommerce_Sales.csv"
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 1. Load raw data
    df = pd.read_csv(data_path)

    # 2. Preprocessing from notebook
    df_clean = df[df['Order ID'] != 'Total'].copy()
    df_clean['Quantity Sold'] = pd.to_numeric(df_clean['Quantity Sold'], errors='coerce')
    df_clean['Unit Price'] = pd.to_numeric(df_clean['Unit Price'], errors='coerce')
    df_clean['Total Sales'] = pd.to_numeric(
        df_clean['Total Sales'].astype(str).str.replace(',', '', regex=False),
        errors='coerce'
    )
    df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'], errors='coerce')

    # Financial field recovery
    m_sales = df_clean['Total Sales'].isna() & df_clean['Quantity Sold'].notna() & df_clean['Unit Price'].notna()
    df_clean.loc[m_sales, 'Total Sales'] = df_clean.loc[m_sales, 'Quantity Sold'] * df_clean.loc[m_sales, 'Unit Price']

    m_price = df_clean['Unit Price'].isna() & df_clean['Total Sales'].notna() & df_clean['Quantity Sold'].notna() & (df_clean['Quantity Sold'] > 0)
    df_clean.loc[m_price, 'Unit Price'] = df_clean.loc[m_price, 'Total Sales'] / df_clean.loc[m_price, 'Quantity Sold']

    m_qty = df_clean['Quantity Sold'].isna() & df_clean['Total Sales'].notna() & df_clean['Unit Price'].notna() & (df_clean['Unit Price'] > 0)
    df_clean.loc[m_qty, 'Quantity Sold'] = df_clean.loc[m_qty, 'Total Sales'] / df_clean.loc[m_qty, 'Unit Price']

    # Impute categoricals & deduplicate
    df_clean['Country'] = df_clean['Country'].fillna('Unknown')
    df_clean['Product Category'] = df_clean['Product Category'].fillna('Unspecified')
    df_clean['Order Status'] = df_clean['Order Status'].fillna('Unknown')
    df_clean = df_clean.drop_duplicates(subset=['Order ID'], keep='first')

    # Outlier clipping using IQR
    q1 = df_clean['Unit Price'].quantile(0.25)
    q3 = df_clean['Unit Price'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)
    df_clean['Unit Price'] = df_clean['Unit Price'].clip(lower=lower_bound, upper=upper_bound)

    # Drop any remaining nulls in core financial/date fields
    df_clean = df_clean.dropna(subset=['Order Date', 'Quantity Sold', 'Total Sales'])
    df_clean['Order Status'] = df_clean['Order Status'].fillna('Unknown')

    # 3. Feature engineering
    df_model = df_clean.dropna(subset=['Order Date', 'Quantity Sold', 'Unit Price', 'Total Sales']).copy()
    df_model['Year'] = df_model['Order Date'].dt.year
    df_model['Month'] = df_model['Order Date'].dt.month
    df_model['DayOfWeek'] = df_model['Order Date'].dt.dayofweek
    df_model['Qty_x_Price'] = df_model['Quantity Sold'] * df_model['Unit Price']

    features = [
        'Quantity Sold', 'Unit Price', 'Qty_x_Price',
        'Year', 'Month', 'DayOfWeek',
        'Country', 'Product Category', 'Order Status'
    ]

    X = pd.get_dummies(df_model[features], drop_first=True)
    y = df_model['Total Sales']

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    # 6. Evaluation metrics
    y_pred = rf_model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # Feature importances
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Sample comparison table
    comparison_df = pd.DataFrame({
        'Actual': y_test.values[:15],
        'Predicted': np.round(y_pred[:15], 2),
        'Difference': np.round(y_pred[:15] - y_test.values[:15], 2)
    })

    # Available categorical options
    countries = sorted(df_clean['Country'].unique().tolist())
    categories = sorted(df_clean['Product Category'].unique().tolist())
    statuses = sorted(df_clean['Order Status'].unique().tolist())

    metadata = {
        'algorithm': 'Random Forest Regressor',
        'n_estimators': 200,
        'random_state': 42,
        'test_size': 0.2,
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'features_count': int(X.shape[1]),
        'feature_columns': X.columns.tolist(),
        'base_features': features,
        'target_variable': 'Total Sales',
        'iqr_lower_bound': lower_bound,
        'iqr_upper_bound': upper_bound,
        'countries': countries,
        'categories': categories,
        'statuses': statuses,
        'top_features': feature_importance_df.head(10).to_dict(orient='records')
    }

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'rf_model.joblib')
    metadata_path = os.path.join(save_dir, 'model_metadata.json')
    comparison_path = os.path.join(save_dir, 'test_predictions.json')

    joblib.dump(rf_model, model_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    comparison_df.to_json(comparison_path, orient='records', indent=4)

    print(f"Model successfully saved to {model_path}")
    print(f"R2 Score: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rf_model, metadata, X_test, y_test, y_pred


def predict_single_order(model, metadata, order_data):
    """
    Transforms a single order into the exact feature format expected by the model
    and returns the prediction alongside base mathematical calculation.
    """
    country = order_data.get('Country', 'Unknown')
    product_category = order_data.get('Product Category', 'Unspecified')
    order_status = order_data.get('Order Status', 'Completed')
    quantity = float(order_data.get('Quantity Sold', 1))
    unit_price = float(order_data.get('Unit Price', 10.0))
    order_date = pd.to_datetime(order_data.get('Order Date', '2026-01-01'))

    lower_bound = metadata['iqr_lower_bound']
    upper_bound = metadata['iqr_upper_bound']

    # Apply IQR outlier clipping exactly as in notebook
    clipped_unit_price = float(np.clip(unit_price, lower_bound, upper_bound))
    qty_x_price = quantity * clipped_unit_price

    # Extract temporal features
    year = int(order_date.year)
    month = int(order_date.month)
    day_of_week = int(order_date.dayofweek)

    # Base feature dictionary
    input_row = {
        'Quantity Sold': quantity,
        'Unit Price': clipped_unit_price,
        'Qty_x_Price': qty_x_price,
        'Year': year,
        'Month': month,
        'DayOfWeek': day_of_week,
        'Country': country,
        'Product Category': product_category,
        'Order Status': order_status
    }

    input_df = pd.DataFrame([input_row])
    # One-hot encode with drop_first=True
    encoded_input = pd.get_dummies(input_df, drop_first=True)

    # Align with the 31 columns used during training
    feature_columns = metadata['feature_columns']
    aligned_input = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    for col in encoded_input.columns:
        if col in aligned_input.columns:
            aligned_input.at[0, col] = float(encoded_input.at[0, col])

    # Ensure numeric columns are filled correctly
    aligned_input['Quantity Sold'] = quantity
    aligned_input['Unit Price'] = clipped_unit_price
    aligned_input['Qty_x_Price'] = qty_x_price
    aligned_input['Year'] = year
    aligned_input['Month'] = month
    aligned_input['DayOfWeek'] = day_of_week

    # Categorical dummy columns: Country_<country>, Product Category_<cat>, Order Status_<status>
    country_col = f"Country_{country}"
    if country_col in aligned_input.columns:
        aligned_input.at[0, country_col] = 1.0

    cat_col = f"Product Category_{product_category}"
    if cat_col in aligned_input.columns:
        aligned_input.at[0, cat_col] = 1.0

    status_col = f"Order Status_{order_status}"
    if status_col in aligned_input.columns:
        aligned_input.at[0, status_col] = 1.0

    # Model prediction
    predicted_sales = float(model.predict(aligned_input)[0])

    # Direct baseline calculation (Quantity * Raw Unit Price)
    baseline_sales = quantity * unit_price
    clipped_baseline_sales = quantity * clipped_unit_price
    variance = predicted_sales - baseline_sales

    is_outlier = unit_price > upper_bound or unit_price < lower_bound

    return {
        'predicted_sales': round(predicted_sales, 2),
        'baseline_sales': round(baseline_sales, 2),
        'clipped_baseline_sales': round(clipped_baseline_sales, 2),
        'variance': round(variance, 2),
        'is_outlier': is_outlier,
        'clipped_unit_price': round(clipped_unit_price, 2),
        'raw_unit_price': round(unit_price, 2),
        'quantity': quantity,
        'country': country,
        'product_category': product_category,
        'order_status': order_status,
        'order_date': str(order_date.date())
    }


if __name__ == "__main__":
    build_and_train_model()

