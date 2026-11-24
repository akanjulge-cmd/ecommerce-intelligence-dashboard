"""
Data loader and preprocessing module for E-Commerce Intelligence Dashboard.
Implements the exact cleaning and recovery operations from the IDS project notebook.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_and_process_data(data_path="data/Global_Ecommerce_Sales.csv"):
    """
    Loads raw CSV and applies the exact data cleaning and feature recovery steps
    from the original notebook.
    """
    if not os.path.exists(data_path):
        if os.path.exists("Global_Ecommerce_Sales.csv"):
            data_path = "Global_Ecommerce_Sales.csv"
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 1. Raw Dataset
    df_raw = pd.read_csv(data_path)

    # 2. Step-by-step cleaning from notebook
    # 2.1 Remove summary row 'Total'
    df_clean = df_raw[df_raw['Order ID'] != 'Total'].copy()

    # 2.2 Fix numeric parsing
    df_clean['Quantity Sold'] = pd.to_numeric(df_clean['Quantity Sold'], errors='coerce')
    df_clean['Unit Price'] = pd.to_numeric(df_clean['Unit Price'], errors='coerce')
    df_clean['Total Sales'] = pd.to_numeric(
        df_clean['Total Sales'].astype(str).str.replace(',', '', regex=False),
        errors='coerce'
    )
    df_clean['Order Date'] = pd.to_datetime(df_clean['Order Date'], errors='coerce')

    # Record missing values before financial recovery
    nulls_before_recovery = df_clean.isnull().sum().to_dict()

    # 2.3 Financial field recovery (three-way cross calculation)
    # Total Sales recovery
    m_sales = df_clean['Total Sales'].isna() & df_clean['Quantity Sold'].notna() & df_clean['Unit Price'].notna()
    recovered_sales_count = int(m_sales.sum())
    df_clean.loc[m_sales, 'Total Sales'] = df_clean.loc[m_sales, 'Quantity Sold'] * df_clean.loc[m_sales, 'Unit Price']

    # Unit Price recovery
    m_price = df_clean['Unit Price'].isna() & df_clean['Total Sales'].notna() & df_clean['Quantity Sold'].notna() & (df_clean['Quantity Sold'] > 0)
    recovered_price_count = int(m_price.sum())
    df_clean.loc[m_price, 'Unit Price'] = df_clean.loc[m_price, 'Total Sales'] / df_clean.loc[m_price, 'Quantity Sold']

    # Quantity Sold recovery
    m_qty = df_clean['Quantity Sold'].isna() & df_clean['Total Sales'].notna() & df_clean['Unit Price'].notna() & (df_clean['Unit Price'] > 0)
    recovered_qty_count = int(m_qty.sum())
    df_clean.loc[m_qty, 'Quantity Sold'] = df_clean.loc[m_qty, 'Total Sales'] / df_clean.loc[m_qty, 'Unit Price']

    # 2.4 Fill categoricals
    df_clean['Country'] = df_clean['Country'].fillna('Unknown')
    df_clean['Product Category'] = df_clean['Product Category'].fillna('Unspecified')
    df_clean['Order Status'] = df_clean['Order Status'].fillna('Unknown')

    # 2.5 Deduplication
    duplicates_count = int(df_clean.duplicated(subset=['Order ID']).sum())
    df_clean = df_clean.drop_duplicates(subset=['Order ID'], keep='first')

    # 2.6 Outlier capping on Unit Price using IQR
    q1 = df_clean['Unit Price'].quantile(0.25)
    q3 = df_clean['Unit Price'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)
    outliers_count = int(((df_clean['Unit Price'] < lower_bound) | (df_clean['Unit Price'] > upper_bound)).sum())
    df_clean['Unit Price'] = df_clean['Unit Price'].clip(lower=lower_bound, upper=upper_bound)

    # 2.7 Final dropna on subset
    df_clean = df_clean.dropna(subset=['Order Date', 'Quantity Sold', 'Total Sales'])
    df_clean['Order Status'] = df_clean['Order Status'].fillna('Unknown')

    # Add temporal columns for convenient analysis
    df_clean['Year'] = df_clean['Order Date'].dt.year
    df_clean['Month'] = df_clean['Order Date'].dt.month
    df_clean['Month_Year'] = df_clean['Order Date'].dt.to_period('M').astype(str)
    df_clean['DayOfWeek'] = df_clean['Order Date'].dt.day_name()

    stats = {
        'raw_shape': df_raw.shape,
        'clean_shape': df_clean.shape,
        'summary_rows_removed': int((df_raw['Order ID'] == 'Total').sum()),
        'duplicates_removed': duplicates_count,
        'outliers_capped': outliers_count,
        'iqr_lower': lower_bound,
        'iqr_upper': upper_bound,
        'recovered_sales': recovered_sales_count,
        'recovered_price': recovered_price_count,
        'recovered_qty': recovered_qty_count,
        'raw_nulls': df_raw.isnull().sum().to_dict(),
        'clean_nulls': df_clean.isnull().sum().to_dict(),
    }

    return df_raw, df_clean, stats


def compute_kpis(df):
    """
    Computes primary summary KPIs from the dataset.
    """
    total_revenue = float(df['Total Sales'].sum())
    total_orders = int(df['Order ID'].nunique())
    total_quantity = float(df['Quantity Sold'].sum())
    avg_order_value = float(df['Total Sales'].mean()) if total_orders > 0 else 0.0
    num_countries = int(df[df['Country'] != 'Unknown']['Country'].nunique())
    num_categories = int(df[df['Product Category'] != 'Unspecified']['Product Category'].nunique())

    return {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'total_quantity': total_quantity,
        'avg_order_value': avg_order_value,
        'num_countries': num_countries,
        'num_categories': num_categories
    }

