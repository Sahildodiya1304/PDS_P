import sys
import io
import os
import base64
import joblib
import pandas as pd
import numpy as np

# Configure Windows UTF-8 stdout safely to avoid charmap / cp1252 crash
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '..', 'models')
data_dir = os.path.join(base_dir, '..', 'data')

# Load trained models & preprocessing assets
try:
    model = joblib.load(os.path.join(models_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    features = joblib.load(os.path.join(models_dir, 'features.pkl'))
    le_brand = joblib.load(os.path.join(models_dir, 'le_brand.pkl'))
    le_condition = joblib.load(os.path.join(models_dir, 'le_condition.pkl'))
    print("[OK] Models and preprocessors loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    sys.exit(1)

# Load dataset
df = pd.read_csv(os.path.join(data_dir, 'smartphone_data.csv'))
print(f"[OK] Dataset loaded: {len(df)} records")

# Professional brand color mapping for executive analytics
BRAND_COLORS = {
    'Apple': '#64748b',    # Slate Silver
    'Samsung': '#2563eb',  # Enterprise Blue
    'OnePlus': '#dc2626',  # Crimson Red
    'Google': '#0d9488',   # Emerald / Teal
    'Xiaomi': '#d97706'    # Amber / Orange
}

CONDITION_COLORS = {
    'Excellent': '#10b981', # Emerald Green
    'Good': '#3b82f6',      # Tech Blue
    'Fair': '#f59e0b',      # Warm Amber
    'Poor': '#ef4444'       # Rose Red
}

# Cache for default charts to make /analytics load instantaneously
DEFAULT_CHARTS_CACHE = None
DEFAULT_STATS_CACHE = None


def apply_chart_theme(fig, ax):
    """Applies a clean, professional dark slate theme to matplotlib figures."""
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    
    # Spines
    for spine in ax.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(1.0)
        
    # Ticks & labels
    ax.tick_params(colors='#94a3b8', labelsize=8.5)
    ax.xaxis.label.set_color('#cbd5e1')
    ax.yaxis.label.set_color('#cbd5e1')
    ax.xaxis.label.set_size(9.0)
    ax.yaxis.label.set_size(9.0)
    ax.xaxis.label.set_weight('600')
    ax.yaxis.label.set_weight('600')
    
    # Title
    ax.title.set_color('#f8fafc')
    ax.title.set_fontsize(11.0)
    ax.title.set_weight('bold')
    
    # Grid
    ax.grid(True, color='#334155', linestyle='--', linewidth=0.6, alpha=0.5)


def fig_to_base64(fig):
    """Encodes a matplotlib figure to a base64 png string and closes the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=95, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def generate_all_charts(filtered_df):
    """Generates 9 high-resolution analytics charts matching the professional dark theme."""
    charts = {}
    
    # 1. Price Distribution
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    apply_chart_theme(fig, ax)
    sns.histplot(filtered_df['resale_price'], bins=22, kde=True,
                 color='#3b82f6', ax=ax, edgecolor='#0f172a', alpha=0.7)
    mean_val = filtered_df['resale_price'].mean()
    median_val = filtered_df['resale_price'].median()
    ax.axvline(mean_val, color='#f59e0b', linestyle='--', linewidth=1.8,
               label=f"Mean: ₹{mean_val:,.0f}")
    ax.axvline(median_val, color='#10b981', linestyle=':', linewidth=2.0,
               label=f"Median: ₹{median_val:,.0f}")
    ax.set_title('Resale Price Distribution')
    ax.set_xlabel('Resale Price (₹)')
    ax.set_ylabel('Number of Devices')
    leg = ax.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155')
    for text in leg.get_texts():
        text.set_color('#f8fafc')
    fig.tight_layout()
    charts['price_distribution'] = fig_to_base64(fig)
    
    # 2. Brand vs Resale Price
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    apply_chart_theme(fig, ax)
    brands = [b for b in ['Apple', 'Samsung', 'OnePlus', 'Google', 'Xiaomi'] if b in filtered_df['brand'].unique()]
    palette = [BRAND_COLORS.get(b, '#3b82f6') for b in brands]
    sns.boxplot(x='brand', y='resale_price', hue='brand', data=filtered_df,
                order=brands, palette=palette, ax=ax, legend=False)
    ax.set_title('Resale Price Range by Brand')
    ax.set_xlabel('Brand')
    ax.set_ylabel('Resale Price (₹)')
    fig.tight_layout()
    charts['brand_price'] = fig_to_base64(fig)
    
    # 3. Age vs Resale Price
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    apply_chart_theme(fig, ax)
    sns.scatterplot(x='age_months', y='resale_price', hue='condition',
                    hue_order=['Excellent', 'Good', 'Fair', 'Poor'],
                    palette=CONDITION_COLORS, data=filtered_df,
                    alpha=0.75, s=28, ax=ax)
    ax.set_title('Device Age vs Resale Value')
    ax.set_xlabel('Age (Months)')
    ax.set_ylabel('Resale Price (₹)')
    leg = ax.legend(title='Condition', fontsize=7.5, title_fontsize=8.5,
                    facecolor='#1e293b', edgecolor='#334155')
    if leg:
        leg.get_title().set_color('#f8fafc')
        for text in leg.get_texts():
            text.set_color('#cbd5e1')
    fig.tight_layout()
    charts['age_price'] = fig_to_base64(fig)
    
    # 4. Condition Impact on Resale Price
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    apply_chart_theme(fig, ax)
    cond_order = ['Excellent', 'Good', 'Fair', 'Poor']
    available_cond = [c for c in cond_order if c in filtered_df['condition'].unique()]
    if available_cond:
        cond_palette = [CONDITION_COLORS.get(c, '#3b82f6') for c in available_cond]
        sns.boxplot(x='condition', y='resale_price', hue='condition',
                    data=filtered_df, order=available_cond,
                    palette=cond_palette, ax=ax, legend=False)
    ax.set_title('Resale Price by Physical Condition')
    ax.set_xlabel('Condition Tier')
    ax.set_ylabel('Resale Price (₹)')
    fig.tight_layout()
    charts['condition_price'] = fig_to_base64(fig)
    
    # 5. Feature Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    apply_chart_theme(fig, ax)
    num_cols = ['age_months', 'ram_gb', 'storage_gb', 'battery_health',
                'original_price', 'resale_price']
    corr = filtered_df[num_cols].corr()
    cmap = sns.color_palette("mako", as_cmap=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, square=True,
                linewidths=0.8, linecolor='#0f172a', ax=ax,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7.5, "weight": "bold", "color": "white"})
    ax.set_title('Feature Correlation Matrix')
    fig.tight_layout()
    charts['correlation'] = fig_to_base64(fig)
    
    # 6. RAM Size vs Resale Price
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    apply_chart_theme(fig, ax)
    ram_order = sorted(filtered_df['ram_gb'].unique())
    sns.boxplot(x='ram_gb', y='resale_price', hue='ram_gb', data=filtered_df,
                order=ram_order, palette='Blues_r', ax=ax, legend=False)
    ax.set_title('Resale Price by RAM Capacity')
    ax.set_xlabel('RAM (GB)')
    ax.set_ylabel('Resale Price (₹)')
    fig.tight_layout()
    charts['ram_price'] = fig_to_base64(fig)
    
    # 7. Storage Size vs Resale Price
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    apply_chart_theme(fig, ax)
    storage_order = sorted(filtered_df['storage_gb'].unique())
    sns.boxplot(x='storage_gb', y='resale_price', hue='storage_gb', data=filtered_df,
                order=storage_order, palette='crest', ax=ax, legend=False)
    ax.set_title('Resale Price by Storage Capacity')
    ax.set_xlabel('Internal Storage (GB)')
    ax.set_ylabel('Resale Price (₹)')
    fig.tight_layout()
    charts['storage_price'] = fig_to_base64(fig)
    
    # 8. Battery Health vs Resale Price
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    apply_chart_theme(fig, ax)
    sns.scatterplot(x='battery_health', y='resale_price', data=filtered_df,
                    alpha=0.6, s=24, color='#38bdf8', ax=ax)
    sns.regplot(x='battery_health', y='resale_price', data=filtered_df,
                scatter=False, ax=ax, color='#0284c7',
                line_kws={'linewidth': 1.8, 'linestyle': '--'})
    ax.set_title('Battery Health vs Resale Price')
    ax.set_xlabel('Battery Health (%)')
    ax.set_ylabel('Resale Price (₹)')
    fig.tight_layout()
    charts['battery_price'] = fig_to_base64(fig)
    
    # 9. Feature Importance (from trained Random Forest Regressor)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    apply_chart_theme(fig, ax)
    raw_importance = pd.Series(model.feature_importances_, index=features)
    display_names = {
        'original_price': 'Original Purchase Price',
        'age_years': 'Age (Years)',
        'condition_encoded': 'Physical Condition',
        'age_months': 'Age (Months)',
        'price_per_gb': 'Price Per GB Storage',
        'battery_health': 'Battery Health (%)',
        'brand_encoded': 'Manufacturer Brand',
        'ram_gb': 'RAM Capacity (GB)',
        'storage_gb': 'Storage Capacity (GB)'
    }
    raw_importance.index = [display_names.get(f, f) for f in raw_importance.index]
    sorted_imp = raw_importance.sort_values(ascending=True)
    
    bars = ax.barh(sorted_imp.index, sorted_imp.values * 100, color='#3b82f6', edgecolor='#1e293b')
    ax.set_title('Random Forest Feature Importance')
    ax.set_xlabel('Importance Weight (%)')
    for bar in bars:
        width = bar.get_width()
        if width > 1:
            ax.text(width + 0.8, bar.get_y() + bar.get_height() / 2,
                    f'{width:.1f}%', va='center', ha='left',
                    color='#94a3b8', fontsize=7.5, weight='bold')
    ax.set_xlim(0, max(sorted_imp.values * 100) * 1.15)
    fig.tight_layout()
    charts['feature_importance'] = fig_to_base64(fig)
    
    return charts


# Pre-compute default charts for instantaneous initial page load
print("[INFO] Pre-generating default analytics charts for fast cache...")
DEFAULT_STATS_CACHE = {
    'total': int(len(df)),
    'avg_price': int(df['resale_price'].mean()),
    'min_price': int(df['resale_price'].min()),
    'max_price': int(df['resale_price'].max()),
    'top_brand': str(df['brand'].mode()[0])
}
DEFAULT_CHARTS_CACHE = generate_all_charts(df)
print(f"[OK] Cached {len(DEFAULT_CHARTS_CACHE)} default charts successfully!")


# Catalog of popular phone models by brand with baseline market specs
PHONE_MODELS_BY_BRAND = {
    'Apple': [
        {'name': 'iPhone 15 Pro Max', 'ram': 8, 'storage': 256, 'age': 8, 'price': 159900},
        {'name': 'iPhone 15 Pro', 'ram': 8, 'storage': 128, 'age': 8, 'price': 134900},
        {'name': 'iPhone 15', 'ram': 6, 'storage': 128, 'age': 8, 'price': 79900},
        {'name': 'iPhone 14 Pro Max', 'ram': 6, 'storage': 128, 'age': 20, 'price': 139900},
        {'name': 'iPhone 14', 'ram': 6, 'storage': 128, 'age': 20, 'price': 69900},
        {'name': 'iPhone 13 Pro', 'ram': 6, 'storage': 128, 'age': 32, 'price': 119900},
        {'name': 'iPhone 13', 'ram': 4, 'storage': 128, 'age': 32, 'price': 59900},
        {'name': 'iPhone 12', 'ram': 4, 'storage': 64, 'age': 44, 'price': 49900},
        {'name': 'iPhone 11', 'ram': 4, 'storage': 64, 'age': 56, 'price': 41900},
        {'name': 'iPhone SE (3rd Gen)', 'ram': 4, 'storage': 64, 'age': 28, 'price': 43900}
    ],
    'Samsung': [
        {'name': 'Galaxy S23 Ultra', 'ram': 12, 'storage': 256, 'age': 14, 'price': 124999},
        {'name': 'Galaxy S23', 'ram': 8, 'storage': 128, 'age': 14, 'price': 74999},
        {'name': 'Galaxy S22 Ultra', 'ram': 12, 'storage': 256, 'age': 26, 'price': 109999},
        {'name': 'Galaxy S22', 'ram': 8, 'storage': 128, 'age': 26, 'price': 62999},
        {'name': 'Galaxy S21 FE 5G', 'ram': 8, 'storage': 128, 'age': 28, 'price': 49999},
        {'name': 'Galaxy Z Fold 4', 'ram': 12, 'storage': 256, 'age': 22, 'price': 154999},
        {'name': 'Galaxy Z Flip 4', 'ram': 8, 'storage': 128, 'age': 22, 'price': 89999},
        {'name': 'Galaxy A54 5G', 'ram': 8, 'storage': 128, 'age': 14, 'price': 38999},
        {'name': 'Galaxy A34 5G', 'ram': 8, 'storage': 128, 'age': 14, 'price': 30999},
        {'name': 'Galaxy Note 20 Ultra', 'ram': 12, 'storage': 256, 'age': 46, 'price': 104999}
    ],
    'OnePlus': [
        {'name': 'OnePlus 11 5G', 'ram': 12, 'storage': 256, 'age': 15, 'price': 61999},
        {'name': 'OnePlus 11R', 'ram': 8, 'storage': 128, 'age': 14, 'price': 39999},
        {'name': 'OnePlus 10 Pro', 'ram': 8, 'storage': 128, 'age': 26, 'price': 66999},
        {'name': 'OnePlus 10T', 'ram': 8, 'storage': 128, 'age': 22, 'price': 49999},
        {'name': 'OnePlus 9 Pro', 'ram': 8, 'storage': 128, 'age': 38, 'price': 64999},
        {'name': 'OnePlus 9R', 'ram': 8, 'storage': 128, 'age': 38, 'price': 39999},
        {'name': 'OnePlus Nord 3 5G', 'ram': 8, 'storage': 128, 'age': 10, 'price': 33999},
        {'name': 'OnePlus Nord CE 3', 'ram': 8, 'storage': 128, 'age': 10, 'price': 26999},
        {'name': 'OnePlus 8T', 'ram': 8, 'storage': 128, 'age': 44, 'price': 42999}
    ],
    'Google': [
        {'name': 'Pixel 8 Pro', 'ram': 12, 'storage': 128, 'age': 7, 'price': 106999},
        {'name': 'Pixel 8', 'ram': 8, 'storage': 128, 'age': 7, 'price': 75999},
        {'name': 'Pixel 7 Pro', 'ram': 12, 'storage': 128, 'age': 19, 'price': 84999},
        {'name': 'Pixel 7', 'ram': 8, 'storage': 128, 'age': 19, 'price': 59999},
        {'name': 'Pixel 7a', 'ram': 8, 'storage': 128, 'age': 12, 'price': 43999},
        {'name': 'Pixel 6 Pro', 'ram': 12, 'storage': 128, 'age': 31, 'price': 72000},
        {'name': 'Pixel 6', 'ram': 8, 'storage': 128, 'age': 31, 'price': 49000},
        {'name': 'Pixel 6a', 'ram': 6, 'storage': 128, 'age': 24, 'price': 31999}
    ],
    'Xiaomi': [
        {'name': 'Xiaomi 13 Pro', 'ram': 12, 'storage': 256, 'age': 14, 'price': 79999},
        {'name': 'Xiaomi 12 Pro', 'ram': 8, 'storage': 256, 'age': 25, 'price': 62999},
        {'name': 'Xiaomi 11T Pro', 'ram': 8, 'storage': 128, 'age': 28, 'price': 39999},
        {'name': 'Redmi Note 13 Pro+', 'ram': 8, 'storage': 256, 'age': 4, 'price': 31999},
        {'name': 'Redmi Note 12 Pro', 'ram': 6, 'storage': 128, 'age': 16, 'price': 24999},
        {'name': 'Redmi Note 11 Pro', 'ram': 6, 'storage': 128, 'age': 26, 'price': 18999},
        {'name': 'POCO F5 5G', 'ram': 8, 'storage': 256, 'age': 12, 'price': 29999},
        {'name': 'POCO X5 Pro', 'ram': 6, 'storage': 128, 'age': 15, 'price': 22999}
    ]
}


# Web Routes
@app.route('/')
def home():
    return render_template('index.html', active='home')


@app.route('/predict')
def predict_page():
    return render_template('predict.html', active='predict', models_by_brand=PHONE_MODELS_BY_BRAND)


@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html', active='analytics')


@app.route('/about')
def about_page():
    return render_template('about.html', active='about')


@app.route('/api/models', methods=['GET'])
def get_phone_models():
    """Returns the catalog of phone models by brand."""
    return jsonify({'success': True, 'models': PHONE_MODELS_BY_BRAND})


# API Endpoints
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        
        # Validations
        brand = str(data.get('brand', '')).strip()
        phone_model = str(data.get('phone_model', '')).strip()
        condition = str(data.get('condition', '')).strip()
        
        if brand not in le_brand.classes_:
            return jsonify({'success': False, 'error': f"Unknown brand '{brand}'. Choose from: {list(le_brand.classes_)}"})
        if condition not in le_condition.classes_:
            return jsonify({'success': False, 'error': f"Unknown condition '{condition}'. Choose from: {list(le_condition.classes_)}"})
            
        original_price = float(data.get('original_price', 0))
        if original_price <= 0:
            return jsonify({'success': False, 'error': 'Original purchase price must be greater than zero.'})
            
        storage = int(data.get('storage', 128))
        if storage <= 0:
            return jsonify({'success': False, 'error': 'Storage must be greater than zero.'})
            
        ram = int(data.get('ram', 8))
        age = max(0, int(data.get('age', 12)))
        battery = max(10, min(100, int(data.get('battery', 85))))
        
        brand_code = le_brand.transform([brand])[0]
        condition_code = le_condition.transform([condition])[0]
        
        input_data = pd.DataFrame([{
            'brand_encoded': brand_code,
            'age_months': age,
            'ram_gb': ram,
            'storage_gb': storage,
            'battery_health': battery,
            'condition_encoded': condition_code,
            'original_price': original_price,
            'price_per_gb': original_price / storage,
            'age_years': age / 12.0
        }])
        
        X_scaled = scaler.transform(input_data[features])
        raw_pred = float(model.predict(X_scaled)[0])
        
        # Defensive price bounds: A used phone shouldn't exceed 98% of original price or dip below ₹500
        min_floor = max(500.0, original_price * 0.04)
        max_ceiling = original_price * 0.98
        bounded_pred = max(min_floor, min(raw_pred, max_ceiling))
        
        # Depreciation & metrics
        depreciation = ((original_price - bounded_pred) / original_price) * 100.0
        depreciation = max(2.0, min(96.0, depreciation))
        value_retained = 100.0 - depreciation
        
        # Fair market range (+-4%)
        price_low = round(bounded_pred * 0.96, 2)
        price_high = round(bounded_pred * 1.04, 2)
        
        # Monthly drop rate
        monthly_depreciation = round((original_price - bounded_pred) / max(age, 1), 2)
        
        device_name = phone_model if phone_model and brand in phone_model else (f"{brand} {phone_model}".strip() if phone_model else brand)
        
        return jsonify({
            'success': True,
            'predicted_price': round(bounded_pred, 2),
            'price_range_low': price_low,
            'price_range_high': price_high,
            'depreciation': round(depreciation, 1),
            'value_retained': round(value_retained, 1),
            'original_price': round(original_price, 2),
            'monthly_depreciation': monthly_depreciation,
            'brand': brand,
            'phone_model': phone_model,
            'device_name': device_name,
            'condition': condition
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/analytics', methods=['POST'])
def get_analytics():
    try:
        filters = request.json or {}
        
        # Check if default filter (serve from instant cache)
        is_default = (
            filters.get('brand', 'All') == 'All' and
            filters.get('condition', 'All') == 'All' and
            int(filters.get('age_max', 48)) >= 48 and
            float(filters.get('price_max', 100000)) >= 100000 and
            not filters.get('ram') and
            not filters.get('storage')
        )
        
        if is_default and DEFAULT_CHARTS_CACHE is not None:
            return jsonify({
                'success': True,
                'total': DEFAULT_STATS_CACHE['total'],
                'stats': DEFAULT_STATS_CACHE,
                'charts': DEFAULT_CHARTS_CACHE
            })
            
        filtered = df.copy()
        
        if filters.get('brand') and filters['brand'] != 'All':
            filtered = filtered[filtered['brand'] == filters['brand']]
        
        if filters.get('condition') and filters['condition'] != 'All':
            filtered = filtered[filtered['condition'] == filters['condition']]
        
        if filters.get('ram') and filters['ram'] != 'All':
            filtered = filtered[filtered['ram_gb'] == int(filters['ram'])]
        
        if filters.get('storage') and filters['storage'] != 'All':
            filtered = filtered[filtered['storage_gb'] == int(filters['storage'])]
        
        if filters.get('age_max'):
            filtered = filtered[filtered['age_months'] <= int(filters['age_max'])]
        
        if filters.get('price_max'):
            filtered = filtered[filtered['original_price'] <= float(filters['price_max'])]
        
        if len(filtered) == 0:
            return jsonify({'success': True, 'total': 0})
        
        stats = {
            'total': int(len(filtered)),
            'avg_price': int(filtered['resale_price'].mean()),
            'min_price': int(filtered['resale_price'].min()),
            'max_price': int(filtered['resale_price'].max()),
            'top_brand': str(filtered['brand'].mode()[0]) if len(filtered['brand'].mode()) > 0 else 'N/A'
        }
        
        charts = generate_all_charts(filtered)
        
        return jsonify({
            'success': True,
            'total': int(len(filtered)),
            'stats': stats,
            'charts': charts
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0' if os.environ.get('PORT') else '127.0.0.1'
    print("\n" + "="*55)
    print("  🚀 PhoneValue AI Server Running Successfully!")
    print(f"  👉 Open in your browser: http://127.0.0.1:{port}")
    print(f"  👉 Or alternatively:     http://localhost:{port}")
    print("="*55 + "\n")
    app.run(debug=True, host=host, port=port)
