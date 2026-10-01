from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '..', 'models')
data_dir = os.path.join(base_dir, '..', 'data')

try:
    model = joblib.load(os.path.join(models_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    features = joblib.load(os.path.join(models_dir, 'features.pkl'))
    le_brand = joblib.load(os.path.join(models_dir, 'le_brand.pkl'))
    le_condition = joblib.load(os.path.join(models_dir, 'le_condition.pkl'))
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

df = pd.read_csv(os.path.join(data_dir, 'smartphone_data.csv'))
print(f"✅ Dataset loaded: {len(df)} records")

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 9


@app.route('/')
def home():
    return render_template('index.html', active='home')

@app.route('/predict')
def predict_page():
    return render_template('predict.html', active='predict')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html', active='analytics')

@app.route('/about')
def about_page():
    return render_template('about.html', active='about')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        brand_code = le_brand.transform([data['brand']])[0]
        condition_code = le_condition.transform([data['condition']])[0]
        
        input_data = pd.DataFrame([{
            'brand_encoded': brand_code,
            'age_months': int(data['age']),
            'ram_gb': int(data['ram']),
            'storage_gb': int(data['storage']),
            'battery_health': int(data['battery']),
            'condition_encoded': condition_code,
            'original_price': float(data['original_price']),
            'price_per_gb': float(data['original_price']) / int(data['storage']),
            'age_years': int(data['age']) / 12
        }])
        
        X_scaled = scaler.transform(input_data[features])
        prediction = model.predict(X_scaled)[0]
        original_price = float(data['original_price'])
        depreciation = ((original_price - prediction) / original_price) * 100
        
        return jsonify({
            'success': True,
            'predicted_price': float(round(prediction, 2)),
            'depreciation': float(round(depreciation, 1)),
            'original_price': float(original_price),
            'value_retained': float(round(100 - depreciation, 1))
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=75, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def generate_all_charts(filtered_df):
    charts = {}
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(filtered_df['resale_price'], bins=20, kde=True,
                 color='#8b7355', ax=ax, edgecolor='white')
    ax.axvline(filtered_df['resale_price'].mean(), color='#c9a86a',
               linestyle='--', linewidth=2,
               label=f"Mean: ₹{filtered_df['resale_price'].mean():,.0f}")
    ax.axvline(filtered_df['resale_price'].median(), color='#a8b5a0',
               linestyle='--', linewidth=2,
               label=f"Median: ₹{filtered_df['resale_price'].median():,.0f}")
    ax.set_title('Smartphone Resale Price Distribution',
                 fontsize=11, fontweight='bold', color='#4a3f35')
    ax.set_xlabel('Resale Price (₹)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    charts['price_distribution'] = fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='brand', y='resale_price', data=filtered_df,
                palette='Set3', ax=ax)
    ax.set_title('Resale Price by Brand',
                 fontsize=11, fontweight='bold', color='#4a3f35')
    ax.set_xlabel('Brand')
    ax.set_ylabel('Resale Price (₹)')
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    charts['brand_price'] = fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x='age_months', y='resale_price', hue='condition',
                    data=filtered_df, alpha=0.6, s=25, ax=ax)
    ax.set_title('Phone Age vs Resale Price',
                 fontsize=11, fontweight='bold', color='#4a3f35')
    ax.set_xlabel('Age (months)')
    ax.set_ylabel('Resale Price (₹)')
    ax.legend(title='Condition', fontsize=8, title_fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    charts['age_price'] = fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ['Excellent', 'Good', 'Fair', 'Poor']
    available = [c for c in order if c in filtered_df['condition'].unique()]
    if available:
        sns.boxplot(x='condition', y='resale_price', data=filtered_df,
                    order=available, palette='Set2', ax=ax)
    ax.set_title('Resale Price by Condition',
                 fontsize=11, fontweight='bold', color='#4a3f35')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Resale Price (₹)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    charts['condition_price'] = fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    numeric_cols = ['age_months', 'ram_gb', 'storage_gb', 'battery_health',
                    'original_price', 'resale_price']
    corr = filtered_df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
    ax.set_title('Feature Correlation Heatmap',
                 fontsize=11, fontweight='bold', color='#4a3f35')
    fig.tight_layout()
    charts['correlation'] = fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='ram_gb', y='resale_price', data=filtered_df,
                palette='coolwarm', ax=ax)
    ax.set_title('Resale Price by RAM Size',
                 fontsize=11, fontweight='bold', color='#4a3f35')
    ax.set_xlabel('RAM (GB)')
    ax.set_ylabel('Resale Price (₹)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    charts['ram_price'] = fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='storage_gb', y='resale_price', data=filtered_df,
                palette='viridis', ax=ax)
    ax.set_title('Resale Price by Storage Size',
                 fontsize=11, fontweight='bold', color='#4a3f35')
    ax.set_xlabel('Storage (GB)')
    ax.set_ylabel('Resale Price (₹)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    charts['storage_price'] = fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x='battery_health', y='resale_price',
                    data=filtered_df, alpha=0.5, s=25,
                    color='#8b7355', ax=ax)
    ax.set_title('Battery Health vs Resale Price',
                 fontsize=11, fontweight='bold', color='#4a3f35')
    ax.set_xlabel('Battery Health (%)')
    ax.set_ylabel('Resale Price (₹)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    charts['battery_price'] = fig_to_base64(fig)
    
    return charts


@app.route('/api/analytics', methods=['POST'])
def get_analytics():
    try:
        filters = request.json or {}
        
        filtered = df.copy()
        
        if filters.get('brand') and filters['brand'] != 'All':
            filtered = filtered[filtered['brand'] == filters['brand']]
        
        if filters.get('condition') and filters['condition'] != 'All':
            filtered = filtered[filtered['condition'] == filters['condition']]
        
        if filters.get('ram'):
            filtered = filtered[filtered['ram_gb'] == int(filters['ram'])]
        
        if filters.get('storage'):
            filtered = filtered[filtered['storage_gb'] == int(filters['storage'])]
        
        if filters.get('age_max'):
            filtered = filtered[filtered['age_months'] <= int(filters['age_max'])]
        
        if filters.get('price_max'):
            filtered = filtered[filtered['original_price'] <= float(filters['price_max'])]
        
        if len(filtered) == 0:
            return jsonify({'success': True, 'total': 0})
        
        # ===== FIX: Convert to Python native types =====
        stats = {
            'total': int(len(filtered)),
            'avg_price': int(filtered['resale_price'].mean()),
            'min_price': int(filtered['resale_price'].min()),
            'max_price': int(filtered['resale_price'].max()),
            'top_brand': str(filtered['brand'].mode()[0])
        }
        
        print(f"🎨 Generating charts for {len(filtered)} records...")
        charts = generate_all_charts(filtered)
        print(f"✅ Generated {len(charts)} charts")
        
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
    print("🚀 Server: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)