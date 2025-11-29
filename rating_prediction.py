"""
High Rating Prediction Based on Business Attributes
===================================================

This script:
1. Loads business data from Yelp dataset
2. Extracts features: location, number of reviews, attributes (WiFi, Parking, Price Range, etc.)
3. Creates binary classification target: high rating (≥4 stars) vs not high rating (<4 stars)
4. Trains a classification model to predict high ratings
5. Analyzes which features are most important for high ratings
6. Generates visualizations and analysis reports
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tarfile
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuration
TAR_FILE = "../Module6_new/data/yelp_dataset.tar"
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
THAI_RESTAURANTS_FILE = "../Module6_new/data/thai_restaurants.csv"

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_business_data(tar_path):
    """Load business data from tar archive."""
    print(f"\n{'='*80}")
    print("STEP 1: Loading Business Data from Yelp Dataset")
    print(f"{'='*80}")
    
    businesses = []
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            business_file = None
            for member in tar.getmembers():
                if 'business' in member.name and member.name.endswith('.json'):
                    business_file = member
                    break
            
            if not business_file:
                print("ERROR: Could not find business.json in tar file")
                return None
            
            print(f"Reading {business_file.name} from tar archive...")
            file_obj = tar.extractfile(business_file)
            
            for line in file_obj:
                try:
                    business = json.loads(line.decode('utf-8'))
                    businesses.append(business)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        
        print(f"Loaded {len(businesses):,} businesses from dataset")
        return businesses
    
    except Exception as e:
        print(f"ERROR: Could not read tar file: {e}")
        return None


def extract_features(businesses):
    """
    Extract features from business data for classification.
    
    Features:
    - Location: city, state
    - Number of reviews: review_count
    - Attributes: WiFi, Parking, Price Range, Accepts Credit Cards, etc.
    - Categories: business categories
    """
    print(f"\n{'='*80}")
    print("STEP 2: Extracting Features from Business Data")
    print(f"{'='*80}")
    
    feature_data = []
    
    for business in businesses:
        # Basic info
        business_id = business.get('business_id', '')
        name = business.get('name', '')
        stars = business.get('stars', 0)
        review_count = business.get('review_count', 0)
        city = business.get('city', '')
        state = business.get('state', '')
        
        # Attributes
        attributes = business.get('attributes', {})
        if attributes is None:
            attributes = {}
        
        # Extract specific attributes
        wifi = attributes.get('WiFi', 'None')
        parking = attributes.get('Parking', {})
        price_range = attributes.get('RestaurantsPriceRange2', None)
        accepts_credit_cards = attributes.get('BusinessAcceptsCreditCards', None)
        outdoor_seating = attributes.get('OutdoorSeating', None)
        good_for_groups = attributes.get('RestaurantsGoodForGroups', None)
        reservations = attributes.get('RestaurantsReservations', None)
        delivery = attributes.get('RestaurantsDelivery', None)
        takeout = attributes.get('RestaurantsTakeOut', None)
        waiter_service = attributes.get('RestaurantsTableService', None)
        alcohol = attributes.get('Alcohol', 'None')
        noise_level = attributes.get('NoiseLevel', 'None')
        ambience = attributes.get('Ambience', {})
        
        # Handle parking (can be dict or string)
        parking_val = None
        if isinstance(parking, dict):
            parking_val = 'Yes' if any(parking.values()) else 'No'
        elif isinstance(parking, str):
            parking_val = parking
        else:
            parking_val = 'None'
        
        # Categories
        categories = business.get('categories', '')
        category_list = [cat.strip() for cat in categories.split(',')] if categories else []
        is_restaurant = 'Restaurants' in category_list
        is_thai = 'Thai' in category_list
        
        # Target variable: high rating (≥4 stars) vs not high (<4 stars)
        high_rating = 1 if stars >= 4.0 else 0
        
        feature_data.append({
            'business_id': business_id,
            'name': name,
            'stars': stars,
            'high_rating': high_rating,  # Target variable
            'review_count': review_count,
            'city': city,
            'state': state,
            'wifi': wifi,
            'parking': parking_val,
            'price_range': price_range,
            'accepts_credit_cards': accepts_credit_cards,
            'outdoor_seating': outdoor_seating,
            'good_for_groups': good_for_groups,
            'reservations': reservations,
            'delivery': delivery,
            'takeout': takeout,
            'waiter_service': waiter_service,
            'alcohol': alcohol,
            'noise_level': noise_level,
            'is_restaurant': is_restaurant,
            'is_thai': is_thai,
            'num_categories': len(category_list)
        })
    
    df = pd.DataFrame(feature_data)
    print(f"Extracted features for {len(df):,} businesses")
    print(f"\nTarget Distribution:")
    print(df['high_rating'].value_counts())
    print(f"High rating (≥4 stars): {df['high_rating'].sum():,} ({df['high_rating'].mean()*100:.1f}%)")
    print(f"Not high rating (<4 stars): {(~df['high_rating'].astype(bool)).sum():,} ({(1-df['high_rating'].mean())*100:.1f}%)")
    
    return df


def prepare_features(df):
    """Prepare features for machine learning model."""
    print(f"\n{'='*80}")
    print("STEP 3: Preparing Features for Classification")
    print(f"{'='*80}")
    
    # Create a copy for feature engineering
    df_features = df.copy()
    
    # Encode categorical variables
    categorical_cols = ['wifi', 'parking', 'alcohol', 'noise_level', 'city', 'state']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df_features.columns:
            le = LabelEncoder()
            # Handle NaN values
            df_features[col] = df_features[col].fillna('Unknown')
            df_features[col] = le.fit_transform(df_features[col].astype(str))
            label_encoders[col] = le
    
    # Handle boolean/None attributes
    bool_cols = ['accepts_credit_cards', 'outdoor_seating', 'good_for_groups', 
                 'reservations', 'delivery', 'takeout', 'waiter_service']
    
    for col in bool_cols:
        if col in df_features.columns:
            # Convert to binary: True/1 -> 1, False/0/None -> 0
            df_features[col] = df_features[col].apply(
                lambda x: 1 if x in [True, 'True', 'true', 1, '1'] else 0
            )
    
    # Handle price_range (1-4 scale, None -> 0)
    if 'price_range' in df_features.columns:
        df_features['price_range'] = df_features['price_range'].fillna(0)
        df_features['price_range'] = pd.to_numeric(df_features['price_range'], errors='coerce').fillna(0)
    
    # Select features for model
    feature_cols = ['review_count', 'price_range', 'wifi', 'parking', 'alcohol', 
                   'noise_level', 'accepts_credit_cards', 'outdoor_seating', 
                   'good_for_groups', 'reservations', 'delivery', 'takeout', 
                   'waiter_service', 'num_categories']
    
    # Add city and state if we want location features
    # Note: City/state might have too many categories, so we'll use them selectively
    
    # Filter to restaurants only for better analysis
    if 'is_restaurant' in df_features.columns:
        df_features = df_features[df_features['is_restaurant'] == True]
        print(f"Filtered to restaurants: {len(df_features):,} businesses")
    
    # Remove rows with missing target
    df_features = df_features.dropna(subset=['high_rating'])
    
    # Prepare X and y
    X = df_features[feature_cols].fillna(0)
    y = df_features['high_rating']
    
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"Sample size: {len(X):,} restaurants")
    print(f"Target distribution:")
    print(y.value_counts())
    
    return X, y, df_features, feature_cols, label_encoders


def train_classification_model(X, y, feature_cols):
    """Train Random Forest classifier to predict high ratings."""
    print(f"\n{'='*80}")
    print("STEP 4: Training Classification Model")
    print(f"{'='*80}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Detailed metrics
    print(f"\n{'='*80}")
    print("Detailed Classification Report (Test Set)")
    print(f"{'='*80}")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['Not High Rating', 'High Rating']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Not High  High")
    print(f"Actual Not High   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       High        {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    return rf_model, X_test, y_test, y_test_pred, feature_importance, X_train, y_train


def create_visualizations(df_features, feature_importance, model, X_test, y_test, y_pred):
    """Create comprehensive visualizations."""
    print(f"\n{'='*80}")
    print("STEP 5: Creating Visualizations")
    print(f"{'='*80}")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Target Distribution
    plt.subplot(4, 3, 1)
    target_dist = df_features['high_rating'].value_counts()
    colors = ['#ff6b6b', '#51cf66']
    plt.bar(['Not High (<4)', 'High (≥4)'], [target_dist[0], target_dist[1]], 
            color=colors, edgecolor='black', alpha=0.7)
    plt.title('Target Distribution\n(High Rating vs Not High)', fontsize=12, fontweight='bold')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    
    # 2. Feature Importance
    plt.subplot(4, 3, 2)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    # 3. Review Count Distribution by Rating
    plt.subplot(4, 3, 3)
    high_reviews = df_features[df_features['high_rating'] == 1]['review_count']
    not_high_reviews = df_features[df_features['high_rating'] == 0]['review_count']
    plt.hist([not_high_reviews, high_reviews], bins=50, alpha=0.7, 
             label=['Not High', 'High'], color=['#ff6b6b', '#51cf66'], edgecolor='black')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Frequency')
    plt.title('Review Count Distribution by Rating', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. Price Range vs High Rating
    plt.subplot(4, 3, 4)
    if 'price_range' in df_features.columns:
        price_high = df_features[df_features['high_rating'] == 1]['price_range'].value_counts().sort_index()
        price_not_high = df_features[df_features['high_rating'] == 0]['price_range'].value_counts().sort_index()
        x = range(len(price_high))
        width = 0.35
        plt.bar([i - width/2 for i in x], price_not_high.values, width, 
                label='Not High', color='#ff6b6b', alpha=0.7)
        plt.bar([i + width/2 for i in x], price_high.values, width, 
                label='High', color='#51cf66', alpha=0.7)
        plt.xlabel('Price Range')
        plt.ylabel('Count')
        plt.title('Price Range vs High Rating', fontsize=12, fontweight='bold')
        plt.xticks(x, price_high.index)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
    
    # 5. WiFi Availability
    plt.subplot(4, 3, 5)
    if 'wifi' in df_features.columns:
        wifi_cross = pd.crosstab(df_features['high_rating'], df_features['wifi'])
        wifi_cross.plot(kind='bar', ax=plt.gca(), color=['#ff6b6b', '#51cf66'], alpha=0.7)
        plt.title('WiFi Availability vs High Rating', fontsize=12, fontweight='bold')
        plt.xlabel('High Rating')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Not High', 'High'], rotation=0)
        plt.legend(title='WiFi')
        plt.grid(axis='y', alpha=0.3)
    
    # 6. Parking Availability
    plt.subplot(4, 3, 6)
    if 'parking' in df_features.columns:
        parking_cross = pd.crosstab(df_features['high_rating'], df_features['parking'])
        parking_cross.plot(kind='bar', ax=plt.gca(), color=['#ff6b6b', '#51cf66'], alpha=0.7)
        plt.title('Parking Availability vs High Rating', fontsize=12, fontweight='bold')
        plt.xlabel('High Rating')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Not High', 'High'], rotation=0)
        plt.legend(title='Parking')
        plt.grid(axis='y', alpha=0.3)
    
    # 7. Confusion Matrix Heatmap
    plt.subplot(4, 3, 7)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks([0.5, 1.5], ['Not High', 'High'])
    plt.yticks([0.5, 1.5], ['Not High', 'High'], rotation=0)
    
    # 8. Attributes Comparison
    plt.subplot(4, 3, 8)
    attr_cols = ['accepts_credit_cards', 'outdoor_seating', 'good_for_groups', 
                'reservations', 'delivery', 'takeout', 'waiter_service']
    attr_high = df_features[df_features['high_rating'] == 1][attr_cols].mean() * 100
    attr_not_high = df_features[df_features['high_rating'] == 0][attr_cols].mean() * 100
    x = np.arange(len(attr_cols))
    width = 0.35
    plt.bar(x - width/2, attr_not_high, width, label='Not High', color='#ff6b6b', alpha=0.7)
    plt.bar(x + width/2, attr_high, width, label='High', color='#51cf66', alpha=0.7)
    plt.xlabel('Attributes')
    plt.ylabel('% Available')
    plt.title('Attribute Availability by Rating', fontsize=12, fontweight='bold')
    plt.xticks(x, [col.replace('_', ' ').title() for col in attr_cols], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 9. Star Rating Distribution
    plt.subplot(4, 3, 9)
    star_dist = df_features['stars'].value_counts().sort_index()
    colors_stars = ['#ff6b6b' if x < 4 else '#51cf66' for x in star_dist.index]
    plt.bar(star_dist.index, star_dist.values, color=colors_stars, alpha=0.7, edgecolor='black')
    plt.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Threshold (4 stars)')
    plt.xlabel('Star Rating')
    plt.ylabel('Count')
    plt.title('Star Rating Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 10. Review Count vs Rating (Scatter)
    plt.subplot(4, 3, 10)
    high_scatter = df_features[df_features['high_rating'] == 1]
    not_high_scatter = df_features[df_features['high_rating'] == 0]
    plt.scatter(not_high_scatter['review_count'], not_high_scatter['stars'], 
               alpha=0.5, s=20, label='Not High', color='#ff6b6b')
    plt.scatter(high_scatter['review_count'], high_scatter['stars'], 
               alpha=0.5, s=20, label='High', color='#51cf66')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Star Rating')
    plt.title('Review Count vs Star Rating', fontsize=12, fontweight='bold')
    plt.axhline(y=4, color='red', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xscale('log')
    
    # 11. Model Performance Metrics
    plt.subplot(4, 3, 11)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    plt.bar(metrics.keys(), metrics.values(), color='steelblue', alpha=0.7, edgecolor='black')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics', fontsize=12, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    for i, (k, v) in enumerate(metrics.items()):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 12. Feature Importance (Top 5)
    plt.subplot(4, 3, 12)
    top5 = feature_importance.head(5)
    plt.pie(top5['importance'], labels=top5['feature'], autopct='%1.1f%%', startangle=90)
    plt.title('Top 5 Features (Pie Chart)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "rating_prediction_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualizations saved to {output_path}")
    plt.close()


def analyze_incorrect_predictions(df_features, X_test, y_test, y_pred, feature_cols):
    """Analyze samples that were incorrectly predicted."""
    print(f"\n{'='*80}")
    print("STEP 6: Analyzing Incorrect Predictions")
    print(f"{'='*80}")
    
    # Get test indices
    test_indices = X_test.index
    
    # Create results dataframe
    results_df = df_features.loc[test_indices].copy()
    results_df['predicted_high_rating'] = y_pred
    results_df['actual_high_rating'] = y_test.values
    results_df['correct'] = (results_df['predicted_high_rating'] == results_df['actual_high_rating'])
    
    # Find incorrect predictions
    incorrect = results_df[~results_df['correct']].copy()
    
    print(f"\nIncorrect Predictions: {len(incorrect)} / {len(results_df)} ({len(incorrect)/len(results_df)*100:.1f}%)")
    
    # Analyze patterns
    print(f"\nFalse Positives (Predicted High, Actually Not High):")
    false_positives = incorrect[(incorrect['predicted_high_rating'] == 1) & (incorrect['actual_high_rating'] == 0)]
    print(f"  Count: {len(false_positives)}")
    if len(false_positives) > 0:
        print(f"  Avg Stars: {false_positives['stars'].mean():.2f}")
        print(f"  Avg Review Count: {false_positives['review_count'].mean():.1f}")
    
    print(f"\nFalse Negatives (Predicted Not High, Actually High):")
    false_negatives = incorrect[(incorrect['predicted_high_rating'] == 0) & (incorrect['actual_high_rating'] == 1)]
    print(f"  Count: {len(false_negatives)}")
    if len(false_negatives) > 0:
        print(f"  Avg Stars: {false_negatives['stars'].mean():.2f}")
        print(f"  Avg Review Count: {false_negatives['review_count'].mean():.1f}")
    
    # Save incorrect predictions
    if len(incorrect) > 0:
        incorrect_output = incorrect[['name', 'city', 'state', 'stars', 'review_count', 
                                     'actual_high_rating', 'predicted_high_rating'] + feature_cols].head(10)
        incorrect_path = os.path.join(OUTPUT_DIR, "incorrect_predictions.csv")
        incorrect_output.to_csv(incorrect_path, index=False)
        print(f"\n✓ Sample incorrect predictions saved to {incorrect_path}")
        print(f"\nTop 5 Incorrect Predictions:")
        print(incorrect_output[['name', 'city', 'stars', 'review_count', 
                               'actual_high_rating', 'predicted_high_rating']].to_string(index=False))
    
    return incorrect


def save_results(df_features, feature_importance, model, X_test, y_test, y_pred, incorrect):
    """Save all results to files."""
    print(f"\n{'='*80}")
    print("STEP 7: Saving Results")
    print(f"{'='*80}")
    
    # Save full dataset
    output_path = os.path.join(OUTPUT_DIR, "rating_prediction_data.csv")
    df_features.to_csv(output_path, index=False)
    print(f"✓ Full dataset saved to {output_path}")
    
    # Save feature importance
    importance_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved to {importance_path}")
    
    # Save summary statistics
    summary_stats = pd.DataFrame({
        'Metric': [
            'Total Restaurants',
            'High Rating (≥4 stars)',
            'Not High Rating (<4 stars)',
            'Test Accuracy',
            'Test Precision',
            'Test Recall',
            'Test F1 Score'
        ],
        'Value': [
            len(df_features),
            df_features['high_rating'].sum(),
            (~df_features['high_rating'].astype(bool)).sum(),
            f"{accuracy_score(y_test, y_pred):.4f}",
            f"{precision_score(y_test, y_pred):.4f}",
            f"{recall_score(y_test, y_pred):.4f}",
            f"{f1_score(y_test, y_pred):.4f}"
        ]
    })
    summary_path = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
    summary_stats.to_csv(summary_path, index=False)
    print(f"✓ Summary statistics saved to {summary_path}")
    
    # Save predictions
    test_indices = X_test.index
    predictions_df = df_features.loc[test_indices].copy()
    predictions_df['predicted_high_rating'] = y_pred
    predictions_df['actual_high_rating'] = y_test.values
    predictions_df['correct'] = (predictions_df['predicted_high_rating'] == predictions_df['actual_high_rating'])
    
    predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predictions saved to {predictions_path}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("HIGH RATING PREDICTION BASED ON BUSINESS ATTRIBUTES")
    print("Binary Classification: High Rating (≥4 stars) vs Not High Rating (<4 stars)")
    print("="*80)
    
    # Load business data
    businesses = load_business_data(TAR_FILE)
    if not businesses:
        return
    
    # Extract features
    df_features = extract_features(businesses)
    if df_features is None or len(df_features) == 0:
        return
    
    # Prepare features
    X, y, df_features, feature_cols, label_encoders = prepare_features(df_features)
    if X is None or len(X) == 0:
        return
    
    # Train model
    model, X_test, y_test, y_pred, feature_importance, X_train, y_train = train_classification_model(
        X, y, feature_cols
    )
    
    # Create visualizations
    create_visualizations(df_features, feature_importance, model, X_test, y_test, y_pred)
    
    # Analyze incorrect predictions
    incorrect = analyze_incorrect_predictions(df_features, X_test, y_test, y_pred, feature_cols)
    
    # Save results
    save_results(df_features, feature_importance, model, X_test, y_test, y_pred, incorrect)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print(f"  - rating_prediction_data.csv (full dataset)")
    print(f"  - rating_prediction_analysis.png (visualizations)")
    print(f"  - feature_importance.csv (feature rankings)")
    print(f"  - summary_statistics.csv (model metrics)")
    print(f"  - predictions.csv (test set predictions)")
    print(f"  - incorrect_predictions.csv (misclassified samples)")
    print()


if __name__ == "__main__":
    main()

