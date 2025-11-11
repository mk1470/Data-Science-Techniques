"""
k-Means Clustering Analysis for Sci-Fi Books - Version 2
Using features: categories, publishedDate (year), authors, 
pageCount, publisher, language, and ratings
All features weighted equally (1.0x)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

# Load the data
print("=" * 60)
print("Loading Sci-Fi Books Data - Version 2 (Equal Weights)")
print("=" * 60)

df = pd.read_csv("data/scifi_books.csv")
print(f"Loaded {len(df)} books")

# Prepare data
print("\nPreparing feature matrix...")

# 1. DESCRIPTION - Not used (using Euclidean distance on structured features only)
print("\n1. Skipping description features (using Euclidean distance on structured features only)...")

# 2. CATEGORIES + MAINCATEGORY → genre/subgenre anchors
print("\n2. Processing categories and mainCategory...")
all_categories = set()
for cats in df['categories'].dropna():
    if cats:
        all_categories.update([c.strip() for c in str(cats).split(',')])

# Add mainCategory if available
for main_cat in df['mainCategory'].dropna():
    if main_cat:
        all_categories.add(str(main_cat).strip())

all_categories = sorted(list(all_categories))
print(f"Found {len(all_categories)} unique categories (including mainCategory)")

# Create binary category matrix
category_vectors = []
for idx, row in df.iterrows():
    cats = str(row['categories']) if pd.notna(row['categories']) else ''
    cat_list = [c.strip() for c in cats.split(',')] if cats else []
    # Add mainCategory if available
    if pd.notna(row.get('mainCategory', None)) and row['mainCategory']:
        cat_list.append(str(row['mainCategory']).strip())
    vector = {cat: 1 if cat in cat_list else 0 for cat in all_categories}
    category_vectors.append(vector)

df_categories = pd.DataFrame(category_vectors, index=df.index)
print(f"Category matrix shape: {df_categories.shape}")

# 3. PUBLISHEDDATE (year extracted) → classic vs. contemporary eras
print("\n3. Extracting year from publishedDate...")
def extract_year(date_str):
    """Extract year from publishedDate string"""
    if pd.isna(date_str) or not date_str:
        return None
    # Try to extract 4-digit year
    match = re.search(r'\d{4}', str(date_str))
    if match:
        return int(match.group())
    return None

df['year'] = df['publishedDate'].apply(extract_year)
# Fill missing years with median
median_year = df['year'].median()
df['year'] = df['year'].fillna(median_year)
# Normalize year (0-1 scale, where 0 is oldest, 1 is newest)
df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1)
print(f"Year range: {int(df['year'].min())} - {int(df['year'].max())}")
print(f"Year normalized (range: {df['year_norm'].min():.3f} - {df['year_norm'].max():.3f})")

# 4. AUTHORS[] → author communities/series universes
print("\n4. Processing authors...")
all_authors = set()
for authors in df['authors'].dropna():
    if authors:
        all_authors.update([a.strip() for a in str(authors).split(',')])

all_authors = sorted(list(all_authors))
print(f"Found {len(all_authors)} unique authors")

# Create binary author matrix
author_vectors = []
for idx, row in df.iterrows():
    authors = str(row['authors']) if pd.notna(row['authors']) else ''
    author_list = [a.strip() for a in authors.split(',')] if authors else []
    vector = {author: 1 if author in author_list else 0 for author in all_authors}
    author_vectors.append(vector)

df_authors = pd.DataFrame(author_vectors, index=df.index)
print(f"Author matrix shape: {df_authors.shape}")

# 5. PAGECOUNT → epics vs. short works
print("\n5. Processing pageCount...")
df['pageCount_norm'] = df['pageCount'].fillna(df['pageCount'].median())
df['pageCount_norm'] = (df['pageCount_norm'] - df['pageCount_norm'].min()) / (df['pageCount_norm'].max() - df['pageCount_norm'].min() + 1)
print(f"PageCount normalized (range: {df['pageCount_norm'].min():.3f} - {df['pageCount_norm'].max():.3f})")

# 6. PUBLISHER → house style (Tor/Del Rey etc.)
print("\n6. Processing publisher...")
all_publishers = set()
for pub in df['publisher'].dropna():
    if pub:
        all_publishers.add(str(pub).strip())

# Handle missing values
all_publishers.add('Unknown')  # Default
all_publishers = sorted(list(all_publishers))
print(f"Found {len(all_publishers)} unique publishers")

# Create binary publisher matrix
publisher_vectors = []
for idx, row in df.iterrows():
    pub = str(row['publisher']) if pd.notna(row['publisher']) and row['publisher'] else 'Unknown'
    vector = {p: 1 if p == pub else 0 for p in all_publishers}
    publisher_vectors.append(vector)

df_publisher = pd.DataFrame(publisher_vectors, index=df.index)
print(f"Publisher matrix shape: {df_publisher.shape}")

# 7. LANGUAGE → original vs. translated SF
print("\n7. Processing language...")
all_languages = set()
for lang in df['language'].dropna():
    if lang:
        all_languages.add(str(lang).strip())

# Handle missing values
all_languages.add('en')  # Default to English
all_languages = sorted(list(all_languages))
print(f"Found {len(all_languages)} unique languages: {all_languages}")

# Create binary language matrix
language_vectors = []
for idx, row in df.iterrows():
    lang = str(row['language']) if pd.notna(row['language']) else 'en'
    vector = {l: 1 if l == lang else 0 for l in all_languages}
    language_vectors.append(vector)

df_language = pd.DataFrame(language_vectors, index=df.index)
print(f"Language matrix shape: {df_language.shape}")

# 8. RATINGS → averageRating, ratingsCount
print("\n8. Processing ratings...")
df['ratingsCount_norm'] = (df['ratingsCount'] - df['ratingsCount'].min()) / (df['ratingsCount'].max() - df['ratingsCount'].min() + 1)
df['averageRating_norm'] = df['averageRating'].fillna(0) / 5.0
print(f"Ratings normalized")

# Combine all features
print("\n" + "=" * 60)
print("Combining all features...")
print("=" * 60)

# Apply weights: Everything gets 1.0x weight (equal weights)
# Note: k-means uses Euclidean distance by default for all features
print("\nWeight distribution:")
print("  - Categories + mainCategory: 1.0x")
print("  - PublishedDate (year): 1.0x")
print("  - Authors: 1.0x")
print("  - PageCount: 1.0x")
print("  - Publisher: 1.0x")
print("  - Language: 1.0x")
print("  - Ratings (ratingsCount + averageRating): 1.0x")
print("\nNote: Using Euclidean distance for k-means clustering with equal feature weights")

# Set weights - all equal
category_weight = 1.0
year_weight = 1.0
author_weight = 1.0
pagecount_weight = 1.0
publisher_weight = 1.0
language_weight = 1.0
rating_weight = 1.0

# Weight the feature matrices
df_categories_weighted = df_categories * category_weight
df_year_weighted = df[['year_norm']] * year_weight
df_authors_weighted = df_authors * author_weight
df_pagecount_weighted = df[['pageCount_norm']] * pagecount_weight
df_publisher_weighted = df_publisher * publisher_weight
df_language_weighted = df_language * language_weight
df_ratings_weighted = df[['ratingsCount_norm', 'averageRating_norm']] * rating_weight

# Combine all features with weights (no description/TF-IDF)
feature_matrix = pd.concat([
    df_categories_weighted,
    df_year_weighted,
    df_authors_weighted,
    df_pagecount_weighted,
    df_publisher_weighted,
    df_language_weighted,
    df_ratings_weighted
], axis=1)

print(f"\nCombined feature matrix shape: {feature_matrix.shape}")
print(f"  - Categories: {df_categories.shape[1]} features (weight: {category_weight}x)")
print(f"  - Year: 1 feature (weight: {year_weight}x)")
print(f"  - Authors: {df_authors.shape[1]} features (weight: {author_weight}x)")
print(f"  - PageCount: 1 feature (weight: {pagecount_weight}x)")
print(f"  - Publisher: {df_publisher.shape[1]} features (weight: {publisher_weight}x)")
print(f"  - Language: {df_language.shape[1]} features (weight: {language_weight}x)")
print(f"  - Ratings: 2 features (weight: {rating_weight}x)")

# Apply dimensionality reduction using SVD
print("\n" + "=" * 60)
print("Applying SVD for dimensionality reduction...")
print("=" * 60)

matrix_dense = feature_matrix.to_numpy()
svd = TruncatedSVD(n_components=10, random_state=42)
matrix_reduced = svd.fit_transform(matrix_dense)

print(f"Reduced matrix shape: {matrix_reduced.shape}")
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

# Visualize in 2D for inspection
svd_2d = TruncatedSVD(n_components=2, random_state=42)
matrix_2d = svd_2d.fit_transform(matrix_dense)

plt.figure(figsize=(10, 8))
plt.scatter(matrix_2d[:, 0], matrix_2d[:, 1], alpha=0.6)
plt.title("Sci-Fi Books in 2D SVD Space (Equal Weights)")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.savefig("results/books_2d_scatter.png", dpi=150, bbox_inches='tight')
print("Saved 2D visualization to results/books_2d_scatter.png")

# Find optimal k using elbow method and silhouette score
print("\n" + "=" * 60)
print("Finding optimal k using elbow method and silhouette score...")
print("=" * 60)

inertia_scores = []
silhouette_scores = []

k_range = range(2, 11)
for k in k_range:
    print(f"Testing k={k}...")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(matrix_reduced)
    
    inertia_scores.append((k, model.inertia_))
    sil_score = silhouette_score(matrix_reduced, labels)
    silhouette_scores.append((k, sil_score))
    print(f"  k={k}: Inertia={model.inertia_:.2f}, Silhouette={sil_score:.3f}")

inertia_df = pd.DataFrame(inertia_scores, columns=['k', 'inertia'])
silhouette_df = pd.DataFrame(silhouette_scores, columns=['k', 'silhouette'])

# Plot elbow method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(inertia_df['k'], inertia_df['inertia'], marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(silhouette_df['k'], silhouette_df['silhouette'], marker='o', color='green')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/k_selection.png", dpi=150, bbox_inches='tight')
print("Saved k selection plots to results/k_selection.png")

# Create standalone elbow method plot
plt.figure(figsize=(10, 6))
plt.plot(inertia_df['k'], inertia_df['inertia'], marker='o', linewidth=2, markersize=8, color='#2E86AB')
plt.xlabel('Number of clusters (k)', fontsize=12, fontweight='bold')
plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12, fontweight='bold')
plt.title('Elbow Method for Optimal k\n(Sci-Fi Books Clustering - Equal Weights)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(inertia_df['k'])

# Add value labels on points
for _, row in inertia_df.iterrows():
    plt.text(row['k'], row['inertia'], f'{row["inertia"]:.0f}', 
             ha='center', va='bottom', fontsize=9)

# Highlight k=5
k5_inertia = inertia_df[inertia_df['k'] == 5]['inertia'].values[0]
plt.scatter([5], [k5_inertia], color='red', s=150, zorder=5, marker='*', 
           label='Selected k=5')
plt.legend()
plt.tight_layout()
plt.savefig("results/elbow_method.png", dpi=150, bbox_inches='tight')
print("Saved elbow method plot to results/elbow_method.png")

# Select optimal k (highest silhouette score)
optimal_k = silhouette_df.loc[silhouette_df['silhouette'].idxmax(), 'k']
print(f"\nOptimal k based on silhouette score: {optimal_k}")

# Use k=5 as specified
k = 5
print(f"\nUsing k={k} for clustering")

# Apply k-means
print("\n" + "=" * 60)
print(f"Applying k-Means clustering with k={k}...")
print("=" * 60)

kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans_model.fit_predict(matrix_reduced)

df['cluster'] = cluster_labels

# Visualize clusters in 2D
plt.figure(figsize=(12, 8))
scatter = plt.scatter(matrix_2d[:, 0], matrix_2d[:, 1], c=cluster_labels, 
                      cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='Cluster')
plt.title(f'Sci-Fi Books Clusters (k={k}) - Equal Weights')
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")

# Add cluster centers in 2D space
centers_original_approx = svd.inverse_transform(kmeans_model.cluster_centers_)
centers_2d = svd_2d.transform(centers_original_approx)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', 
           s=200, linewidths=3, label='Cluster Centers', zorder=10)
plt.legend()
plt.savefig("results/clusters_2d.png", dpi=150, bbox_inches='tight')
print("Saved cluster visualization to results/clusters_2d.png")

# Analyze clusters
print("\n" + "=" * 60)
print("CLUSTER ANALYSIS RESULTS")
print("=" * 60)

cluster_results = []

for cluster_id in sorted(df['cluster'].unique()):
    cluster_books = df[df['cluster'] == cluster_id]
    cluster_size = len(cluster_books)
    
    print(f"\n{'='*60}")
    print(f"Cluster {cluster_id} - Size: {cluster_size} books")
    print(f"{'='*60}")
    
    # Count categories in this cluster
    category_counter = Counter()
    for cats in cluster_books['categories'].dropna():
        if cats:
            category_counter.update([c.strip() for c in str(cats).split(',')])
    
    print("\nTop Categories:")
    for cat, count in category_counter.most_common(10):
        pct = (count / cluster_size) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    # Count authors in this cluster
    author_counter = Counter()
    for authors in cluster_books['authors'].dropna():
        if authors:
            author_counter.update([a.strip() for a in str(authors).split(',')])
    
    print("\nTop Authors:")
    for author, count in author_counter.most_common(10):
        pct = (count / cluster_size) * 100
        print(f"  {author}: {count} ({pct:.1f}%)")
    
    # Publisher distribution
    publisher_counter = Counter()
    for pub in cluster_books['publisher'].dropna():
        if pub:
            publisher_counter[str(pub).strip()] += 1
    
    if publisher_counter:
        print("\nTop Publishers:")
        for pub, count in publisher_counter.most_common(5):
            pct = (count / cluster_size) * 100
            print(f"  {pub}: {count} ({pct:.1f}%)")
    
    # Year statistics
    avg_year = cluster_books['year'].mean()
    print(f"\nAverage Publication Year: {int(avg_year)}")
    
    # Rating statistics
    avg_rating_count = cluster_books['ratingsCount'].mean()
    avg_rating = cluster_books['averageRating'].mean()
    books_with_ratings = cluster_books['ratingsCount'].gt(0).sum()
    
    print(f"\nRating Statistics:")
    print(f"  Average ratingsCount: {avg_rating_count:.1f}")
    if pd.notna(avg_rating):
        print(f"  Average rating: {avg_rating:.2f}")
    else:
        print(f"  Average rating: N/A")
    print(f"  Books with ratings: {books_with_ratings}/{cluster_size}")
    
    # Sample books
    print(f"\nSample Books (showing up to 10):")
    sample_size = min(10, cluster_size)
    for idx, row in cluster_books.head(sample_size).iterrows():
        title = row['title'][:50]  # Truncate long titles
        authors = str(row['authors'])[:40] if pd.notna(row['authors']) else "Unknown"
        year = int(row['year']) if pd.notna(row['year']) else "N/A"
        rating_count = row['ratingsCount']
        avg_rating = row['averageRating'] if pd.notna(row['averageRating']) else "N/A"
        print(f"  - {title} by {authors} ({year})")
        print(f"    (ratingsCount: {rating_count}, avgRating: {avg_rating})")
    
    # Store results
    avg_rating_value = None
    if pd.notna(avg_rating):
        try:
            avg_rating_value = float(avg_rating)
        except (ValueError, TypeError):
            avg_rating_value = None
    
    cluster_results.append({
        'cluster': int(cluster_id),
        'size': int(cluster_size),
        'top_categories': dict(category_counter.most_common(5)),
        'top_authors': dict(author_counter.most_common(5)),
        'top_publishers': dict(publisher_counter.most_common(3)),
        'avg_year': float(avg_year),
        'avg_ratingsCount': float(avg_rating_count),
        'avg_rating': avg_rating_value,
        'books_with_ratings': int(books_with_ratings)
    })

# Save results
print("\n" + "=" * 60)
print("Saving results...")
print("=" * 60)

# Save cluster assignments
df_output = df[['title', 'authors', 'categories', 'publishedDate', 'publisher', 
                'averageRating', 'ratingsCount', 'cluster']].copy()
df_output.to_csv('results/books_with_clusters.csv', index=False)
print("Saved cluster assignments to results/books_with_clusters.csv")

# Save cluster summary
with open('results/cluster_summary.json', 'w') as f:
    json.dump(cluster_results, f, indent=2)
print("Saved cluster summary to results/cluster_summary.json")

# Create summary table
summary_data = []
for result in cluster_results:
    summary_data.append({
        'Cluster': result['cluster'],
        'Size': result['size'],
        'Top Category': list(result['top_categories'].keys())[0] if result['top_categories'] else 'N/A',
        'Top Author': list(result['top_authors'].keys())[0] if result['top_authors'] else 'N/A',
        'Avg Year': int(result['avg_year']),
        'Avg RatingsCount': f"{result['avg_ratingsCount']:.1f}",
        'Avg Rating': f"{result['avg_rating']:.2f}" if result['avg_rating'] else 'N/A',
        'Books with Ratings': result['books_with_ratings']
    })

summary_df = pd.DataFrame(summary_data)
print("\nCluster Summary Table:")
print(summary_df.to_string(index=False))
summary_df.to_csv('results/cluster_summary_table.csv', index=False)
print("\nSaved cluster summary table to results/cluster_summary_table.csv")

print("\n" + "=" * 60)
print("CLUSTERING ANALYSIS COMPLETE!")
print("=" * 60)

