"""
Module 3: Career Similarity Analysis - Streamlined Version
Uses full O*NET 30.0 database to find similar roles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load O*NET 30.0 data from processed files"""
    print("Loading O*NET 30.0 data...")
    
    # Load from the processed data files
    occupations = pd.read_csv("../data/Occupation_Data.txt", sep="\t")
    print(f"✓ Loaded {len(occupations)} occupations")
    
    skills = pd.read_csv("../data/Skills.txt", sep="\t")
    print(f"✓ Loaded {len(skills)} skill ratings")
    
    return occupations, skills

def create_skill_matrix(skills_data):
    """Create occupation x skill matrix"""
    print("\nCreating skill matrix...")
    
    # Pivot to create matrix
    skill_matrix = skills_data.pivot_table(
        index='O*NET-SOC Code',
        columns='Element Name',
        values='Data Value',
        aggfunc='mean'
    ).fillna(0)
    
    print(f"✓ Matrix shape: {skill_matrix.shape}")
    
    # Normalize using Min-Max scaling
    scaler = MinMaxScaler()
    skill_matrix_normalized = pd.DataFrame(
        scaler.fit_transform(skill_matrix),
        index=skill_matrix.index,
        columns=skill_matrix.columns
    )
    
    return skill_matrix, skill_matrix_normalized

def compute_similarity(skill_matrix_normalized):
    """Compute cosine similarity between all occupations"""
    print("\nComputing cosine similarity...")
    
    sim = cosine_similarity(skill_matrix_normalized)
    similarity_matrix = pd.DataFrame(
        sim,
        index=skill_matrix_normalized.index,
        columns=skill_matrix_normalized.index
    )
    
    print(f"✓ Computed similarity for {len(similarity_matrix)} occupations")
    return similarity_matrix

def create_composite_role(skill_matrix_normalized, occupations, soc_codes, role_name):
    """Create a composite role by averaging multiple occupations"""
    print(f"\nCreating composite role: {role_name}")
    
    vectors = []
    for soc in soc_codes:
        if soc in skill_matrix_normalized.index:
            vectors.append(skill_matrix_normalized.loc[soc].values)
            occ_title = occupations[occupations['O*NET-SOC Code'] == soc]['Title'].values[0]
            print(f"  + {occ_title}")
    
    if not vectors:
        raise ValueError(f"None of the SOC codes found: {soc_codes}")
    
    # Average the vectors
    composite_vector = np.mean(vectors, axis=0)
    skill_matrix_normalized.loc[role_name] = composite_vector
    
    return skill_matrix_normalized

def find_similar(similarity_matrix, occupations, query_soc, top_n=10):
    """Find the most similar occupations to a query occupation"""
    similarities = similarity_matrix.loc[query_soc].sort_values(ascending=False)
    similarities = similarities[similarities.index != query_soc]  # Exclude self
    top_similar = similarities.head(top_n)
    
    results = []
    for soc_code, similarity in top_similar.items():
        if soc_code in occupations['O*NET-SOC Code'].values:
            title = occupations[occupations['O*NET-SOC Code'] == soc_code]['Title'].values[0]
        else:
            title = soc_code  # Composite role
        
        results.append({
            'Rank': len(results) + 1,
            'O*NET-SOC Code': soc_code,
            'Title': title,
            'Similarity': similarity
        })
    
    return pd.DataFrame(results)

def create_bar_chart(df, title, output_file):
    """Create a bar chart for similarity results"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(df)), df['Similarity'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(df))))
    
    # Customize the plot
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{row['Title'][:45]}{'...' if len(row['Title']) > 45 else ''}" 
                       for _, row in df.iterrows()], fontsize=11)
    ax.set_xlabel('Cosine Similarity Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Top 10 Similar Roles to {title}', fontsize=16, fontweight='bold', pad=20)
    
    # Add similarity scores on the bars
    for i, (bar, similarity) in enumerate(zip(bars, df['Similarity'])):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{similarity:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # Set x-axis limits and formatting
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Saved {output_file}")
    plt.close()

def main():
    """Main analysis workflow"""
    print("=" * 70)
    print("Module 3: Career Similarity Analysis - Streamlined")
    print("=" * 70)
    
    # Load data
    occupations, skills = load_data()
    
    # Create skill matrix
    skill_matrix, skill_matrix_normalized = create_skill_matrix(skills)
    
    # Compute similarity
    similarity_matrix = compute_similarity(skill_matrix_normalized)
    
    # Create composite roles
    print("\n" + "=" * 70)
    print("DEFINING QUERY ROLES")
    print("=" * 70)
    
    # SAP Consultant (composite)
    skill_matrix_normalized = create_composite_role(
        skill_matrix_normalized, occupations,
        ["15-1211.00", "13-1111.00", "15-1241.00"],  # Computer Systems Analysts, Management Analysts, Computer Network Architects
        "SAP Consultant"
    )
    
    # Supply Chain Analyst (composite)
    skill_matrix_normalized = create_composite_role(
        skill_matrix_normalized, occupations,
        ["13-1081.02", "15-2031.00", "17-2112.00"],  # Logistics Analysts, Operations Research Analysts, Industrial Engineers
        "Supply Chain Analyst"
    )
    
    # Recompute similarity with composite roles
    similarity_matrix = compute_similarity(skill_matrix_normalized)
    
    # Define queries
    queries = [
        ("15-2051.01", "Data Analyst"),  # Business Intelligence Analysts
        ("SAP Consultant", "SAP Consultant"),
        ("Supply Chain Analyst", "Supply Chain Analyst")
    ]
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    # Find similar occupations and create charts
    print("\n" + "=" * 70)
    print("FINDING SIMILAR OCCUPATIONS")
    print("=" * 70)
    
    for query_code, query_name in queries:
        print(f"\nQuery: {query_name}")
        print("-" * 50)
        
        # Find similar roles
        similar = find_similar(similarity_matrix, occupations, query_code, top_n=10)
        
        # Save CSV
        csv_file = output_dir / f"{query_name.replace(' ', '_').lower()}_similar.csv"
        similar.to_csv(csv_file, index=False)
        print(f"✓ Saved {csv_file}")
        
        # Create bar chart
        chart_file = graphs_dir / f"{query_name.replace(' ', '_').lower()}_chart.png"
        create_bar_chart(similar, query_name, chart_file)
        
        # Show results
        print(similar.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to {output_dir}/")
    print(f"Charts saved to {graphs_dir}/")

if __name__ == "__main__":
    main()