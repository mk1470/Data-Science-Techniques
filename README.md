# Module 3: Career Similarity Analysis - Streamlined Version

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python career_similarity.py
```

## What This Does

- Loads O*NET 30.0 data (894 occupations, 35 skills)
- Creates composite roles for SAP Consultant and Supply Chain Analyst
- Finds top 10 similar roles for each query
- Generates CSV files and bar charts
- Saves results to `output/` directory

## Output Files

- `output/data_analyst_similar.csv` - Data Analyst results
- `output/sap_consultant_similar.csv` - SAP Consultant results  
- `output/supply_chain_analyst_similar.csv` - Supply Chain Analyst results
- `output/graphs/` - Individual bar charts for each role

## Requirements

- Python 3.8+
- O*NET 30.0 database in `db_30_0_text/` folder
