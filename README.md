# Mental Health @ Work — Dual-Dataset EDA

This project explores the relationship between workplace factors, employee stress, and attrition using two Kaggle datasets. The analysis focuses on **resignation behavior** and **access to mental health resources**, producing both statistical results and visualizations.

---

## 📂 Datasets Used
1. **Dataset A** – *Employees Attrition & Leadership Impact HR Data*  
   - File: `hr analytics data - employees attrition and leadership impact.csv`  
   - Used to analyze stress vs. job satisfaction among employees who resigned.  

2. **Dataset B** – *Impact of Remote Work on Mental Health*  
   - File: `Impact_of_Remote_Work_on_Mental_Health.csv`  
   - Used to test whether access to mental health resources is linked to lower stress levels.  

---

## ⚙️ Data Cleaning
The script performs several preprocessing steps:
- Converts messy text values to numeric (e.g., stress/satisfaction scores).
- Clamps stress and satisfaction scores to the valid **0–10 range**.
- Drops rows missing key values (stress or satisfaction).
- Standardizes categorical responses:
  - Stress: `"Low" → 1`, `"Medium" → 2`, `"High" → 3`
  - Access to resources: `"yes/true/available" → 1`, `"no/false/none" → 0`
- Creates derived features:
  - **Quadrant categories** (low/high stress × low/high satisfaction).  
  - **LowStressFlag** (1 if stress is low, 0 otherwise).  

---

## 🔎 Analysis

### Part 1: Dataset A (Attrition – Resignations Only)
- Focus: Relationship between **stress** and **job satisfaction** for employees who resigned.  
- Groups employees into **four quadrants** (low/high stress × low/high satisfaction).  
- Outputs:
  - **Scatter plot** with quadrant boundaries and trend line.  
  - **Summary table** with counts and percentages by quadrant.  

### Part 2: Dataset B (Mental Health & Remote Work)
- Focus: Whether **access to mental health resources** is associated with **low stress**.  
- Builds a **2×2 contingency table**: Access (Yes/No) × Stress (Low/Not Low).  
- Statistical tests:
  - **Chi-square test of independence**.  
  - **Cramér’s V** to measure strength of association.  
- Outputs:
  - **Observed and expected tables**, residuals, contributions.  
  - Visualizations:
    - Bar chart of low-stress counts.  
    - Percent low-stress with **95% Wilson CIs**.  
    - 100% stacked bar (low vs not low).  

---

## 📊 Outputs
All results are saved in the `output_new/` directory:
- **Tables** (`.csv`): summaries, quadrant counts, chi-square components.  
- **Figures** (`.png`): scatter plots, bar charts, stacked compositions.  

---

## ▶️ How to Run
1. Place the two CSV datasets in the same folder as `analysis_new.py`.  
2. Run the script:
   ```bash
   python analysis_new.py