# Empathy & AI Emotional Interaction Analysis

Research application and empirical analysis evaluating user perceptions, personality traits, empathy ratings, and trust dynamics in AI vs. human emotional interactions.

## Research Objectives & Analytical Scope

1. **Non-User Perceptions**: Analyzing reluctance factors and barriers to adoption among non-users.
2. **Personality & Usage Patterns**: K-Means clustering mapping Big Five personality traits to chat interaction frequency.
3. **Motivation Factor Analysis**: Exploratory Factor Analysis (EFA) and regression modeling identifying drivers of AI interaction.
4. **Empathy & Authenticity Evaluations**: ANOVA and interaction plots measuring perceived empathy ratings.
5. **Trust Dynamics (AI vs Human)**: Mediation analysis evaluating trust scores across scenario contexts.

## Analytical Module Matrix

| Script Location | Objective Scope | Methodology |
|---|---|---|
| `analysis/Objective_1_NonUser_Perceptions.py` | Non-user reluctance | Chi-square tests & frequency analysis |
| `analysis/Objective_2_Usage_Patterns_by_Personality.py` | Personality clustering | K-Means clustering & PCA visualization |
| `analysis/Objective_3_Motivation_Factors.py` | Interaction drivers | Exploratory Factor Analysis & Multiple Regression |
| `analysis/Objective_4_Personality_Frequency_Empathy_Authenticity.py` | Empathy ratings | Two-Way ANOVA & interaction plots |
| `analysis/Objective_5_Trust_AI_vs_Human.py` | Trust evaluation | Paired t-tests & mediation modeling |

## Project Structure

```
new_emotional_interaction_chatbot/
├── analysis/
│   ├── Objective_1_NonUser_Perceptions.py
│   ├── Objective_2_Usage_Patterns_by_Personality.py
│   ├── Objective_3_Motivation_Factors.py
│   ├── Objective_4_Personality_Frequency_Empathy_Authenticity.py
│   └── Objective_5_Trust_AI_vs_Human.py
├── data/
├── questions.py
├── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch Streamlit Application
```bash
streamlit run app.py
```

### Run Objective Analysis Script (Example: Objective 1)
```bash
python analysis/Objective_1_NonUser_Perceptions.py
```

## Author

- Sanman Kadam (MSc Statistics | Data Analyst)
- Rutuja Shinde (MSc Statistics | Data Analyst)
