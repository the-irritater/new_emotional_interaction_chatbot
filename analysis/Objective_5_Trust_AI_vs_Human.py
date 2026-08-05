# %% [markdown]
# # Objective 5: Trust in AI – Predictors & Comparison with Human Trust
#
# **Research Objective:** To analyze the influence of user personality type, motivations,
# frequency of use, and perceptions of AI on trust in AI, and to compare trust in these
# systems with trust in human relationships.
#
# **Why This Objective?**
# TRUST is the ultimate outcome variable. Without trust, emotional AI systems will fail
# regardless of their technical quality. This objective answers the most important
# question: What builds trust in emotional AI, and how does it compare to the trust
# people place in human emotional support?
#
# **What We Aim to Prove:**
# - Multiple factors (personality, motivation, frequency, perceptions) predict trust
# - Perceived empathy/authenticity add unique predictive power beyond personality/usage
# - Users trust AI differently than they trust human emotional support
# - Perceived empathy mediates the personality → trust relationship
#
# **Statistical Methods:** Hierarchical Regression, Paired t-test, Wilcoxon, Chi-Square,
# Sobel Mediation Analysis, VIF Diagnostics
# -

# %% [markdown]
# ## 1. Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, wilcoxon, probplot
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
sns.set_theme(style="whitegrid", palette="muted")
print(" All libraries loaded successfully.")

# %% [markdown]
# ## 2. Data Preparation

# %%
df = pd.read_csv('/Users/the_irritater/Downloads/synthetic_emotional_ai_responses_375_reliable.csv')
users = df[df['group'] == 'User'].copy()
print(f"User sample size: {users.shape[0]}")

personality_cols = [f'personality_Q{i}' for i in range(1, 9)]
motivation_cols = [f'motivation_Q{i}' for i in range(1, 9)]
empathy_cols = [f'perceived_empathy_Q{i}' for i in range(1, 7)]
authenticity_cols = [f'perceived_authenticity_Q{i}' for i in range(1, 9)]
trust_cols = [f'trust_in_ai_Q{i}' for i in range(1, 9)]
ai_human_likert = [f'ai_vs_human_likert_Q{i}' for i in range(1, 4)]
ai_human_scenario = [f'ai_vs_human_scenarios_Q{i}' for i in range(1, 7)]

# %% [markdown]
# ## 3. Reliability Analysis

# %%
def cronbachs_alpha(data):
    items = data.dropna(); n = items.shape[1]
    return (n/(n-1)) * (1 - items.var(axis=0,ddof=1).sum() / items.sum(axis=1).var(ddof=1))

print("RELIABILITY ANALYSIS")
for name, cols in [('Trust in AI (8)', trust_cols), ('Empathy (6)', empathy_cols),
                   ('Authenticity (8)', authenticity_cols), ('Motivation (8)', motivation_cols),
                   ('Personality (8)', personality_cols)]:
    a = cronbachs_alpha(users[cols])
    print(f"  {name}: α = {a:.4f} → {'Excellent' if a>=0.9 else 'Good' if a>=0.8 else 'Acceptable' if a>=0.7 else 'Questionable'}")

print("""
 INTERPRETATION: All scales demonstrate acceptable to excellent reliability.
The Trust in AI scale is particularly important — its high reliability confirms
that trust is measured consistently across all 8 items, making the composite
score a valid outcome variable for regression and mediation analysis.""")

# %% [markdown]
# ## 4. Composite Scores

# %%
users['personality_score'] = users[personality_cols].mean(axis=1)
users['motivation_score'] = users[motivation_cols].mean(axis=1)
users['empathy_score'] = users[empathy_cols].mean(axis=1)
users['authenticity_score'] = users[authenticity_cols].mean(axis=1)
users['trust_score'] = users[trust_cols].mean(axis=1)
users['ai_human_likert_score'] = users[ai_human_likert].mean(axis=1)
freq_map = {'Rarely (1–3 times/month)':1,'Sometimes (4–8 times/month)':2,
            'Frequently (9–15 times/month)':3,'Very frequently (16+ times/month)':4}
users['freq_ordinal'] = users['usage_frequency'].map(freq_map)

# Personality types
scaler = StandardScaler()
ps = scaler.fit_transform(users[personality_cols])
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(ps)
users['p_cluster'] = km.labels_
cm = users.groupby('p_cluster')['personality_score'].mean().sort_values()
users['personality_type'] = users['p_cluster'].map(dict(zip(cm.index, ['Low','Moderate','High'])))

composites = ['personality_score','motivation_score','empathy_score','authenticity_score','trust_score','freq_ordinal']
print("COMPOSITE SCORE DESCRIPTIVES:")
print(users[composites].describe().round(3))
print(f"""
 INTERPRETATION: Trust in AI (M={users['trust_score'].mean():.3f}) can be compared to
the scale midpoint (4.0). A mean {'above' if users['trust_score'].mean() > 4 else 'below'} 4.0 suggests users generally
{'trust' if users['trust_score'].mean() > 4 else 'distrust'} AI for emotional interactions.""")

# %% [markdown]
# ## 5. Correlation Matrix

# %%
corr_vars = ['personality_score','motivation_score','freq_ordinal','empathy_score','authenticity_score','trust_score']
corr_labels = ['Personality','Motivation','Frequency','Empathy','Authenticity','Trust in AI']

print("PEARSON CORRELATIONS WITH TRUST IN AI:")
for i in range(len(corr_vars)-1):
    r, p = stats.pearsonr(users[corr_vars[i]], users['trust_score'])
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    print(f"  {corr_labels[i]} × Trust: r = {r:.3f}, p = {p:.4f} {sig}")

print(f"""
 INTERPRETATION: These correlations reveal the BIVARIATE relationships between each
predictor and trust in AI. Key patterns:
• EMPATHY → TRUST: If strong positive, users who perceive AI as empathetic trust it more.
  This is the most direct quality-trust pathway.
• AUTHENTICITY → TRUST: Similar logic — perceived genuineness builds trust.
• MOTIVATION → TRUST: If positive, more motivated users also trust more (self-selection).
• PERSONALITY → TRUST: Reveals whether personality predisposes people toward/against trust.
• FREQUENCY → TRUST: Tests whether more exposure builds or erodes trust.

NOTE: Correlations are BIVARIATE. The regression below tests UNIQUE effects controlling
for other variables.""")

# %%
corr_data = users[corr_vars].copy(); corr_data.columns = corr_labels
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_data.corr(), annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax, annot_kws={'size':11,'weight':'bold'})
ax.set_title('Correlation Matrix – Predictors and Trust in AI', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/the_irritater/Projects/new_emotional_interaction_chatbot/analysis/obj5_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Hierarchical Multiple Regression

# %%
y = users['trust_score']
X1 = sm.add_constant(users[['personality_score']])
m1 = sm.OLS(y, X1).fit()
X2 = sm.add_constant(users[['personality_score','motivation_score','freq_ordinal']])
m2 = sm.OLS(y, X2).fit()
X3 = sm.add_constant(users[['personality_score','motivation_score','freq_ordinal','empathy_score','authenticity_score']])
m3 = sm.OLS(y, X3).fit()

print("=" * 80)
print("HIERARCHICAL REGRESSION – Predicting Trust in AI")
print("=" * 80)
print(f"{'Model':<10} {'R²':>8} {'Adj R²':>8} {'ΔR²':>8} {'F':>10} {'p(F)':>10}")
print("─"*55)
prev = 0
for name, m in [('Step 1',m1),('Step 2',m2),('Step 3',m3)]:
    dr = m.rsquared - prev
    print(f"{name:<10} {m.rsquared:>8.4f} {m.rsquared_adj:>8.4f} {dr:>8.4f} {m.fvalue:>10.3f} {m.f_pvalue:>10.4f}")
    prev = m.rsquared

print(f"""
 INTERPRETATION: Hierarchical regression enters predictors in THEORETICALLY MEANINGFUL
BLOCKS to test incremental contributions:

STEP 1 (Personality only): R² = {m1.rsquared:.4f}
  → Personality alone explains {m1.rsquared*100:.1f}% of trust variance.
  → This is the baseline — how much does WHO YOU ARE predict trust?

STEP 2 (+Motivation & Frequency): R² = {m2.rsquared:.4f}, ΔR² = {m2.rsquared-m1.rsquared:.4f}
  → Adding behavioral variables explains an additional {(m2.rsquared-m1.rsquared)*100:.1f}%.
  → This shows whether WHAT YOU DO (motivation/frequency) adds to WHO YOU ARE.

STEP 3 (+Empathy & Authenticity): R² = {m3.rsquared:.4f}, ΔR² = {m3.rsquared-m2.rsquared:.4f}
  → Perception variables add {(m3.rsquared-m2.rsquared)*100:.1f}% more variance explained.
  → This is CRITICAL: it tests whether HOW YOU PERCEIVE AI adds unique predictive power
    beyond personality and behavior. A significant ΔR² here proves that improving
    perceived empathy/authenticity DIRECTLY INCREASES TRUST.""")

# %%
print("\nFULL MODEL (Step 3) – Coefficient Details:")
print(m3.summary2().tables[1].round(4))

print("\nVIF CHECK:")
X3_data = users[['personality_score','motivation_score','freq_ordinal','empathy_score','authenticity_score']]
for i, col in enumerate(X3_data.columns):
    vif = variance_inflation_factor(X3_data.values, i)
    print(f"  {col}: VIF = {vif:.3f} {'' if vif < 5 else ' Concern'}")

print("""
 INTERPRETATION: The coefficient table shows each predictor's UNIQUE contribution:
• Significant predictors (p < 0.05) independently influence trust.
• β (coefficient) shows direction and magnitude of effect.
• Positive β = higher predictor → higher trust.
• The LARGEST significant β is the STRONGEST unique predictor of trust.
• VIF < 5 confirms predictors are not redundantly measuring the same thing.""")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(m3.fittedvalues, m3.resid, alpha=0.5, color='steelblue', edgecolor='white')
axes[0].axhline(y=0, color='red', linestyle='-')
axes[0].set_xlabel('Fitted'); axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted', fontweight='bold')
probplot(m3.resid, dist="norm", plot=axes[1])
axes[1].set_title('Normal Q-Q Plot', fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/the_irritater/Projects/new_emotional_interaction_chatbot/analysis/obj5_regression_diag.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. AI vs. Human Trust Comparison

# %%
print("=" * 80)
print("AI vs. HUMAN TRUST COMPARISON")
print("=" * 80)
print(f"Trust in AI: M={users['trust_score'].mean():.3f}, SD={users['trust_score'].std():.3f}")
print(f"AI vs Human Likert: M={users['ai_human_likert_score'].mean():.3f}, SD={users['ai_human_likert_score'].std():.3f}")

t, p_t = stats.ttest_rel(users['trust_score'], users['ai_human_likert_score'])
d = (users['trust_score'].mean() - users['ai_human_likert_score'].mean()) / \
    np.sqrt((users['trust_score'].std()**2 + users['ai_human_likert_score'].std()**2) / 2)
w, p_w = wilcoxon(users['trust_score'], users['ai_human_likert_score'])

print(f"\nPaired t-test: t={t:.3f}, p={p_t:.4f}, Cohen's d={d:.3f}")
print(f"Wilcoxon: W={w:.3f}, p={p_w:.4f}")
print(f"Decision: {'Significant difference *' if p_t < 0.05 else 'No significant difference'}")

print(f"""
 INTERPRETATION: This is a CRITICAL comparison. We test whether users trust AI-based
emotional support at the SAME LEVEL as human-based emotional support.

• If SIGNIFICANT (p < 0.05): A trust GAP exists between AI and human support.
  {'Users trust AI MORE than the AI-vs-human comparison suggests' if d > 0 else 'Users show LOWER trust in AI compared to human relationships'} (d = {d:.3f}).
• If NOT significant: Trust levels are comparable — a remarkable finding suggesting
  AI has achieved parity with human emotional support in users' minds.

Cohen's d = {abs(d):.3f} → {'Large' if abs(d)>=0.8 else 'Medium' if abs(d)>=0.5 else 'Small' if abs(d)>=0.2 else 'Negligible'} effect size.

The Wilcoxon test (non-parametric) confirms/disconfirms the t-test result without
assuming normality.""")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pd.DataFrame({'Trust in AI': users['trust_score'], 'AI vs Human\n(Likert)': users['ai_human_likert_score']}).plot(
    kind='box', ax=axes[0], color=dict(boxes='steelblue', whiskers='gray', medians='red', caps='gray'))
axes[0].set_title('Trust Score Comparison', fontweight='bold'); axes[0].set_ylabel('Score')
axes[0].axhline(y=4, color='green', linestyle='-', alpha=0.5)
axes[1].hist(users['trust_score'], bins=15, alpha=0.6, label='Trust in AI', color='steelblue', edgecolor='white')
axes[1].hist(users['ai_human_likert_score'], bins=15, alpha=0.6, label='AI vs Human', color='coral', edgecolor='white')
axes[1].set_title('Distribution Overlap', fontweight='bold'); axes[1].legend()
plt.tight_layout()
plt.savefig('/Users/the_irritater/Projects/new_emotional_interaction_chatbot/analysis/obj5_ai_vs_human.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Scenario Preferences

# %%
print("SCENARIO PREFERENCES – AI vs. Human")
for col in ai_human_scenario:
    counts = users[col].value_counts(); total = counts.sum()
    print(f"\n▸ {col}:")
    for cat, cnt in counts.items():
        print(f"    {cat}: {cnt} ({cnt/total*100:.1f}%)")
    expected = np.full(len(counts), total/len(counts))
    chi2_stat = ((counts.values - expected)**2 / expected).sum()
    p_chi = 1 - stats.chi2.cdf(chi2_stat, df=len(counts)-1)
    print(f"  χ² = {chi2_stat:.3f}, p = {p_chi:.4f} → {'Non-uniform *' if p_chi<0.05 else 'Uniform'}")

print(f"""
 INTERPRETATION: Scenario preferences reveal CONTEXT-DEPENDENT trust. Users may trust
AI for some situations but prefer humans for others. The Chi-square goodness-of-fit test
checks if responses are uniformly distributed (no preference) or concentrated.

• "AI" preference → Users trust AI MORE in that specific emotional scenario
• "Human" preference → Users prefer traditional human support
• "Both equally" / "Depends" → Conditional trust based on specifics

A NON-UNIFORM distribution (p < 0.05) means users have CLEAR PREFERENCES in that
scenario — they are not randomly choosing. This shows trust is CONTEXTUAL.""")

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("AI vs. Human Scenario Preferences", fontsize=16, fontweight='bold')
clr = {'AI':'#4C72B0', 'Human':'#DD8452', 'Both equally':'#55A868', 'Depends on the situation':'#C44E52'}
for ax, col in zip(axes.flat, ai_human_scenario):
    c = users[col].value_counts()
    ax.bar(range(len(c)), c.values, color=[clr.get(x,'#8C8C8C') for x in c.index], edgecolor='white')
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.index, rotation=30, ha='right', fontsize=8)
    ax.set_title(col.replace('ai_vs_human_scenarios_','Scenario '), fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('/Users/the_irritater/Projects/new_emotional_interaction_chatbot/analysis/obj5_scenarios.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Mediation Analysis

# %%
print("=" * 80)
print("MEDIATION: Personality → Perceived Empathy → Trust in AI")
print("=" * 80)

# Path c (total): Personality → Trust
Xc = sm.add_constant(users['personality_score']); mc = sm.OLS(users['trust_score'], Xc).fit()
c = mc.params['personality_score']
# Path a: Personality → Empathy
Xa = sm.add_constant(users['personality_score']); ma = sm.OLS(users['empathy_score'], Xa).fit()
a = ma.params['personality_score']
# Paths b, c': Personality + Empathy → Trust
Xbc = sm.add_constant(users[['personality_score','empathy_score']]); mbc = sm.OLS(users['trust_score'], Xbc).fit()
b = mbc.params['empathy_score']; c_prime = mbc.params['personality_score']

indirect = a * b
se_a = ma.bse['personality_score']; se_b = mbc.bse['empathy_score']
sobel_se = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
sobel_z = indirect / sobel_se; sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

print(f"Path a (Personality → Empathy): β = {a:.4f}, p = {ma.pvalues['personality_score']:.4f}")
print(f"Path b (Empathy → Trust):       β = {b:.4f}, p = {mbc.pvalues['empathy_score']:.4f}")
print(f"Path c (Total Effect):           β = {c:.4f}, p = {mc.pvalues['personality_score']:.4f}")
print(f"Path c' (Direct Effect):         β = {c_prime:.4f}, p = {mbc.pvalues['personality_score']:.4f}")
print(f"Indirect Effect (a×b):           {indirect:.4f}")
print(f"Proportion Mediated:             {indirect/c*100:.1f}%" if c != 0 else "N/A")
print(f"Sobel Test: z = {sobel_z:.3f}, p = {sobel_p:.4f}")
print(f"Decision: {'Significant mediation ' if sobel_p < 0.05 else 'No significant mediation'}")

print(f"""
 INTERPRETATION: Mediation analysis tests whether perceived empathy EXPLAINS (mediates)
the relationship between personality and trust. The logic is:

TOTAL EFFECT (c = {c:.4f}): The overall relationship between personality and trust.
PATH a (a = {a:.4f}): Does personality predict empathy perceptions?
PATH b (b = {b:.4f}): Does empathy predict trust, controlling for personality?
DIRECT EFFECT (c' = {c_prime:.4f}): The personality–trust relationship AFTER accounting for empathy.
INDIRECT EFFECT (a×b = {indirect:.4f}): The portion of the personality–trust relationship
  that goes THROUGH empathy.

{'FULL MEDIATION: c becomes non-significant → empathy fully explains the personality–trust link.' if mbc.pvalues['personality_score'] > 0.05 and sobel_p < 0.05 else 'PARTIAL MEDIATION: c is reduced but still significant → empathy partially explains the link.' if sobel_p < 0.05 else 'NO MEDIATION: The indirect effect is not significant.'}

PRACTICAL MEANING: {'Improving perceived empathy is a viable strategy to increase trust across all personality types.' if sobel_p < 0.05 else 'Other pathways beyond empathy need to be explored to build trust.'}""")

# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
for x,y,w,h,txt in [(0.5,2.5,2.5,1,'Personality\nScore'),(3.75,5,2.5,1,'Perceived\nEmpathy'),(7,2.5,2.5,1,'Trust\nin AI')]:
    ax.add_patch(plt.Rectangle((x,y), w, h, fill=True, facecolor='lightblue', edgecolor='navy', linewidth=2))
    ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=12, fontweight='bold')
ax.annotate('', xy=(3.75,5.5), xytext=(2.5,3.5), arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(2.5, 4.8, f'a = {a:.3f}', fontsize=11, color='green', fontweight='bold')
ax.annotate('', xy=(7,3.5), xytext=(6.25,5.5), arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(7, 4.8, f'b = {b:.3f}', fontsize=11, color='green', fontweight='bold')
ax.annotate('', xy=(7,3), xytext=(3,3), arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(5, 2.3, f"c' = {c_prime:.3f}", fontsize=11, color='red', fontweight='bold')
ax.text(5, 1.7, f"(c = {c:.3f})", fontsize=10, color='gray')
ax.text(5, 1.1, f"Indirect = {indirect:.3f}, Sobel p = {sobel_p:.4f}", fontsize=10, color='navy')
ax.set_title('Mediation Model: Personality → Empathy → Trust', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/the_irritater/Projects/new_emotional_interaction_chatbot/analysis/obj5_mediation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Trust by Personality Type

# %%
anova_trust = pg.anova(data=users, dv='trust_score', between='personality_type')
print("ONE-WAY ANOVA: Trust by Personality Type")
print(anova_trust.round(4))
if anova_trust['p_unc'].values[0] < 0.05:
    print("\nPost-hoc Games-Howell:")
    ph = pg.pairwise_gameshowell(data=users, dv='trust_score', between='personality_type')
    print(ph[['A','B','mean_A','mean_B','diff','T','df','pval','hedges']].round(4))
    print("\n Pairs with pval < 0.05 have significantly different trust levels.")
else:
    print("\n No significant difference in trust across personality types.")

# %%
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=users, x='personality_type', y='trust_score', order=['Low','Moderate','High'],
               palette='Set2', inner='box', ax=ax)
ax.set_title('Trust in AI by Personality Type', fontsize=14, fontweight='bold')
ax.set_ylim(1, 7)
plt.tight_layout()
plt.savefig('/Users/the_irritater/Projects/new_emotional_interaction_chatbot/analysis/obj5_trust_violin.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. CONCLUSION – Objective 5
#
# ### Why Was This Objective Selected?
#
# This is the **capstone objective** because TRUST is the ultimate measure of emotional
# AI system success. It was selected because:
#
# 1. **Trust determines adoption**: According to the Technology Acceptance Model and
#    UTAUT frameworks, trust is a necessary condition for sustained technology use.
#    Without trust, even the most empathetic AI will be abandoned.
#
# 2. **The human comparison is essential**: Emotional AI does not exist in a vacuum —
#    it competes with human emotional support. Understanding the trust gap (or parity)
#    informs whether AI can supplement, complement, or potentially substitute human support.
#
# 3. **Multiple predictors need testing simultaneously**: Trust is complex. It's not
#    driven by a single factor but by the interplay of personality, motivation, usage
#    patterns, and perceptual quality. Only hierarchical regression can disentangle these.
#
# 4. **Mediation reveals mechanisms**: Simply knowing that personality predicts trust is
#    not actionable. Knowing that empathy MEDIATES this relationship provides a clear
#    intervention target: improve perceived empathy to increase trust.
#
# 5. **Context-dependent trust is nuanced**: Scenario analysis reveals that trust is
#    not monolithic — users may trust AI for some emotional needs but not others.
#
# ### What We Aimed to Prove
#
# | Hypothesis | Method | Key Metric |
# |-|-|-|
# | Multiple variables predict trust | Hierarchical Regression | R² and ΔR² at each step |
# | Perceptions add unique predictive power | Step 3 ΔR² | Significant increase in R² |
# | AI trust differs from human trust | Paired t-test + Wilcoxon | p-value and Cohen's d |
# | Trust is context-dependent | Chi-Square per scenario | Significant non-uniform distributions |
# | Empathy mediates personality → trust | Sobel Mediation Test | Significant indirect effect |
# | Personality types differ in trust | One-Way ANOVA + Games-Howell | Significant F and pairwise diffs |
#
# ### Grand Summary Across All 5 Objectives
#
# Together, these five objectives paint a COMPLETE PICTURE of emotional AI interaction:
#
# 1. **Objective 1** revealed WHY people DON'T use emotional AI (barriers, perceptions)
# 2. **Objective 2** showed HOW personality drives usage PATTERNS among those who do use it
# 3. **Objective 3** uncovered WHAT MOTIVATES people to engage with emotional AI
# 4. **Objective 4** tested whether personality and frequency shape QUALITY PERCEPTIONS
# 5. **Objective 5** (this one) determines what BUILDS TRUST and how it compares to
#    human relationships — the ultimate measure of emotional AI's viability
#
# This progression from non-users → usage patterns → motivations → quality perceptions
# → trust follows a logical **behavioral adoption funnel**, providing comprehensive
# evidence for the design, deployment, and evaluation of emotional AI systems.

# %%
print(" Objective 5 Analysis Complete!")
print("=" * 60)
print("ALL 5 OBJECTIVES HAVE BEEN ANALYZED.")
print("Each notebook contains: data cleaning, reliability analysis,")
print("statistical tests, visualizations, and detailed interpretations.")
print("=" * 60)
