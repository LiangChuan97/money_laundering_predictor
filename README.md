<h1 align="center">🏦 Money Laundering Detection with Machine Learning</h1>

<p align="center">
Imbalance-resilient modelling for transaction-level AML detection
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.14-blue">
<img src="https://img.shields.io/badge/Models-Logistic%20Regression%20%7C%20⭐%20Balanced%20Random%20Forest%20%7C%20XGBoost-blue">
<img src="https://img.shields.io/badge/Domain-AML-orange">
<img src="https://img.shields.io/badge/Dataset-IBM%20Transactions-purple">
</p>

## 🚀 Project Summary

This project builds a machine learning system to detect suspicious financial transactions in highly imbalanced AML data.

Key highlights:

- Dataset with **0.2% laundering rate**
- Behavioural feature engineering using **rolling transaction signals**
- Compared **Logistic Regression, Balanced Random Forest, and XGBoost**
- **Balanced Random Forest achieved best performance**
- Reduced **false negatives from 60 → 8**
- Achieved **lowest operational cost in evaluation framework**

## 🧠 Skills Demonstrated

- Imbalanced classification modelling
- AML behavioural feature engineering
- Precision–Recall evaluation for rare events
- Time-aware model validation
- Ensemble modelling (Balanced Random Forest, XGBoost) and linear baseline modelling (Logistic Regression)
- Cost-sensitive model evaluation

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Data Cleaning](#-data-cleaning)
- [Exploratory Data Analysis](-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Model Development](#-model-development)
- [Model Evaluation](#-model-evaluation)
- [Key Results](#-key-results)
- [Feature Importance](#-feature-importance)
- [Tech Stack](#-tech-stack)
- [Strengths](#-strengths)
- [Limitations](#-limitations)
- [Recommendations](#-recommendations)
- [Conclusion](#-conclusion)
- [Project Presentation](#-project-presentation)

## 📌 Project Overview

Anti-Money Laundering (AML) systems aim to identify suspicious financial transactions indicative of illicit activity. Traditional rule-based systems generate excessive false positives and fail to capture evolving laundering strategies.

Machine learning approaches offer improved adaptability but must address significant challenges, particularly extreme class imbalance and temporal dependency.

In the present dataset, laundering transactions constitute approximately 0.2% of total observations, making detection a rare-event classification problem.

<h2 align="center">🌍 Why This Matters </h2>

Money laundering is estimated to represent **2–5% of global GDP annually**.Financial institutions rely on AML monitoring systems to detect suspicious activity while balancing operational investigation costs.

This project demonstrates how **machine learning and behavioural feature engineering** can improve detection performance under extreme class imbalance conditions.

## 🏦 Business Problem

The primary objective of this study is to develop a predictive model capable of identifying laundering transactions with improved recall while maintaining operationally acceptable precision.

Key challenges include:
- Severe class imbalance
- High-cardinality categorical variables
- Temporal data structure
- Risk of information leakage
- Nonlinear behavioral patterns

Goals:
- Develop a leakage-safe predictive modeling pipeline.
- Engineer behaviorally meaningful features.
- Compare linear and tree-based models.
- Evaluate performance using precision-recall metrics.
- Provide actionable recommendations for AML implementation.

## 🗂 Dataset

The dataset contains IBM Transactions for Anti Money Laundering (AML) which was sourced from <a href = "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data"> kaggle. </a> 

Target variable:

Binary Laundering Label (0/1)

The dataset consists of transactional records containing:
- Timestamp
- Sending and receiving account identifiers
- Sending and receiving bank identifiers
- Transaction amounts
- Currency types
- Payment format

All data were historical and anonymized.

## 🧹 Data Cleaning

The following preprocessing steps were applied:
- Conversion of timestamp to datetime format
- Removal of duplicate records
- Handling of missing values
- Data Type Validation

## 🔍 Exploratory Data Analysis

Key questions explored:
- Are the transaction amounts significantly different between normal transactions and laundering transactions?
- How large is the difference in transaction amounts between laundering and normal transactions?
- What's the amount distribution by laundering class?
- What's the laundering rate by payment format?
- What's the laundering rate by receiving currency?
- What's the laundering rate by hour of day?

These questions help identify patterns, anomalies, and key risk indicators associated with laundering transactions, such as unusual transaction amounts, risky payment formats, high-risk currencies, and suspicious transaction timing.


<h2 align="center">💰 Amount distribution by laundering class</h2>

<p align="center"><img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/a07c94d3-6850-4732-85b0-9d2dd2ac1357" /></p>
<p align="center"><img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/80d162fa-242f-4675-9ca2-77a3fa484d79" /></p>

Laundering transactions tend to occur at higher transaction values, with a higher median log transaction amount compared to normal transactions. This suggests that transaction amount may be a useful feature for identifying suspicious financial activity.

<h2 align="center">💳 Laundering Rate by Payment Format and Receiving Currency</h2>

<p align="center"><img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/7c35a49a-8ce4-48a8-91c1-e05c0b026b52" />
<p align="center"><img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/a6e46ad9-4d15-4597-8555-0abec54edf16" />

Laundering activity varies across payment formats and currencies. ACH payments show the highest laundering rate among payment methods, while certain receiving currencies such as Saudi Riyal exhibit higher laundering activity compared to others. These patterns suggest that payment channels and currency usage may serve as useful indicators for identifying suspicious transactions.

<h2 align="center">🕒 Laundering Rate by Hour of Day </h2>

<p align="center"><img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/780d3627-b2f6-4612-af7e-9bed9ec94186" />

Laundering activity is minimal during most hours of the day but shows a sharp spike late at night around hour 23. This suggests that transaction timing may be an important behavioural signal for detecting suspicious activity.

<h2 align="center">🧪 Mann-Whitney U test </h2>

The Mann–Whitney U test produced a p-value of **9.85 × 10⁻²⁹**, indicating a statistically significant difference in transaction amounts between normal and laundering transactions. This suggests that transaction amount is a meaningful feature for identifying suspicious financial activity.

<h2 align="center">📏 Cohen's d</h2>

Cohen’s d of 0.078 indicates a very small effect size, suggesting that although transaction amounts differ statistically between normal and laundering transactions, the magnitude of the difference is small and additional features are needed for effective detection.

<h2 align="center">📊 Effect size</h2>

The median transaction amount for laundering transactions (8,784.93) is significantly higher than normal transactions (1,925.73), indicating that suspicious transactions tend to involve larger payment amounts.

## ⚙️ Feature Engineering

Behavioral indicators were constructed to capture short-term transaction dynamics:
- Rolling 7-day volatility
- Rolling 7-day transaction count
- Rolling 7-day transaction sum
- Unique counterparties count
- Average Transaction Amount
- Transaction frequency per day
- Incoming/outgoing ratio
- Suspicious Format Indicator
  
These features aim to detect structuring, burst activity, and anomalous behavior.

## 🧠 Model Development

Given 0.2% base rate:
- PR-AUC significantly above baseline (0.002)
- Recall > 85% at operational threshold
-	Model stable across time-based split

Dataset split into 80/20 (train/test) - in order of timestamp so as to simulate predicting future transactions

To prevent temporal leakage, a time-based train-test split was implemented using the 80th percentile timestamp.

Linear and tree models were developed and compared. The goal was to identify a model that could achieve high recall for suspicious transactions reasonable precision, handle extreme class imbalance, strong ranking ability(PR-AUC), low false negative and reasonable amount of false positive. 

Different thresholds were populated to find the optimum precision, recall, false positive and false negative. 


<h2 align="center">📈 Logistic Regression (Baseline Model) </h2>

• Baseline linear model  

<p>Logistic Regression was used as the initial benchmark model. It provides a simple and interpretable approach for binary classification by modelling the probability that a transaction is classified as laundering or normal.

• Class-weighted  

<p>Because the dataset is highly imbalanced (with far fewer laundering transactions than normal ones), class weights were applied. This assigns a higher penalty to misclassifying laundering transactions, helping the model focus more on detecting suspicious activity.

• Scaled features  

<p>Numerical features were standardised before training the model. Scaling ensures that variables with larger magnitudes do not dominate the optimisation process and helps the model converge more effectively.

• Interpretable coefficients 

<p>One advantage of logistic regression is its interpretability. Each feature has a coefficient that indicates how it influences the probability of a transaction being classified as laundering. Positive coefficients increase the likelihood of suspicious activity, while negative coefficients reduce it. This makes the model useful for understanding which factors contribute most to potential money laundering behaviour.


<h2 align="center">🌳 Balanced Random Forest</h2>

• Undersampling per tree 

<p>Balanced Random Forest addresses the extreme class imbalance in AML datasets by randomly undersampling the majority class (normal transactions) when building each decision tree. This ensures that each tree is trained on a more balanced subset of the data, allowing the model to better learn patterns associated with laundering transactions.

• Strong recall performance 

<p>Because the model focuses more on the minority class during training, it tends to achieve higher recall for laundering transactions. This is important in AML systems where missing suspicious activity (false negatives) can have serious regulatory and financial consequences.

• Stable minority detection 

<p>By combining predictions from many decision trees, Balanced Random Forest reduces variance and improves robustness when identifying rare events such as laundering transactions. This ensemble approach allows the model to consistently detect suspicious behaviour across different transaction patterns.

<h2 align="center">⚡ XGBoost</h2>

• Gradient boosting algorithm 

<p>XGBoost is an ensemble learning method that builds decision trees sequentially, where each new tree focuses on correcting the errors made by previous trees. This approach allows the model to capture complex patterns in transaction behaviour.

• Captures nonlinear relationships 

<p>Unlike linear models, XGBoost can learn interactions between multiple features such as transaction amount, payment format, currency, and time of transaction. This makes it useful for identifying subtle patterns associated with suspicious financial activity.

• Hyperparameter tuning  

<p>Key hyperparameters such as tree depth, learning rate, number of estimators, and regularisation parameters were tuned to optimise model performance. Hyperparameter tuning helps control model complexity, improve prediction accuracy, and reduce the risk of overfitting.

The <code>scale_pos_weight</code> parameter was also used to address the severe class imbalance in the dataset by giving higher importance to laundering transactions during training. This helps the model focus more on detecting rare suspicious activity.

• Strong predictive performance

<p>XGBoost achieved strong predictive performance and improved upon the baseline logistic regression model. However, the Balanced Random Forest ultimately performed best in detecting laundering transactions due to its ability to handle class imbalance more effectively.

## 📊 Model Evaluation

MAS requires that:
- Institutions must detect suspicious transactions
- Systems must be calibrated to the institution’s risk profile
- False negatives (missed laundering) must be controlled
- Model governance and validation must be robust
  
They do NOT prescribe numerical ML thresholds.

Metrics for success:
- PR-AUC: How well the model balances catching positives while avoiding false positives. It focuses only on minority class performance
- ROC-AUC: How well the model can separate positives (laundering) from negatives (normal) very well
probabilistically.
- Recall: The proportion of actual laundering transactions correctly identified.
- Precision: Precision: The proportion of predicted laundering transactions that are actually laundering.
- False Negative: A laundering transaction incorrectly classified as normal.
- False Positive: A normal transaction incorrectly classified as laundering.

| | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| Logistics Regression | 0.93005 | 0.01355 | 0.8309 | 0.0154 |
| ⭐ Balanced RandomForest| 0.95936 | 0.052514 | 0.98 | 0.1544 |
| XGBoost | 0.92651 | 0.04974| 0.1544 | 0.0602 |

Cost assumptions:
<p>Falae Negative = $50,000 
<p>False Positice = $50

| | False Negative | False Positive | Total Cost | 
|---|---|---|---|
| Logistics Regression | 60 | 21675 | 4083.75 | 
| ⭐ Balanced RandomForest| 8 | 29786 | 1889.30 | 
| XGBoost | 345 | 984 | 17299.20 | 

## 🏆 Key Results

- Balanced Random Forest achieved **98% recall**
- Reduced **false negatives from 60 → 8**
- Achieved **lowest operational cost**
- Identified behavioural indicators of suspicious transactions
- Achieved **highest PR-AUC**

Its ability to handle class imbalance through undersampling makes it particularly effective for rare-event detection in AML systems.

## 🌟 Feature Importance

Payment format and high-risk payment indicators are the strongest predictors of laundering activity. 

Behavioural features such as transaction frequency, rolling transaction volatility, and recent transaction totals also play an important role, suggesting that unusual transaction patterns are key signals for detecting suspicious financial activity.

## 🧰 Tech Stack

🐍 Python  
📊 Pandas  
🤖 Scikit-learn  
🌲 XGBoost  
📈 Matplotlib / Seaborn

## 💪 Strengths

<h2 align="center"><u>⏳ Time-Aware Modeling Framework</u></h2>

A chronological train-test split was implemented using the 80th percentile timestamp to simulate real-world deployment conditions. This approach prevents temporal leakage, which is a common source of inflated performance in financial crime modeling.

Key benefits include:
- Ensures that future transactions do not influence past feature calculations.
- Mimics production inference conditions.
- Reduces overestimation of model performance.
- Enhances regulatory defensibility.

Many AML modeling failures stem from inadvertent leakage via global aggregations; this framework explicitly mitigates that risk.

<h2 align="center"><u>🔬 Comprehensive Behavioral Feature Engineering</u></h2>

The model incorporates short-term behavioral indicators such as:
- Rolling 7-day transaction count
- Rolling 7-day transaction sum
- Rolling 7-day volatility
- Transaction intensity ratios
- Counterparty diversity metrics

These features capture dynamic laundering behaviors including:
- Structuring (smurfing)
- Burst activity
- Rapid layering
- Abnormal transaction acceleration

Behavioral modeling enhances detection of new or emerging laundering patterns that may not yet be reflected in historical risk scores.

<h2 align="center"><u>📊 Appropriate Rare-Event Evaluation Metrics </u></h2>

Traditional metrics such as accuracy are misleading in datasets with <1% positive class prevalence.
This study appropriately emphasizes:
- Precision-Recall AUC (PR-AUC)
- Recall for minority class

PR-AUC is especially relevant because it directly evaluates minority-class ranking performance, which aligns with AML investigation workflows.

<h2 align="center"><u>⚖️ Comparative Model Benchmarking</u></h2>

Multiple modeling approaches were evaluated:
- Logistic Regression (linear baseline)
- Balanced Random Forest (imbalance-aware ensemble)
- Gradient Boosting (XGBoost)

This comparative framework:
- Validates that improvements are model-agnostic.
- Prevents over-reliance on a single algorithm.
- Supports governance review through benchmarking.

## ⚠️ Limitations

<h2 align="center"><u>🌍 Lack of External Risk Indicators </u></h2>

The dataset excludes key contextual variables such as country risk ratings, sanctions exposure, Know YourCustomer(KYC) attributes, Politically Exposed Person(PEP) flags, and beneficial ownership data.

Without these enrichment features, risk scoring lacks broader regulatory context.

<h2 align="center"><u>⚠️ Structural Constraints from Extreme Imbalance </u></h2>

With a laundering rate of approximately 0.2%:
- Precision is inherently limited.
- Threshold tuning is highly sensitive.
- False positives remain operationally challenging.

This limitation is intrinsic to rare-event classification rather than purely methodological.

## 💡 Recommendations

<h2 align="center"><u>🚀 Model Selection for Production </u></h2>

Based on comparative evaluation, the recommended model for initial deployment is:
- Balanced Random Forest (BRF)

Balanced Random Forest demonstrates strong recall under extreme imbalance conditions due to per-tree class rebalancing. Gradient boosting models provide improved ranking capability when properly tuned with class weighting and regularization.

The final production selection should consider:
- Precision-Recall AUC performance
- Stability across temporal splits
- Computational efficiency
- Governance interpretability requirements

<h2 align="center"><u>📉 Use Model Output as a Risk Score</u></h2>

The predictive model should not function as a deterministic binary classifier. Instead, it should generate a continuous risk score.

Advantages:
- Enables prioritization rather than rigid filtering.
- Supports adjustable risk tolerance.
- Facilitates dynamic investigation scaling.

This approach aligns with modern risk-based AML frameworks.

<h2 align="center"><u>🔗 Combine Model with Rule-Based Monitoring </u></h2>

Pure machine learning systems may overlook edge-case regulatory rules. A hybrid system is recommended:
- Machine learning model for probabilistic ranking.
- Rule-based triggers for regulatory compliance requirements.
- Combined alert scoring mechanism.
Hybrid architecture improves robustness and regulatory acceptance.

## 🏁 Conclusion

This study demonstrates that behaviorally engineered features combined with smoothed entity risk encoding significantly enhance laundering detection under extreme class imbalance.
Balanced ensemble methods provide superior recall compared to linear models. However, further gains require incorporation of network-level features and additional contextual risk signals.
The proposed framework provides a scalable foundation for AML transaction monitoring and future model refinement.

## 📑 Project Presentation

🔗 [View Full Presentation]([https://your-link-here](https://drive.google.com/file/d/1f0JEwUXKK38_Ht9tP5XhfAfj8KnBBYhy/view?usp=sharing])






