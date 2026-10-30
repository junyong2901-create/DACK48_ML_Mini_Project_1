import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
# warnings.formatwarning('ignore')

df = pd.read_csv(r'C:\Users\dungt\Documents\DA\ML_mini_project_1\customer_churn_dataset.csv')

#Quick overview

print(df.head())
print(df.shape)
print(df.info())

# Checking missing values
print(df.isna().sum())

print(df[df['CustomerID'].isna()])

df = df.dropna() #filter df with notna()
print(df.isnull().sum())

df.columns = df.columns.str.lower().str.replace(' ', '_') #Remove the empty space in title 
print(df.columns)

#Checking duplicated values
print(df['customerid'].nunique())
print(df['customerid'].count())

test_df = df.groupby('customerid')['churn'].count().reset_index()
print(test_df[test_df['churn']>1])

print(df[df['customerid']==64370.0])

print(f"Ratio of duplicated customerid in total: {len(test_df[test_df['churn'] > 1]['customerid'].unique())/len(test_df['customerid'].unique())*100:.2f}%")

duplicated_customerid_list = test_df[test_df['churn']>1]['customerid'].unique()
df_final = df[~df['customerid'].isin(duplicated_customerid_list)]
print(df_final.shape)

#Quick univariate analyse
#Check unique values of category columns:
str_cols = df.select_dtypes(include=['str']).columns
for c in str_cols:
    print(f'\nColumn: {c}')
    print(df[c].unique())

#Check the distribution of numeric columns using histograms
df.select_dtypes(include='float').hist(figsize=(10,8), bins=30, layout=(-1,3))
plt.tight_layout()
plt.show()

#Check the outlier of numeric columns using IQR
summary = []
num_cols = df.select_dtypes(include='float').columns

for c in num_cols:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    n_ouliers = ((df[c] < lower) | (df[c] > upper)).sum()

    summary.append({
        "Column": c,
        "Outliers": n_ouliers,
        "Percentage": round(n_ouliers/len(df) * 100, 2),
        "Has Outliers":n_ouliers > 0
    })
outlier_summary = pd.DataFrame(summary)
print(outlier_summary)

#Overal Churn Rate
churn_rate = df_final['churn'].value_counts(normalize=True)
print(f'Overal Churn Rate: {churn_rate[1]*100:.2f}%')

#Churn rate by subscription type and contract length
cat_cols = ['subscription_type', 'contract_length']
for c in cat_cols:
    summary = (df_final.groupby(c).agg(Total_Users=('churn', 'count'), Churn_Users=('churn','sum')))
    summary['Churn_Rate'] = (summary['Churn_Users']/summary['Total_Users'] * 100).round(2)
    print(f'\n===== {c} =====')
    print(summary)

churn_rate = (df_final.groupby('contract_length')['churn'].mean().reset_index())
plt.figure(figsize=(7,5))
ax = sns.barplot(data=churn_rate, x='contract_length', y='churn')

plt.ylabel('Churn Rate')
plt.title('Churn Rate by Contract Length')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# #Add percentage labels on top of each bar
for p in ax.patches:
    height=p.get_height()
    ax.annotate(f'{height:.1%}',
                (p.get_x() + p.get_width() /2, height),
                ha='center',
                va='bottom',
                fontsize=10,
                xytext=(0,3),
                textcoords='offset points')

plt.figtext(
    0.5, -0.05,
    "Figure 2. Churn rate by contract length",
    ha='center',
    fontsize=10,
    style='italic'
)
plt.tight_layout()
plt.show()

#Churn rate by payment delay bucket
print(df_final['payment_delay'].describe())

bins = [0,5,10,15,20,25,30]
labels = [
    '0-5',
    '6-10',
    '11-15',
    '16-20',
    '21-25',
    '26-30'
]

df_final['delay_bucket']=pd.cut(df['payment_delay'], bins=bins, labels=labels,include_lowest=True)

summary = (df_final.groupby('delay_bucket').agg(Total_Users=('churn','count'), Churn_Users=('churn','sum')))

summary['Churn_Rate'] = (summary['Churn_Users']/summary['Total_Users'] * 100).round(2)

print('===== Payment Delay Bucket =====')
print(summary)

churn_rate = (df_final.groupby('delay_bucket')['churn'].mean().reset_index())

plt.figure(figsize=(7,5))
ax = sns.barplot(data=churn_rate, x='delay_bucket',y='churn')

plt.ylabel('Churn Rate')
plt.title('Churn Rate by Delay Bucket')
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f'{y: .0%}')
)

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1%}',
                (p.get_x() + p.get_width() / 2,height),
                ha='center',
                va='bottom',
                fontsize=10,
                xytext=(0,3),
                textcoords='offset points')

plt.figtext(
    0.5, -0.05,
    "Figure 3. Churn rate by delay bucket.",
    ha='center',
    fontsize=10,
    style='italic'
)

plt.tight_layout()
plt.show()

#Correlation matrix
#Calculate the correlation matrix for numeric columns and visualize it using a heatmap
corr_df = df_final.select_dtypes(include='float').corr()

plt.figure(figsize=(8,8))
sns.heatmap(
    corr_df,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    square=True
)

plt.figtext(
    0.5, -0.05,
    "Figure 4. Correlation heatmap shows three most correlation features: 'payment_delay', 'monthly_charges', 'total_charges'.",
    ha='center',
    fontsize=10,
    style='italic'
)

plt.title('Correlation Heatmap')
# plt.show()
print(
    'From the heatmap, we can have following obeservations:' \
    '- Support Call has moderate positive correlation with Churn (0.59), it represents that customers who contact support more frequently are likely to churn. It indicates unresolved issues and poor customer experiences.' \
    '- Total Spend has weak negative correlation with Churn (-0.44), it indicates that customers who spend higher money tend to be more loyal, while low spending customers tend to leave.' \
    '- Payment Delay has weak positive correlation with Churn (0.32), it shows that customers who frequently delay payments are at higher risk of leaving, possibly due to lower engagement.'
)

# #Relationship between Churn and Support Call
churn_rate = (df_final.groupby('support_calls')['churn'].mean().reset_index())
plt.figure(figsize=(7,5))
sns.barplot(data=churn_rate, x='support_calls', y='churn')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Number of Support Calls')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _:f'{y:.0%}'))

plt.figtext(
    0.5, -0.05,
    "Figure 5. Churn rate rises dramatically after 5 support calls (over 80%), indicating a clear threshold for high churn risk.",
    ha='center',
    fontsize=10,
    style='italic'
)
plt.show()

#Relationship between Churn rate and Total spend
plt.figure(figsize=(7,5))
sns.kdeplot(
    data=df,
    x='total_spend',
    hue='churn',
    fill=True,
    common_norm=False,
    alpha=0.4
)

plt.title('Distribution of Total Spend by Churn Status')
plt.xlabel('Total Spend')
plt.ylabel('Density')

plt.figtext(
    0.5,-0.05,
    "Figure 6. Lower-spending customers are more likely to churn, but the overlap between two groups \n suggests that total_spend alone is not a strong predictor.",
    fontsize=10,
    style='italic'
)
plt.show()

print(df_final.groupby('churn')['total_spend'].agg(['mean','median','std']))

#Relationship between Age and Churn
churn_rate = df_final.groupby('age')['churn'].mean().reset_index()

plt.figure(figsize=(14,5))
sns.barplot(data=churn_rate, x='age', y='churn')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Age')
plt.xticks(rotation=45)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.figtext(
    0.5,-0.05,
    "Figure 7. Customers aged 51 and above (account for about 20% of total users) are significantly more likely to churn than younger customers.",
    fontsize=10,
    style='italic'
)
plt.show()

pct_51_plus = (df['age'] >= 51).mean() * 100 
print (f'Users aged 51+: {pct_51_plus:.2f}%')

print('===== INSIGHT =====')
print('1. Customers with more than five support calls show a high tendency to churn, which suggests that support calls are a clear indicator of churn.')
print('2. Payment delay shows a distinct threshold effect. The trend indicates that nearly 100% of customers whose payments are delayed by more than 21 days choose to churn, whereas those who delay payment by only a few days tend to stay.')
print('3. Customers with lower spending levels tend to churn at a higher rate than those with higher spending, but this trend does not fully support the idea that low spending automatically leads to churn.')
print('4. Contract length is closely tied to customer retention. Customers on quarterly and annual contracts have low churn rates, while most of the customers who choose to leave are on monthly contracts.')
print('5. Customers over the age of 51 show a relatively higher churn rate than younger customers.')

df_final.to_csv(r'C:\Users\dungt\Documents\DA\ML_mini_project_1\customer_churn_dataset_cleaned.csv', index=False)