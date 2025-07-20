import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LassoCV  # Added missing imports
import io  # For capturing df.info() output

#page title
st.title("Data Analysis On Titanic Data")

#Load the Titanic dataset using seaborn
data = sns.load_dataset('titanic')
df = data.copy()

# Markdown sections
st.markdown("## Asking the basic questions")

st.markdown("### How big is data?")
st.write("Shape of the dataset:", df.shape)

st.markdown("### How does the data look like?")
st.dataframe(df.head())
st.dataframe(df.sample(5))
st.dataframe(df.tail())

st.markdown("### What is the data type of columns?")
# Fixed the df.info() display
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

st.markdown("### Are there any missing values?")
missing_percent = df.isnull().sum() / df.shape[0] * 100
st.dataframe(missing_percent[missing_percent > 0])

st.markdown("### How does the data look like mathematically?")
st.dataframe(df.describe())
st.write("Age column description:")
st.write(df['age'].describe())

st.markdown("### Are there duplicate values?")
st.write("Number of duplicate rows:", df.duplicated().sum())

# Drop duplicates
df = df.drop_duplicates()

st.markdown("### How is the correlation between numerical columns?")
st.dataframe(df.select_dtypes(include='number').corr())

# Handle missing values in age
st.markdown("#### Handling the missing values")
df['age'] = df['age'].fillna(df['age'].mean())
st.write("Filled missing values in 'age' with mean:", df['age'].mean())

st.markdown("### Analysis of missing values in 'embarked'")
missing_embarked = df['embarked'].isnull().mean() * 100
unique_embarked = df['embarked'].nunique()
survived_by_embarked = df.groupby('embarked')['survived'].mean() * 100

st.write(f"Percentage of missing values in 'embarked': {missing_embarked:.2f}%")
st.write(f"Number of unique values in 'embarked': {unique_embarked}")
st.write("Survival rate by 'embarked':")
st.dataframe(survived_by_embarked)

st.markdown("#### From the above analysis, 'embarked' is relevant to our analysis and does not need to be dropped.")

# Impute embarked with mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
st.markdown("### Missing Values After Imputation (embarked)")
st.dataframe(df.isnull().sum() / df.shape[0] * 100)

st.markdown("### Deck column has very high missing values")

# Data type of 'deck'
st.write("Data type of 'deck':", df['deck'].dtype)

# Unique values in 'deck'
st.write("Number of unique values in 'deck':", df['deck'].nunique())

# Grouped survival rates and counts by deck and class
st.markdown("### Survival Rate by Deck and Passenger Class")
survival_rate_by_deck = df.groupby(['deck', 'pclass'], observed=True)['survived'].mean() * 100
st.dataframe(survival_rate_by_deck)

st.markdown("### Count of Passengers by Deck and Passenger Class")
survival_count_by_deck = df.groupby(['deck', 'pclass'], observed=True)['survived'].count()
st.dataframe(survival_count_by_deck)

# Plot
st.markdown("### Visualization: Survival Rate by Deck and pClass")
sns.set(style="whitegrid")
fig, ax = plt.subplots()
sns.barplot(x='deck', y='survived', hue='pclass', data=df, ax=ax)
ax.set_title('Survival Rate by Deck and pClass')
ax.set_ylabel('Survival Rate')
st.pyplot(fig)

# Drop 'deck' column
df = df.drop('deck', axis=1)
st.success("'deck' column dropped successfully.")
st.markdown("## Univariate Analysis")

st.markdown("#### Outlier Removal")

st.markdown("#### Fare")
# Boxplot & KDE before outlier removal
fig1, ax1 = plt.subplots()
sns.boxplot(x='fare', data=df, ax=ax1)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.kdeplot(x='fare', data=df, ax=ax2)
st.pyplot(fig2)

st.markdown("Since there are extreme outliers and it is right-skewed, we can choose a log transformation or IQR method.")

# IQR Method
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df_fare_no_outliers = df[(df['fare'] >= lower) & (df['fare'] <= upper)]

st.write("Fare Outlier Bounds:")
st.write(f"Lower: {lower}, Upper: {upper}")

st.markdown("#### Age")
fig3, ax3 = plt.subplots()
sns.boxplot(x='age', data=df, ax=ax3)
st.pyplot(fig3)

fig4, ax4 = plt.subplots()
sns.kdeplot(x='age', data=df, ax=ax4)
st.pyplot(fig4)

st.markdown("There are a few outliers in 'age', but they likely represent real elderly individuals, so we retain them.")

st.markdown("#### SibSp (Siblings/Spouses Aboard)")
fig5, ax5 = plt.subplots()
sns.boxplot(x='sibsp', data=df, ax=ax5)
st.pyplot(fig5)

fig6, ax6 = plt.subplots()
sns.kdeplot(x='sibsp', data=df, ax=ax6)
st.pyplot(fig6)

st.markdown("The 'sibsp' outliers may reflect real data. We will use binning later if needed.")

# Histogram of Age
st.markdown("### Histogram of Age")
fig7, ax7 = plt.subplots(figsize=(10, 6))
ax7.hist(df['age'], bins=30, color='skyblue', edgecolor='black')
ax7.set_title('Age distribution of Titanic Passengers')
ax7.set_xlabel('Age')
ax7.set_ylabel('Count')
ax7.grid(alpha=0.3)
st.pyplot(fig7)

# Survival by Passenger Class
fig8, ax8 = plt.subplots(figsize=(10, 6))
df.groupby('pclass')['survived'].mean().plot(kind='bar', color=['red', 'green', 'blue'], ax=ax8)
ax8.set_title('Survival with respect to Passenger Class')
ax8.set_xlabel('Passenger Class')
ax8.set_ylabel('Survival Rate')
ax8.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
st.pyplot(fig8)

# Summary Statistics
st.markdown("### Summary Statistics")
summary_stats = df.describe()
st.dataframe(summary_stats)

st.markdown("### Distribution Analysis")
st.write("Skewness of age and fare:")
st.write(df[['age', 'fare']].skew())

st.write("Kurtosis of age and fare:")
st.write(df[['age', 'fare']].kurtosis())

st.markdown("### Survival Rates by Categories")
st.write("By Gender:")
st.dataframe(df.groupby('sex')['survived'].mean())

st.write("By Class:")
st.dataframe(df.groupby('pclass')['survived'].mean())

st.write("By Embarkation Port:")
st.dataframe(df.groupby('embark_town')['survived'].mean())

# Boxplot without outliers (fare)
st.markdown("### Fare Boxplot Without Outliers")
fig9, ax9 = plt.subplots()
sns.boxplot(data=df_fare_no_outliers, ax=ax9)
st.pyplot(fig9)

# Crosstabs and Dropping Redundant Columns
st.markdown("### Removing Redundant Columns")

st.write("Survived vs Alive Crosstab:")
st.dataframe(pd.crosstab(df['survived'], df['alive']))

df = df.drop('alive', axis=1)
st.success("Dropped 'alive' column.")

st.write("Sex vs Who Crosstab:")
st.dataframe(pd.crosstab(df['sex'], df['who']))
df = df.drop('who', axis=1)
st.success("Dropped 'who' column.")

st.write("Pclass vs Class Crosstab:")
st.dataframe(pd.crosstab(df['pclass'], df['class']))
df = df.drop('class', axis=1)
st.success("Dropped 'class' column.")

st.write("SibSp & Parch vs Alone Crosstab:")
st.dataframe(pd.crosstab(index=[df['sibsp'], df['parch']], columns=df['alone']))
df = df.drop(['alone', 'adult_male', 'embark_town'], axis=1)
st.success("Dropped 'alone', 'adult_male', and 'embark_town' columns.")

st.markdown("### Starting Encoding for the Continuation of the Analysis")

# Data types
st.markdown("#### Data Types Before Encoding")
st.dataframe(df.dtypes)

# Sampling categorical columns
st.markdown("#### Sample Values from Categorical Columns")
st.write("Sample from 'sex':")
st.write(df['sex'].sample(5))
st.write("Sample from 'embarked':")
st.write(df['embarked'].sample(5))

st.markdown("#### One-Hot Encoding of Nominal Categorical Columns ('sex', 'embarked')")
df_encoded = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)
st.dataframe(df_encoded.head())

st.markdown("##### Changing boolean columns to integers")
for col in df_encoded.select_dtypes(include='bool'):
    df_encoded[col] = df_encoded[col].astype(int)

# Correlation Matrix
st.markdown("### Correlation Matrix")
corr_matrix = df_encoded.corr()
fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
ax_corr.set_title('Correlation Matrix')
st.pyplot(fig_corr)

# Covariance Matrix
st.markdown("### Covariance Matrix")
cov_matrix = df_encoded.cov()
fig_cov, ax_cov = plt.subplots(figsize=(12, 8))
sns.heatmap(cov_matrix, annot=True, cmap='viridis', fmt=".2f", ax=ax_cov)
ax_cov.set_title('Covariance Matrix')
st.pyplot(fig_cov)

# Spearman Correlation Matrix
st.markdown("### Spearman Correlation Matrix (Non-Parametric)")
spearman_corr = df_encoded.corr(method='spearman')
fig_spear, ax_spear = plt.subplots(figsize=(12, 8))
sns.heatmap(spearman_corr, annot=True, cmap='Blues', fmt=".2f", ax=ax_spear)
ax_spear.set_title('Spearman Correlation Matrix')
st.pyplot(fig_spear)

### Feature Engineering
st.markdown("### Feature Engineering")
df_encoded = df_encoded.drop_duplicates().copy()
st.write("Duplicates:", df_encoded.duplicated().sum())
st.write("Shape after dropping duplicates:", df_encoded.shape)

# Creating new features to reduce complexity
st.write("Sample data:")
st.dataframe(df_encoded.sample(5))

# Sex and pclass shows survival patterns
st.write("Survival patterns by sex and pclass:")
st.dataframe(df_encoded.groupby(['sex_male', 'pclass'])['survived'].mean())

# Family size feature
df_encoded['family'] = df_encoded['parch'] + df_encoded['sibsp']
st.write("Survival by family size:")
st.dataframe(df_encoded.groupby('family')['survived'].mean())
st.dataframe(df_encoded.head())

# Checking for highly correlated features
st.markdown("### Feature Correlation Heatmap")
fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', ax=ax_feat)
ax_feat.set_title("Feature Correlation Heatmap")
st.pyplot(fig_feat)

# Drop 'sibsp' and 'parch' after creating 'family'
df_encoded = df_encoded.drop(['sibsp', 'parch'], axis=1)

# Splitting data
st.markdown("### Data Splitting and Preprocessing")
x = df_encoded.drop('survived', axis=1)
y = df_encoded['survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

st.write(f"Training set shape: {x_train.shape}")
st.write(f"Test set shape: {x_test.shape}")

# Log transform to reduce skewness
log = FunctionTransformer(np.log1p, validate=True)
x_train['fare'] = log.fit_transform(x_train[['fare']])
x_test['fare'] = log.transform(x_test[['fare']])

# Standardization
sc = StandardScaler()
x_train[['fare', 'age', 'family']] = sc.fit_transform(x_train[['fare', 'age', 'family']])
x_test[['fare', 'age', 'family']] = sc.transform(x_test[['fare', 'age', 'family']])

st.write("Preprocessed training data:")
st.dataframe(x_train.head())

### Feature Selection
st.markdown("### Feature Selection")

# Correlation matrix
corr_matrix = x_train.corr().abs()
fig_multi, ax_multi = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, ax=ax_multi)
ax_multi.set_title("Correlation Matrix for Multicollinearity Check")
st.pyplot(fig_multi)

# Recursive Feature Elimination (RFE)
model = LinearRegression()
rfe = RFE(model, n_features_to_select=5)
rfe.fit(x_train, y_train)

selected_columns = x_train.columns[rfe.support_]
st.write("Selected features using RFE:", selected_columns.tolist())

# Lasso Feature Selection
lasso = LassoCV().fit(x_train, y_train)
selected = x_train.columns[(lasso.coef_ != 0)]
st.write("Selected features using Lasso:", selected.tolist())

# Using Lasso-selected features
x_train_selected = x_train[selected]
x_test_selected = x_test[selected]

st.write("Final selected features shape:", x_train_selected.shape)
st.dataframe(x_train_selected.head())