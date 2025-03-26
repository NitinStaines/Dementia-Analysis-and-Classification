import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(r'C:\Users\nitin\OneDrive\Desktop\dav\dementia_dataset.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Reading the CSV file into a DataFrame
df1 = pd.read_csv(r'C:\Users\nitin\OneDrive\Desktop\dav\dementia_dataset.csv')

# Printing the shape of the DataFrame
print(df1.shape)

# Checking unique values in column 'Group'
df1['Group'].unique()

# Filtering rows in df1 where the 'Group' column is equal to 'Converted' and assigning them to df2
df2 = df1.loc[df1['Group'] == 'Converted']

# Dropping the rows from df1 that have been assigned to df2 using the corresponding index values
df1 = df1.drop(df2.index)

df2.head(10)

# Creating a new column 'Last_Visit' to identify the last visit for each patient
df2['Last_Visit'] = df2.groupby('Subject ID')['Visit'].transform('max')

# Updating the 'Group' column based on 'Visit' and 'Last_Visit' conditions
df2.loc[df2['Visit'] < df2['Last_Visit'], 'Group'] = 'Nondemented'
df2.loc[df2['Visit'] == df2['Last_Visit'], 'Group'] = 'Demented'

# Dropping the 'Last_Visit' column
df2.drop('Last_Visit', axis=1, inplace=True)

# Displaying the updated DataFrame
df2.head(5)

# Combining the DataFrames df1 and df2
frames = [df1, df2]
df = pd.concat(frames)

df['Group'].unique()

# Renaming the 'M/F' column to 'Gender' in the DataFrame
df.rename(columns={'M/F': 'Gender'}, inplace=True)

# Display the current column names
print(df.columns)

# Drop unnecessary columns from the DataFrame if they exist
columns_to_drop = ['Subject ID', 'MRI ID', 'Hand', 'Visit', 'MR Delay']
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df.drop(columns=existing_columns_to_drop, inplace=True)

# Imputing missing values in the 'SES' column with the mode
df.SES.fillna(df.SES.mode()[0], inplace=True)

# Imputing missing values in the 'MMSE' column with the mean
df.MMSE.fillna(df.MMSE.mean(), inplace=True)

df.isna().sum()

print(df.columns)

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Your preprocessed DataFrame `df` goes here

# Adding title for the dashboard
st.title("Dementia Analysis Dashboard")

tab1, tab2, tab3 = st.tabs(["Visual Analysis", "Hypothesis Testing", "Risk Prediction"])

# === Tab 1: Visual Analysis (all plots) ===
with tab1:
    st.header("Exploratory Data Analysis")

    # Create two columns for the first row (plots side by side)
    col1 = st.columns(1)[0]

    # Plot 1: Countplot for 'Group'
    with col1:
        st.subheader("Distribution of Dementia Groups")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Group', palette='Set2', ax=ax1)
        ax1.set_title('Dementia Group')
        st.pyplot(fig1)
        # Adding description (subtext) for Plot 1
        st.text("This plot shows the distribution of subjects in different dementia groups (Demented vs Nondemented).")

    col2 = st.columns(1)[0]
    # Plot 2: Distribution of 'Group' by 'Gender'
    with col2:
        st.subheader("Distribution of Dementia Group by Gender")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='Group', palette='Set2', hue='Gender', ax=ax2)
        ax2.set_title('Dementia Group by Gender')
        st.pyplot(fig2)
        # Adding description (subtext) for Plot 2
        st.text("""Dementia Group by Gender:
                
    The distribution of gender across the 'Nondemented' and 'Demented' groups is unbalanced.
    Nondemented: More females are present than males.
    Demented: The male count is higher compared to females.
                
    Inference: This suggests a potential gender difference in the prevalence of dementia within the dataset. It may be useful to explore further whether gender plays a significant role in the progression or risk of dementia.""")

    # Create another row for the next two plots
    col3 = st.columns(1)[0]

    # Plot 3: Distribution of Education for each 'Gender' and 'Group'
    with col3:
        st.subheader("Distribution of Education by Gender and Group")
        fig3 = sns.displot(data=df, x='EDUC', col='Gender', palette='Set2', hue='Group', kind='kde', height=5, aspect=1.5)
        st.pyplot(fig3)
        # Adding description (subtext) for Plot 3
        st.text("""Distribution of Education by Gender and Group:

    For both genders, the non-demented group tends to have slightly higher levels of education compared to the demented group. This difference is more noticeable in females than in males.
    In males, there is a broader distribution of education levels across both groups, but the non-demented group still shows a slight tendency toward higher education levels.
    In females, there is a noticeable peak in education levels for the non-demented group around 15 years of education, whereas the demented group shows a peak at a lower education level.
    Inference: Higher education levels might be associated with a lower likelihood of dementia, particularly noticeable in the female subgroup.""")

    col4 = st.columns(1)[0]
    # Plot 4: Distribution of Age for each Group
    with col4:
        st.subheader("Distribution of Age by Group")
        fig4 = sns.displot(data=df, x='Age', hue='Group', kind="kde", palette='Set2', height=5, aspect=1.5)
        st.pyplot(fig4)
        # Adding description (subtext) for Plot 4
        st.text("""Distribution of Age by Group:
                
        The demented group has a distribution skewed towards older ages compared to the non-demented group. The peak age for the demented group is around 80, while the non-demented group has a peak slightly younger, around 75.
        Both distributions taper off at higher ages, but the non-demented group has a wider spread across younger ages.
        Inference: Higher age is associated with a higher likelihood of dementia, with a notable difference between the typical ages of demented and non-demented individuals. This suggests that age is a strong factor in dementia risk.""")

    # Create another row for the final plot
    col5 = st.columns(1)[0]

    # Plot 5: Correlation Matrix Heatmap
    with col5:
        st.subheader("Correlation Matrix Heatmap")
        fig5, ax5 = plt.subplots(figsize=(10, 8))  # Adjust the size of the heatmap
        sns.heatmap(df.corr(numeric_only=True), vmin=-1, vmax=1, cmap='coolwarm', annot=True, fmt=".2f", ax=ax5)
        ax5.set_title("Correlation Matrix")
        st.pyplot(fig5)
        # Adding description (subtext) for Plot 5
        st.text("""Key Correlations:
                
    ASF (Atlas Scaling Factor) is almost perfectly inversely correlated with eTIV (Estimated Total Intracranial Volume) (-0.99), indicating a strong negative relationship.
    MMSE (Mini-Mental State Examination) has a strong negative correlation with CDR (Clinical Dementia Rating) (-0.68), suggesting that lower MMSE scores are associated with higher dementia severity.
    EDUC (Years of Education) and SES (Socioeconomic Status) are negatively correlated (-0.69), indicating that individuals with higher education may have better socioeconomic status.
    Age shows a moderate negative correlation with nWBV (Normalized Whole Brain Volume) (-0.52), which suggests brain volume decreases with age.
    Inference: The strong correlation between MMSE and CDR implies that MMSE scores could be a reliable indicator of dementia severity.
    The relationship between brain volume and age might indicate neurodegeneration associated with aging.""")

    col6 = st.columns(1)[0]
    # Plot 6: Scatter plot of MMSE vs CDR colored by 'Group'
    with col6:
        st.subheader("Relationship Between MMSE and CDR")
        fig6, ax6 = plt.subplots()
        sns.scatterplot(data=df, x='MMSE', y='CDR', palette='Set2', hue='Group', ax=ax6)
        ax6.set_title('MMSE vs CDR')
        st.pyplot(fig6)
        # Adding description (subtext) for Plot 6
        st.text("""MMSE vs. CDR Scatter Plot:

    Patients with higher MMSE scores (close to 30) are predominantly classified as Nondemented, while those with lower scores (below 15) are generally classified as Demented.
    There are some overlaps for scores between 15 and 25, indicating mixed classification where patients might fall into either category.
    Inference: The MMSE score is a strong differentiator for dementia classification, but the overlap suggests it might not be the sole factor, especially in borderline cases.
    Further analysis could explore the impact of other variables (like education or brain volume) to improve classification accuracy.""")



import scipy.stats as stats

# === Tab 2: Hypothesis Testing ===
with tab2:
    st.header("Hypothesis Testing")
    
    # Extracting CDR scores for 'Demented' and 'Nondemented' groups
    demented_group_cdr = df[df['Group'] == 'Demented']['CDR']
    nondemented_group_cdr = df[df['Group'] == 'Nondemented']['CDR']
    
    # Performing Mann-Whitney U test (non-parametric test)
    u_stat, p_value = stats.mannwhitneyu(demented_group_cdr, nondemented_group_cdr)
    
    # Displaying the test result
    st.subheader("Mann-Whitney U Test Results for CDR Scores")
    st.write(f"U-statistic: {u_stat}")
    st.write(f"P-value: {p_value}")

    # Interpreting the p-value
    if p_value < 0.05:
        st.write("**Result:** Reject the null hypothesis. There is a significant difference in CDR scores between Demented and Nondemented groups.")
    else:
        st.write("**Result:** Fail to reject the null hypothesis. There is no significant difference in CDR scores between Demented and Nondemented groups.")
    

    # ===================
    # Independent T-Test: Age comparison between Demented and Nondemented groups
    st.subheader("Independent T-Test Results for Age")

    demented_group_age = df[df['Group'] == 'Demented']['Age']
    nondemented_group_age = df[df['Group'] == 'Nondemented']['Age']

    # Performing Independent T-Test
    t_stat, p_value_age = stats.ttest_ind(demented_group_age, nondemented_group_age)
    
    st.write(f"T-statistic: {t_stat}")
    st.write(f"P-value: {p_value_age}")

    # Interpreting the p-value
    if p_value_age < 0.05:
        st.write("**Result:** Reject the null hypothesis. There is a significant difference in Age between Demented and Nondemented groups.")
    else:
        st.write("**Result:** Fail to reject the null hypothesis. There is no significant difference in Age between Demented and Nondemented groups.")
    
    # ===================
    # Chi-Square Test: Gender vs Group (Demented vs Nondemented)
    st.subheader("Chi-Square Test Results for Gender vs Group")

    # Contingency table for Gender vs Group
    contingency_table = pd.crosstab(df['Gender'], df['Group'])
    
    # Performing Chi-Square Test
    chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    
    st.write(f"Chi2 Statistic: {chi2_stat}")
    st.write(f"P-value: {p_value_chi2}")

    # Interpreting the p-value
    if p_value_chi2 < 0.05:
        st.write("**Result:** Reject the null hypothesis. There is an association between Gender and Group.")
    else:
        st.write("**Result:** Fail to reject the null hypothesis. There is no association between Gender and Group.")
    
    # ===================
    # Spearman's Rank Correlation: CDR vs Age (monotonic relationship)
    st.subheader("Spearman's Rank Correlation Results for CDR vs Age")

    # Performing Spearman's rank correlation test
    correlation, p_value_corr = stats.spearmanr(df['CDR'], df['Age'])
    
    st.write(f"Correlation coefficient: {correlation}")
    st.write(f"P-value: {p_value_corr}")

    # Interpreting the p-value
    if p_value_corr < 0.05:
        st.write("**Result:** Reject the null hypothesis. There is a significant monotonic relationship between CDR and Age.")
    else:
        st.write("**Result:** Fail to reject the null hypothesis. There is no significant monotonic relationship between CDR and Age.")
    
    # ===================
    # Kolmogorov-Smirnov Test: Testing the distribution of CDR for Demented and Nondemented groups
    st.subheader("Kolmogorov-Smirnov Test for CDR Distribution")

    # Performing the Kolmogorov-Smirnov test
    ks_stat, p_value_ks = stats.ks_2samp(demented_group_cdr, nondemented_group_cdr)
    
    st.write(f"KS Statistic: {ks_stat}")
    st.write(f"P-value: {p_value_ks}")

    # Interpreting the p-value
    if p_value_ks < 0.05:
        st.write("**Result:** Reject the null hypothesis. The distributions of CDR scores between Demented and Nondemented groups are significantly different.")
    else:
        st.write("**Result:** Fail to reject the null hypothesis. The distributions of CDR scores between Demented and Nondemented groups are not significantly different.")
    

    # ===================
    # Inference Notes for Each Test
    st.text("""
    - The Mann-Whitney U test is used to compare CDR scores between Demented and Nondemented groups without assuming normality.
    - The Independent T-Test checks if there is a significant difference in Age between the two groups.
    - The Chi-Square Test determines if there is an association between Gender and Group.
    - Spearman's Rank Correlation tests if there is a monotonic relationship between CDR and Age.
    - The Kolmogorov-Smirnov Test checks if the distributions of CDR for Demented and Nondemented groups are different.
    """)


# Check if 'ASF' column exists before dropping
if 'ASF' in df.columns:
    df.drop(columns=['ASF'], inplace=True)
else:
    print("Column 'ASF' not found in the DataFrame.")

# Importing the necessary library for label encoding
from sklearn.preprocessing import LabelEncoder

# Creating an instance of the LabelEncoder class
le = LabelEncoder()

# Encoding the 'Gender' column in the DataFrame
df.Gender = le.fit_transform(df.Gender.values)

# Printing the mapping of encoded values to original classes for 'Gender'
print(f'Sex:\n0 : {le.classes_[0]}\n1 : {le.classes_[1]}\n\n')

df.Group = le.fit_transform(df.Group.values)
print(f'Group:\n0 : {le.classes_[0]}\n1 : {le.classes_[1]}')

# Importing the necessary library for train-test split
from sklearn.model_selection import train_test_split

# Assigning the 'Group' column as the target variable
y = df.Group

# Assigning the remaining columns as the features
X = df.drop(['Group'], axis=1)

# Performing the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Importing the necessary library for Random Forest classification
from sklearn.ensemble import RandomForestClassifier

# Creating an instance of the RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Fitting the model to the training data
model.fit(X_train, y_train)

# Predicting the target variable for the test data
y_hat = model.predict(X_test)

# Importing the necessary libraries for performance evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

print('Accuracy Score:', accuracy_score(y_test, y_hat))
print('Precision:', precision_score(y_test, y_hat, average='binary'))
print('Recall:', recall_score(y_test, y_hat, average='binary'))
print('F1 Score:', f1_score(y_test, y_hat, average='binary') )


with tab3:
    st.title("Dementia Risk Prediction")

    # Collecting user input using Streamlit UI components
    st.header("Enter Patient Details")
    
    gender = st.selectbox("Gender", ['M', 'F'])
    age = st.slider("Age", 1, 100, value=70)
    educ = st.slider("Years of Education", 0, 20, value=12)
    ses = st.slider("Socioeconomic Status (SES)", 1.0, 5.0, step=0.5, value=2.0)
    mmse = st.slider("MMSE Score", 0, 30, value=28)
    cdr = st.slider("CDR Score", 0.0, 3.0, step=0.1, value=0.5)
    etiv = st.number_input("Estimated Total Intracranial Volume (eTIV)", min_value=1000, max_value=2000, value=1600)
    nwbv = st.slider("Normalized Whole Brain Volume (nWBV)", 0.0, 1.0, step=0.01, value=0.7)

    # Button to run the prediction
    if st.button("Predict"):
        # Function to predict dementia
        import pandas as pd

        def predict_dementia(gender, age, educ, ses, mmse, cdr, etiv, nwbv):
            # Encoding gender using LabelEncoder
            gender_encoded = 0  # Default value for Male
            
            if gender.lower() == 'f':
                gender_encoded = 1  # Set to 1 for Female
            
            # Creating a DataFrame with the input details
            input_data = pd.DataFrame({
                'Gender': [gender_encoded],
                'Age': [age],
                'EDUC': [educ],
                'SES': [ses],
                'MMSE': [mmse],
                'CDR': [cdr],
                'eTIV': [etiv],
                'nWBV': [nwbv]
            })

            # Predicting dementia group using the trained model
            probability_demented = model.predict_proba(input_data)[:, 1][0]
            probability_percentage = probability_demented * 100
            
            # Categorizing into Low, Mild, and High risk categories
            if probability_demented < 0.33:
                risk_category = "Low"
            elif probability_demented < 0.66:
                risk_category = "Mild"
            else:
                risk_category = "High"

            # Formatting the probability as a percentage
            probability_formatted = f"{probability_percentage:.2f}%"
            
            return probability_formatted, risk_category

        # Run the prediction function
        probability, risk_category = predict_dementia(gender, age, educ, ses, mmse, cdr, etiv, nwbv)
        
        # Display the results
        st.subheader("Prediction Results")
        st.write(f"**Probability of being demented:** {probability}")
        st.write(f"**Risk Category:** {risk_category}")