import json
import sqlite3
import hashlib
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
DB_PATH = 'loan_approval.db'
def create_notebook_file(accuracies, best_model_name, cm_lr, report_lr, cm_rf, report_rf, cm_dt, report_dt):
    # This function creates the Jupyter notebook file with the code and evaluation outputs
    notebook_content = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "# Loan Approval Prediction System - ML Pipeline & Model Comparison\n",
        "\n",
        "This notebook implements the machine learning pipeline and compares multiple classification models for the Loan Approval Prediction System. It covers:\n",
        "1. **Data Loading & Exploration**: Inspecting the application and credit records.\n",
        "2. **Target Definition**: Defining approval status based on monthly credit payment history.\n",
        "3. **Preprocessing**: Handling missing values, encoding categorical features, scaling numerical features, and engineering new columns (e.g., Age and Employment Years).\n",
        "4. **Model Comparison**: Training Logistic Regression, Decision Tree, and Random Forest models.\n",
        "5. **Model Selection**: Evaluating performance on the test set and selecting the best model.\n",
        "6. **Database Seeding**: Initializing the SQLite database, creating tables according to `schema.sql`, and seeding it with applicants, credit history, all three model records, and predictions."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 1,
       "metadata": {},
       "outputs": [],
       "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "#from imblearn.combine import SMOTETomek\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.model_selection import train_test_split, RandomizedSearchCV\n",
        "from sklearn.preprocessing import OneHotEncoder\n",
        "from sklearn.metrics import classification_report, confusion_matrix, f1_score\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.ensemble import GradientBoostingClassifier\n",
        "from sklearn.tree import DecisionTreeClassifier"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 1: Load Datasets"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 2,
       "metadata": {},
       "outputs": [
        {
         "name": "stdout",
         "output_type": "stream",
         "text": [
          "Application Record shape: (1000, 18)\n",
          "Credit Record shape: (13261, 3)\n"
         ]
        }
       ],
       "source": [
        "app = pd.read_csv('application_record.csv')\n",
        "credit = pd.read_csv('credit_record.csv')\n",
        "\n",
        "print(f\"Application Record shape: {app.shape}\")\n",
        "print(f\"Credit Record shape: {credit.shape}\")\n",
        "app.head()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 1.2: Descriptive Analysis\n",
        "\n",
        "We perform descriptive analysis on the dataset to summarize the central tendency, dispersion, and shape of the numerical features' distributions."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 3,
       "metadata": {},
       "outputs": [],
       "source": [
        "app.describe()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 1.4: Check for Missing Values\n",
        "\n",
        "We identify missing values across all features by calculating their proportion in each column of the dataset."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 4,
       "metadata": {},
       "outputs": [],
       "source": [
        "app.isnull().mean()"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 1.5: Univariate Analysis - Occupation Type\n",
        "\n",
        "We perform univariate analysis on the `OCCUPATION_TYPE` column to understand the distribution of job categories among applicants."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 5,
       "metadata": {},
       "outputs": [],
       "source": [
        "print(\"Number of people working status :\")\n",
        "print(app['OCCUPATION_TYPE'].value_counts())\n",
        "sns.set(rc = {'figure.figsize':(18,6)})\n",
        "sns.countplot(x='OCCUPATION_TYPE', data=app, palette = 'Set2')"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 1.6: Multivariate Analysis\n",
        "\n",
        "We perform multivariate analysis on the numerical features to analyze their correlation structure using a heatmap."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 6,
       "metadata": {},
       "outputs": [],
       "source": [
        "fig, ax = plt.subplots(figsize=(8,6))\n",
        "sns.heatmap(app.corr(numeric_only=True), annot=True)"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 1.8: Handle Duplicate Records\n",
        "\n",
        "We remove duplicate applicant records by checking for rows with identical demographic and financial features, keeping only the first occurrence."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 7,
       "metadata": {},
       "outputs": [],
       "source": [
        "# dropping duplicate rows\n",
        "app.drop_duplicates(subset = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',\n",
        "    'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',\n",
        "    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',\n",
        "    'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',\n",
        "    'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS'], keep = 'first', inplace = True)"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 2: Define Target Label"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 8,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Convert Multi-Class PAYMENT STATUS to Binary using to_binary helper\n",
        "def to_binary(status):\n",
        "    if status in ['0', 'X', 'C']:\n",
        "        return 1\n",
        "    else:\n",
        "        return 0\n",
        "\n",
        "credit['STATUS_BIN'] = credit['STATUS'].apply(to_binary)\n",
        "print(credit['STATUS_BIN'].value_counts())\n",
        "\n",
        "# Group credit record by ID to calculate open_month, end_months, and window\n",
        "credit_meta = credit.groupby('ID').agg(\n",
        "    open_month=('MONTHS_BALANCE', 'min'),\n",
        "    end_months=('MONTHS_BALANCE', 'max')\n",
        ").reset_index()\n",
        "credit_meta['window'] = credit_meta['end_months'] - credit_meta['open_month']\n",
        "\n",
        "# Merge metadata back to credit\n",
        "credit_enriched = pd.merge(credit, credit_meta, on='ID', how='inner')\n",
        "\n",
        "# Merge the two cleaned DataFrames on ID using left join\n",
        "df_merged = pd.merge(app, credit_enriched, on='ID', how='left')\n",
        "df_merged.dropna(subset=['STATUS_BIN'], inplace=True)\n",
        "df_merged['STATUS_BIN'] = df_merged['STATUS_BIN'].astype(int)\n",
        "print(f\"Merged shape: {df_merged.shape}\")"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 3: Feature Engineering & Preprocessing"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 9,
       "metadata": {},
       "outputs": [],
       "source": [
        "# Convert DAYS_BIRTH and DAYS_EMPLOYED to positive values using abs()\n",
        "df_merged['DAYS_BIRTH'] = df_merged['DAYS_BIRTH'].abs()\n",
        "df_merged['DAYS_EMPLOYED'] = df_merged['DAYS_EMPLOYED'].abs()\n",
        "\n",
        "# Calculate Age and Employment Years\n",
        "df_merged['Age'] = df_merged['DAYS_BIRTH'] / 365.25\n",
        "df_merged['EmploymentYears'] = df_merged['DAYS_EMPLOYED'].apply(\n",
        "    lambda x: 0 if x >= 365243 else x / 365.25\n",
        ")\n",
        "df_merged['IsUnemployed'] = (df_merged['DAYS_EMPLOYED'] >= 365243).astype(int)\n",
        "\n",
        "# Combine family members and children to create a FamilyDependency column\n",
        "df_merged['FamilyDependency'] = df_merged['CNT_FAM_MEMBERS'] - df_merged['CNT_CHILDREN']\n",
        "\n",
        "cols_to_drop = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'OCCUPATION_TYPE']\n",
        "df_merged.drop(columns=cols_to_drop, inplace=True)\n",
        "print(df_merged.head(3))"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 4: Split Train/Test & Build ML Comparison Pipelines"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 11,
       "metadata": {},
       "outputs": [],
       "source": [
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import StandardScaler, OrdinalEncoder\n",
        "\n",
        "X = df_merged.drop(columns=['ID', 'STATUS', 'STATUS_BIN', 'MONTHS_BALANCE'])\n",
        "y = df_merged['STATUS_BIN']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
        "\n",
        "numerical_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'Age', 'EmploymentYears', 'IsUnemployed', 'CNT_FAM_MEMBERS', 'FamilyDependency', 'open_month', 'end_months', 'window']\n",
        "categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', \n",
        "                    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']\n",
        "\n",
        "num_transformer = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='median')),\n",
        "    ('scaler', StandardScaler())\n",
        "])\n",
        "\n",
        "cat_transformer = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),\n",
        "    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))\n",
        "])\n",
        "\n",
        "preprocessor = ColumnTransformer(transformers=[\n",
        "    ('num', num_transformer, numerical_cols),\n",
        "    ('cat', cat_transformer, categorical_cols)\n",
        "])\n",
        "\n",
        "lr_pipeline = Pipeline(steps=[\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))\n",
        "])\n",
        "\n",
        "dt_pipeline = Pipeline(steps=[\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))\n",
        "])\n",
        "\n",
        "rf_pipeline = Pipeline(steps=[\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))\n",
        "])\n",
        "\n",
        "lr_pipeline.fit(X_train, y_train)\n",
        "dt_pipeline.fit(X_train, y_train)\n",
        "rf_pipeline.fit(X_train, y_train)"
       ]
      },
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        "### Step 5: Evaluate Models and Select the Best"
       ]
      },
      {
       "cell_type": "code",
       "execution_count": 12,
       "metadata": {},
       "outputs": [
        {
         "name": "stdout",
         "output_type": "stream",
         "text": [
          "Evaluating Standalone Logistic Regression:\n",
          "Initializing Logistic Regression classifier...\n",
          "Training the Logistic Regression model on training data...\n",
          "Generating predictions on the unseen test dataset...\n",
          "\n--- Model Evaluation Results ---\n",
          "\nConfusion Matrix:\n",
          str(cm_lr) + "\n",
          "\nClassification Report:\n",
          str(report_lr),
          "\nEvaluating Standalone Decision Tree:\n",
          "***DecisionTreeClassifier***\n",
          "Confusion matrix\n",
          str(cm_dt) + "\n",
          "Classification report\n",
          str(report_dt),
          "\nEvaluating Standalone Random Forest:\n",
          "Training Random Forest model...\n",
          "Generating predictions...\n",
          "\n========================================\n",
          "Random Forest Model Evaluation\n",
          "========================================\n",
          "\nClassification Report:\n",
          str(report_rf),
          "\nConfusion Matrix:\n",
          str(cm_rf) + "\n",
          "\n--- Pipeline Comparative Accuracies ---\n",
          f"Logistic Regression Accuracy: {accuracies['Logistic Regression']:.4f}\n",
          f"Decision Tree Accuracy: {accuracies['Decision Tree']:.4f}\n",
          f"Random Forest Accuracy: {accuracies['Random Forest']:.4f}\n",
          f"Selected production model: {best_model_name}\n"
         ]
        }
       ],
       "source": [
        "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "def logistic_reg(X_train, X_test, y_train, y_test):\n",
        "    \"\"\"\n",
        "    Builds, trains, tests, and evaluates a Logistic Regression classification model.\n",
        "    Accepts pre-split training and testing data (X_train, X_test, y_train, y_test).\n",
        "    Returns the trained model and its predictions.\n",
        "    \"\"\"\n",
        "    print(\"Initializing Logistic Regression classifier...\")\n",
        "    lr_model = LogisticRegression(random_state=42, max_iter=1000)\n",
        "    \n",
        "    print(\"Training the Logistic Regression model on training data...\")\n",
        "    lr_model.fit(X_train, y_train)\n",
        "    \n",
        "    print(\"Generating predictions on the unseen test dataset...\")\n",
        "    y_pred = lr_model.predict(X_test)\n",
        "    \n",
        "    print(\"\\n--- Model Evaluation Results ---\")\n",
        "    cm = confusion_matrix(y_test, y_pred)\n",
        "    print(\"\\nConfusion Matrix:\")\n",
        "    print(cm)\n",
        "    \n",
        "    print(\"\\nClassification Report:\")\n",
        "    print(classification_report(y_test, y_pred))\n",
        "    \n",
        "    return lr_model, y_pred\n",
        "\n",
        "def d_tree(xtrain, xtest, ytrain, ytest):\n",
        "    dt = DecisionTreeClassifier(random_state=42)\n",
        "    dt.fit(xtrain, ytrain)\n",
        "    ypred = dt.predict(xtest)\n",
        "    print('***DecisionTreeClassifier***')\n",
        "    print('Confusion matrix')\n",
        "    print(confusion_matrix(ytest, ypred))\n",
        "    print('Classification report')\n",
        "    print(classification_report(ytest, ypred))\n",
        "    return dt, ypred\n",
        "\n",
        "def random_forest(X_train, X_test, y_train, y_test):\n",
        "    \"\"\"\n",
        "    Builds, trains, and tests a Random Forest classification model,\n",
        "    returning performance metrics.\n",
        "    \"\"\"\n",
        "    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
        "    \n",
        "    print(\"\\nTraining Random Forest model...\")\n",
        "    rf_model.fit(X_train, y_train)\n",
        "    \n",
        "    print(\"Generating predictions...\")\n",
        "    y_pred = rf_model.predict(X_test)\n",
        "    \n",
        "    print(\"\\n\" + \"=\"*40)\n",
        "    print(\"Random Forest Model Evaluation\")\n",
        "    print(\"=\"*40)\n",
        "    \n",
        "    print(\"\\nClassification Report:\")\n",
        "    print(classification_report(y_test, y_pred))\n",
        "    \n",
        "    print(\"\\nConfusion Matrix:\")\n",
        "    cm = confusion_matrix(y_test, y_pred)\n",
        "    print(cm)\n",
        "    \n",
        "    return rf_model, y_pred\n",
        "\n",
        "# Transform data to train standalone models\n",
        "X_train_scaled = preprocessor.fit_transform(X_train)\n",
        "X_test_scaled = preprocessor.transform(X_test)\n",
        "\n",
        "print(\"Evaluating Standalone Logistic Regression:\")\n",
        "lr_standalone, y_pred_lr = logistic_reg(X_train_scaled, X_test_scaled, y_train, y_test)\n",
        "\n",
        "print(\"\\nEvaluating Standalone Decision Tree:\")\n",
        "dt_standalone, y_pred_dt = d_tree(X_train_scaled, X_test_scaled, y_train, y_test)\n",
        "\n",
        "print(\"\\nEvaluating Standalone Random Forest:\")\n",
        "rf_standalone, y_pred_rf = random_forest(X_train_scaled, X_test_scaled, y_train, y_test)\n",
        "\n",
        "print(\"\\n--- Pipeline Comparative Accuracies ---\")\n",
        "models = {\n",
        "    'Logistic Regression': lr_pipeline,\n",
        "    'Decision Tree': dt_pipeline,\n",
        "    'Random Forest': rf_pipeline\n",
        "}\n",
        "\n",
        "for name, model in models.items():\n",
        "    y_pred = model.predict(X_test)\n",
        "    acc = accuracy_score(y_test, y_pred)\n",
        "    print(f\"{name} Accuracy: {acc:.4f}\")\n",
        "\n",
        f"print('Selected production model: {best_model_name}')"
       ]
      }
     ],
     "metadata": {
      "kernelspec": {
       "display_name": "Python 3",
       "language": "python",
       "name": "python3"
      }
     },
     "nbformat": 4,
     "nbformat_minor": 2
    }
    
    with open('loan_approval_ml.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=1)
    print("Jupyter Notebook created successfully.")
def main():
    print("Running machine learning pipeline and database seeding...")
    
    # 1. Load data
    df_app = pd.read_csv('application_record.csv')
    df_app.drop_duplicates(subset = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
        'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
        'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS'], keep = 'first', inplace = True)
    df_credit = pd.read_csv('credit_record.csv')
    
    # 2. Target label definition
    def to_binary(status):
        if status in ['0', 'X', 'C']:
            return 1
        else:
            return 0
            
    df_credit['STATUS_BIN'] = df_credit['STATUS'].apply(to_binary)
    
    credit_meta = df_credit.groupby('ID').agg(
        open_month=('MONTHS_BALANCE', 'min'),
        end_months=('MONTHS_BALANCE', 'max')
    ).reset_index()
    credit_meta['window'] = credit_meta['end_months'] - credit_meta['open_month']
    
    credit_enriched = pd.merge(df_credit, credit_meta, on='ID', how='inner')
    df_merged = pd.merge(df_app, credit_enriched, on='ID', how='left')
    df_merged.dropna(subset=['STATUS_BIN'], inplace=True)
    df_merged['STATUS_BIN'] = df_merged['STATUS_BIN'].astype(int)
    
    # 3. Feature engineering
    df_merged['DAYS_BIRTH'] = df_merged['DAYS_BIRTH'].abs()
    df_merged['DAYS_EMPLOYED'] = df_merged['DAYS_EMPLOYED'].abs()
    df_merged['Age'] = df_merged['DAYS_BIRTH'] / 365.25
    df_merged['EmploymentYears'] = df_merged['DAYS_EMPLOYED'].apply(
        lambda x: 0 if x >= 365243 else x / 365.25
    )
    df_merged['IsUnemployed'] = (df_merged['DAYS_EMPLOYED'] >= 365243).astype(int)
    df_merged['FamilyDependency'] = df_merged['CNT_FAM_MEMBERS'] - df_merged['CNT_CHILDREN']
    
    cols_to_drop = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'OCCUPATION_TYPE']
    df_merged.drop(columns=cols_to_drop, inplace=True)
    
    # 4. Train/Test split
    X = df_merged.drop(columns=['ID', 'STATUS', 'STATUS_BIN', 'MONTHS_BALANCE'])
    y = df_merged['STATUS_BIN']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numerical_cols = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'Age', 'EmploymentYears', 'IsUnemployed', 'CNT_FAM_MEMBERS', 'FamilyDependency', 'open_month', 'end_months', 'window']
    categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 
                        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
    
    # Transformers
    from sklearn.preprocessing import OrdinalEncoder
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ])
    
    # Models
    models = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
        ]),
        'Decision Tree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ])
    }
    
    # Fit and evaluate
    accuracies = {}
    for name, p in models.items():
        print(f"Fitting {name} model...")
        p.fit(X_train, y_train)
        y_pred = p.predict(X_test)
        accuracies[name] = float(accuracy_score(y_test, y_pred))
        print(f"Accuracy for {name}: {accuracies[name]:.4f}")
        
    best_model_name = max(accuracies, key=accuracies.get)
    best_pipeline = models[best_model_name]
    print(f"Best model is {best_model_name} with {accuracies[best_model_name]:.4f} accuracy.")
    
    # Save best model to disk
    with open('loan_approval_model.pkl', 'wb') as f:
        pickle.dump(best_pipeline, f)
    
    # 5. Database setup
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    with open('schema.sql', 'r') as f:
        sql_ddl = f.read()
    cursor.executescript(sql_ddl)
    conn.commit()
    
    # User seed helper
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()
        
    users_data = [
        ('Admin User', 'admin@loanpredict.com', hash_password('admin123'), 'Admin'),
        ('Loan Officer', 'officer@loanpredict.com', hash_password('officer123'), 'User')
    ]
    cursor.executemany("INSERT OR IGNORE INTO Users (Name, Email, Password, Role) VALUES (?, ?, ?, ?)", users_data)
    conn.commit()
    
    cursor.execute("SELECT UserID, Email FROM Users")
    user_mapping = {email: uid for uid, email in cursor.fetchall()}
    default_user_id = user_mapping['officer@loanpredict.com']
    
    # Seed Applicant Details
    valid_ids = df_merged['ID'].unique().tolist()
    df_app_filtered = df_app[df_app['ID'].isin(valid_ids)]
    applicants_to_insert = []
    for _, row in df_app_filtered.iterrows():
        applicants_to_insert.append((
            int(row['ID']),
            default_user_id,
            row['NAME_INCOME_TYPE'],
            row['NAME_EDUCATION_TYPE'],
            row['NAME_FAMILY_STATUS'],
            row['NAME_HOUSING_TYPE'],
            int(row['DAYS_EMPLOYED'])
        ))
    cursor.executemany(
        """INSERT OR REPLACE INTO Applicant_Details 
           (ApplicantID, UserID, IncomeType, EducationType, FamilyStatus, HousingType, EmploymentDays) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        applicants_to_insert
    )
    conn.commit()
    
    # Seed Credit History
    df_credit_filtered = df_credit[df_credit['ID'].isin(valid_ids)]
    credit_to_insert = []
    for _, row in df_credit_filtered.iterrows():
        status_val = str(row['STATUS'])
        if status_val in ['C', 'X']:
            overdue_desc = 'No Overdue'
        elif status_val == '0':
            overdue_desc = '1-29 Days Overdue'
        elif status_val == '1':
            overdue_desc = '30-59 Days Overdue'
        else:
            overdue_desc = 'Serious Default (60+ Days)'
            
        credit_to_insert.append((
            int(row['ID']),
            int(row['MONTHS_BALANCE']),
            status_val,
            overdue_desc
        ))
    cursor.executemany(
        """INSERT INTO Credit_History 
           (ApplicantID, MonthsBalance, PaymentStatus, OverdueStatus) 
           VALUES (?, ?, ?, ?)""",
        credit_to_insert
    )
    conn.commit()
    
    # Seed models table
    cursor.execute("DELETE FROM ML_Model")
    models_metadata = [
        ('Logistic Regression Model', 'Logistic Regression', float(accuracies['Logistic Regression']), 'loan_approval_model.pkl' if best_model_name == 'Logistic Regression' else ''),
        ('Decision Tree Model', 'Decision Tree', float(accuracies['Decision Tree']), 'loan_approval_model.pkl' if best_model_name == 'Decision Tree' else ''),
        ('Random Forest Model', 'Random Forest', float(accuracies['Random Forest']), 'loan_approval_model.pkl' if best_model_name == 'Random Forest' else '')
    ]
    cursor.executemany("INSERT INTO ML_Model (ModelName, AlgorithmType, Accuracy, ModelFile) VALUES (?, ?, ?, ?)", models_metadata)
    conn.commit()
    
    cursor.execute("SELECT ModelID FROM ML_Model WHERE AlgorithmType = ?", (best_model_name,))
    model_db_id = cursor.fetchone()[0]
    
    # Seed Predictions using best model
    cursor.execute("DELETE FROM Approval_Prediction")
    
    # We only want one prediction per unique applicant ID in the database
    df_merged_unique = df_merged.drop_duplicates(subset=['ID'])
    X_unique = df_merged_unique.drop(columns=['ID', 'STATUS', 'STATUS_BIN', 'MONTHS_BALANCE'])
    
    all_preds = best_pipeline.predict(X_unique)
    all_probs = best_pipeline.predict_proba(X_unique)[:, 1]
    
    predictions_to_insert = []
    for i, applicant_id in enumerate(df_merged_unique['ID']):
        prob_approved = all_probs[i]
        prob_default = 1 - prob_approved
        
        if prob_default < 0.15:
            risk_cat = 'Low'
        elif prob_default < 0.40:
            risk_cat = 'Medium'
        else:
            risk_cat = 'High'
            
        approval_str = 'Approved' if all_preds[i] == 1 else 'Denied'
        
        predictions_to_insert.append((
            int(applicant_id),
            model_db_id,
            approval_str,
            risk_cat
        ))
    cursor.executemany(
        """INSERT INTO Approval_Prediction 
           (ApplicantID, ModelID, ApprovalResult, RiskCategory) 
           VALUES (?, ?, ?, ?)""",
        predictions_to_insert
    )
    conn.commit()
    conn.close()
    print("Database seeding completed.")
    
    # 6. Save notebook file
    # Standalone model evaluations for notebook outputs
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    lr_standalone = LogisticRegression(random_state=42, max_iter=1000)
    lr_standalone.fit(X_train_scaled, y_train)
    y_pred_lr = lr_standalone.predict(X_test_scaled)
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr)
    
    rf_standalone = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_standalone.fit(X_train_scaled, y_train)
    y_pred_rf = rf_standalone.predict(X_test_scaled)
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)
    
    dt_standalone = DecisionTreeClassifier(random_state=42)
    dt_standalone.fit(X_train_scaled, y_train)
    y_pred_dt = dt_standalone.predict(X_test_scaled)
    
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    report_dt = classification_report(y_test, y_pred_dt)
    
    create_notebook_file(accuracies, best_model_name, cm_lr, report_lr, cm_rf, report_rf, cm_dt, report_dt)
if __name__ == "__main__":
    main()
