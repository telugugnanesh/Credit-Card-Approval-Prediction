# Credit Card Approval Prediction System — Project Development Phase

## Project Overview

**Credit Card Approval Prediction** is an AI-powered system built using Machine Learning that analyzes an applicant's financial and personal information to predict whether a credit card application should be **Approved** or **Rejected** — along with a **Risk Category** (Low / Medium / High).

The system processes real-world applicant attributes such as:

| Feature | Description |
|---|---|
| Gender | Male / Female |
| Own Car | Whether the applicant owns a car |
| Own Realty | Whether the applicant owns property |
| Number of Children | Count of dependent children |
| Annual Income | Total declared income (₹) |
| Income Type | Working / Commercial / Pensioner / Student / State servant |
| Education Type | Secondary / Higher / Incomplete Higher / Lower Secondary / Academic |
| Family Status | Married / Single / Civil marriage / Separated / Widow |
| Housing Type | House/apartment / With parents / Municipal / Rented / Office / Co-op |
| Age | Applicant's age in years |
| Employment Status | Employed / Unemployed |
| Employment Years | Years in current employment |
| Occupation | Occupation category from 18 types |
| Credit History | Monthly repayment record (last N months) |

To output a **probability confidence score** and a final **Approved / Rejected** decision.

---

## Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Flask |
| Database | SQLite |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn (Logistic Regression, Decision Tree, Random Forest) |
| Visualization | Plotly Express |
| Frontend | HTML5, CSS3 (Dark Mode), Vanilla JavaScript |
| Notebook | Jupyter Notebook (ipykernel) |
| Deployment | Render.com (render.yaml) |

---

## Machine Learning Models

Three models are trained and compared:

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline model |
| Decision Tree | Interpretable rule-based model |
| **Random Forest** | **Best model — selected for production** |

**Preprocessing Pipeline:**
- Categorical features: `OneHotEncoder`
- Numerical features: `StandardScaler`
- Missing values: `SimpleImputer`
- Uses `sklearn.pipeline.Pipeline` for end-to-end transformation

**Performance Metrics (Approximate):**
- Training Accuracy: ~85–90%
- Testing Accuracy: ~82–88%
- Risk scoring via `predict_proba()` probability of default

---

## Project Structure

```
5.Project Deveolpment Phase/
│
├── Dataset/
│   ├── application_record.csv          # 1,000 applicant records (18 features)
│   └── credit_record.csv               # 13,261 monthly credit payment records
│
├── Flask/
│   ├── app.py                          # Flask web application (routes + logic)
│   ├── schema.sql                      # SQLite database schema (5 tables)
│   ├── requirements.txt                # Python dependencies
│   ├── generate_data.py                # Synthetic data generation script
│   ├── run_pipeline.py                 # ML training pipeline + DB seeding
│   ├── templates/
│   │   ├── base.html                   # Base layout with navbar
│   │   ├── login.html                  # Officer login page
│   │   ├── home.html                   # Home / landing page
│   │   ├── dashboard.html              # Analytics dashboard (Plotly charts)
│   │   ├── index.html                  # Credit card application form
│   │   ├── predict.html                # Prediction form (alternate)
│   │   ├── applicants.html             # Applicant database browser
│   │   ├── result.html                 # Prediction result page
│   │   └── logs.html                   # System logs (model info, DB stats)
│   └── static/
│       └── style.css                   # Dark-mode CSS design system
│
├── Training/
│   ├── loan_approval_ml.ipynb          # Jupyter notebook — EDA + model training
│   └── run_notebook.py                 # Script to execute notebook programmatically
│
├── README.md                           # This file
└── render.yaml                         # Render.com deployment configuration
```

---

## Database Schema

The SQLite database (`loan_approval.db`) has **5 tables**:

```
Users               → System login accounts (Name, Email, Password, Role)
Applicant_Details   → Applicant profile (Income, Education, Housing, Employment)
Credit_History      → Monthly repayment history per applicant
ML_Model            → Registered ML models (algorithm, accuracy, file path)
Approval_Prediction → Predictions per applicant (Approved/Denied, Risk Level)
```

---

## Application Features

### 1. 🔐 Secure Login System
- SHA-256 password hashing
- Session-based authentication
- Role-based access (Officer / Admin)
- Default: `officer@loanpredict.com` / `officer123`

### 2. 📊 Analytics Dashboard
- **Total Applications** processed
- **Approval Rate** (%)
- **High-Risk Applicants** (%)
- **Model Accuracy** (latest trained model)
- Interactive Plotly charts:
  - Risk Category histogram (Approved vs Denied)
  - Education level vs. Approval rate bar chart
  - Employment Years box plot by decision
  - Income Type donut chart

### 3. 📝 Credit Card Application Form
- 13-field application form
- Real-time ML prediction on submission
- Approval probability score
- Risk category assignment (Low / Medium / High)
- Result saved automatically to database

### 4. 🗂️ Applicant Database Browser
- Search by ID or Income Type
- View full applicant profile
- View monthly credit history
- View individual prediction result

### 5. 📋 System Logs
- Registered ML models and accuracy scores
- Database file size
- Active user count

---

## Setup & Run Instructions

### Prerequisites
```bash
Python 3.11+
pip
```

### Step 1: Install Dependencies
```bash
pip install -r Flask/requirements.txt
```

### Step 2: Train the Model & Seed the Database
```bash
python Flask/run_pipeline.py
```
This will:
- Load `Dataset/application_record.csv` and `Dataset/credit_record.csv`
- Train Logistic Regression, Decision Tree, and Random Forest models
- Save the best model as `loan_approval_model.pkl`
- Create and seed `loan_approval.db` with applicants, credit history, and predictions

### Step 3: Run the Flask Application
```bash
python Flask/app.py
```

Open your browser at: **http://localhost:5000**

---

## Deployment (Render.com)

See [`render.yaml`](./render.yaml) for the deployment configuration.

```bash
# Build command
pip install -r Flask/requirements.txt && python Flask/run_pipeline.py

# Start command
python Flask/app.py
```

---

## Functional Features Count

| # | Feature |
|---|---|
| 1 | User Authentication (Login / Logout) |
| 2 | Session Management & Route Protection |
| 3 | Credit Card Application Form (13 fields) |
| 4 | Real-time ML Prediction (RF Classifier) |
| 5 | Probability Score & Risk Category |
| 6 | Result Persistence to SQLite |
| 7 | Analytics Dashboard with 4 Plotly Charts |
| 8 | Applicant Search & Lookup |
| 9 | Credit History Viewer |
| 10 | System Logs Page |
| 11 | Multi-model Training & Comparison |
| 12 | Automated Database Seeding |

---

## Contributors

- **Project:** Credit Card Approval Prediction System
- **Tech Stack:** Flask + Scikit-Learn + SQLite + Plotly
