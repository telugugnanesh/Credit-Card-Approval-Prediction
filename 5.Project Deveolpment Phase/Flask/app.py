from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import sqlite3
import hashlib
import pickle
import os
from functools import wraps
# pyrefly: ignore [missing-import]
import plotly.express as px
app = Flask(__name__)
app.secret_key = "loan_approval_system_secret_key_12345"
DB_PATH = "loan_approval.db"
MODEL_PATH = "loan_approval_model.pkl"
# Constants matching dataset
INCOME_TYPES = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student']
EDUCATION_TYPES = ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
FAMILY_STATUSES = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
HOUSING_TYPES = ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment']
OCCUPATIONS = ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 'Waiters/barmen staff', 'Secretaries', 'HR staff', 'Realty agents', 'IT staff']
# Database Helpers
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
def load_prediction_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None
# Authentication Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('authenticated'):
        return redirect(url_for('dashboard'))
        
    error = None
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        hashed = hash_password(password)
        cursor.execute("SELECT * FROM Users WHERE Email = ? AND Password = ?", (email, hashed))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['authenticated'] = True
            session['user_email'] = user['Email']
            session['user_name'] = user['Name']
            session['user_role'] = user['Role']
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid email or password. Hint: officer@loanpredict.com / officer123"
            
    return render_template('login.html', error=error)
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
@app.route('/')
@app.route('/home')
@login_required
def home():
    return render_template('home.html', active_page='home')
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    df_applicants = pd.read_sql_query("SELECT * FROM Applicant_Details", conn)
    df_predictions = pd.read_sql_query("SELECT * FROM Approval_Prediction", conn)
    df_models = pd.read_sql_query("SELECT * FROM ML_Model", conn)
    conn.close()
    
    if df_applicants.empty or df_predictions.empty:
        return render_template('dashboard.html', total_apps=0, approval_rate=0.0, high_risk_pct=0.0, latest_accuracy=0.0, active_page='dashboard')
        
    # Merged dashboard data
    df_dash = pd.merge(df_applicants, df_predictions, on="ApplicantID")
    
    total_apps = len(df_dash)
    approved_apps = len(df_dash[df_dash["ApprovalResult"] == "Approved"])
    approval_rate = (approved_apps / total_apps * 100) if total_apps > 0 else 0.0
    
    high_risk_count = len(df_dash[df_dash["RiskCategory"] == "High"])
    high_risk_pct = (high_risk_count / total_apps * 100) if total_apps > 0 else 0.0
    
    latest_accuracy = df_models["Accuracy"].iloc[-1] if not df_models.empty else 0.0
    
    # 1. Risk category histogram
    fig_risk = px.histogram(
        df_dash,
        x="RiskCategory",
        color="ApprovalResult",
        barmode="group",
        category_orders={"RiskCategory": ["Low", "Medium", "High"]},
        color_discrete_map={"Approved": "#10b981", "Denied": "#ef4444"},
        labels={"RiskCategory": "Risk Level", "ApprovalResult": "Decision"}
    )
    fig_risk.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)", 
        margin=dict(t=10, b=10, l=10, r=10), 
        height=300
    )
    chart_risk_html = fig_risk.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 2. Education level bar chart
    df_edu = df_dash.groupby("EducationType")["ApprovalResult"].value_counts(normalize=True).unstack().fillna(0) * 100
    df_edu = df_edu.reset_index()
    if "Approved" not in df_edu.columns:
        df_edu["Approved"] = 0.0
    fig_edu = px.bar(
        df_edu,
        x="EducationType",
        y="Approved",
        labels={"Approved": "Approval Rate (%)", "EducationType": "Education Level"},
        color_discrete_sequence=["#3b82f6"]
    )
    fig_edu.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)", 
        margin=dict(t=10, b=10, l=10, r=10), 
        height=300
    )
    chart_edu_html = fig_edu.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 3. Box plot of Employment length vs Decision
    df_dash["EmploymentYears"] = df_dash["EmploymentDays"].apply(lambda x: 0 if x >= 365243 else -x/365.25)
    fig_emp = px.box(
        df_dash,
        x="ApprovalResult",
        y="EmploymentYears",
        color="ApprovalResult",
        color_discrete_map={"Approved": "#10b981", "Denied": "#ef4444"},
        labels={"ApprovalResult": "Decision", "EmploymentYears": "Years Employed"}
    )
    fig_emp.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)", 
        margin=dict(t=10, b=10, l=10, r=10), 
        height=300
    )
    chart_emp_html = fig_emp.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 4. Pie chart of Income Type Representation
    fig_inc = px.pie(
        df_dash,
        names="IncomeType",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_inc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        margin=dict(t=20, b=10, l=10, r=10), 
        height=300
    )
    chart_inc_html = fig_inc.to_html(full_html=False, include_plotlyjs='cdn')
    
    return render_template(
        'dashboard.html',
        total_apps=total_apps,
        approval_rate=approval_rate,
        high_risk_pct=high_risk_pct,
        latest_accuracy=latest_accuracy,
        chart_risk_html=chart_risk_html,
        chart_edu_html=chart_edu_html,
        chart_emp_html=chart_emp_html,
        chart_inc_html=chart_inc_html,
        active_page='dashboard'
    )
@app.route('/applicants')
@login_required
def applicants():
    search_query = request.args.get('search', '')
    lookup_id = request.args.get('lookup_id', type=int)
    
    conn = get_db_connection()
    
    # Filtered SQL query
    if search_query:
        cursor = conn.execute(
            """SELECT a.*, p.ApprovalResult, p.RiskCategory 
               FROM Applicant_Details a
               JOIN Approval_Prediction p ON a.ApplicantID = p.ApplicantID
               WHERE a.ApplicantID LIKE ? OR a.IncomeType LIKE ?""",
            (f"%{search_query}%", f"%{search_query}%")
        )
    else:
        cursor = conn.execute(
            """SELECT a.*, p.ApprovalResult, p.RiskCategory 
               FROM Applicant_Details a
               JOIN Approval_Prediction p ON a.ApplicantID = p.ApplicantID
               ORDER BY a.ApplicantID DESC LIMIT 200"""
        )
        
    applicants_list = [dict(row) for row in cursor.fetchall()]
    
    lookup_applicant = None
    lookup_history = []
    lookup_prediction = None
    
    if lookup_id:
        cursor = conn.execute("SELECT * FROM Applicant_Details WHERE ApplicantID = ?", (lookup_id,))
        row = cursor.fetchone()
        if row:
            lookup_applicant = dict(row)
            cursor = conn.execute("SELECT * FROM Credit_History WHERE ApplicantID = ? ORDER BY MonthsBalance DESC", (lookup_id,))
            lookup_history = [dict(r) for r in cursor.fetchall()]
            cursor = conn.execute("SELECT * FROM Approval_Prediction WHERE ApplicantID = ?", (lookup_id,))
            pred_row = cursor.fetchone()
            if pred_row:
                lookup_prediction = dict(pred_row)
                
    conn.close()
    
    return render_template(
        'applicants.html',
        applicants=applicants_list,
        search_query=search_query,
        lookup_applicant=lookup_applicant,
        lookup_history=lookup_history,
        lookup_prediction=lookup_prediction,
        active_page='applicants'
    )
@app.route('/index')
@login_required
def index_page():
    return render_template(
        'index.html',
        INCOME_TYPES=INCOME_TYPES,
        EDUCATION_TYPES=EDUCATION_TYPES,
        FAMILY_STATUSES=FAMILY_STATUSES,
        HOUSING_TYPES=HOUSING_TYPES,
        OCCUPATIONS=OCCUPATIONS,
        active_page='predict'
    )
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return redirect(url_for('index_page'))
        
    model = load_prediction_model()
    if model is None:
        flash("Model is not loaded.", "danger")
        return redirect(url_for('index_page'))
        
    applicant_name = request.form.get('applicant_name', 'Unnamed Applicant')
    gender = request.form.get('gender')
    own_car = request.form.get('own_car')
    own_realty = request.form.get('own_realty')
    children = int(request.form.get('children', 0))
    income = float(request.form.get('income', 150000))
    income_type = request.form.get('income_type')
    education = request.form.get('education')
    family_status = request.form.get('family_status')
    housing = request.form.get('housing')
    age = float(request.form.get('age', 35))
    emp_status = request.form.get('emp_status')
    
    if emp_status == "Unemployed":
        emp_years = 0.0
        occupation = ""
        days_employed = 365243
    else:
        emp_years = float(request.form.get('emp_years', 3.0))
        occupation = request.form.get('occupation', '')
        days_employed = int(-emp_years * 365.25)
        
    family_members = children + (2 if family_status in ["Married", "Civil marriage"] else 1)
    
    # Prepare input for model pipeline
    input_data = pd.DataFrame([{
        'CODE_GENDER': gender,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': children,
        'AMT_INCOME_TOTAL': income,
        'NAME_INCOME_TYPE': income_type,
        'NAME_EDUCATION_TYPE': education,
        'NAME_FAMILY_STATUS': family_status,
        'NAME_HOUSING_TYPE': housing,
        'CNT_FAM_MEMBERS': family_members,
        'Age': age,
        'EmploymentYears': emp_years,
        'IsUnemployed': 1 if emp_status == "Unemployed" else 0,
        'FamilyDependency': family_members - children,
        'open_month': 0,
        'end_months': 0,
        'window': 0
    }])
    
    pred_class = model.predict(input_data)[0]
    pred_prob = model.predict_proba(input_data)[0][1] # Probability of Approval
    prob_default = 1 - pred_prob # Probability of Default (Risk)
    
    if prob_default < 0.15:
        risk_cat = 'Low'
    elif prob_default < 0.40:
        risk_cat = 'Medium'
    else:
        risk_cat = 'High'
        
    decision = "Approved" if pred_class == 1 else "Denied"
    
    # Write to SQLite Database
    conn = get_db_connection()
    cursor = conn.cursor()
    new_id = 6000000
    try:
        cursor.execute("SELECT MAX(ApplicantID) FROM Applicant_Details")
        max_id = cursor.fetchone()[0]
        new_id = (max_id + 1) if max_id else 6000000
        
        # Save applicant profile
        cursor.execute(
            """INSERT INTO Applicant_Details 
               (ApplicantID, UserID, IncomeType, EducationType, FamilyStatus, HousingType, EmploymentDays)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (new_id, 2, income_type, education, family_status, housing, days_employed)
        )
        
        # Retrieve the active model's ModelID from the database
        cursor.execute("SELECT ModelID FROM ML_Model WHERE ModelFile = 'loan_approval_model.pkl'")
        model_row = cursor.fetchone()
        model_id = model_row['ModelID'] if model_row else 1

        # Save model prediction
        cursor.execute(
            """INSERT INTO Approval_Prediction 
               (ApplicantID, ModelID, ApprovalResult, RiskCategory)
               VALUES (?, ?, ?, ?)""",
            (new_id, model_id, decision, risk_cat)
        )
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        flash(f"Error saving to database: {e}", "danger")
    finally:
        conn.close()
        
    prediction_result = {
        'decision': decision,
        'prob_approved': float(pred_prob),
        'prob_default': float(prob_default),
        'risk_cat': risk_cat,
        'new_id': new_id,
        'applicant_name': applicant_name
    }
    
    prediction_text = "Credit Card Approved" if decision == "Approved" else "Credit Card Rejected"
    
    return render_template(
        'result.html',
        prediction=prediction_result,
        prediction_text=prediction_text,
        active_page='predict'
    )
@app.route('/logs')
@login_required
def logs():
    conn = get_db_connection()
    cursor = conn.execute("SELECT * FROM ML_Model")
    models_list = [dict(row) for row in cursor.fetchall()]
    
    cursor = conn.execute("SELECT COUNT(*) FROM Users")
    user_count = cursor.fetchone()[0]
    conn.close()
    
    db_size_kb = 0.0
    if os.path.exists(DB_PATH):
        db_size_kb = os.path.getsize(DB_PATH) / 1024.0
        
    return render_template(
        'logs.html',
        models=models_list,
        db_size_kb=db_size_kb,
        user_count=user_count,
        active_page='logs'
    )
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
