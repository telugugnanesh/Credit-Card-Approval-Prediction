import csv
import random
import os
# Set seed for reproducibility
random.seed(42)
# Configuration
NUM_APPLICANTS = 1000
OUTPUT_DIR = "."
APP_RECORD_FILE = os.path.join(OUTPUT_DIR, "application_record.csv")
CREDIT_RECORD_FILE = os.path.join(OUTPUT_DIR, "credit_record.csv")
# Constants
INCOME_TYPES = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student']
EDUCATION_TYPES = ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree']
FAMILY_STATUSES = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
HOUSING_TYPES = ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment']
OCCUPATIONS = ['Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 'Waiters/barmen staff', 'Secretaries', 'HR staff', 'Realty agents', 'IT staff']
def generate_applicant_data():
    applicants = []
    start_id = 5008804
    
    for i in range(NUM_APPLICANTS):
        applicant_id = start_id + i
        gender = random.choice(['M', 'F'])
        own_car = random.choices(['Y', 'N'], weights=[0.35, 0.65])[0]
        own_realty = random.choices(['Y', 'N'], weights=[0.68, 0.32])[0]
        children = random.choices([0, 1, 2, 3], weights=[0.70, 0.18, 0.10, 0.02])[0]
        
        # Income generation (influenced by education and housing)
        edu = random.choices(EDUCATION_TYPES, weights=[0.60, 0.25, 0.10, 0.04, 0.01])[0]
        income_type = random.choice(INCOME_TYPES)
        
        base_income = 120000
        if edu == 'Higher education':
            base_income += 80000
        elif edu == 'Academic degree':
            base_income += 150000
        
        if income_type == 'Commercial associate':
            base_income += 30000
        elif income_type == 'Pensioner':
            base_income -= 20000
            
        income = round(random.normalvariate(base_income, 40000), 2)
        income = max(30000.0, income)  # Min income clamp
        
        fam_status = random.choice(FAMILY_STATUSES)
        housing = random.choice(HOUSING_TYPES)
        
        # Age (DAYS_BIRTH: negative days, e.g. -25180 for 69 years, -10000 for 27 years)
        age_years = random.uniform(21, 68)
        days_birth = int(-age_years * 365.25)
        
        # Employment Days
        if income_type == 'Pensioner':
            days_employed = 365243  # Standard representation for unemployed/retired in Kaggle dataset
            occupation = ""
        else:
            emp_years = random.uniform(0.5, age_years - 18)
            days_employed = int(-emp_years * 365.25)
            occupation = random.choice(OCCUPATIONS)
            
        mobil = 1
        work_phone = random.choice([0, 1])
        phone = random.choice([0, 1])
        email = random.choices([0, 1], weights=[0.9, 0.1])[0]
        
        fam_members = children + (2 if fam_status == 'Married' or fam_status == 'Civil marriage' else 1)
        
        applicants.append({
            'ID': applicant_id,
            'CODE_GENDER': gender,
            'FLAG_OWN_CAR': own_car,
            'FLAG_OWN_REALTY': own_realty,
            'CNT_CHILDREN': children,
            'AMT_INCOME_TOTAL': income,
            'NAME_INCOME_TYPE': income_type,
            'NAME_EDUCATION_TYPE': edu,
            'NAME_FAMILY_STATUS': fam_status,
            'NAME_HOUSING_TYPE': housing,
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'FLAG_MOBIL': mobil,
            'FLAG_WORK_PHONE': work_phone,
            'FLAG_PHONE': phone,
            'FLAG_EMAIL': email,
            'OCCUPATION_TYPE': occupation,
            'CNT_FAM_MEMBERS': fam_members
        })
        
    return applicants
def generate_credit_data(applicants):
    credit_records = []
    
    for app in applicants:
        app_id = app['ID']
        income = app['AMT_INCOME_TOTAL']
        edu = app['NAME_EDUCATION_TYPE']
        
        # Determine "default risk" based on features
        # Higher score = lower risk (better client)
        score = 50
        
        # Income impact
        if income > 250000:
            score += 20
        elif income > 150000:
            score += 10
        elif income < 80000:
            score -= 15
            
        # Education impact
        if edu in ['Higher education', 'Academic degree']:
            score += 15
        elif edu == 'Lower secondary':
            score -= 10
            
        # Age impact (older is usually slightly lower risk)
        age = abs(app['DAYS_BIRTH']) / 365.25
        if age > 45:
            score += 10
        elif age < 30:
            score -= 10
            
        # Determine client quality
        # Probability of default
        prob_default = 0.05  # Base default probability
        if score > 80:
            prob_default = 0.01
        elif score > 60:
            prob_default = 0.03
        elif score < 30:
            prob_default = 0.15
        elif score < 15:
            prob_default = 0.30
        is_defaulter = random.random() < prob_default
        
        # Generate monthly records (random history length between 1 and 24 months)
        num_months = random.randint(3, 24)
        
        for m in range(num_months):
            month_balance = -m
            
            if is_defaulter:
                # If they are a defaulter, they have a chance of high status values ('1', '2', '3', '4', '5')
                # status: 0: 1-29 days overdue, 1: 30-59, 2: 60-89, 3: 90-119, 4: 120-149, 5: >150 overdue
                status = random.choices(
                    ['C', 'X', '0', '1', '2', '3', '4', '5'],
                    weights=[0.1, 0.1, 0.3, 0.2, 0.15, 0.08, 0.05, 0.02]
                )[0]
            else:
                # Good client, mostly paid off ('C') or no loan ('X') or minor delay ('0')
                status = random.choices(
                    ['C', 'X', '0', '1'],
                    weights=[0.5, 0.2, 0.28, 0.02]
                )[0]
                
            credit_records.append({
                'ID': app_id,
                'MONTHS_BALANCE': month_balance,
                'STATUS': status
            })
            
    return credit_records
def main():
    print("Generating synthetic data...")
    applicants = generate_applicant_data()
    credit_records = generate_credit_data(applicants)
    
    # Save applications
    with open(APP_RECORD_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=applicants[0].keys())
        writer.writeheader()
        writer.writerows(applicants)
    print(f"Saved {len(applicants)} applicant records to {APP_RECORD_FILE}")
    
    # Save credit history
    with open(CREDIT_RECORD_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=credit_records[0].keys())
        writer.writeheader()
        writer.writerows(credit_records)
    print(f"Saved {len(credit_records)} credit history records to {CREDIT_RECORD_FILE}")
    print("Data generation complete!")
if __name__ == "__main__":
    main()