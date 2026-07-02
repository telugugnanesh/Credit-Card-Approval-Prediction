-- SQL Schema for Loan Approval Prediction System
-- Compatible with SQLite, PostgreSQL, and MySQL
-- 1. Users Table
CREATE TABLE IF NOT EXISTS Users (
    UserID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name VARCHAR(100) NOT NULL,
    Email VARCHAR(100) UNIQUE NOT NULL,
    Password VARCHAR(255) NOT NULL,
    Role VARCHAR(50) NOT NULL DEFAULT 'User'
);
-- 2. Applicant Details Table
CREATE TABLE IF NOT EXISTS Applicant_Details (
    ApplicantID INTEGER PRIMARY KEY, -- Matches the ID in the dataset
    UserID INTEGER,
    IncomeType VARCHAR(100) NOT NULL,
    EducationType VARCHAR(100) NOT NULL,
    FamilyStatus VARCHAR(100) NOT NULL,
    HousingType VARCHAR(100) NOT NULL,
    EmploymentDays INTEGER NOT NULL,
    FOREIGN KEY (UserID) REFERENCES Users(UserID) ON DELETE SET NULL
);
-- 3. Credit History Table
CREATE TABLE IF NOT EXISTS Credit_History (
    HistoryID INTEGER PRIMARY KEY AUTOINCREMENT,
    ApplicantID INTEGER NOT NULL,
    MonthsBalance INTEGER NOT NULL,
    PaymentStatus VARCHAR(10) NOT NULL, -- e.g., '0', '1', '2', '3', '4', '5', 'C', 'X'
    OverdueStatus VARCHAR(50) NOT NULL, -- Derived categorization, e.g., 'No Overdue', 'Past Due', 'Serious Default'
    FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details(ApplicantID) ON DELETE CASCADE
);
-- 4. ML Model Table
CREATE TABLE IF NOT EXISTS ML_Model (
    ModelID INTEGER PRIMARY KEY AUTOINCREMENT,
    ModelName VARCHAR(100) NOT NULL,
    AlgorithmType VARCHAR(100) NOT NULL,
    Accuracy REAL,
    ModelFile VARCHAR(255) NOT NULL
);
-- 5. Approval Prediction Table
CREATE TABLE IF NOT EXISTS Approval_Prediction (
    PredictionID INTEGER PRIMARY KEY AUTOINCREMENT,
    ApplicantID INTEGER NOT NULL,
    ModelID INTEGER NOT NULL,
    ApprovalResult VARCHAR(50) NOT NULL, -- e.g., 'Approved', 'Denied'
    RiskCategory VARCHAR(50) NOT NULL, -- e.g., 'Low', 'Medium', 'High'
    PredictionDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ApplicantID) REFERENCES Applicant_Details(ApplicantID) ON DELETE CASCADE,
    FOREIGN KEY (ModelID) REFERENCES ML_Model(ModelID)
);
