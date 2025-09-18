@echo OFF
REM ============================================================================
REM  NFL Sports Analytics Data Pipeline
REM  ----------------------------------------------------------------------------
REM  This script automates the entire process of fetching, processing, 
REM  and uploading NFL data for your web application.
REM  
REM  Instructions:
REM  1. Make sure you have Python, Node.js, and R installed.
REM  2. Install Python dependencies: pip install -r requirements.txt
REM  3. Install Node.js dependencies: npm install firebase-admin
REM  4. Place your serviceAccountKey.json file in the same directory.
REM  5. Run this script from the command line.
REM ============================================================================

ECHO [STEP 1/4] Starting raw data fetch...
ECHO.

ECHO [1a] Fetching NFL Schedules and Depth Charts...
python nfl_data.py
IF ERRORLEVEL 1 (
    ECHO [ERROR] Failed to fetch schedule and depth chart data. Halting pipeline.
    GOTO :EOF
)
ECHO [SUCCESS] Schedules and depth charts fetched.
ECHO.

ECHO [1b] Fetching NFL Player Gamelogs...
python nfl_gamelogs.py
IF ERRORLEVEL 1 (
    ECHO [ERROR] Failed to fetch player gamelogs. Halting pipeline.
    GOTO :EOF
)
ECHO [SUCCESS] Gamelogs fetched.
ECHO.

ECHO [STEP 2/4] Generating predictions and analysis...
python nflplan.py
IF ERRORLEVEL 1 (
    ECHO [ERROR] Failed to generate predictions with nflplan.py. Halting pipeline.
    GOTO :EOF
)
ECHO [SUCCESS] Predictions and analysis complete.
ECHO.

ECHO [STEP 3/4] Uploading all generated data to Firebase Firestore...
node nfl_upload.js
IF ERRORLEVEL 1 (
    ECHO [ERROR] Failed to upload data to Firestore. Please check nfl_upload.js and serviceAccountKey.json. Halting pipeline.
    GOTO :EOF
)
ECHO [SUCCESS] All data has been uploaded to Firestore.
ECHO.

ECHO [STEP 4/4] Pipeline finished successfully!
ECHO Your website data is now up to date.
ECHO The Streamlit app (1TD.py) runs independently and can be started with 'streamlit run 1TD.py'.

PAUSE
