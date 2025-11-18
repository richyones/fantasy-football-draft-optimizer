README.txt
==========

DESCRIPTION
-----------
The Fantasy Football Draft Optimizer is a comprehensive web-based application that provides data-driven player recommendations and real-time analytics during fantasy football drafts. The system combines machine learning models with an intuitive React-based user interface to help users make optimal draft decisions.

The application features a live draft board with player rankings, position-based roster management, and intelligent "Top 3" recommendations that adapt based on draft position and roster needs. It includes comprehensive analytics visualizations showing player distributions, position availability, and draft trends. The system processes historical NFL play-by-play data to generate fantasy point projections and feature engineering for all offensive positions (QB, RB, WR, TE), kickers, and defense/special teams.

The project consists of two main components: (1) a Python-based data processing and modeling pipeline that extracts features from NFL data and generates predictions, and (2) a React web application that provides the interactive draft interface. The system is deployed as a live web application accessible via GitHub Pages, allowing users to access the tool without local installation.


INSTALLATION
------------
Prerequisites:
- Node.js (v14 or higher) and npm
  * To check if you have them installed, run:
    node --version
    npm --version
  * If not installed, download from https://nodejs.org/
- Python 3.7 or higher (optional, only needed for modeling)
- pip (Python package manager, optional)

Step 1: Extract the zip file
    Unzip the provided zip file to your desired location.

Step 2: Open a terminal/command prompt and navigate to the extracted folder
    cd fantasy-football-draft-optimizer

Step 3: Install frontend dependencies (REQUIRED - do not skip this step!)
    cd frontend
    npm install
    
    IMPORTANT: This step is mandatory before running the application. It installs all 
    required packages including react-scripts, react, react-dom, and recharts. 
    This may take a few minutes to complete.
    
    If you see an error like "react-scripts: command not found" when trying to run 
    npm start, it means you haven't run npm install yet. Make sure you complete 
    this step first.

Step 4: Install Python dependencies for modeling (OPTIONAL)
    cd ../modeling
    pip install -r requirements.txt

The required Python packages include: numpy, pandas, and nflreadpy.

Note: The application includes pre-loaded player data, so the modeling step is optional 
if you only want to run the web application.


EXECUTION
---------
To run the Fantasy Football Draft Optimizer:

Option 1: Use the live web application (recommended)
    The application is available at:
    https://richyones.github.io/fantasy-football-draft-optimizer
    
    No installation required - simply open the URL in your web browser.

Option 2: Run locally
    Navigate to the frontend directory:
        cd frontend
    
    IMPORTANT: Make sure you've run "npm install" first (see Step 3 in Installation).
    If you haven't, you'll get an error "react-scripts: command not found".
    
    Start the development server:
        npm start
    
    The application will automatically open in your browser at http://localhost:3000

Demo Workflow:
1. Upon launching, you'll see the draft setup screen
2. Configure your draft settings:
   - Select pick time (30-90 seconds recommended)
   - Choose number of teams (8, 10, 12, 14, or 16)
   - Set your starting pick number (1-16)
3. Click "Start Draft" to begin
4. During the draft:
   - View your recommended "Top 3" players in the left panel
   - Browse the full player board with filtering options
   - Monitor your roster as you draft
   - Check the Analytics tab for data visualizations
   - Review pick history and alerts in the right sidebar
5. Click "DRAFT" on any available player when it's your turn
6. The system automatically handles CPU picks for other teams

The application includes pre-loaded player data from the 2024 season, so no additional data processing is required to run the demo. However, if you wish to regenerate predictions using the modeling pipeline, navigate to the modeling directory and run:
    python scripts/"Fantasy PPR Scoring and Dataframes.py"


[Optional, but recommended] DEMO VIDEO
--------------------------------------
[Include URL of a 1-minute unlisted YouTube video here]

The video demonstrates:
- Extracting the zip file
- Installation steps (npm install, pip install)
- Launching the application (npm start)
- Configuring draft settings
- Running a sample draft with player selection
- Exploring the analytics visualizations
- Reviewing the roster and pick history

Note: This video is optional for submission but recommended as it helps demonstrate the full workflow and user experience of the system.


TROUBLESHOOTING
---------------
Common errors and solutions:

Error: "react-scripts: command not found"
Solution: Run "npm install" in the frontend directory. This installs all required 
         dependencies. Make sure you're in the frontend folder when running this command.

Error: "Cannot find module 'react'" or similar module errors
Solution: Make sure you're in the frontend directory and run "npm install" again. 
         This will reinstall all dependencies.

Error: Port 3000 already in use
Solution: Either stop the other process using port 3000, or set a different port:
         On Mac/Linux: PORT=3001 npm start
         On Windows: set PORT=3001 && npm start

CSV files not loading in the application
Solution: Make sure the CSV files are in the frontend/public/ directory:
         - player_adp_optimized_FINAL.csv
         - nfl_player_information.csv
         These files should be included in the zip file. If they're missing, check 
         the data/outputs/ folder and copy them to frontend/public/.

