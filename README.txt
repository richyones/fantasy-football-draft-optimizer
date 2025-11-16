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
- Python 3.7 or higher
- pip (Python package manager)

Step 1: Extract the zip file
    Unzip the provided zip file to your desired location.

Step 2: Open a terminal/command prompt and navigate to the extracted folder
    cd fantasy-football-draft-optimizer

Step 3: Install frontend dependencies
    cd frontend
    npm install

Step 4: Install Python dependencies for modeling
    cd ../modeling
    pip install -r requirements.txt

The required Python packages include: numpy, pandas, and nflreadpy.

Note: The application includes pre-loaded player data, so the modeling step is optional if you only want to run the web application.


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

