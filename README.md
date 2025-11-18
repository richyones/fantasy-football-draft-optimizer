# Fantasy Football Draft Optimizer

A comprehensive fantasy football draft optimization tool with data-driven player recommendations and analytics.

## Project Structure

```
fantasy-football-draft-optimizer/
├── frontend/          # React web application
├── modeling/          # Data science and modeling code
├── data/              # Data files (raw, processed, outputs)
│   ├── raw/          # Original data files
│   ├── processed/    # Generated feature files
│   └── outputs/      # Final outputs for web app
└── docs/             # Documentation
```

## Features

- **Draft Status Bar**: Shows current round, timer, and upcoming picks
- **Your Roster**: Position-based roster management
- **Your Top 3**: Recommended player suggestions
- **Your Big Board**: Dynamic player rankings
- **Picks History**: Recent draft selections
- **Alerts**: Important notifications and warnings
- **Analytics**: Data visualization and analysis

## Quick Start

### Prerequisites

- **Node.js** (v14 or higher) and **npm** installed on your machine
- To check if you have them installed, run:
  ```bash
  node --version
  npm --version
  ```
- If not installed, download from [nodejs.org](https://nodejs.org/)

### Frontend (Web Application)

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. **IMPORTANT: Install dependencies first** (this step is required before running the app):
```bash
npm install
```
This will install all required packages including `react-scripts`, `react`, `react-dom`, and `recharts`. This may take a few minutes.

3. Start the development server:
```bash
npm start
```

The app will open automatically at [http://localhost:3000](http://localhost:3000)

**Note:** If you see an error like `react-scripts: command not found`, it means dependencies haven't been installed. Make sure you've run `npm install` in the `frontend` directory first.

### Modeling

1. Navigate to the modeling directory:
```bash
cd modeling
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the scoring script:
```bash
python scripts/"Fantasy PPR Scoring and Dataframes.py"
```

See `modeling/README.md` for more details.

## Deployment

The web application is deployed to GitHub Pages at:
https://richyones.github.io/fantasy-football-draft-optimizer

To deploy updates:
```bash
cd frontend
npm run deploy
```

## Data Files

- **Raw Data**: `data/raw/` - Original data files
- **Processed Features**: `data/processed/` - Generated feature files from modeling
- **Web App Outputs**: `data/outputs/` - Final CSV files used by the web application

## Documentation

- Project documentation: `docs/CSE 6242 - Group Project.pdf`
- Progress report: `docs/team044progress.pdf`

## Technologies

- **Frontend**: React, Create React App
- **Visualization**: Recharts
- **Modeling**: Python, Pandas, nflreadpy
- **Deployment**: GitHub Pages

## Troubleshooting

### Error: `react-scripts: command not found`
**Solution:** Run `npm install` in the `frontend` directory. This installs all required dependencies.

### Error: `Cannot find module 'react'` or similar
**Solution:** Make sure you're in the `frontend` directory and run `npm install` again.

### Port 3000 already in use
**Solution:** Either stop the other process using port 3000, or set a different port:
```bash
PORT=3001 npm start
```

### CSV files not loading
**Solution:** Make sure the CSV files are in the `frontend/public/` directory:
- `player_adp_optimized_FINAL.csv`
- `nfl_player_information.csv`

## License

This project is part of CSE 6242 Group Project.
