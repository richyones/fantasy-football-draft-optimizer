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

### Frontend (Web Application)

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000)

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

## License

This project is part of CSE 6242 Group Project.
