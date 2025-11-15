import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import OverallPicks from './components/OverallPicks';
import BigBoard from './components/BigBoard';
import Analytics from './components/Analytics';

function App() {
  const [draftStarted, setDraftStarted] = useState(false);
  const [draftComplete, setDraftComplete] = useState(false);
  const [draftSettings, setDraftSettings] = useState({
    pickTime: 3,
    numTeams: 12,
    startingPickNumber: 1
  });
  const [activeView, setActiveView] = useState('big-board');
  const [pickHistory, setPickHistory] = useState([]);
  const [draftedPlayerNames, setDraftedPlayerNames] = useState([]);
  const [currentPickNumber, setCurrentPickNumber] = useState(1);
  const [currentRound, setCurrentRound] = useState(1);
  const [timeRemaining, setTimeRemaining] = useState(3);
  const [isUserTurn, setIsUserTurn] = useState(false);
  const [rejectedPlayerNames, setRejectedPlayerNames] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [currentTop3, setCurrentTop3] = useState([]);
  const [roster, setRoster] = useState({
    QB: null,
    RB1: null,
    RB2: null,
    WR1: null,
    WR2: null,
    TE: null,
    FLX: null,
    DST: null,
    K: null,
    BE1: null,
    BE2: null,
    BE3: null,
    BE4: null,
    BE5: null,
    BE6: null,
    BE7: null
  });

  const teamNames = [
    'Team Schwab', 'Team Naughton', 'Team Jacobi', 'Team Jarrett',
    'Team Carter', 'Team Garcia Jr', 'Team Tompkins', 'Team Kephart',
    'Team Demman', 'Team Wilson', 'Team Anderson', 'Team Martinez',
    'Team Taylor', 'Team Thomas', 'Team Moore', 'Team Jackson'
  ];

  const handleStartDraft = () => {
    setDraftStarted(true);
    setDraftComplete(false);
    setTimeRemaining(draftSettings.pickTime);
    // Check if user picks first
    const userPickPosition = draftSettings.startingPickNumber;
    setIsUserTurn(userPickPosition === 1);
  };

  const handleSettingChange = (setting, value) => {
    setDraftSettings({
      ...draftSettings,
      [setting]: value
    });
  };

  const handleRejectPlayer = (playerName) => {
    setRejectedPlayerNames([...rejectedPlayerNames, playerName]);
  };

  // Calculate how many picks until user's next turn
  const getPicksUntilUserTurn = () => {
    if (isUserTurn) return 0;
    
    const { numTeams, startingPickNumber } = draftSettings;
    let picksAway = 0;
    let checkPickNum = currentPickNumber;
    
    // Look ahead up to 2 rounds to find next user pick
    for (let i = 0; i < numTeams * 2; i++) {
      checkPickNum++;
      picksAway++;
      
      if (checkIfUserTurn(checkPickNum)) {
        return picksAway;
      }
    }
    
    return picksAway;
  };

  // Get smart Top 3 based on projected availability
  const getSmartTop3 = () => {
    // Ensure allPlayersRef.current is initialized and is an array
    if (!allPlayersRef.current || !Array.isArray(allPlayersRef.current)) {
      return [];
    }
    
    const availablePlayers = allPlayersRef.current.filter(
      p => !draftedPlayerNames.includes(p.name) && !rejectedPlayerNames.includes(p.name)
    );
    
    // If no available players, return empty array
    if (availablePlayers.length === 0) {
      return [];
    }
    
    // Check if bench is full
    const benchIsFull = roster.BE1 && roster.BE2 && roster.BE3 && roster.BE4 && 
                       roster.BE5 && roster.BE6 && roster.BE7;
    
    // Filter out positions that are filled in starting lineup AND bench is full
    const filteredPlayers = availablePlayers.filter(player => {
      if (!benchIsFull) return true; // If bench isn't full, recommend all positions
      
      const position = player.position;
      
      // Check if position is filled in starting lineup
      if (position === 'QB') {
        return !roster.QB; // Don't recommend if QB is filled
      } else if (position === 'RB') {
        // RB can fill RB1, RB2, or FLX - don't recommend if all are filled
        return !(roster.RB1 && roster.RB2 && roster.FLX);
      } else if (position === 'WR') {
        // WR can fill WR1, WR2, or FLX - don't recommend if all are filled
        return !(roster.WR1 && roster.WR2 && roster.FLX);
      } else if (position === 'TE') {
        // TE can fill TE or FLX - don't recommend if both are filled
        return !(roster.TE && roster.FLX);
      } else if (position === 'DST') {
        return !roster.DST; // Don't recommend if DST is filled
      } else if (position === 'K') {
        return !roster.K; // Don't recommend if K is filled
      }
      
      return true; // Default: include position
    });
    
    // If no filtered players, return empty array
    if (filteredPlayers.length === 0) {
      return [];
    }
    
    const picksAway = getPicksUntilUserTurn();
    
    // If it's user's turn, show top 3 available (or fewer if not enough players)
    if (picksAway === 0) {
      return filteredPlayers.slice(0, Math.min(3, filteredPlayers.length));
    }
    
    // Otherwise, estimate which players will still be available
    // Add buffer: assume best players will be taken before user's turn
    const estimatedPicksBeforeUserTurn = Math.max(0, picksAway - 1);
    const maxStartIndex = Math.max(0, filteredPlayers.length - 3);
    const startIndex = Math.min(estimatedPicksBeforeUserTurn, maxStartIndex);
    const endIndex = Math.min(startIndex + 3, filteredPlayers.length);
    
    return filteredPlayers.slice(startIndex, endIndex);
  };

  // Check if a player from user's Top 3 was stolen
  const checkTop3Stolen = (newPick) => {
    // Only check if it wasn't the user who drafted
    if (isUserTurn) return;
    
    const top3Names = currentTop3.map(p => p.name);
    
    if (top3Names.includes(newPick.name)) {
      const newAlert = {
        id: Date.now(),
        type: 'stolen',
        message: `😱 Your Top 3 was stolen! ${newPick.name} (${newPick.position}) was taken by ${newPick.teamName}`,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setAlerts([newAlert, ...alerts]);
    }
  };

  // Check for herd warning (3+ consecutive picks of same position)
  const checkHerdWarning = (newPick) => {
    if (pickHistory.length < 2) return;

    const recentPicks = [newPick, ...pickHistory.slice(0, 2)];
    const positions = recentPicks.map(p => p.position);
    
    // Check if all 3 positions are the same
    if (positions[0] === positions[1] && positions[1] === positions[2]) {
      const positionName = positions[0] === 'RB' ? 'Running Backs' :
                          positions[0] === 'WR' ? 'Wide Receivers' :
                          positions[0] === 'QB' ? 'Quarterbacks' :
                          positions[0] === 'TE' ? 'Tight Ends' :
                          positions[0] === 'K' ? 'Kickers' :
                          positions[0] === 'DST' ? 'Defenses' : positions[0];
      
      const newAlert = {
        id: Date.now(),
        type: 'herd',
        message: `🚨 Herd Warning: 3 consecutive ${positionName} picked!`,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setAlerts([newAlert, ...alerts]);
    }
  };

  // Get current team name based on pick number
  const getCurrentTeamName = (pickNum) => {
    const { numTeams, startingPickNumber } = draftSettings;
    const round = Math.ceil(pickNum / numTeams);
    const isSnakeDraft = round % 2 === 0; // Even rounds go in reverse
    
    let positionInRound = ((pickNum - 1) % numTeams) + 1;
    if (isSnakeDraft) {
      positionInRound = numTeams - positionInRound + 1;
    }
    
    // Check if it's user's turn
    const isUserPick = positionInRound === startingPickNumber;
    if (isUserPick) {
      return 'Your team';
    }
    
    return `Team ${positionInRound}`;
  };

  // Check if current pick is user's turn
  const checkIfUserTurn = (pickNum) => {
    const { numTeams, startingPickNumber } = draftSettings;
    const round = Math.ceil(pickNum / numTeams);
    const isSnakeDraft = round % 2 === 0;
    
    let positionInRound = ((pickNum - 1) % numTeams) + 1;
    if (isSnakeDraft) {
      positionInRound = numTeams - positionInRound + 1;
    }
    
    return positionInRound === startingPickNumber;
  };

  // Load CSV data for all players
  useEffect(() => {
    const loadCSVData = async () => {
      try {
        // Load both CSVs
        const [draftBoardResponse, playerInfoResponse] = await Promise.all([
          fetch('/fantasy_predictions_2024_full.csv'),
          fetch('/nfl_player_information.csv')
        ]);
        
        const draftBoardText = await draftBoardResponse.text();
        const playerInfoText = await playerInfoResponse.text();
        
        // Parse player information CSV to create lookup map
        const playerInfoLines = playerInfoText.split('\n').filter(line => line.trim());
        const playerInfoHeaders = playerInfoLines[0].split(',');
        const playerInfoNameIndex = playerInfoHeaders.findIndex(h => h.includes('Player Name'));
        const playerInfoTeamIndex = playerInfoHeaders.findIndex(h => h === 'Team');
        const playerInfoByeIndex = playerInfoHeaders.findIndex(h => h.includes('Bye'));
        
        const playerInfoMap = {};
        for (let i = 1; i < playerInfoLines.length; i++) {
          const line = playerInfoLines[i];
          if (!line.trim()) continue;
          
          const values = line.split(',').map(v => v.trim());
          if (values.length > Math.max(playerInfoNameIndex, playerInfoTeamIndex, playerInfoByeIndex)) {
            const name = values[playerInfoNameIndex];
            const team = values[playerInfoTeamIndex];
            const bye = parseInt(values[playerInfoByeIndex]) || 0;
            
            if (name) {
              playerInfoMap[name] = { team, bye };
            }
          }
        }
        
        // Parse draft board CSV
        const lines = draftBoardText.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',');
        
        // Find column indices (support both old and new CSV formats)
        const nameIndex = headers.findIndex(h => h.includes('Player Name') || h.toLowerCase() === 'player');
        const positionIndex = headers.findIndex(h => h.includes('Player Position') || h.toLowerCase() === 'position');
        const pointsIndex = headers.findIndex(h => h.includes('Projected Average Fantasy Points Per Week') || h.toLowerCase().includes('proj_points') || h.toLowerCase().includes('proj points'));
        
        // Parse CSV data
        const parsedData = [];
        for (let i = 1; i < lines.length; i++) {
          const line = lines[i];
          if (!line.trim()) continue;
          
          const values = line.split(',').map(v => v.trim());
          
          if (values.length > Math.max(nameIndex, positionIndex, pointsIndex)) {
            const name = values[nameIndex];
            let position = values[positionIndex];
            const weeklyPts = parseFloat(values[pointsIndex]) || 0;
            
            // Normalize position values (handle D/ST vs DST)
            if (position && (position.toUpperCase() === 'D/ST' || position.toUpperCase() === 'DST')) {
              position = 'DST';
            }
            
            if (name && position && !isNaN(weeklyPts)) {
              // Convert weekly points to season total (17 weeks)
              const seasonPts = weeklyPts * 17;
              
              // Join with player information
              const playerInfo = playerInfoMap[name] || {};
              
              parsedData.push({
                name: name,
                position: position,
                pts: seasonPts,
                team: playerInfo.team || '—',
                bye: playerInfo.bye || 0
              });
            }
          }
        }
        
        // Keep the CSV order as rankings (don't sort by points)
        const rankedData = parsedData.map((player, index) => ({
          rank: index + 1,
          name: player.name,
          position: player.position,
          pts: parseFloat(player.pts.toFixed(1)),
          team: player.team,
          bye: player.bye,
          rush: '-',
          com: '-',
          att: '-',
          payd: '-',
          patd: '-',
          int: '-',
          ru: '-'
        }));
        
        allPlayersRef.current = rankedData;
      } catch (error) {
        console.error('Error loading CSV data:', error);
        // Keep fallback data if CSV fails
      }
    };
    
    loadCSVData();
  }, []);

  // All available players for CPU to draft from (fallback data)
  const fallbackPlayers = [
    { rank: 1, name: 'Christian McCaffrey', team: 'SF', position: 'RB', bye: 9, pts: 285.5, rush: 305 },
    { rank: 2, name: 'Tyreek Hill', team: 'MIA', position: 'WR', bye: 10, pts: 278.2, rush: '-' },
    { rank: 3, name: 'Austin Ekeler', team: 'LAC', position: 'RB', bye: 8, pts: 272.8, rush: 280 },
    { rank: 4, name: 'Justin Jefferson', team: 'MIN', position: 'WR', bye: 13, pts: 268.9, rush: '-' },
    { rank: 5, name: 'Cooper Kupp', team: 'LAR', position: 'WR', bye: 7, pts: 265.4, rush: '-' },
    { rank: 6, name: 'Derrick Henry', team: 'TEN', position: 'RB', bye: 7, pts: 262.1, rush: 320 },
    { rank: 7, name: 'Travis Kelce', team: 'KC', position: 'TE', bye: 8, pts: 258.7, rush: '-' },
    { rank: 8, name: 'Stefon Diggs', team: 'BUF', position: 'WR', bye: 7, pts: 255.3, rush: '-' },
    { rank: 9, name: 'Davante Adams', team: 'LV', position: 'WR', bye: 6, pts: 252.9, rush: '-' },
    { rank: 10, name: 'Josh Allen', team: 'BUF', position: 'QB', bye: 7, pts: 251.5, rush: '-' },
    { rank: 11, name: 'Saquon Barkley', team: 'NYG', position: 'RB', bye: 13, pts: 248.1, rush: 250 },
    { rank: 12, name: 'Ja\'Marr Chase', team: 'CIN', position: 'WR', bye: 7, pts: 244.7, rush: '-' },
    { rank: 13, name: 'Nick Chubb', team: 'CLE', position: 'RB', bye: 5, pts: 241.3, rush: 290 },
    { rank: 14, name: 'A.J. Brown', team: 'PHI', position: 'WR', bye: 10, pts: 237.9, rush: '-' },
    { rank: 15, name: 'Patrick Mahomes', team: 'KC', position: 'QB', bye: 8, pts: 234.5, rush: '-' },
    { rank: 16, name: 'Tony Pollard', team: 'DAL', position: 'RB', bye: 7, pts: 231.1, rush: 265 },
    { rank: 17, name: 'CeeDee Lamb', team: 'DAL', position: 'WR', bye: 7, pts: 227.7, rush: '-' },
    { rank: 18, name: 'Najee Harris', team: 'PIT', position: 'RB', bye: 9, pts: 224.3, rush: 270 },
    { rank: 19, name: 'Amon-Ra St. Brown', team: 'DET', position: 'WR', bye: 6, pts: 220.9, rush: '-' },
    { rank: 20, name: 'Travis Etienne', team: 'JAX', position: 'RB', bye: 11, pts: 217.5, rush: 245 },
    { rank: 21, name: 'Garrett Wilson', team: 'NYJ', position: 'WR', bye: 7, pts: 214.1, rush: '-' },
    { rank: 22, name: 'Joe Mixon', team: 'CIN', position: 'RB', bye: 7, pts: 210.7, rush: 255 },
    { rank: 23, name: 'Mark Andrews', team: 'BAL', position: 'TE', bye: 13, pts: 207.3, rush: '-' },
    { rank: 24, name: 'Jalen Hurts', team: 'PHI', position: 'QB', bye: 10, pts: 203.9, rush: '-' },
    { rank: 25, name: 'Tee Higgins', team: 'CIN', position: 'WR',bye: 7, pts: 200.5, rush: '-' },
    { rank: 26, name: 'DeVonta Smith', team: 'PHI', position: 'WR', bye: 10, pts: 197.1, rush: '-' },
    { rank: 27, name: 'Kenneth Walker', team: 'SEA', position: 'RB', bye: 11, pts: 193.7, rush: 235 },
    { rank: 28, name: 'Bijan Robinson', team: 'ATL', position: 'RB', bye: 11, pts: 190.3, rush: 225 },
    { rank: 29, name: 'DK Metcalf', team: 'SEA', position: 'WR', bye: 11, pts: 186.9, rush: '-' },
    { rank: 30, name: 'Lamar Jackson', team: 'BAL', position: 'QB', bye: 13, pts: 183.5, rush: '-' },
    { rank: 31, name: 'Chris Olave', team: 'NO', position: 'WR', bye: 6, pts: 180.1, rush: '-' },
    { rank: 32, name: 'Rhamondre Stevenson', team: 'NE', position: 'RB', bye: 14, pts: 176.7, rush: 220 },
    { rank: 33, name: 'Jaylen Waddle', team: 'MIA', position: 'WR', bye: 10, pts: 173.3, rush: '-' },
    { rank: 34, name: 'Miles Sanders', team: 'CAR', position: 'RB', bye: 11, pts: 169.9, rush: 210 },
    { rank: 35, name: 'George Kittle', team: 'SF', position: 'TE', bye: 9, pts: 166.5, rush: '-' },
    { rank: 36, name: 'Breece Hall', team: 'NYJ', position: 'RB', bye: 7, pts: 163.1, rush: 200 },
    { rank: 37, name: 'Calvin Ridley', team: 'JAX', position: 'WR', bye: 11, pts: 159.7, rush: '-' },
    { rank: 38, name: 'Justin Fields', team: 'CHI', position: 'QB', bye: 14, pts: 156.3, rush: '-' },
    { rank: 39, name: 'Terry McLaurin', team: 'WAS', position: 'WR', bye: 14, pts: 152.9, rush: '-' },
    { rank: 40, name: 'Joe Burrow', team: 'CIN', position: 'QB', bye: 7, pts: 149.5, rush: '-' },
    { rank: 41, name: 'Dameon Pierce', team: 'HOU', position: 'RB', bye: 7, pts: 146.1, rush: 195 },
    { rank: 42, name: 'Mike Evans', team: 'TB', position: 'WR', bye: 11, pts: 142.7, rush: '-' },
    { rank: 43, name: 'Christian Kirk', team: 'JAX', position: 'WR', bye: 11, pts: 139.3, rush: '-' },
    { rank: 44, name: 'Aaron Jones', team: 'GB', position: 'RB', bye: 10, pts: 135.9, rush: 190 },
    { rank: 45, name: 'T.J. Hockenson', team: 'MIN', position: 'TE', bye: 13, pts: 132.5, rush: '-' },
    { rank: 46, name: 'Amari Cooper', team: 'CLE', position: 'WR', bye: 5, pts: 129.1, rush: '-' },
    { rank: 47, name: 'Dallas Goedert', team: 'PHI', position: 'TE', bye: 10, pts: 125.7, rush: '-' },
    { rank: 48, name: 'Drake London', team: 'ATL', position: 'WR', bye: 11, pts: 122.3, rush: '-' },
    { rank: 49, name: 'Trevor Lawrence', team: 'JAX', position: 'QB', bye: 11, pts: 118.9, rush: '-' },
    { rank: 50, name: 'Dalvin Cook', team: 'MIN', position: 'RB', bye: 13, pts: 115.5, rush: 185 },
    { rank: 51, name: 'Keenan Allen', team: 'LAC', position: 'WR', bye: 8, pts: 112.1, rush: '-' },
    { rank: 52, name: 'DJ Moore', team: 'CHI', position: 'WR', bye: 14, pts: 108.7, rush: '-' },
    { rank: 53, name: 'Kyle Pitts', team: 'ATL', position: 'TE', bye: 11, pts: 105.3, rush: '-' },
    { rank: 54, name: 'Dak Prescott', team: 'DAL', position: 'QB', bye: 7, pts: 101.9, rush: '-' },
    { rank: 55, name: 'Alexander Mattison', team: 'MIN', position: 'RB', bye: 13, pts: 98.5, rush: 180 },
    { rank: 56, name: 'Brandon Aiyuk', team: 'SF', position: 'WR', bye: 9, pts: 95.1, rush: '-' },
    { rank: 57, name: 'Gabe Davis', team: 'BUF', position: 'WR', bye: 7, pts: 91.7, rush: '-' },
    { rank: 58, name: 'Deebo Samuel', team: 'SF', position: 'WR', bye: 9, pts: 88.3, rush: '-' },
    { rank: 59, name: 'Darren Waller', team: 'NYG', position: 'TE', bye: 13, pts: 84.9, rush: '-' },
    { rank: 60, name: 'Deshaun Watson', team: 'CLE', position: 'QB', bye: 5, pts: 81.5, rush: '-' },
    { rank: 61, name: '49ers D/ST', team: 'SF', position: 'DST', bye: 9, pts: 138.0, rush: '-' },
    { rank: 62, name: 'Bills D/ST', team: 'BUF', position: 'DST', bye: 7, pts: 135.0, rush: '-' },
    { rank: 63, name: 'Cowboys D/ST', team: 'DAL', position: 'DST', bye: 7, pts: 132.0, rush: '-' },
    { rank: 64, name: 'Justin Tucker', team: 'BAL', position: 'K', bye: 13, pts: 142.0, rush: '-' },
    { rank: 65, name: 'Harrison Butker', team: 'KC', position: 'K', bye: 8, pts: 138.0, rush: '-' },
  ];

  // All available players for CPU to draft from
  const allPlayersRef = useRef(fallbackPlayers);

  // Timer effect
  useEffect(() => {
    if (!draftStarted || draftComplete) return;
    
    // Check if draft is complete
    const totalPicks = draftSettings.numTeams * 16;
    if (currentPickNumber > totalPicks) {
      return;
    }

    const timer = setInterval(() => {
      setTimeRemaining((prev) => {
        // Check again if draft is complete
        const totalPicks = draftSettings.numTeams * 16;
        if (currentPickNumber > totalPicks || draftComplete) {
          return prev;
        }
        
        if (prev <= 1) {
          // Time's up! Auto-draft if it's user's turn, otherwise CPU drafts
          if (isUserTurn) {
            autoDraftForUser();
          } else {
            cpuDraft();
          }
          return draftSettings.pickTime;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [draftStarted, draftComplete, isUserTurn, currentPickNumber, draftSettings.pickTime, draftSettings.numTeams]);

  // Auto-draft for user when time runs out
  const autoDraftForUser = () => {
    if (!allPlayersRef.current || !Array.isArray(allPlayersRef.current)) {
      return;
    }
    
    const availablePlayers = allPlayersRef.current.filter(
      p => !draftedPlayerNames.includes(p.name)
    );
    if (availablePlayers.length > 0) {
      const bestPlayer = availablePlayers[0];
      draftPlayer(bestPlayer, true);
    }
  };

  // CPU drafting logic
  const cpuDraft = () => {
    if (!allPlayersRef.current || !Array.isArray(allPlayersRef.current)) {
      return;
    }
    
    const availablePlayers = allPlayersRef.current.filter(
      p => !draftedPlayerNames.includes(p.name)
    );
    
    if (availablePlayers.length > 0) {
      // CPU picks best available with slight randomization
      const randomIndex = Math.floor(Math.random() * Math.min(3, availablePlayers.length));
      const cpuPick = availablePlayers[randomIndex];
      draftPlayerCPU(cpuPick);
    }
  };

  // Draft player (called by user clicking DRAFT button)
  const draftPlayer = (player, isAutoPick = false) => {
    // Check if draft is complete
    if (draftComplete) return;
    const totalPicks = draftSettings.numTeams * 16;
    if (currentPickNumber > totalPicks) {
      return;
    }
    
    const teamName = getCurrentTeamName(currentPickNumber);
    const newPick = {
      ...player,
      pickNumber: currentPickNumber,
      teamName: teamName,
      round: currentRound
    };
    
    // Check for alerts before adding to history
    checkTop3Stolen(newPick);
    checkHerdWarning(newPick);
    
    setPickHistory([newPick, ...pickHistory]);
    setDraftedPlayerNames([...draftedPlayerNames, player.name]);
    
    // Only update roster if it's the user's pick
    if (isUserTurn || isAutoPick) {
      updateRoster(player);
    }

    // Move to next pick
    advancePick();
  };

  // Draft player (called by CPU)
  const draftPlayerCPU = (player) => {
    // Check if draft is complete
    if (draftComplete) return;
    const totalPicks = draftSettings.numTeams * 16;
    if (currentPickNumber > totalPicks) {
      return;
    }
    
    const teamName = getCurrentTeamName(currentPickNumber);
    const newPick = {
      ...player,
      pickNumber: currentPickNumber,
      teamName: teamName,
      round: currentRound
    };
    
    // Check for alerts before adding to history
    checkTop3Stolen(newPick);
    checkHerdWarning(newPick);
    
    setPickHistory([newPick, ...pickHistory]);
    setDraftedPlayerNames([...draftedPlayerNames, player.name]);

    // Move to next pick
    advancePick();
  };

  // Update user's roster
  const updateRoster = (player) => {
    const position = player.position;
    
    // Try to fill starting positions first
    if (position === 'QB' && !roster.QB) {
      setRoster({...roster, QB: player});
    } else if (position === 'RB') {
      if (!roster.RB1) {
        setRoster({...roster, RB1: player});
      } else if (!roster.RB2) {
        setRoster({...roster, RB2: player});
      } else if (!roster.FLX) {
        setRoster({...roster, FLX: player});
      } else {
        fillBench(player);
      }
    } else if (position === 'WR') {
      if (!roster.WR1) {
        setRoster({...roster, WR1: player});
      } else if (!roster.WR2) {
        setRoster({...roster, WR2: player});
      } else if (!roster.FLX) {
        setRoster({...roster, FLX: player});
      } else {
        fillBench(player);
      }
    } else if (position === 'TE') {
      if (!roster.TE) {
        setRoster({...roster, TE: player});
      } else if (!roster.FLX) {
        setRoster({...roster, FLX: player});
      } else {
        fillBench(player);
      }
    } else if (position === 'DST' && !roster.DST) {
      setRoster({...roster, DST: player});
    } else if (position === 'K' && !roster.K) {
      setRoster({...roster, K: player});
    } else {
      fillBench(player);
    }
  };

  // Advance to next pick
  const advancePick = () => {
    if (draftComplete) return;
    
    const totalPicks = draftSettings.numTeams * 16; // 16 rounds = 16 players per team
    
    // Check if draft is complete
    if (currentPickNumber >= totalPicks) {
      // Draft is complete, but don't change draftStarted - let the useEffect handle it
      return;
    }
    
    const nextPickNumber = currentPickNumber + 1;
    const nextRound = Math.ceil(nextPickNumber / draftSettings.numTeams);
    
    // Don't advance if we've exceeded total picks
    if (nextPickNumber > totalPicks) {
      return;
    }
    
    setCurrentPickNumber(nextPickNumber);
    setCurrentRound(nextRound);
    setTimeRemaining(draftSettings.pickTime);
    
    // Check if next pick is user's turn
    const nextIsUserTurn = checkIfUserTurn(nextPickNumber);
    setIsUserTurn(nextIsUserTurn);
  };

  // Handle CPU auto-drafting when it's not user's turn
  useEffect(() => {
    if (!draftStarted || isUserTurn || draftComplete) return;
    
    // Check if draft is complete
    const totalPicks = draftSettings.numTeams * 16;
    if (currentPickNumber > totalPicks) {
      return;
    }
    
    if (!allPlayersRef.current || !Array.isArray(allPlayersRef.current)) {
      return;
    }

    const cpuDraftTimer = setTimeout(() => {
      // Check again if draft is complete
      const totalPicks = draftSettings.numTeams * 16;
      if (currentPickNumber > totalPicks || draftComplete) {
        return;
      }
      
      const availablePlayers = allPlayersRef.current.filter(
        p => !draftedPlayerNames.includes(p.name)
      );
      if (availablePlayers.length > 0) {
        const randomIndex = Math.floor(Math.random() * Math.min(3, availablePlayers.length));
        const cpuPick = availablePlayers[randomIndex];
        draftPlayerCPU(cpuPick);
      }
    }, 2000); // 2 second delay for CPU picks

    return () => clearTimeout(cpuDraftTimer);
  }, [currentPickNumber, isUserTurn, draftStarted, draftComplete, draftSettings.numTeams]);

  // Update Top 3 whenever draft state changes
  useEffect(() => {
    if (draftStarted) {
      const newTop3 = getSmartTop3();
      setCurrentTop3(newTop3);
    }
  }, [draftStarted, currentPickNumber, draftedPlayerNames, rejectedPlayerNames, isUserTurn]);

  const fillBench = (player) => {
    if (!roster.BE1) {
      setRoster({...roster, BE1: player});
    } else if (!roster.BE2) {
      setRoster({...roster, BE2: player});
    } else if (!roster.BE3) {
      setRoster({...roster, BE3: player});
    } else if (!roster.BE4) {
      setRoster({...roster, BE4: player});
    } else if (!roster.BE5) {
      setRoster({...roster, BE5: player});
    } else if (!roster.BE6) {
      setRoster({...roster, BE6: player});
    } else if (!roster.BE7) {
      setRoster({...roster, BE7: player});
    }
  };

  // Check if roster is full (all 16 spots filled: 9 starting + 7 bench)
  const isRosterFull = () => {
    const startingPositions = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE', 'FLX', 'DST', 'K'];
    const benchPositions = ['BE1', 'BE2', 'BE3', 'BE4', 'BE5', 'BE6', 'BE7'];
    const allPositions = [...startingPositions, ...benchPositions];
    
    return allPositions.every(pos => roster[pos] !== null);
  };

  // Check if all teams have filled their rosters (16 picks each)
  const areAllTeamsFull = () => {
    const numTeams = draftSettings.numTeams;
    const picksPerTeam = {};
    
    // Initialize pick counts for all teams
    for (let i = 1; i <= numTeams; i++) {
      picksPerTeam[i] = 0;
    }
    
    // Count picks per team from pickHistory
    pickHistory.forEach(pick => {
      const pickNum = pick.pickNumber;
      const round = pick.round || Math.ceil(pickNum / numTeams);
      const isSnakeDraft = round % 2 === 0; // Even rounds go in reverse
      
      // Calculate which team made this pick (same logic as getCurrentTeamName)
      let positionInRound = ((pickNum - 1) % numTeams) + 1;
      if (isSnakeDraft) {
        positionInRound = numTeams - positionInRound + 1;
      }
      
      // positionInRound is the team number (1 to numTeams)
      if (positionInRound >= 1 && positionInRound <= numTeams) {
        picksPerTeam[positionInRound]++;
      }
    });
    
    // Check if all teams have 16 picks
    return Object.values(picksPerTeam).every(count => count >= 16);
  };

  // Check if draft should end (user roster full AND all teams full)
  useEffect(() => {
    if (!draftStarted) return;
    
    const userRosterFull = isRosterFull();
    const allTeamsFull = areAllTeamsFull();
    
    console.log('Draft status check:', {
      draftStarted,
      userRosterFull,
      allTeamsFull,
      pickHistoryLength: pickHistory.length,
      numTeams: draftSettings.numTeams
    });
    
    if (userRosterFull && allTeamsFull) {
      console.log('Draft complete! User roster full and all teams full.');
      setDraftComplete(true);
      // Don't set draftStarted to false - keep the draft interface visible
    }
  }, [draftStarted, pickHistory, roster, draftSettings.numTeams]);

  const renderBigBoardContent = () => {
    if (activeView === 'overall-picks') {
      return <OverallPicks pickHistory={pickHistory} draftSettings={draftSettings} />;
    } else if (activeView === 'analytics') {
      return <Analytics allPlayers={allPlayersRef.current || []} draftedPlayerNames={draftedPlayerNames} draftSettings={draftSettings} roster={roster} />;
    } else {
      // Check if roster is full
      if (isRosterFull()) {
        return (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '100%',
            fontSize: '24px',
            fontWeight: 'bold',
            color: 'rgb(179, 163, 105)'
          }}>
            Draft Finished
          </div>
        );
      }
      
      return (
        <BigBoard 
          onDraftPlayer={draftPlayer} 
          draftedPlayerNames={draftedPlayerNames}
          isUserTurn={isUserTurn}
        />
      );
    }
  };

  // Show setup screen if draft hasn't started (but not if draft is complete)
  if (!draftStarted && !draftComplete) {
    return (
      <div className="App">
        <div className="header">
          <h1>Fantasy Football Draft Optimizer</h1>
        </div>
        
        <div className="setup-container">
          <div className="setup-card">
            <h2>Draft Setup</h2>
            <p className="setup-subtitle">Configure your draft settings to begin</p>
            
            <div className="setup-form">
              <div className="form-group">
                <label htmlFor="pickTime">Pick Time (seconds)</label>
                <select
                  id="pickTime"
                  value={draftSettings.pickTime}
                  onChange={(e) => handleSettingChange('pickTime', parseInt(e.target.value))}
                  className="form-input"
                >
                  <option value="1">1 Second (Testing)</option>
                  <option value="3">3 Seconds (Testing)</option>
                  <option value="300">5 Minutes (Testing)</option>
                  <option value="30">30 Seconds</option>
                  <option value="60">60 Seconds</option>
                  <option value="90">90 Seconds</option>
                </select>
                <span className="form-help">Time allowed per pick</span>
              </div>
              
              <div className="form-group">
                <label htmlFor="numTeams">Number of Teams</label>
                <select
                  id="numTeams"
                  value={draftSettings.numTeams}
                  onChange={(e) => handleSettingChange('numTeams', parseInt(e.target.value))}
                  className="form-input"
                >
                  <option value="1">1 Team (Testing)</option>
                  <option value="8">8 Teams</option>
                  <option value="10">10 Teams</option>
                  <option value="12">12 Teams</option>
                  <option value="14">14 Teams</option>
                  <option value="16">16 Teams</option>
                </select>
                <span className="form-help">Total number of teams in the draft</span>
              </div>
              
              <div className="form-group">
                <label htmlFor="startingPick">Your Starting Pick Number</label>
                <input
                  type="number"
                  id="startingPick"
                  value={draftSettings.startingPickNumber}
                  onChange={(e) => handleSettingChange('startingPickNumber', parseInt(e.target.value))}
                  min="1"
                  max={draftSettings.numTeams}
                  className="form-input"
                />
                <span className="form-help">Your position in the draft order (1-{draftSettings.numTeams})</span>
              </div>
              
              <button onClick={handleStartDraft} className="start-draft-button">
                Start Draft
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="App">
      <div className="header">
        <h1>Fantasy Football Draft Optimizer</h1>
      </div>
      
      {draftComplete && (
        <div style={{
          backgroundColor: 'rgb(179, 163, 105)',
          color: 'white',
          padding: '15px',
          textAlign: 'center',
          fontSize: '20px',
          fontWeight: 'bold',
          marginBottom: '10px'
        }}>
          🎉 Draft Complete! All teams have filled their rosters. 🎉
        </div>
      )}
      
      <div className="draft-status-bar">
        <div className="draft-info">
          <div className="round-info">
            <div className="round-text">RND {currentRound} OF 16</div>
            <div className={`timer ${timeRemaining <= 10 ? 'urgent' : ''}`}>
              {Math.floor(timeRemaining / 60).toString().padStart(2, '0')}:
              {(timeRemaining % 60).toString().padStart(2, '0')}
            </div>
          </div>
          <div className="current-pick">
            <div className="helmet-icon">🏈</div>
            <div className="pick-info">
              <div className="on-clock">ON THE CLOCK: PICK {currentPickNumber}</div>
              <div className="team-name">{getCurrentTeamName(currentPickNumber)}</div>
            </div>
          </div>
        </div>
        
        <div className="upcoming-picks">
          {[...Array(Math.max(0, Math.min(9, draftSettings.numTeams * 16 - currentPickNumber)))].map((_, i) => {
            const pickNum = currentPickNumber + i + 1;
            const round = Math.ceil(pickNum / draftSettings.numTeams);
            const prevRound = Math.ceil((pickNum - 1) / draftSettings.numTeams);
            const showRoundSeparator = round !== prevRound && i > 0;
            const teamName = getCurrentTeamName(pickNum);
            const isUserPick = checkIfUserTurn(pickNum);
            
            return (
              <React.Fragment key={pickNum}>
                {showRoundSeparator && <div className="round-separator">ROUND {round}</div>}
                <div className={`pick-box ${isUserPick ? 'user-team' : ''}`}>
                  <div className="pick-number">PICK {pickNum}</div>
                  {isUserPick ? (
                    <div className="helmet-small">🏈</div>
                  ) : (
                    <div className="auto-indicator">AUTO</div>
                  )}
                  <div className="team-name">{teamName}</div>
                </div>
              </React.Fragment>
            );
          })}
        </div>
      </div>
      
      <div className="main-layout">
        <div className="main-content">
          <div className="roster-panel">
            <h3>Your Roster</h3>
            {(() => {
              const totalPoints = Object.values(roster)
                .filter(player => player !== null)
                .reduce((sum, player) => sum + (parseFloat(player.pts) || 0), 0);
              return (
                <div className="total-points">
                  <span className="total-points-label">Total Proj. Points:</span>
                  <span className="total-points-value">{totalPoints.toFixed(1)}</span>
                </div>
              );
            })()}
            <div className="position-slots">
              <div className="position-slot">
                <span className="position-label">QB</span>
                <div className={`player-field ${roster.QB ? 'filled' : ''}`}>
                  {roster.QB ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.QB.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.QB.bye} <span className="stat-label">Proj. Pts.:</span> {roster.QB.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">RB</span>
                <div className={`player-field ${roster.RB1 ? 'filled' : ''}`}>
                  {roster.RB1 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.RB1.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.RB1.bye} <span className="stat-label">Proj. Pts.:</span> {roster.RB1.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">RB</span>
                <div className={`player-field ${roster.RB2 ? 'filled' : ''}`}>
                  {roster.RB2 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.RB2.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.RB2.bye} <span className="stat-label">Proj. Pts.:</span> {roster.RB2.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">WR</span>
                <div className={`player-field ${roster.WR1 ? 'filled' : ''}`}>
                  {roster.WR1 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.WR1.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.WR1.bye} <span className="stat-label">Proj. Pts.:</span> {roster.WR1.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">WR</span>
                <div className={`player-field ${roster.WR2 ? 'filled' : ''}`}>
                  {roster.WR2 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.WR2.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.WR2.bye} <span className="stat-label">Proj. Pts.:</span> {roster.WR2.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">TE</span>
                <div className={`player-field ${roster.TE ? 'filled' : ''}`}>
                  {roster.TE ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.TE.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.TE.bye} <span className="stat-label">Proj. Pts.:</span> {roster.TE.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">FLX</span>
                <div className={`player-field ${roster.FLX ? 'filled' : ''}`}>
                  {roster.FLX ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.FLX.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.FLX.bye} <span className="stat-label">Proj. Pts.:</span> {roster.FLX.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">D/ST</span>
                <div className={`player-field ${roster.DST ? 'filled' : ''}`}>
                  {roster.DST ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.DST.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.DST.bye} <span className="stat-label">Proj. Pts.:</span> {roster.DST.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">K</span>
                <div className={`player-field ${roster.K ? 'filled' : ''}`}>
                  {roster.K ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.K.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.K.bye} <span className="stat-label">Proj. Pts.:</span> {roster.K.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              
              <div className="bench-divider">BENCH</div>
              
              <div className="position-slot">
                <span className="position-label">BE</span>
                <div className={`player-field ${roster.BE1 ? 'filled' : ''}`}>
                  {roster.BE1 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.BE1.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.BE1.bye} <span className="stat-label">Proj. Pts.:</span> {roster.BE1.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">BE</span>
                <div className={`player-field ${roster.BE2 ? 'filled' : ''}`}>
                  {roster.BE2 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.BE2.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.BE2.bye} <span className="stat-label">Proj. Pts.:</span> {roster.BE2.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">BE</span>
                <div className={`player-field ${roster.BE3 ? 'filled' : ''}`}>
                  {roster.BE3 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.BE3.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.BE3.bye} <span className="stat-label">Proj. Pts.:</span> {roster.BE3.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">BE</span>
                <div className={`player-field ${roster.BE4 ? 'filled' : ''}`}>
                  {roster.BE4 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.BE4.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.BE4.bye} <span className="stat-label">Proj. Pts.:</span> {roster.BE4.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">BE</span>
                <div className={`player-field ${roster.BE5 ? 'filled' : ''}`}>
                  {roster.BE5 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.BE5.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.BE5.bye} <span className="stat-label">Proj. Pts.:</span> {roster.BE5.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">BE</span>
                <div className={`player-field ${roster.BE6 ? 'filled' : ''}`}>
                  {roster.BE6 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.BE6.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.BE6.bye} <span className="stat-label">Proj. Pts.:</span> {roster.BE6.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="position-slot">
                <span className="position-label">BE</span>
                <div className={`player-field ${roster.BE7 ? 'filled' : ''}`}>
                  {roster.BE7 ? (
                    <div className="player-info-container">
                      <div className="player-name-row">{roster.BE7.name}</div>
                      <div className="player-stats">
                        <span className="stat-label">BYE:</span> {roster.BE7.bye} <span className="stat-label">Proj. Pts.:</span> {roster.BE7.pts}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
          
          <div className="top3-panel">
            <h3>Your Top 3</h3>
            <div className="player-cards">
              {currentTop3.map((player, index) => (
                  <div key={index} className="top3-player-card">
                    <div className="top3-player-icon">👤</div>
                    <div className="top3-player-info">
                      <div className="top3-player-name">{player.name}</div>
                      <div className="top3-player-position">{player.position} - {player.team}</div>
                      <div className="top3-player-stats">
                        <span className="top3-stat-label">BYE:</span> {player.bye} <span className="top3-stat-label">Proj. Pts.:</span> {player.pts}
                      </div>
                    </div>
                    <div className="top3-card-actions">
                      <button 
                        className="draft-button"
                        onClick={() => isUserTurn && draftPlayer(player)}
                        disabled={!isUserTurn}
                      >
                        Draft
                      </button>
                      <button 
                        className="close-button"
                        onClick={() => handleRejectPlayer(player.name)}
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                ))}
            </div>
          </div>
          
          <div className="big-board-panel">
            <div className="panel-tabs">
              <div 
                className={`panel-tab ${activeView === 'big-board' ? 'active' : ''}`}
                onClick={() => setActiveView('big-board')}
              >
                Your Big Board
              </div>
              <div 
                className={`panel-tab ${activeView === 'overall-picks' ? 'active' : ''}`}
                onClick={() => setActiveView('overall-picks')}
              >
                Overall Picks
              </div>
              <div 
                className={`panel-tab ${activeView === 'analytics' ? 'active' : ''}`}
                onClick={() => setActiveView('analytics')}
              >
                Analytics
              </div>
            </div>
            {renderBigBoardContent()}
          </div>
        </div>
        
        <div className="sidebar">
            <div className="picks-panel">
              <h3>Picks</h3>
              <div className="picks-list">
                {pickHistory.length === 0 ? (
                  <div className="no-picks">No picks yet. Start drafting!</div>
                ) : (
                  pickHistory.map((pick, index) => (
                    <div key={index} className="pick-item">
                      <div className="pick-icon">👤</div>
                      <div className="pick-details">
                        <div className="pick-player">{pick.name} / {pick.team} {pick.position}</div>
                        <div className="pick-info">(Pick {pick.pickNumber} - {pick.teamName})</div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
            
            <div className="alerts-panel">
              <h3>Alerts</h3>
              <div className="alerts-list">
                {alerts.length === 0 ? (
                  <div className="no-alerts">No alerts yet.</div>
                ) : (
                  alerts.map((alert) => (
                    <div key={alert.id} className={`alert-item ${alert.type}`}>
                      <div className="alert-message">{alert.message}</div>
                      <div className="alert-time">{alert.timestamp}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
    </div>
  );
}

export default App;