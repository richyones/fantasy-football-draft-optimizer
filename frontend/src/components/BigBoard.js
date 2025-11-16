import React, { useState, useEffect } from 'react';
import './BigBoard.css';

const BigBoard = ({ onDraftPlayer, draftedPlayerNames = [], isUserTurn = false }) => {
  const [selectedPosition, setSelectedPosition] = useState('All Pos.');
  const [selectedTeam, setSelectedTeam] = useState('All NFL Teams');
  const [showDrafted, setShowDrafted] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [playerData, setPlayerData] = useState([]);
  const [loading, setLoading] = useState(true);

  // Load CSV data
  useEffect(() => {
    const loadCSVData = async () => {
      try {
        // Load both CSVs
        const [draftBoardResponse, playerInfoResponse] = await Promise.all([
          fetch(`${process.env.PUBLIC_URL}/player_adp_optimized_FINAL.csv`),
          fetch(`${process.env.PUBLIC_URL}/nfl_player_information.csv`)
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
          
          // Simple CSV parsing (split by comma, trim whitespace)
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
              // Both "Projected Average Fantasy Points Per Week" and "proj_points" are weekly averages
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
        
        setPlayerData(rankedData);
        setLoading(false);
      } catch (error) {
        console.error('Error loading CSV data:', error);
        // Fallback to empty array or default data
        setPlayerData([]);
        setLoading(false);
      }
    };
    
    loadCSVData();
  }, []);

  // Fallback player database (if CSV fails to load)
  const fallbackPlayerData = [
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
    { rank: 25, name: 'Tee Higgins', team: 'CIN', position: 'WR', bye: 7, pts: 200.5, rush: '-' },
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

  // Use loaded playerData or fallback
  const dataToUse = playerData.length > 0 ? playerData : fallbackPlayerData;
  
  const players = dataToUse.map(p => ({
    ...p,
    drafted: false,
    image: '🏈'
  }));

  const handleDraft = (player) => {
    if (onDraftPlayer && isUserTurn) {
      onDraftPlayer(player);
    }
  };

  const filteredPlayers = players.filter(player => {
    const matchesPosition = selectedPosition === 'All Pos.' || player.position === selectedPosition;
    const matchesTeam = selectedTeam === 'All NFL Teams' || player.team === selectedTeam;
    const matchesSearch = player.name.toLowerCase().includes(searchTerm.toLowerCase());
    const isDrafted = draftedPlayerNames.includes(player.name);
    const matchesDrafted = showDrafted || !isDrafted;
    
    return matchesPosition && matchesTeam && matchesSearch && matchesDrafted;
  });

  // Get best available player for autopick suggestion
  const bestAvailable = filteredPlayers.find(p => !draftedPlayerNames.includes(p.name));

  if (loading) {
    return (
      <div className="big-board">
        <div className="loading-message">Loading player data...</div>
      </div>
    );
  }

  return (
    <div className="big-board">
      <div className="big-board-header">
        {isUserTurn && bestAvailable && (
          <div className="draft-banner">
            <div className="banner-content">
              <div className="banner-text">
                <h2>You are on the clock!</h2>
                <p>Your autopick would be: {bestAvailable.name} / {bestAvailable.team} {bestAvailable.position}</p>
              </div>
              <div className="banner-player">
                <div className="player-image">🏈</div>
              </div>
              <button className="draft-button" onClick={() => handleDraft(bestAvailable)}>DRAFT</button>
            </div>
          </div>
        )}
        
        <div className="filters">
          <select value={selectedPosition} onChange={(e) => setSelectedPosition(e.target.value)}>
            <option value="All Pos.">All Pos.</option>
            <option value="QB">QB</option>
            <option value="RB">RB</option>
            <option value="WR">WR</option>
            <option value="TE">TE</option>
            <option value="K">K</option>
            <option value="DST">DST</option>
          </select>
          
          <select value={selectedTeam} onChange={(e) => setSelectedTeam(e.target.value)}>
            <option value="All NFL Teams">All NFL Teams</option>
            <option value="SF">SF</option>
            <option value="MIA">MIA</option>
            <option value="LAC">LAC</option>
            <option value="MIN">MIN</option>
            <option value="LAR">LAR</option>
            <option value="TEN">TEN</option>
            <option value="KC">KC</option>
            <option value="BUF">BUF</option>
            <option value="LV">LV</option>
            <option value="NYG">NYG</option>
            <option value="CIN">CIN</option>
            <option value="CLE">CLE</option>
            <option value="PHI">PHI</option>
            <option value="DAL">DAL</option>
            <option value="PIT">PIT</option>
          </select>
          
          <input 
            type="text" 
            placeholder="Q Player Name" 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          
          <label className="toggle-label">
            <input 
              type="checkbox" 
              checked={showDrafted}
              onChange={(e) => setShowDrafted(e.target.checked)}
            />
            Show Drafted
          </label>
        </div>
      </div>
      
      <div className="players-table">
        <table>
          <thead>
            <tr>
              <th>RK</th>
              <th>PLAYER</th>
              <th>NFL TEAM</th>
              <th>BYE</th>
              <th>PROJ. POINTS</th>
              <th>ACTION</th>
            </tr>
          </thead>
          <tbody>
            {filteredPlayers.map((player, index) => (
              <tr key={index}>
                <td>{player.rank}</td>
                <td className="player-cell">
                  <div className="player-info">
                    <div className="player-image">{player.image}</div>
                    <div className="player-details">
                      <div className="player-name">{player.name}</div>
                      <div className="player-position">{player.position}</div>
                    </div>
                  </div>
                </td>
                <td>{player.team}</td>
                <td>{player.bye}</td>
                <td>{player.pts}</td>
                <td>
                  <button 
                    className="draft-btn" 
                    onClick={() => handleDraft(player)}
                    disabled={draftedPlayerNames.includes(player.name) || !isUserTurn}
                  >
                    {draftedPlayerNames.includes(player.name) ? 'DRAFTED' : 'DRAFT'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default BigBoard;
