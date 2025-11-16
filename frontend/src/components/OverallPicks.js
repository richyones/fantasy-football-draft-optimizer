import React from 'react';
import './OverallPicks.css';

const OverallPicks = ({ pickHistory = [], draftSettings }) => {
  const numTeams = draftSettings?.numTeams || 12;
  const startingPickNumber = draftSettings?.startingPickNumber || 1;
  const totalRounds = 16;

  // Generate team names - identify user's team
  const teams = Array.from({ length: numTeams }, (_, i) => {
    const teamNumber = i + 1;
    const isUserTeam = teamNumber === startingPickNumber;
    return {
      name: isUserTeam ? 'Your team' : `Team ${teamNumber}`,
      icon: '🏈',
      isUserTeam: isUserTeam
    };
  });

  // Organize picks by team (columns)
  const organizePicksByTeam = () => {
    const teamPicks = [];
    
    for (let teamIndex = 0; teamIndex < numTeams; teamIndex++) {
      const picks = [];
      
      for (let round = 1; round <= totalRounds; round++) {
        const isSnakeDraft = round % 2 === 0; // Even rounds go reverse
        let actualTeamIndex = teamIndex;
        
        // Adjust team index for snake draft
        if (isSnakeDraft) {
          actualTeamIndex = numTeams - 1 - teamIndex;
        }
        
        const pickNumber = (round - 1) * numTeams + actualTeamIndex;
        const pick = pickHistory.find(p => p.pickNumber === pickNumber + 1);
        
        if (pick) {
          const positionColor = 
            pick.position === 'QB' ? 'pink' :
            pick.position === 'RB' ? 'lightgreen' :
            pick.position === 'WR' ? 'darkblue' :
            pick.position === 'TE' ? 'orange' :
            pick.position === 'K' ? 'purple' :
            pick.position === 'DST' ? 'teal' :
            'gray';
          
          picks.push({
            player: pick.name,
            pos: pick.position,
            team: pick.team,
            pick: `${round}.${actualTeamIndex + 1}`,
            round: round,
            color: positionColor
          });
        } else {
          // Empty slot for picks not made yet
          picks.push({
            player: '-',
            pos: '',
            team: '',
            pick: `${round}.${actualTeamIndex + 1}`,
            round: round,
            color: 'gray'
          });
        }
      }
      
      teamPicks.push(picks);
    }
    
    return teamPicks;
  };

  const teamPicks = organizePicksByTeam();

  return (
    <div className="overall-picks">
      <div className="draft-board">
        {/* Teams as columns with picks stacked vertically */}
        <div 
          className="teams-container"
          style={{ gridTemplateColumns: `repeat(${numTeams}, 90px)` }}
        >
          {teams.map((team, teamIndex) => (
            <div key={teamIndex} className="team-column">
              {/* Team header */}
              <div className={`team-header ${team.isUserTeam ? 'user-team' : ''}`}>
                <div className="team-icon">{team.icon}</div>
                <div className="team-name">{team.name}</div>
              </div>
              
              {/* Picks for this team */}
              <div className="team-picks">
                {teamPicks[teamIndex].map((pick, pickIndex) => (
                  <div key={pickIndex} className="player-card-container">
                    <div className={`player-card ${pick.color}`}>
                      <div className="pick-number">{pick.pick}</div>
                      {pick.player === '-' ? (
                        <div className="player-name empty-slot">Empty</div>
                      ) : (
                        <>
                          <div className="player-name">{pick.player}</div>
                          <div className="player-info">{pick.pos || ''}</div>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default OverallPicks;
