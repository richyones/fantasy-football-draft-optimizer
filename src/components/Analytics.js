import React, { useMemo, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ReferenceLine, ComposedChart, Cell } from 'recharts';
import './Analytics.css';

const Analytics = ({ allPlayers = [], draftedPlayerNames = [], draftSettings = null, roster = null }) => {
  const [activeSubTab, setActiveSubTab] = useState('distribution');
  // Filter available players (not drafted)
  const availablePlayers = useMemo(() => {
    return allPlayers.filter(player => !draftedPlayerNames.includes(player.name));
  }, [allPlayers, draftedPlayerNames]);

  // Group players by position and calculate statistics
  const positionData = useMemo(() => {
    const positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'];
    const data = {};

    positions.forEach(pos => {
      const playersInPosition = availablePlayers.filter(p => p.position === pos);
      const points = playersInPosition.map(p => p.pts).filter(p => p != null && !isNaN(p));
      
      if (points.length > 0) {
        points.sort((a, b) => a - b);
        const min = Math.min(...points);
        const max = Math.max(...points);
        const q1 = points[Math.floor(points.length * 0.25)];
        const median = points[Math.floor(points.length * 0.5)];
        const q3 = points[Math.floor(points.length * 0.75)];
        const mean = points.reduce((a, b) => a + b, 0) / points.length;

        data[pos] = {
          position: pos,
          count: points.length,
          min,
          max,
          q1,
          median,
          q3,
          mean,
          points: points
        };
      }
    });

    return data;
  }, [availablePlayers]);

  // Calculate global min and max across all positions for consistent scaling
  const globalRange = useMemo(() => {
    const allPoints = availablePlayers
      .map(p => p.pts)
      .filter(p => p != null && !isNaN(p));
    
    if (allPoints.length === 0) return { min: 0, max: 100 };
    
    const min = Math.min(...allPoints);
    const max = Math.max(...allPoints);
    
    // Round min down and max up to nice numbers
    const roundedMin = Math.floor(min / 50) * 50;
    const roundedMax = Math.ceil(max / 50) * 50;
    
    return { min: roundedMin, max: roundedMax };
  }, [availablePlayers]);

  // Prepare histogram data for each position with consistent bucket sizes
  const histogramData = useMemo(() => {
    const positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'];
    const histograms = {};
    const bins = 10;
    const binWidth = (globalRange.max - globalRange.min) / bins;

    // Create consistent bin structure for all positions
    const createBins = () => {
      return Array(bins).fill(0).map((_, i) => {
        const binStart = globalRange.min + i * binWidth;
        const binEnd = globalRange.min + (i + 1) * binWidth;
        return {
          bin: i,
          range: `${Math.round(binStart)}-${Math.round(binEnd)}`,
          binStart: binStart,
          binEnd: binEnd,
          count: 0,
          players: [] // Store player objects in each bin
        };
      });
    };

    positions.forEach(pos => {
      // Get all players for this position with their full data
      const playersInPosition = availablePlayers
        .filter(p => p.position === pos && p.pts != null && !isNaN(p.pts))
        .sort((a, b) => b.pts - a.pts); // Sort by points descending
      
      const binsData = createBins();
      
      playersInPosition.forEach((player, index) => {
        const point = player.pts;
        // Find which bin this point belongs to
        const binIndex = Math.min(
          Math.floor((point - globalRange.min) / binWidth),
          bins - 1
        );
        if (binIndex >= 0 && binIndex < bins) {
          binsData[binIndex].count++;
          // Store player info with rank (index + 1 since sorted by points)
          binsData[binIndex].players.push({
            name: player.name,
            rank: index + 1,
            pts: point
          });
        }
      });

      histograms[pos] = binsData;
    });

    return histograms;
  }, [positionData, globalRange, availablePlayers]);

  // Prepare ranking bucket data (for Analytics 2)
  const rankingBucketData = useMemo(() => {
    const positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'];
    const bucketSize = 10; // 1-10, 11-20, 21-30, etc.
    const rankingData = {};

    positions.forEach(pos => {
      // Get all players for this position, sorted by projected points (descending)
      const playersInPosition = availablePlayers
        .filter(p => p.position === pos && p.pts != null && !isNaN(p.pts))
        .sort((a, b) => b.pts - a.pts);

      if (playersInPosition.length === 0) {
        rankingData[pos] = [];
        return;
      }

      // Group into buckets of 5
      const buckets = [];
      for (let i = 0; i < playersInPosition.length; i += bucketSize) {
        const bucketPlayers = playersInPosition.slice(i, i + bucketSize);
        const avgPoints = bucketPlayers.reduce((sum, p) => sum + p.pts, 0) / bucketPlayers.length;
        const rankStart = i + 1;
        const rankEnd = Math.min(i + bucketSize, playersInPosition.length);
        
        buckets.push({
          rankRange: `${pos}${rankStart}-${rankEnd}`,
          rankStart,
          rankEnd,
          avgPoints: avgPoints,
          count: bucketPlayers.length
        });
      }

      rankingData[pos] = buckets;
    });

    return rankingData;
  }, [availablePlayers]);

  // Calculate global y-axis range for ranking charts (Analytics 2)
  const rankingYAxisRange = useMemo(() => {
    const allAvgPoints = [];
    
    Object.values(rankingBucketData).forEach(buckets => {
      buckets.forEach(bucket => {
        if (bucket.avgPoints != null && !isNaN(bucket.avgPoints)) {
          allAvgPoints.push(bucket.avgPoints);
        }
      });
    });

    if (allAvgPoints.length === 0) return { min: 0, max: 100 };

    const min = Math.min(...allAvgPoints);
    const max = Math.max(...allAvgPoints);
    
    // Round min down and max up to nice numbers
    const roundedMin = Math.max(0, Math.floor(min / 50) * 50);
    const roundedMax = Math.ceil(max / 50) * 50;
    
    return { min: roundedMin, max: roundedMax };
  }, [rankingBucketData]);

  // Prepare box plot data
  const boxPlotData = useMemo(() => {
    return Object.values(positionData).map(data => ({
      position: data.position,
      min: data.min,
      q1: data.q1,
      median: data.median,
      q3: data.q3,
      max: data.max,
      mean: data.mean,
      count: data.count
    }));
  }, [positionData]);

  // Calculate points added chart data (Analytics 3)
  const pointsAddedData = useMemo(() => {
    const positions = ['QB', 'RB', 'WR', 'TE'];
    const numTeams = draftSettings?.numTeams || 12;
    const data = [];

    positions.forEach(pos => {
      const playersInPosition = availablePlayers
        .filter(p => p.position === pos && p.pts != null && !isNaN(p.pts))
        .sort((a, b) => b.pts - a.pts);

      if (playersInPosition.length === 0) return;

      // Calculate how many starting spots are filled for this position
      let startingSpotsFilled = 0;
      let requiredSpots = 0;
      let shouldShowBench = false;

      if (pos === 'QB') {
        // QB can be in QB slot or FLX slot
        startingSpotsFilled = (roster?.QB ? 1 : 0) + (roster?.FLX && roster.FLX.position === 'QB' ? 1 : 0);
        requiredSpots = 1; // QB needs 1 starter
        shouldShowBench = startingSpotsFilled >= requiredSpots;
      } else if (pos === 'RB') {
        // RB: bench points show if both RB slots are filled AND flex is filled (by WR or RB)
        const rb1Filled = roster?.RB1 ? 1 : 0;
        const rb2Filled = roster?.RB2 ? 1 : 0;
        const flexFilled = roster?.FLX ? 1 : 0;
        startingSpotsFilled = rb1Filled + rb2Filled + (roster?.FLX && roster.FLX.position === 'RB' ? 1 : 0);
        requiredSpots = 2; // RB needs 2 starters
        shouldShowBench = (rb1Filled === 1 && rb2Filled === 1 && flexFilled === 1);
      } else if (pos === 'WR') {
        // WR: bench points show if both WR slots are filled AND flex is filled (by WR or RB)
        const wr1Filled = roster?.WR1 ? 1 : 0;
        const wr2Filled = roster?.WR2 ? 1 : 0;
        const flexFilled = roster?.FLX ? 1 : 0;
        startingSpotsFilled = wr1Filled + wr2Filled + (roster?.FLX && roster.FLX.position === 'WR' ? 1 : 0);
        requiredSpots = 2; // WR needs 2 starters
        shouldShowBench = (wr1Filled === 1 && wr2Filled === 1 && flexFilled === 1);
      } else if (pos === 'TE') {
        // TE can be in TE slot or FLX slot
        startingSpotsFilled = (roster?.TE ? 1 : 0) + (roster?.FLX && roster.FLX.position === 'TE' ? 1 : 0);
        requiredSpots = 1; // TE needs 1 starter
        shouldShowBench = startingSpotsFilled >= requiredSpots;
      }

      let topRankPoints = null;
      let replacementPoints = null;
      let benchPlayerPoints = null;

      // If required starting spots are filled (and for RB/WR, flex must also be filled), show ONLY bench points
      if (shouldShowBench) {
        // Required spots are filled, so the next player would be bench
        if (playersInPosition.length > 0) {
          benchPlayerPoints = playersInPosition[0]?.pts || null;
        }
      } else {
        // Required spots are NOT filled, show top rank and replacement rank
        topRankPoints = playersInPosition[0]?.pts || 0;
        
        // Replacement rank points added
        const replacementRank = numTeams;
        const replacementPlayer = playersInPosition[Math.min(replacementRank - 1, playersInPosition.length - 1)];
        replacementPoints = replacementPlayer?.pts || 0;
      }

      data.push({
        position: pos,
        topRankPoints: topRankPoints,
        replacementPoints: replacementPoints,
        benchPoints: benchPlayerPoints
      });
    });

    return data;
  }, [availablePlayers, draftSettings, roster]);

  // Calculate tier dropoff data (Analytics 3) - individual ranks
  const tierDropoffData = useMemo(() => {
    const positions = ['QB', 'RB', 'WR', 'TE'];
    const maxRank = 30;
    const rankData = [];

    for (let rank = 1; rank <= maxRank; rank++) {
      const rankEntry = { rank: rank };
      
      positions.forEach(pos => {
        const playersInPosition = availablePlayers
          .filter(p => p.position === pos && p.pts != null && !isNaN(p.pts))
          .sort((a, b) => b.pts - a.pts);

        if (rank <= playersInPosition.length) {
          rankEntry[pos] = playersInPosition[rank - 1].pts;
        } else {
          rankEntry[pos] = null;
        }
      });
      
      rankData.push(rankEntry);
    }

    return rankData;
  }, [availablePlayers]);

  // Calculate Value Over Replacement (VOR) (Analytics 4)
  const vorData = useMemo(() => {
    const positions = ['QB', 'RB', 'WR', 'TE'];
    
    // Calculate replacement levels dynamically based on number of teams
    const numTeams = draftSettings?.numTeams || 12;
    const replacementLevels = {
      QB: numTeams,  // QB12 for 12 teams
      RB: numTeams,  // RB12 for 12 teams
      WR: numTeams,  // WR12 for 12 teams
      TE: numTeams   // TE12 for 12 teams
    };

    const vor = [];

    positions.forEach(pos => {
      const playersInPosition = availablePlayers
        .filter(p => p.position === pos && p.pts != null && !isNaN(p.pts))
        .sort((a, b) => b.pts - a.pts);

      if (playersInPosition.length === 0) return;

      const replacementRank = replacementLevels[pos];
      const replacementPlayer = playersInPosition[Math.min(replacementRank - 1, playersInPosition.length - 1)];
      const replacementValue = replacementPlayer?.pts || 0;

      // Calculate VOR for top 15 players at each position
      const topPlayers = playersInPosition.slice(0, 15);
      topPlayers.forEach((player, index) => {
        const rank = index + 1;
        if (!vor.find(v => v.rank === rank)) {
          vor.push({ rank: rank });
        }
        const vorEntry = vor.find(v => v.rank === rank);
        vorEntry[`${pos}_VOR`] = player.pts - replacementValue;
        vorEntry[`${pos}_Points`] = player.pts;
      });
    });

    return vor.sort((a, b) => a.rank - b.rank);
  }, [availablePlayers, draftSettings]);

  // Calculate scarcity heatmap data (Analytics 5)
  const scarcityHeatmapData = useMemo(() => {
    const positions = ['QB', 'RB', 'WR', 'TE'];
    const maxRank = 30;
    const heatmapData = [];

    for (let rank = 1; rank <= maxRank; rank++) {
      const row = { rank: rank };
      positions.forEach(pos => {
        const playersInPosition = availablePlayers
          .filter(p => p.position === pos && p.pts != null && !isNaN(p.pts))
          .sort((a, b) => b.pts - a.pts);

        if (rank <= playersInPosition.length) {
          row[pos] = playersInPosition[rank - 1].pts;
        } else {
          row[pos] = null;
        }
      });
      heatmapData.push(row);
    }

    return heatmapData;
  }, [availablePlayers]);

  const renderDistributionView = () => {
    // Calculate y-axis range for boxplots
    const allValues = boxPlotData.flatMap(d => [d.min, d.max]);
    const yMin = Math.min(...allValues);
    const yMax = Math.max(...allValues);
    const yPadding = (yMax - yMin) * 0.1;
    
    // Custom shape for boxplot - uses recharts coordinate system
    const BoxPlotShape = (props) => {
      const { x, y, width, payload } = props;
      if (!payload) return null;
      
      // x is the left edge of the bar, so center is x + width/2
      const centerX = x + width / 2;
      const boxWidth = width * 0.4;
      const boxX = centerX - boxWidth / 2;
      
      // Calculate relative positions based on data values
      // y is the y-coordinate of the median bar
      // We need to calculate the y-coordinates for other values
      const range = (yMax + yPadding) - (yMin - yPadding);
      const chartHeight = 350; // Approximate chart height
      
      // Calculate the y-coordinate for a given value
      // y increases downward in SVG, but values increase upward
      const getY = (value) => {
        const ratio = (value - (yMin - yPadding)) / range;
        return chartHeight - (ratio * chartHeight);
      };
      
      // Get y positions relative to the median (y)
      // The median bar's y is at the bottom of the bar
      const medianY = y; // This is the bottom of the median bar
      const medianValue = payload.median;
      const valueToPixel = chartHeight / range;
      
      const yMinPos = medianY - (payload.max - medianValue) * valueToPixel;
      const yQ1Pos = medianY - (payload.q1 - medianValue) * valueToPixel;
      const yMedianPos = medianY;
      const yQ3Pos = medianY - (payload.q3 - medianValue) * valueToPixel;
      const yMaxPos = medianY - (payload.min - medianValue) * valueToPixel;
      const boxHeight = Math.abs(yQ3Pos - yQ1Pos);
      
      return (
        <g>
          {/* Whiskers */}
          <line x1={centerX} y1={yMinPos} x2={centerX} y2={yQ1Pos} stroke="rgb(0, 48, 87)" strokeWidth={2} />
          <line x1={centerX} y1={yQ3Pos} x2={centerX} y2={yMaxPos} stroke="rgb(0, 48, 87)" strokeWidth={2} />
          {/* Min/Max lines */}
          <line x1={boxX} y1={yMinPos} x2={boxX + boxWidth} y2={yMinPos} stroke="rgb(0, 48, 87)" strokeWidth={2} />
          <line x1={boxX} y1={yMaxPos} x2={boxX + boxWidth} y2={yMaxPos} stroke="rgb(0, 48, 87)" strokeWidth={2} />
          {/* Box */}
          <rect x={boxX} y={Math.min(yQ1Pos, yQ3Pos)} width={boxWidth} height={boxHeight} fill="rgb(0, 48, 87)" fillOpacity={0.6} stroke="rgb(0, 48, 87)" strokeWidth={2} />
          {/* Median line */}
          <line x1={boxX} y1={yMedianPos} x2={boxX + boxWidth} y2={yMedianPos} stroke="rgb(179, 163, 105)" strokeWidth={2} />
          {/* Mean marker */}
          <circle cx={centerX} cy={medianY - (payload.mean - medianValue) * valueToPixel} r={4} fill="rgb(255, 0, 0)" />
        </g>
      );
    };
    
    return (
      <>
        <h3>Projected Points Distribution by Position</h3>
        <div className="analytics-info">
          <p>Available Players: {availablePlayers.length} | Drafted: {draftedPlayerNames.length}</p>
          <p>Boxplots show min, Q1, median, Q3, max, and mean (red dot) for each position</p>
        </div>

        <div className="charts-container">
          <div className="chart-section">
            <h4>Boxplot Comparison by Position</h4>
            <p style={{ marginBottom: '15px', fontSize: '14px', color: '#666' }}><strong>How to use:</strong> Compare the spread of projected points across positions to identify which positions have the most depth. Positions with tighter spreads indicate more consistent value, delaying a pick at the position might not have that much effect. Larger spreads suggest greater variance and thus these players want to be considered sooner rather than later.</p>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={boxPlotData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="position" />
                <YAxis 
                  label={{ value: 'Projected Points', angle: -90, position: 'insideLeft' }}
                  domain={[yMin - yPadding, yMax + yPadding]}
                  tickFormatter={(value) => Math.round(value).toString()}
                />
                <Tooltip 
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="custom-tooltip">
                          <p className="tooltip-label"><strong>{data.position}</strong></p>
                          <p>Count: {data.count} players</p>
                          <p>Min: {data.min.toFixed(1)}</p>
                          <p>Q1: {data.q1.toFixed(1)}</p>
                          <p>Median: {data.median.toFixed(1)}</p>
                          <p>Q3: {data.q3.toFixed(1)}</p>
                          <p>Max: {data.max.toFixed(1)}</p>
                          <p>Mean: {data.mean.toFixed(1)}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend />
                {/* Use median bar to position boxplots */}
                <Bar dataKey="median" fill="transparent" isAnimationActive={false} shape={<BoxPlotShape />} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Statistics Summary Table */}
          {boxPlotData.length > 0 && (
            <div className="chart-section">
              <h4>Statistics Summary by Position</h4>
              <div className="stats-table-container">
                <table className="stats-table">
                  <thead>
                    <tr>
                      <th>Position</th>
                      <th>Count</th>
                      <th>Min</th>
                      <th>Q1</th>
                      <th>Median</th>
                      <th>Q3</th>
                      <th>Max</th>
                      <th>Mean</th>
                    </tr>
                  </thead>
                  <tbody>
                    {boxPlotData.map(data => (
                      <tr key={data.position}>
                        <td className="position-cell">{data.position}</td>
                        <td>{data.count}</td>
                        <td>{data.min.toFixed(1)}</td>
                        <td>{data.q1.toFixed(1)}</td>
                        <td className="median-cell">{data.median.toFixed(1)}</td>
                        <td>{data.q3.toFixed(1)}</td>
                        <td>{data.max.toFixed(1)}</td>
                        <td className="mean-cell">{data.mean.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </>
    );
  }

  const renderValueDropoffView = () => {
    return (
      <>
        <h3>Points Added to Team & Tier Dropoff</h3>
        <div className="analytics-info">
          <p>Available Players: {availablePlayers.length} | Drafted: {draftedPlayerNames.length}</p>
          <p>Shows projected points added by top rank, replacement rank, and bench players (when starting spots are filled)</p>
        </div>

        <div className="charts-container">
          {/* Points Added Chart */}
          <div className="chart-section">
            <h4>Points Added to Team</h4>
            <p style={{ marginBottom: '15px', fontSize: '14px', color: '#666' }}><strong>How to use:</strong> The visual shows the value gap between the remaining top player and potential replacement player at each position. Larger gaps indicate positions where elite players provide more value. The grey bar shows the points added to the BENCH (non-scoring) by picking the highest ranked player.</p>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={pointsAddedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="position" />
                <YAxis label={{ value: 'Projected Points', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend 
                  content={({ payload }) => {
                    if (!payload || payload.length === 0) return null;
                    
                    // Reorder payload to match desired order
                    const orderedPayload = [
                      payload.find(item => item.dataKey === 'topRankPoints'),
                      payload.find(item => item.dataKey === 'replacementPoints'),
                      payload.find(item => item.dataKey === 'benchPoints')
                    ].filter(Boolean);
                    
                    return (
                      <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '10px', padding: '10px' }}>
                        {orderedPayload.map((entry, index) => {
                          if (!entry) return null;
                          return (
                            <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                              <div style={{ width: '14px', height: '14px', backgroundColor: entry.color || entry.fill }}></div>
                              <span style={{ fontSize: '12px', color: '#333' }}>{entry.name || entry.value}</span>
                            </div>
                          );
                        })}
                      </div>
                    );
                  }}
                />
                <Bar dataKey="topRankPoints" fill="rgb(0, 48, 87)" name="Top Player Points" />
                <Bar dataKey="replacementPoints" fill="rgb(179, 163, 105)" name="Replacement Player Points" />
                <Bar dataKey="benchPoints" fill="#d3d3d3" name="Top Player Bench Points" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Tier Dropoff Chart */}
          <div className="chart-section">
            <h4>Tier Dropoff Comparison</h4>
            <p style={{ marginBottom: '15px', fontSize: '14px', color: '#666' }}><strong>How to use:</strong> The tier dropoff line reveals how quickly value decreases as the draft moves; steeper drops suggest prioritizing thatposition earlier, while flatter declines mean you can wait and still find comparable players.</p>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={tierDropoffData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="rank" label={{ value: 'Player Rank', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Projected Points', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="QB" stroke="#8884d8" strokeWidth={2} name="QB" isAnimationActive={false} />
                <Line type="monotone" dataKey="RB" stroke="#82ca9d" strokeWidth={2} name="RB" isAnimationActive={false} />
                <Line type="monotone" dataKey="WR" stroke="#ffc658" strokeWidth={2} name="WR" isAnimationActive={false} />
                <Line type="monotone" dataKey="TE" stroke="#ff7300" strokeWidth={2} name="TE" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </>
    );
  }

  const renderVORView = () => {
    return (
      <>
        <h3>Value Over Replacement (VOR)</h3>
        <div className="analytics-info">
          <p>Available Players: {availablePlayers.length} | Drafted: {draftedPlayerNames.length}</p>
          {(() => {
            const numTeams = draftSettings?.numTeams || 12;
            return (
              <p>VOR = Player Points - Replacement Level Points (QB{numTeams}, RB{numTeams}, WR{numTeams}, TE{numTeams})</p>
            );
          })()}
        </div>

        <div className="charts-container">
          <div className="chart-section">
            <h4>VOR by Position and Rank</h4>
            <p style={{ marginBottom: '15px', fontSize: '14px', color: '#666' }}><strong>How to use:</strong> VOR measures the gap between a player and their potential replacement at their position. Higher VOR indicates that a position might have more value in drafting now, as the dropoff to the replacement is steeper. </p>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={vorData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="rank" label={{ value: 'Player Rank', position: 'insideBottom', offset: -5 }} />
                <YAxis label={{ value: 'Value Over Replacement', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="QB_VOR" stroke="#8884d8" strokeWidth={2} name="QB VOR" isAnimationActive={false} />
                <Line type="monotone" dataKey="RB_VOR" stroke="#82ca9d" strokeWidth={2} name="RB VOR" isAnimationActive={false} />
                <Line type="monotone" dataKey="WR_VOR" stroke="#ffc658" strokeWidth={2} name="WR VOR" isAnimationActive={false} />
                <Line type="monotone" dataKey="TE_VOR" stroke="#ff7300" strokeWidth={2} name="TE VOR" isAnimationActive={false} />
                <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </>
    );
  }


  // Main render with subtabs
  return (
    <div className="analytics-container">
      <div className="analytics-subtabs">
        <div 
          className={`analytics-subtab ${activeSubTab === 'distribution' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('distribution')}
        >
          Distribution
        </div>
        <div 
          className={`analytics-subtab ${activeSubTab === 'value-dropoff' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('value-dropoff')}
        >
          Points Added & Dropoff
        </div>
        <div 
          className={`analytics-subtab ${activeSubTab === 'vor' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('vor')}
        >
          Value Over Replacement
        </div>
      </div>
      
      {activeSubTab === 'distribution' && renderDistributionView()}
      {activeSubTab === 'value-dropoff' && renderValueDropoffView()}
      {activeSubTab === 'vor' && renderVORView()}
    </div>
  );
};

export default Analytics;

