// src/selfplay/league.js
//
// F数ごとのリーグ管理
//

class LeagueTable {
  constructor() {
    this.leagues = new Map();
  }

  ensureLeague(targetWildcards) {
    const key = `f${targetWildcards}`;

    if (!this.leagues.has(key)) {
      this.leagues.set(key, {
        targetWildcards,
        games: 0,
        solved: 0,
        bestScore: -Infinity,
        bestPuzzle: null,
        totalScore: 0,
        totalExploredNodes: 0,
        totalBacktracks: 0,
        totalDeadEnds: 0,
      });
    }

    return this.leagues.get(key);
  }

  record({
    targetWildcards,
    puzzle,
    score,
    solverStats,
    solved,
  }) {
    const league =
      this.ensureLeague(targetWildcards);

    league.games++;

    if (solved) {
      league.solved++;
    }

    league.totalScore += score;

    league.totalExploredNodes +=
      solverStats?.exploredNodes || 0;

    league.totalBacktracks +=
      solverStats?.backtracks || 0;

    league.totalDeadEnds +=
      solverStats?.deadEnds || 0;

    if (score > league.bestScore) {
      league.bestScore = score;
      league.bestPuzzle = puzzle;
    }
  }

  getSummary() {
    return [...this.leagues.values()].map((l) => {
      const games = Math.max(1, l.games);

      return {
        targetWildcards: l.targetWildcards,
        games: l.games,
        solved: l.solved,
        solveRate: l.solved / games,
        bestScore: l.bestScore,
        avgScore: l.totalScore / games,
        avgExploredNodes:
          l.totalExploredNodes / games,
        avgBacktracks:
          l.totalBacktracks / games,
        avgDeadEnds:
          l.totalDeadEnds / games,
      };
    });
  }

  printSummary() {
    console.log("\n=== LEAGUE SUMMARY ===\n");

    const summary = this.getSummary();

    for (const row of summary) {
      console.log(
        [
          `F=${row.targetWildcards}`,
          `games=${row.games}`,
          `solved=${row.solved}`,
          `solveRate=${row.solveRate.toFixed(2)}`,
          `best=${row.bestScore.toFixed(2)}`,
          `avg=${row.avgScore.toFixed(2)}`,
          `avgNodes=${row.avgExploredNodes.toFixed(1)}`,
          `avgBacktracks=${row.avgBacktracks.toFixed(1)}`,
          `avgDeadEnds=${row.avgDeadEnds.toFixed(1)}`,
        ].join(" | ")
      );
    }
  }
}

module.exports = {
  LeagueTable,
};