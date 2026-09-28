// src/selfplay/curriculum.js
//
// Curriculum Learning
//
// F数ごとに難易度を段階化する
//

function getTargetWildcards(epoch) {
  if (epoch < 20) return 0;
  if (epoch < 50) return 1;
  if (epoch < 100) return 2;
  if (epoch < 200) return 3;

  return 4;
}

function getEpochConfig(epoch) {
  const targetWildcards = getTargetWildcards(epoch);

  if (targetWildcards === 0) {
    return {
      targetWildcards,
      generatorIterations: 30,
      timeLimitMs: 10000,
      solverMaxNodes: 30000,
      solverMaxDepth: 80,
      solverMaxPathsPerWord: 30,
    };
  }

  if (targetWildcards === 1) {
    return {
      targetWildcards,
      generatorIterations: 40,
      timeLimitMs: 12000,
      solverMaxNodes: 50000,
      solverMaxDepth: 90,
      solverMaxPathsPerWord: 40,
    };
  }

  if (targetWildcards === 2) {
    return {
      targetWildcards,
      generatorIterations: 50,
      timeLimitMs: 15000,
      solverMaxNodes: 80000,
      solverMaxDepth: 100,
      solverMaxPathsPerWord: 50,
    };
  }

  if (targetWildcards === 3) {
    return {
      targetWildcards,
      generatorIterations: 60,
      timeLimitMs: 20000,
      solverMaxNodes: 120000,
      solverMaxDepth: 110,
      solverMaxPathsPerWord: 60,
    };
  }

  return {
    targetWildcards,
    generatorIterations: 80,
    timeLimitMs: 30000,
    solverMaxNodes: 200000,
    solverMaxDepth: 120,
    solverMaxPathsPerWord: 80,
  };
}

function getLeagueName(targetWildcards) {
  return `f${targetWildcards}`;
}

module.exports = {
  getTargetWildcards,
  getEpochConfig,
  getLeagueName,
};