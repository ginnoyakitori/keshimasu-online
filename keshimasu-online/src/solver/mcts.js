// src/solver/mcts.js
//
// 将来用 MCTS 雛形
//

class MCTSNode {
  constructor(board, parent = null) {
    this.board = board;
    this.parent = parent;

    this.children = [];

    this.visits = 0;
    this.value = 0;
  }
}

function ucb(node, totalVisits) {
  if (node.visits === 0) {
    return Infinity;
  }

  return (
    node.value / node.visits +
    Math.sqrt(
      (2 * Math.log(totalVisits)) /
        node.visits
    )
  );
}

module.exports = {
  MCTSNode,
  ucb,
};