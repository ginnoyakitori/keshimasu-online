// src/generator/genetic.js

function selectTop(population, size = 10) {
  return [...population]
    .sort((a, b) => b.score - a.score)
    .slice(0, size);
}

function crossover(a, b) {
  const rows = a.length;
  const cols = a[0].length;

  const child = [];

  for (let r = 0; r < rows; r++) {
    const row = [];

    for (let c = 0; c < cols; c++) {
      row.push(Math.random() < 0.5 ? a[r][c] : b[r][c]);
    }

    child.push(row);
  }

  return child;
}

module.exports = {
  selectTop,
  crossover,
};