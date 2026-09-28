// src/neural/value-network.js
//
// value network
//
// 局面難易度予測
//

const tf = require("@tensorflow/tfjs");

function createValueNetwork() {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [4, 4, 1],

      filters: 32,

      kernelSize: 3,

      activation: "relu",

      padding: "same",
    })
  );

  model.add(
    tf.layers.flatten()
  );

  model.add(
    tf.layers.dense({
      units: 64,

      activation: "relu",
    })
  );

  //
  // difficulty value
  //

  model.add(
    tf.layers.dense({
      units: 1,

      activation: "linear",
    })
  );

  model.compile({
    optimizer: "adam",

    loss: "meanSquaredError",
  });

  return model;
}

module.exports = {
  createValueNetwork,
};
