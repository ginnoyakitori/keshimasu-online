// src/neural/policy-network.js
//
// policy network
//

const tf = require("@tensorflow/tfjs");

//
// Input:
// 4x4 board tensor
//
// Output:
// next cell prediction
//

function createPolicyNetwork() {
  const model = tf.sequential();

  //
  // [4,4,1]
  //

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
    tf.layers.conv2d({
      filters: 64,

      kernelSize: 3,

      activation: "relu",

      padding: "same",
    })
  );

  model.add(tf.layers.flatten());

  model.add(
    tf.layers.dense({
      units: 128,

      activation: "relu",
    })
  );

  //
  // 4x4 = 16 cells
  //

  model.add(
    tf.layers.dense({
      units: 16,

      activation: "softmax",
    })
  );

  model.compile({
    optimizer: "adam",

    loss:
      "sparseCategoricalCrossentropy",

    metrics: ["accuracy"],
  });

  return model;
}

module.exports = {
  createPolicyNetwork,
};
