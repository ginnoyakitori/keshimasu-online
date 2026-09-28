// src/neural/train-policy.js
//
// Policy model training
//
// 使用:
//   node src/neural/train-policy.js --epochs 50 --batchSize 32 --maxNegativeRatio 3
//
// 注意:
// - @tensorflow/tfjs-node は使わない
// - @tensorflow/tfjs を使う
// - model.save("file://") は使わず、自前で model.json / weights.bin を保存する
//

const fs = require("fs");
const path = require("path");

const {
  tf,
  createPolicyModel,
  exampleToFeatures,
  createMetadataFromSummary,
} = require("./policy-model");

function datasetDir() {
  return path.join(
    __dirname,
    "../../data/datasets/policy"
  );
}

function modelDir() {
  return path.join(
    __dirname,
    "../../data/models/policy"
  );
}

function readJsonl(file) {
  if (!fs.existsSync(file)) {
    throw new Error(`File not found: ${file}`);
  }

  const text = fs.readFileSync(file, "utf8");

  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function loadSummary(file) {
  if (!fs.existsSync(file)) {
    throw new Error(`File not found: ${file}`);
  }

  return JSON.parse(
    fs.readFileSync(file, "utf8")
  );
}

function shuffle(array) {
  const a = [...array];

  for (let i = a.length - 1; i > 0; i--) {
    const j =
      Math.floor(Math.random() * (i + 1));

    [a[i], a[j]] = [a[j], a[i]];
  }

  return a;
}

function balanceExamples(examples, options = {}) {
  const maxNegativeRatio =
    options.maxNegativeRatio ?? 3;

  const positives = examples.filter(
    (ex) => ex.label === 1
  );

  const negatives = examples.filter(
    (ex) => ex.label === 0
  );

  const maxNegatives =
    positives.length * maxNegativeRatio;

  const sampledNegatives =
    shuffle(negatives).slice(0, maxNegatives);

  return shuffle([
    ...positives,
    ...sampledNegatives,
  ]);
}

function makeTensors(examples, metadata) {
  const xs = examples.map((ex) =>
    exampleToFeatures(ex, metadata)
  );

  const ys = examples.map((ex) => [
    ex.label === 1 ? 1 : 0,
  ]);

  return {
    xTensor: tf.tensor2d(xs),
    yTensor: tf.tensor2d(ys),
  };
}

async function saveModelToDisk(model, outDir, metadata) {
  fs.mkdirSync(outDir, {
    recursive: true,
  });

  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      const modelJson = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: [
          {
            paths: ["weights.bin"],
            weights: artifacts.weightSpecs,
          },
        ],
      };

      fs.writeFileSync(
        path.join(outDir, "model.json"),
        JSON.stringify(modelJson, null, 2),
        "utf8"
      );

      fs.writeFileSync(
        path.join(outDir, "weights.bin"),
        Buffer.from(artifacts.weightData)
      );

      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
          weightDataBytes:
            artifacts.weightData.byteLength,
        },
      };
    })
  );

  fs.writeFileSync(
    path.join(outDir, "metadata.json"),
    JSON.stringify(metadata, null, 2),
    "utf8"
  );
}

async function trainPolicy(options = {}) {
  console.log("=== TRAIN POLICY START ===");

  const ddir = datasetDir();

  const moveFile = path.join(
    ddir,
    "move-examples.jsonl"
  );

  const summaryFile = path.join(
    ddir,
    "summary.json"
  );

  console.log("Dataset dir:", ddir);
  console.log("Move examples:", moveFile);
  console.log("Summary:", summaryFile);

  const rawExamples = readJsonl(moveFile);
  const summary = loadSummary(summaryFile);

  console.log("Raw examples:", rawExamples.length);

  if (rawExamples.length === 0) {
    throw new Error(
      "No move examples found. Run node src/neural/build-policy-dataset.js first."
    );
  }

  const metadata =
    createMetadataFromSummary(summary);

  const balancedExamples =
    balanceExamples(rawExamples, {
      maxNegativeRatio:
        options.maxNegativeRatio ?? 3,
    });

  const positiveCount =
    balancedExamples.filter(
      (ex) => ex.label === 1
    ).length;

  const negativeCount =
    balancedExamples.length - positiveCount;

  console.log("Balanced examples:", balancedExamples.length);
  console.log("Positive:", positiveCount);
  console.log("Negative:", negativeCount);

  const shuffled =
    shuffle(balancedExamples);

  const validationSplit =
    options.validationSplit ?? 0.15;

  const validationSize = Math.max(
    1,
    Math.floor(
      shuffled.length * validationSplit
    )
  );

  const validationExamples =
    shuffled.slice(0, validationSize);

  const trainExamples =
    shuffled.slice(validationSize);

  console.log("Train examples:", trainExamples.length);
  console.log("Validation examples:", validationExamples.length);

  const trainTensors =
    makeTensors(trainExamples, metadata);

  const valTensors =
    makeTensors(validationExamples, metadata);

  const model = createPolicyModel();

  model.summary();

  console.log("\nTraining...");

  await model.fit(
    trainTensors.xTensor,
    trainTensors.yTensor,
    {
      epochs: options.epochs ?? 30,
      batchSize: options.batchSize ?? 32,
      shuffle: true,
      validationData: [
        valTensors.xTensor,
        valTensors.yTensor,
      ],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const loss =
            logs.loss?.toFixed(4);

          const acc =
            logs.acc?.toFixed(4) ??
            logs.accuracy?.toFixed(4);

          const valLoss =
            logs.val_loss?.toFixed(4);

          const valAcc =
            logs.val_acc?.toFixed(4) ??
            logs.val_accuracy?.toFixed(4);

          console.log(
            [
              `epoch=${epoch + 1}`,
              `loss=${loss}`,
              `acc=${acc}`,
              `val_loss=${valLoss}`,
              `val_acc=${valAcc}`,
            ].join(" | ")
          );
        },
      },
    }
  );

  const outDir = modelDir();

  const finalMetadata = {
    ...metadata,
    trainedAt: new Date().toISOString(),
    rawExamples: rawExamples.length,
    balancedExamples: balancedExamples.length,
    positiveCount,
    negativeCount,
    epochs: options.epochs ?? 30,
    batchSize: options.batchSize ?? 32,
    maxNegativeRatio:
      options.maxNegativeRatio ?? 3,
  };

  console.log("\nSaving model...");

  await saveModelToDisk(
    model,
    outDir,
    finalMetadata
  );

  trainTensors.xTensor.dispose();
  trainTensors.yTensor.dispose();
  valTensors.xTensor.dispose();
  valTensors.yTensor.dispose();

  console.log("\n=== TRAINING DONE ===");
  console.log("Model dir:", outDir);
  console.log("model.json:", path.join(outDir, "model.json"));
  console.log("weights.bin:", path.join(outDir, "weights.bin"));
  console.log("metadata.json:", path.join(outDir, "metadata.json"));
}

function parseArgs() {
  const args = process.argv.slice(2);

  const options = {};

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === "--epochs") {
      options.epochs = Number(args[++i]);
    } else if (arg === "--batchSize") {
      options.batchSize = Number(args[++i]);
    } else if (arg === "--maxNegativeRatio") {
      options.maxNegativeRatio =
        Number(args[++i]);
    }
  }

  return options;
}

// ここが重要。
// この if がないと、node src/neural/train-policy.js で何も実行されない。
if (require.main === module) {
  trainPolicy(parseArgs()).catch((error) => {
    console.error("\nTRAINING FAILED");
    console.error(error);
    process.exit(1);
  });
}

module.exports = {
  trainPolicy,
  saveModelToDisk,
};