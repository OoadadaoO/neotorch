/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package app.adada.neo4j;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.FixedPerVarTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import app.adada.neo4j.config.PluginSettings;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Collectors;

public final class Gnn {

    private Gnn() {
    }

    public static void main(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        Gnn.runExample(args);
        // System.out.println("Training completed with result: " + result);
    }

    public static void runExample(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        PluginSettings settings = PluginSettings.getInstance();
        String engineName = settings.engineName;
        boolean trainParam = true;
        int seed = 42;
        float learningRate = 0.001f;
        // String modelName = "products_sage";
        String modelName = "test";
        String inputDir = settings.modelHome + "/" + modelName;
        String outputDir = settings.modelHome + "/" + modelName;
        Path outputPath = Paths.get(outputDir);
        Files.createDirectories(outputPath);

        // Load torch model
        System.out.println(">>> Loading model from " + modelName + ".pt");
        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(Paths.get(inputDir))
                .optModelName(modelName + ".pt")
                .optEngine(engineName)
                .optOption("trainParam", String.valueOf(trainParam))
                .optProgress(new ProgressBar())
                .build();

        // Inference
        System.out.println("\n>>> Running inference with model.");
        System.out.println(">>> Initializing model.");
        try (Model model = Model.newInstance(modelName, engineName);
                ZooModel<NDList, NDList> embedding = criteria.loadModel()) {

            /*
             * TODO: The engine is global, so it will effect all procedures that use the
             * engine.
             */
            // Fix seed
            Engine.getEngine(engineName).setRandomSeed(seed);

            // Data
            RandomAccessDataset datasetTest = GnnDataset.builder()
                    .setSampling(1, false, seed)
                    .build();

            // model
            model.setBlock(embedding.getBlock());
            Translator<NDList, NDList> translator = new Translator<NDList, NDList>() {
                @Override
                public NDList processInput(TranslatorContext ctx, NDList input) {
                    // Process input if needed
                    return input;
                }

                @Override
                public NDList processOutput(TranslatorContext ctx, NDList list) {
                    return list;
                }

                @Override
                public Batchifier getBatchifier() {
                    return new GnnBatchifier();
                }
            };
            try (Predictor<NDList, NDList> predictor = model.newPredictor(translator)) {
                predictor.setMetrics(new Metrics());
                Record input = datasetTest.get(model.getNDManager(), 0);
                NDList out = predictor.predict(input.getData());
                // System.out.println(">>> Prediction output:\n" + out.get(0));
            }
        }

        // Train
        System.out.println("\n>>> Running training with model.");
        System.out.println(">>> Initializing model.");
        try (Model model = Model.newInstance(modelName, engineName);
                ZooModel<NDList, NDList> embedding = criteria.loadModel()) {

            /*
             * TODO: The engine is global, so it will effect all procedures that use the
             * engine.
             */
            // Fix seed
            Engine.getEngine(engineName).setRandomSeed(seed);

            // Data
            RandomAccessDataset datasetTrain = GnnDataset.builder()
                    .setSampling(1, false, seed)
                    .build();
            RandomAccessDataset datasetTest = GnnDataset.builder()
                    .setSampling(1, false, seed)
                    .build();

            // model
            model.setBlock(embedding.getBlock());
            DefaultTrainingConfig config = setupTrainingConfig(outputDir, learningRate);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                long NUMBER_OF_NODES = 6; // Example number of nodes
                long DIMENSION_OF_FEATURES = 8; // Example feature dimension
                long NUMBER_OF_EDGES = 12; // Example number of edges
                trainer.initialize(new Shape(NUMBER_OF_NODES, DIMENSION_OF_FEATURES),
                        new Shape(2, NUMBER_OF_EDGES));

                // Train
                TorchTrain.fit(trainer, 5, datasetTrain, datasetTest);

                // Save model
                System.out.println(">>> Saving model parameters.");
                model.save(outputPath, modelName);
            }
        }

        // Inference
        System.out.println("\n>>> Running inference with trained model.");
        System.out.println(">>> Initializing model.");
        try (Model model = Model.newInstance(modelName, engineName);
                ZooModel<NDList, NDList> embedding = criteria.loadModel()) {

            /*
             * TODO: The engine is global, so it will effect all procedures that use the
             * engine.
             */
            // Fix seed
            Engine.getEngine(engineName).setRandomSeed(seed);

            // Data
            RandomAccessDataset datasetTest = GnnDataset.builder()
                    .setSampling(1, false, seed)
                    .build();

            // model
            model.setBlock(embedding.getBlock());
            System.out.println(">>> Loading model parameters.");
            model.load(outputPath, modelName);
            Translator<NDList, NDList> translator = new Translator<NDList, NDList>() {
                @Override
                public NDList processInput(TranslatorContext ctx, NDList input) {
                    // Process input if needed
                    return input;
                }

                @Override
                public NDList processOutput(TranslatorContext ctx, NDList list) {
                    return list;
                }

                @Override
                public Batchifier getBatchifier() {
                    return new GnnBatchifier();
                }
            };
            try (Predictor<NDList, NDList> predictor = model.newPredictor(translator)) {
                predictor.setMetrics(new Metrics());
                Record input = datasetTest.get(model.getNDManager(), 0);
                NDList out = predictor.predict(input.getData());
                // System.out.println(">>> Prediction output:\n" + out.get(0));
            }
        }

        return;
    }

    public static final class MyTranslator extends NoopTranslator {

        @Override
        public NDList processOutput(TranslatorContext ctx, NDList input) {
            return new NDList(
                    input.stream().map(ndArray -> ndArray.duplicate()).collect(Collectors.toList()));
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(String outputDir, float lr) {

        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        DefaultTrainingConfig config = new DefaultTrainingConfig(new SoftmaxCrossEntropyLoss("SoftmaxCrossEntropy"))
                .addEvaluator(new Accuracy())
                .optDevices(Engine.getEngine("PyTorch").getDevices(1))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);

        // Customized learning rate support
        /*
         * block-wise learning rate
         */
        // FixedPerVarTracker.Builder learningRateTrackerBuilder =
        // FixedPerVarTracker.builder().setDefaultValue(lr);
        // for (Pair<String, Parameter> paramPair : baseBlock.getParameters()) {
        // learningRateTrackerBuilder.put(paramPair.getValue().getId(), 0.1f * lr);
        // }
        FixedPerVarTracker learningRateTracker = FixedPerVarTracker.builder().setDefaultValue(lr).build();
        Optimizer optimizer = Adam.builder().optLearningRateTracker(learningRateTracker).build();
        config.optOptimizer(optimizer);

        return config;
    }
}