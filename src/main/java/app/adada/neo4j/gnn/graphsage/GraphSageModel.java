package app.adada.neo4j.gnn.graphsage;

import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.FixedPerVarTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import app.adada.neo4j.config.PluginSettings;
import app.adada.neo4j.gnn.GnnModel;
import app.adada.neo4j.gnn.GnnModelConfig;
import app.adada.neo4j.gnn.training.GraphSageUnsupervisedLoss;
import app.adada.neo4j.gnn.training.UnsupervisedTrain;
import app.adada.neo4j.util.ModelBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Concrete implementation of GnnModel for GraphSAGE.
 */
public class GraphSageModel extends GnnModel {

    @Override
    public GnnModel create(GnnModelConfig modelConfig) {
        if (!(modelConfig instanceof GraphSageModelConfig)) {
            throw new IllegalArgumentException("Invalid model configuration type for GraphSAGE.");
        }
        return create((GraphSageModelConfig) modelConfig);
    }

    @Override
    public void train(Object modelConfig, Dataset trainingDataset, Dataset validateDataset) {
        if (!(modelConfig instanceof GraphSageTrainConfig)) {
            throw new IllegalArgumentException("Invalid model configuration type for GraphSAGE.");
        }
        train((GraphSageTrainConfig) modelConfig, trainingDataset, validateDataset);
    }

    @Override
    public String getModelType() {
        return "GraphSAGE";
    }

    protected static final String BUILDER_PY = PluginSettings.getInstance().builderHome + "/graph_sage.py";

    protected final String name;
    protected final String dir;
    protected GraphSageModelConfig config;

    public GraphSageModel(Transaction tx, String modelName) {
        super(tx, modelName);
        this.name = modelName;
        this.dir = PluginSettings.getInstance().modelHome + "/" + modelName;
        try {
            Files.createDirectories(Path.of(dir));
        } catch (IOException e) {
            throw new RuntimeException("Failed to create directories for model: " + dir, e);
        }
    }

    public GraphSageModelConfig config() {
        return config;
    }

    public GraphSageModel create(GraphSageModelConfig modelConfig) {
        config = modelConfig;

        // Check if the model configuration file already exists
        String filePath = getModelConfigFilePath();
        if (Files.exists(Path.of(filePath))) {
            throw new RuntimeException("Configuration file already exists for model: " + modelName);
        }

        // Build the model using the Python script
        try {
            List<String> command = new ArrayList<>();
            command.add("--name");
            command.add(name);
            command.add("--output_dir");
            command.add(dir);
            command.add("--in_dim");
            command.add(String.valueOf(config.featureDimension()));
            command.add("--hidden_dim");
            command.add(String.valueOf(config.hiddenDimension()));
            command.add("--out_dim");
            command.add(config.supervised() ? String.valueOf(config.classDimension())
                    : String.valueOf(config.embeddingDimension()));
            command.add("--num_layers");
            command.add(String.valueOf(config.sampleSizes().size()));
            command.add("--aggr");
            command.add(config.aggregator());
            command.add("--activ");
            command.add(config.activationFunction());
            command.add("--num_pre_linears");
            command.add(String.valueOf(config.preLinearLayers()));
            command.add("--num_post_linears");
            command.add(String.valueOf(config.postLinearLayers()));
            command.add("--dropout");
            command.add(String.valueOf(config.dropoutRate()));
            command.add("--norm");
            command.add(config.layerNormalization() ? "layer_norm" : "none");

            if (config.residualConnection()) {
                command.add("--residual");
            }
            ModelBuilder.run("graph_sage.py", command);
        } catch (Exception e) {
            throw new RuntimeException("Failed to execute Python script", e);
        }

        // Create the model configuration file
        createModelConfig(modelConfig);
        return this;
    }

    public GraphSageModel load() {
        config = (GraphSageModelConfig) loadModelConfig();
        return this;
    }

    public void delete() {
        try {
            Files.walk(Path.of(dir))
                    .sorted((path1, path2) -> path2.compareTo(path1)) // Delete files before directories
                    .forEach(path -> {
                        try {
                            Files.deleteIfExists(path);
                        } catch (IOException e) {
                            throw new RuntimeException("Failed to delete path: " + path, e);
                        }
                    });
        } catch (IOException e) {
            throw new RuntimeException("Failed to delete model directory: " + dir, e);
        }
    }

    public void train(GraphSageTrainConfig trainingConfig, Dataset trainingDataset, Dataset validateDataset) {
        if (config == null) {
            throw new IllegalStateException("Model configuration is not set. Please create or load the model first.");
        }
        PluginSettings settings = PluginSettings.getInstance();

        System.out.println(">>> Loading model from " + modelName + ".pt");
        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(Paths.get(dir))
                .optEngine(settings.engineName)
                .optOption("trainParam", String.valueOf(true))
                .optProgress(new ProgressBar())
                .build();

        try (Model model = Model.newInstance(modelName, settings.engineName);
                ZooModel<NDList, NDList> embedding = criteria.loadModel()) {

            // Fix seed
            Engine.getEngine(settings.engineName).setRandomSeed(trainingConfig.randomSeed().intValue());

            // model
            model.setBlock(embedding.getBlock());
            DefaultTrainingConfig tConfig = setupTrainingConfig(dir, trainingConfig.maxGpus().intValue(),
                    trainingConfig.learningRate().floatValue(),
                    trainingConfig.negativeSampleWeight().floatValue(),
                    config.supervised());
            try (Trainer trainer = model.newTrainer(tConfig)) {
                trainer.setMetrics(new Metrics());

                // Initialize trainer with input and output shapes
                trainer.initialize(new Shape(1, config.featureDimension()), new Shape(2, 1));

                // Train
                if (config.supervised()) {
                    throw new UnsupportedOperationException("Supervised training is not implemented yet.");
                } else {
                    UnsupervisedTrain.fit(trainer, trainingConfig.epochs().intValue(), trainingDataset,
                            validateDataset);
                }

                // Save model
                System.out.println(">>> Saving model parameters.");
                model.save(Paths.get(dir), modelName);
            } catch (IOException e) {
                e.printStackTrace();
            } catch (TranslateException e) {
                e.printStackTrace();
            }
        } catch (ModelNotFoundException e1) {
            e1.printStackTrace();
        } catch (MalformedModelException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }

    public void infer(Iterable<Node> nodes, Object predictConfig) {
        if (config == null) {
            throw new IllegalStateException("Model configuration is not set. Please create or load the model first.");
        }
        throw new UnsupportedOperationException("Inference not implemented yet");
    }

    private static DefaultTrainingConfig setupTrainingConfig(String outputDir, int maxGpus, float lr,
            float negativeSampleWeight,
            boolean supervised) {

        PluginSettings settings = PluginSettings.getInstance();

        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    // float accuracy = result.getValidateEvaluation("Accuracy");
                    // model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        DefaultTrainingConfig config = new DefaultTrainingConfig(
                supervised ? new SoftmaxCrossEntropyLoss() : new GraphSageUnsupervisedLoss(negativeSampleWeight))
                // .addEvaluator(new Accuracy())
                .optDevices(Engine.getEngine(settings.engineName).getDevices(maxGpus))
                // .optExecutorService()
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
