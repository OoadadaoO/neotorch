package app.adada.neo4j.gnn;

import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.Node;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import com.fasterxml.jackson.core.JsonProcessingException;

import ai.djl.training.dataset.Dataset;
import app.adada.neo4j.config.PluginSettings;

/**
 * Abstract base class for GNN models.
 * Defines the common operations and structure for different GNN architectures.
 */
public abstract class GnnModel {
    protected final Transaction tx;
    protected final String modelName;

    public GnnModel(Transaction tx, String modelName) {
        this.tx = tx;
        this.modelName = modelName;
    }

    /**
     * Gets the specific type of the GNN model (e.g., "GraphSAGE", "GCN").
     * 
     * @return The model type string.
     */
    public abstract String getModelType();

    // --- Result Inner Classes ---
    // These classes define the structure of results returned by model operations.

    public static class CreateResult {
        public final String modelName;
        public final String message;

        public CreateResult(String modelName, String message) {
            this.modelName = modelName;
            this.message = message;
        }
    }

    public static class TrainResult {
        public final String modelName;
        public final String status;
        // public final Map<String, Object> metrics; // Example: training loss, accuracy

        public TrainResult(String modelName, String status /* , Map<String, Object> metrics */) {
            this.modelName = modelName;
            this.status = status;
            // this.metrics = metrics;
        }
    }

    public static class Prediction {
        public final Long nodeId;
        public final Object predictionValue; // Can be a label, embedding, score, etc.

        public Prediction(Long nodeId, Object predictionValue) {
            this.nodeId = nodeId;
            this.predictionValue = predictionValue;
        }
    }

    // --- Abstract GNN Operations ---
    // These methods must be implemented by concrete GNN model classes.

    /**
     * Creates a new GNN model.
     * 
     * @return A CreateResult object indicating the outcome of the creation.
     */
    public abstract GnnModel create(GnnModelConfig modelConfig);

    /**
     * Loads an existing GNN model.
     * This method should retrieve the model's metadata and possibly its weights.
     * 
     * @return A CreateResult object indicating the outcome of the load operation.
     */
    public abstract GnnModel load();

    /**
     * Trains the GNN model with the provided configuration.
     * 
     * @param trainingConfig A map containing training-specific parameters (e.g.,
     *                       epochs, learning rate).
     * @return A TrainResult object indicating the outcome of the training.
     */
    public abstract void train(Object trainingConfig, Dataset trainingDataset, Dataset validateDataset);

    /**
     * Infers predictions for a set of nodes using the GNN model.
     * 
     * @param nodes         An iterable collection of Node objects to infer on.
     * @param predictConfig A configuration object for inference (e.g., threshold,
     *                      batch size).
     * @return A stream of Prediction objects containing node IDs and their
     *         corresponding prediction values.
     */
    public abstract void infer(Iterable<Node> nodes, Object predictConfig);

    // --- Helper Methods for Model Persistence (Conceptual) ---

    protected String getModelConfigFilePath() {
        PluginSettings settings = PluginSettings.getInstance();
        return settings.modelHome + "/" + modelName + "/config.json";
    }

    /**
     * Create and store the model configuration as a JSON file.
     * 
     * @param filePath The path where the configuration file should be saved.
     * @param config   The configuration object to be serialized and stored.
     */
    protected GnnModelConfig createModelConfig(GnnModelConfig config) {
        String filePath = getModelConfigFilePath();
        if (Files.exists(Path.of(filePath))) {
            throw new RuntimeException("Configuration file already exists for model: " + modelName);
        }
        try {
            Files.writeString(Path.of(filePath), config.toJson());
            System.out.printf("Configuration for model '%s' stored at '%s'%n", modelName, filePath);
            return config;
        } catch (JsonProcessingException e) {
            System.err.printf("Error serializing configuration for model '%s': %s%n", modelName, e.getMessage());
            throw new RuntimeException("Error serializing configuration for " + modelName, e);
        } catch (IOException e) {
            System.err.printf("Error writing configuration to file '%s': %s%n", filePath, e.getMessage());
            throw new RuntimeException("Error writing configuration to file " + filePath, e);
        }
    }

    /**
     * Loads the model configuration from a JSON file.
     * 
     * @return A GnnModelConfig object containing the model's configuration.
     */
    protected GnnModelConfig loadModelConfig() {
        String filePath = getModelConfigFilePath();
        if (!Files.exists(Path.of(filePath))) {
            System.err.printf("Configuration file '%s' does not exist for model '%s'%n", filePath, modelName);
            throw new RuntimeException("Configuration file not found for " + modelName);
        }
        try {
            String json = Files.readString(Path.of(filePath));
            return GnnModelConfig.fromJson(json);
        } catch (IOException e) {
            System.err.printf("Error reading configuration file '%s': %s%n", filePath, e.getMessage());
            throw new RuntimeException("Error reading configuration file " + filePath, e);
        }
    }

    /**
     * Deletes the model configuration file.
     * 
     * @return A boolean indicating whether the deletion was successful.
     */
    protected boolean deleteModelConfig() {
        PluginSettings settings = PluginSettings.getInstance();
        String filePath = settings.modelHome + "/" + modelName + "/config.json";
        try {
            if (Files.deleteIfExists(Path.of(filePath))) {
                System.out.printf("Configuration file '%s' deleted successfully for model '%s'%n", filePath, modelName);
                return true;
            } else {
                System.out.printf("Configuration file '%s' does not exist for model '%s'%n", filePath, modelName);
                return false;
            }
        } catch (IOException e) {
            System.err.printf("Error deleting configuration file '%s': %s%n", filePath, e.getMessage());
            throw new RuntimeException("Error deleting configuration file " + filePath, e);
        }
    }
}
