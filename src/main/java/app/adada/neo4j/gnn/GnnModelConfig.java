package app.adada.neo4j.gnn;

import java.util.List;
import java.util.Map;

public interface GnnModelConfig {
    /**
     * Converts a map of configuration parameters to a ModelConfig instance.
     *
     * @param config the configuration parameters
     * @return a ModelConfig instance
     */
    static GnnModelConfig fromMap(Map<String, Object> config) {
        throw new UnsupportedOperationException("This method should be implemented by subclasses");
    }

    /**
     * Converts a JSON string to a ModelConfig instance.
     *
     * @param json the JSON string representing the model configuration
     * @return a ModelConfig instance
     */
    static GnnModelConfig fromJson(String json) {
        throw new UnsupportedOperationException("This method should be implemented by subclasses");
    }

    /**
     * Converts the model configuration to a JSON string.
     *
     * @return a JSON representation of the model configuration
     */
    String toJson();

    List<String> featureProperties();

    List<String> relationshipTypes();

    List<String> nodeLabels();

    List<Long> sampleSizes();
}
