package app.adada.neo4j.gnn.graphsage;

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import app.adada.neo4j.gnn.GnnModelConfig;
import app.adada.neo4j.util.TypeParser;

public record GraphSageModelConfig(
        List<String> featureProperties,
        Long featureDimension,
        List<String> nodeLabels,
        List<String> relationshipTypes,
        Boolean supervised,
        Long embeddingDimension,
        String classProperties,
        Long classDimension,
        Long hiddenDimension,
        List<Long> sampleSizes,
        String aggregator,
        String activationFunction,
        Long preLinearLayers,
        Long postLinearLayers,
        Float dropoutRate,
        Boolean layerNormalization,
        Boolean residualConnection) implements GnnModelConfig {

    public static GraphSageModelConfig fromMap(Map<String, Object> config) {
        if (config.get("featureProperties") == null) {
            throw new IllegalArgumentException("`featureProperties` must be provided in the configuration");
        }
        if (config.get("featureDimension") == null) {
            throw new IllegalArgumentException("`featureDimension` must be provided in the configuration");
        }
        return new GraphSageModelConfig(
                TypeParser.parseList(config.get("featureProperties"), String.class),
                TypeParser.parse(config.get("featureDimension"), Long.class),
                TypeParser.parseList(config.get("nodeLabels"), String.class, List.of("*")),
                TypeParser.parseList(config.get("relationshipTypes"), String.class, List.of("*")),
                TypeParser.parse(config.get("supervised"), Boolean.class, false),
                TypeParser.parse(config.get("embeddingDimension"), Long.class, 64L),
                TypeParser.parse(config.get("classProperties"), String.class, "y"),
                TypeParser.parse(config.get("classDimension"), Long.class, 1L),
                TypeParser.parse(config.get("hiddenDimension"), Long.class, 1024L),
                TypeParser.parseList(config.get("sampleSizes"), Long.class, List.of(10L, 5L)),
                TypeParser.parse(config.get("aggregator"), String.class, "mean"),
                TypeParser.parse(config.get("activationFunction"), String.class, "relu"),
                TypeParser.parse(config.get("preLinearLayers"), Long.class, 0L),
                TypeParser.parse(config.get("postLinearLayers"), Long.class, 0L),
                TypeParser.parse(config.get("dropoutRate"), Float.class, 0.0f),
                TypeParser.parse(config.get("layerNormalization"), Boolean.class, false),
                TypeParser.parse(config.get("residualConnection"), Boolean.class, false));
    }

    public static GraphSageModelConfig fromJson(String json) {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            return objectMapper.readValue(json, GraphSageModelConfig.class);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to deserialize JSON to GraphSageModelConfig", e);
        }
    }

    @Override
    public String toJson() {
        try {
            ObjectMapper objectMapper = new ObjectMapper();
            return objectMapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to serialize GraphSageModelConfig to JSON", e);
        }
    }
}
