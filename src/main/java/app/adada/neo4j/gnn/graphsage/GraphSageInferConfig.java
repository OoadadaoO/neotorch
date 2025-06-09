package app.adada.neo4j.gnn.graphsage;

import java.util.Map;

import app.adada.neo4j.util.TypeParser;

public record GraphSageInferConfig(
        Integer maxGpus,
        Integer randomSeed,
        Integer batchSize) {

    public static GraphSageInferConfig fromMap(Map<String, Object> config) {
        return new GraphSageInferConfig(
                TypeParser.parse(config.get("maxGpus"), Integer.class, 1),
                TypeParser.parse(config.get("randomSeed"), Integer.class, null),
                TypeParser.parse(config.get("batchSize"), Integer.class, 100));
    }
}
