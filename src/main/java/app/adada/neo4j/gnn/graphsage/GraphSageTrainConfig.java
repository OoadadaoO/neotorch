package app.adada.neo4j.gnn.graphsage;

import java.util.Map;
import java.util.Random;

import app.adada.neo4j.util.TypeParser;

public record GraphSageTrainConfig(
        Long maxGpus,
        Long randomSeed,
        Long batchSize,
        Long epochs,
        String optimizer,
        Float learningRate) {

    public static GraphSageTrainConfig fromMap(Map<String, Object> config) {
        return new GraphSageTrainConfig(
                TypeParser.parse(config.get("maxGpus"), Long.class, 1L),
                TypeParser.parse(config.get("randomSeed"), Long.class, new Random().nextLong()),
                TypeParser.parse(config.get("batchSize"), Long.class, 100L),
                TypeParser.parse(config.get("epochs"), Long.class, 10L),
                TypeParser.parse(config.get("optimizer"), String.class, "adam"),
                TypeParser.parse(config.get("learningRate"), Float.class, 0.001f));
    }
}
