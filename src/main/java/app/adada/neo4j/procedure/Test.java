package app.adada.neo4j.procedure;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import app.adada.neo4j.Gnn;
import app.adada.neo4j.algo.BatchSampler;
import app.adada.neo4j.algo.DenseGraph;
import app.adada.neo4j.algo.NeighborSampler;
import app.adada.neo4j.algo.BatchSampler.ExtendedBatch;
import app.adada.neo4j.procedure.Test.SampledSubgraphResult;

public class Test {

    // @Context
    // public GraphDatabaseService db;

    @Context
    public Transaction tx;

    @Procedure(name = "neotorch.test.settings", mode = Mode.READ)
    public Stream<TestSettingsResult> testSettings() {
        return Stream.of(new TestSettingsResult(Map.of(
                "modelDirectory", "/var/lib/neo4j/plugins/neotorch/models",
                "pluginVersion", "1.0.0" // Example version, replace with actual version if available
        )));
    }

    public class TestSettingsResult {
        public final Map<String, String> settings;

        public TestSettingsResult(Map<String, String> settings) {
            this.settings = settings;
        }
    }

    @Procedure(name = "neotorch.test.gnn", mode = Mode.READ)
    public Stream<TestGnnResult> testGnn() {
        try {
            Gnn.runExample(null); // Assuming Gnn.runExample is a static method that runs a test example
        } catch (Exception e) {
            return Stream.of(new TestGnnResult("GraphSAGE", "Test failed: " + e.getMessage()));
        }
        return Stream.of(new TestGnnResult("GraphSAGE", "Test successful"));
    }

    public class TestGnnResult {
        public final String modelType;
        public final String message;

        public TestGnnResult(String modelType, String message) {
            this.modelType = modelType;
            this.message = message;
        }
    }

    @Procedure(name = "neotorch.test.sample", mode = Mode.READ)
    @Description("Performs multi-hop neighbor sampling with positive/negative sampling for GNN training.")
    public Stream<SampledSubgraphResult> sampler(
            @Name("batchNodes") List<Node> batchNodes,
            @Name("featureProperties") List<String> featureProperties,
            @Name("nodeLabels") List<String> nodeLabels,
            @Name("relationshipTypes") List<String> relationshipTypes,
            @Name("sampleSizes") List<Long> sampleSizes,
            @Name("supervised") boolean supervised) {

        long[][] posEdgeIndex = new long[2][0];
        long[][] negEdgeIndex = new long[2][0];
        if (!supervised) {
            long startTime = System.currentTimeMillis();
            BatchSampler batchSampler = new BatchSampler(tx);
            ExtendedBatch extendedBatch = batchSampler.sampleBatch(batchNodes, nodeLabels, relationshipTypes, 1);
            batchNodes = extendedBatch.nodes;
            posEdgeIndex = extendedBatch.positiveEdges;
            negEdgeIndex = extendedBatch.negativeEdges;
            long endTime = System.currentTimeMillis();
            System.out.println("Batch sampling took " + (endTime - startTime) + " ms");
        }

        // Perform multi-hop neighbor sampling
        long startTime = System.currentTimeMillis();
        NeighborSampler neighborSampler = new NeighborSampler();
        DenseGraph sampledData = neighborSampler.sample(batchNodes, featureProperties, nodeLabels,
                relationshipTypes, sampleSizes);
        long endTime = System.currentTimeMillis();
        System.out.println("Neighbor sampling took " + (endTime - startTime) + " ms");
        return Stream.of(new SampledSubgraphResult(sampledData.getFeatures(), sampledData.getEdges(),
                posEdgeIndex, negEdgeIndex));

    }

    public static class SampledSubgraphResult {
        public final List<List<Double>> features;
        public final List<List<Long>> edgeIndex;
        public final List<List<Long>> posEdgeIndex;
        public final List<List<Long>> negEdgeIndex;

        public SampledSubgraphResult(float[][] features, long[][] edgeIndex,
                long[][] posEdgeIndex, long[][] negEdgeIndex) {
            this.features = convertToList(features);
            this.edgeIndex = convertToList(edgeIndex);
            this.posEdgeIndex = convertToList(posEdgeIndex);
            this.negEdgeIndex = convertToList(negEdgeIndex);
        }

        private List<List<Long>> convertToList(long[][] array) {
            List<List<Long>> list = new ArrayList<>();
            for (long[] row : array) {
                List<Long> rowList = new ArrayList<>();
                for (long value : row) {
                    rowList.add(value);
                }
                list.add(rowList);
            }
            return list;
        }

        private List<List<Double>> convertToList(float[][] array) {
            List<List<Double>> list = new ArrayList<>();
            for (float[] row : array) {
                List<Double> rowList = new ArrayList<>();
                for (float value : row) {
                    rowList.add((double) value);
                }
                list.add(rowList);
            }
            return list;
        }
    }
}
