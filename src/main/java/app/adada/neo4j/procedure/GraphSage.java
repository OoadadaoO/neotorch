package app.adada.neo4j.procedure;

import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

import app.adada.neo4j.gnn.dataset.GnnDataset;
import app.adada.neo4j.gnn.graphsage.GraphSageModel;
import app.adada.neo4j.gnn.graphsage.GraphSageModelConfig;
import app.adada.neo4j.gnn.graphsage.GraphSageTrainConfig;

import java.util.Map;
import java.util.List; // For nodeIds in predict, a common way to pass collections
import java.util.stream.Stream;

/**
 * Neo4j Procedures for GraphSAGE models.
 * This class uses the GraphSAGEModel for the actual GNN logic.
 */
public class GraphSage {

    @Context
    public Transaction tx;

    // --- Procedure Result Inner Classes ---
    // These mirror the structure from your original request for procedure outputs.

    public static class TrainResult {
        public final Map<String, Object> modelInfo;
        public final Map<String, Object> configuration;

        public TrainResult(Map<String, Object> modelInfo, Map<String, Object> configuration) {
            this.modelInfo = modelInfo;
            this.configuration = configuration;
        }
    }

    public static class InferResult {
        public final Long nodeId;
        public final Object infer; // Changed from String to Object for more general prediction values

        public InferResult(Long nodeId, Object inferValue) {
            this.nodeId = nodeId;
            this.infer = inferValue;
        }
    }

    public static class DropResult {
        public final String modelName;

        public DropResult(String modelName) {
            this.modelName = modelName;
        }
    }

    // --- GraphSAGE Procedures ---
    @Procedure(name = "neotorch.graphsage.train", mode = Mode.WRITE)
    @Description("CALL neotorch.graphsage.train(modelName, nodeIds, [config]) YIELD model. " +
            "Train a GraphSAGE model. 'config' is an optional map for learning rate, etc.")
    public Stream<TrainResult> train(
            @Name("modelName") String modelName,
            @Name("nodes") List<Node> nodes,
            @Name(value = "config", defaultValue = "{}") Map<String, Object> config) {
        // try {
        // ResourceIterable<Node> allNodes = tx.getAllNodes();
        // List<Node> nodes = allNodes.stream().toList();
        GraphSageModel sageModel = new GraphSageModel(tx, modelName);
        GraphSageModelConfig modelConfig = GraphSageModelConfig.fromMap(config);
        GraphSageTrainConfig trainingConfig = GraphSageTrainConfig.fromMap(config);

        System.out.println("GraphSAGEProcedures.train: Training with " + nodes.size() + " nodes.");

        GnnDataset trainingDataset = GnnDataset.builder(trainingConfig.randomSeed().intValue())
                .setTransaction(tx)
                .setNodes(nodes)
                .setConfig(modelConfig)
                .setSampling(trainingConfig.batchSize().intValue(), true)
                .build(modelConfig.supervised());

        Map<String, Object> modelInfo = sageModel.create(modelConfig).train(trainingConfig, trainingDataset, null);
        // } catch (Exception e) {
        // System.err.println("GraphSAGEProcedures.train: Exception occurred during
        // training.");
        // e.printStackTrace();
        // }
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> configurationMap = mapper.convertValue(modelConfig,
                new TypeReference<Map<String, Object>>() {
                });
        configurationMap.putAll(mapper.convertValue(trainingConfig, new TypeReference<Map<String, Object>>() {
        }));

        return Stream.of(new TrainResult(modelInfo, configurationMap));
    }

    // @Procedure(name = "torch.graphsage.infer", mode = Mode.READ)
    // @Description("CALL torch.graphsage.infer(modelName, nodes, [config]) YIELD
    // nodeId, infer. " +
    // "'nodes' is a list of Neo4j internal node IDs. 'config' is optional.")
    // public Stream<InferResult> infer(
    // @Name("modelName") String modelName,
    // @Name("nodes") List<Node> nodes, // Using List<Long> as it's common for node
    // ID collections
    // @Name(value = "config", defaultValue = "{}") Map<String, Object>
    // predictConfig) {

    // if (nodes == null || nodes.isEmpty()) {
    // System.out.println("GraphSAGEProcedures.predict: Node list is empty or null.
    // Returning empty stream.");
    // return Stream.empty();
    // }

    // GraphSageModel sageModel = new GraphSageModel(tx);

    // return sageModel.predict(modelName, nodes, predictConfig)
    // .map(gnnPrediction -> new InferResult(gnnPrediction.nodeId,
    // gnnPrediction.predictionValue));
    // }

    @Procedure(name = "neotorch.graphsage.drop", mode = Mode.WRITE)
    @Description("CALL neotorch.graphsage.drop(modelName) YIELD model. " +
            "Delete a GraphSAGE model.")
    public Stream<DropResult> drop(
            @Name("modelName") String modelName) {
        GraphSageModel sageModel = new GraphSageModel(tx, modelName);

        sageModel.delete();

        return Stream.of(new DropResult(modelName));
    }
}
