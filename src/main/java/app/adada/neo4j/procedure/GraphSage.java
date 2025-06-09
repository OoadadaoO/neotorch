package app.adada.neo4j.procedure;

import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

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
        public final String model; // Represents modelName

        public TrainResult(String modelName) {
            this.model = modelName;
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

    public static class DeleteResult {
        public final String modelName;

        public DeleteResult(String modelName) {
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
        try {
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

            sageModel.create(modelConfig).train(trainingConfig, trainingDataset, null);
        } catch (Exception e) {
            System.err.println("GraphSAGEProcedures.train: Exception occurred during training.");
            e.printStackTrace();
        }

        // GnnModel.TrainResult gnnTrainResult = sageModel.train(modelName,
        // trainingConfig);
        // System.out.println("GraphSAGEProcedures.train: " + gnnTrainResult.status); //
        // Log the detailed status
        return Stream.of(new TrainResult(modelName));
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

    @Procedure(name = "neotorch.graphsage.delete", mode = Mode.WRITE)
    @Description("CALL neotorch.graphsage.delete(modelName) YIELD model. " +
            "Delete a GraphSAGE model.")
    public Stream<DeleteResult> delete(
            @Name("modelName") String modelName) {
        GraphSageModel sageModel = new GraphSageModel(tx, modelName);

        sageModel.delete();

        return Stream.of(new DeleteResult(modelName));
    }
}
