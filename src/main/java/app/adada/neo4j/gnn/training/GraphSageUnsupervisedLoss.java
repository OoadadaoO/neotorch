package app.adada.neo4j.gnn.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Activation;
import ai.djl.training.loss.Loss;

public class GraphSageUnsupervisedLoss extends Loss {

    public GraphSageUnsupervisedLoss() {
        super("GraphSAGEUnsupervised");
    }

    /**
     * @param label
     * @param prediction
     */
    @Override
    public NDArray evaluate(NDList label, NDList prediction) {
        // 1. parse prediction
        NDArray embeddings = prediction.get(0); // shape [N, d]
        NDArray posEdgeIndex = prediction.get(1); // shape [2, B]
        NDArray negEdgeIndex = prediction.get(2); // shape [2, B]

        // 2. node embeddings: [B, d]
        NDArray posU = embeddings.get(posEdgeIndex.get(0).toType(DataType.INT64, false));
        NDArray posV = embeddings.get(posEdgeIndex.get(1).toType(DataType.INT64, false));
        NDArray negU = embeddings.get(negEdgeIndex.get(0).toType(DataType.INT64, false));
        NDArray negV = embeddings.get(negEdgeIndex.get(1).toType(DataType.INT64, false));

        // 3. inner product: [B]
        NDArray posScore = posU.mul(posV).sum(new int[] { 1 });
        NDArray negScore = negU.mul(negV).sum(new int[] { 1 });

        // 4. logistic loss
        NDArray posLoss = Activation.sigmoid(posScore).log().neg(); // [B]
        NDArray negLoss = Activation.sigmoid(negScore.neg()).log().neg(); // [B]

        // 5. combine losses
        NDArray loss = posLoss.add(negLoss).mean();
        return loss;
    }
}
