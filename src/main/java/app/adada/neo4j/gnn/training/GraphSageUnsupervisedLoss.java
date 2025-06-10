package app.adada.neo4j.gnn.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import ai.djl.training.loss.Loss;

public class GraphSageUnsupervisedLoss extends Loss {

    float negativeSampleWeight;

    public GraphSageUnsupervisedLoss(float negativeSampleWeight) {
        this("GraphSAGEUnsupervisedLoss", negativeSampleWeight);
    }

    public GraphSageUnsupervisedLoss(String name, float negativeSampleWeight) {
        super(name);
        if (negativeSampleWeight <= 0) {
            throw new IllegalArgumentException("`negativeSampleWeight` must be a positive number");
        }
        this.negativeSampleWeight = negativeSampleWeight;
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
        NDArray posU = embeddings.get(posEdgeIndex.get(0));
        NDArray posV = embeddings.get(posEdgeIndex.get(1));
        NDArray negU = embeddings.get(negEdgeIndex.get(0));
        NDArray negV = embeddings.get(negEdgeIndex.get(1));

        // 3. inner product: [B]
        NDArray posScore = posU.mul(posV).sum(new int[] { 1 });
        NDArray negScore = negU.mul(negV).sum(new int[] { 1 });

        // 4. logistic loss
        NDArray posLoss = Activation.sigmoid(posScore).log().neg(); // [B]
        NDArray negLoss = Activation.sigmoid(negScore.neg()).log().neg().mul(negativeSampleWeight); // [B]

        // 5. combine losses
        return posLoss.add(negLoss).mean();
    }
}
