package app.adada.neo4j;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;

public class GnnBatchifier implements Batchifier {
    @Override
    public NDList batchify(NDList[] inputs) {
        int batchSize = inputs.length;
        if (batchSize > 1) {
            throw new IllegalArgumentException("GnnBatchifier only supports batch size of 1");
        }
        if (inputs.length == 0) {
            return new NDList();
        }
        NDList original = inputs[0];
        NDList copy = new NDList();
        for (NDArray array : original) {
            copy.add(array.duplicate());
        }
        return copy;

    }

    @Override
    public NDList[] unbatchify(NDList input) {
        return new NDList[] { input };
    }
}
