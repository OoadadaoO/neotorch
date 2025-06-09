package app.adada.neo4j.gnn.dataset;

import java.util.List;

import org.neo4j.graphdb.Node;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.Record;
import app.adada.neo4j.algo.BatchSampler;
import app.adada.neo4j.algo.BatchSampler.ExtendedBatch;
import app.adada.neo4j.algo.DenseGraph;
import app.adada.neo4j.algo.NeighborSampler;

public class UnsupervisedDataset extends GnnDataset {

    public UnsupervisedDataset(Builder builder) {
        super(builder);
    }

    @Override
    protected Record sampleBatch(NDManager manager, List<Node> batchNodes) {
        BatchSampler batchSampler = new BatchSampler(tx);
        ExtendedBatch extendedBatch = batchSampler.sampleBatch(batchNodes, config.nodeLabels(),
                config.relationshipTypes(), 1);

        NeighborSampler neighborSampler = new NeighborSampler();
        DenseGraph sampledData = neighborSampler.sample(extendedBatch.nodes, config.featureProperties(),
                config.nodeLabels(),
                config.relationshipTypes(), config.sampleSizes());

        NDArray x = manager.create(sampledData.getFeatures());
        NDArray edgeIndex = manager.create(sampledData.getEdges());
        NDArray posEdgeIndex = manager.create(extendedBatch.positiveEdges);
        NDArray negEdgeIndex = manager.create(extendedBatch.negativeEdges);
        NDList data = new NDList(x, edgeIndex, posEdgeIndex, negEdgeIndex);

        NDArray zeroLabels = manager.zeros(new Shape(x.getShape().get(0)), x.getDataType());
        NDList labels = new NDList(zeroLabels);

        return new Record(data, labels);
    }
}
