package app.adada.neo4j.gnn.dataset;

import ai.djl.Device;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.DataIterable;
import ai.djl.training.dataset.Record;
import ai.djl.training.dataset.Sampler;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Pipeline;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;

public class GnnDataIterable extends DataIterable {

    private final GnnDataset gnnDataset;

    /**
     * Creates a new instance of {@code DataIterable} with the given parameters.
     *
     * @param dataset         the dataset to iterate on
     * @param manager         the manager to create the arrays
     * @param sampler         a sampler to sample data with
     * @param dataBatchifier  a batchifier for data
     * @param labelBatchifier a batchifier for labels
     * @param pipeline        the pipeline of transforms to apply on the data
     * @param targetPipeline  the pipeline of transforms to apply on the labels
     * @param executor        an {@link ExecutorService}
     * @param preFetchNumber  the number of samples to prefetch
     * @param device          the {@link Device}
     */
    public GnnDataIterable(
            GnnDataset dataset,
            NDManager manager,
            Sampler sampler,
            Batchifier dataBatchifier,
            Batchifier labelBatchifier,
            Pipeline pipeline,
            Pipeline targetPipeline,
            ExecutorService executor,
            int preFetchNumber,
            Device device) {
        super(
                dataset,
                manager,
                sampler,
                dataBatchifier,
                labelBatchifier,
                pipeline,
                targetPipeline,
                executor,
                preFetchNumber,
                device);
        this.gnnDataset = dataset;
    }

    @Override
    protected Batch fetch(List<Long> indices, int progress) throws IOException {
        NDManager subManager = manager.newSubManager();
        subManager.setName("dataIter fetch");
        int batchSize = indices.size();
        // NDList[] data = new NDList[batchSize];
        // NDList[] labels = new NDList[batchSize];
        // for (int i = 0; i < batchSize; i++) {
        // Record record = dataset.get(subManager, indices.get(i));
        // data[i] = record.getData();
        // // apply transform
        // if (pipeline != null) {
        // data[i] = pipeline.transform(data[i]);
        // }

        // labels[i] = record.getLabels();
        // }
        Record record = gnnDataset.getBatch(subManager, indices);
        NDList batchData = record.getData();
        NDList batchLabels = record.getLabels();

        // Arrays.stream(data).forEach(NDList::close);
        // Arrays.stream(labels).forEach(NDList::close);

        // apply label transform
        if (targetPipeline != null) {
            batchLabels = targetPipeline.transform(batchLabels);
        }
        // pin to a specific device
        if (device != null) {
            batchData = batchData.toDevice(device, false);
            batchLabels = batchLabels.toDevice(device, false);
        }
        return new Batch(
                subManager,
                batchData,
                batchLabels,
                batchSize,
                dataBatchifier,
                labelBatchifier,
                progress,
                dataset.size(),
                indices);
    }
}
