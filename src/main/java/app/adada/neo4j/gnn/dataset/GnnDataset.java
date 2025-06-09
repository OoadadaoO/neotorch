package app.adada.neo4j.gnn.dataset;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;

import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Transaction;

import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.BatchSampler;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.training.dataset.Sampler;
import ai.djl.training.dataset.SequenceSampler;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import app.adada.neo4j.gnn.GnnModelConfig;

public abstract class GnnDataset extends RandomAccessDataset {

    protected final Transaction tx;
    protected final int seed;
    protected final List<Node> nodes;
    protected final GnnModelConfig config;

    @Override
    public void prepare(Progress progress) throws IOException {
        // No preparation needed for this dataset
    }

    protected GnnDataset(Builder builder) {
        super(builder);
        this.tx = builder.tx;
        this.seed = builder.seed;
        this.nodes = builder.nodes;
        this.config = builder.config;
    }

    public static Builder builder(int seed) {
        return new Builder(seed);
    }

    @Override
    protected long availableSize() {
        return nodes.size();
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        throw new UnsupportedOperationException("Use getBatch method instead.");
    }

    @Override
    public Iterable<Batch> getData(
            NDManager manager, Sampler sampler, ExecutorService executorService)
            throws IOException, TranslateException {
        prepare();
        return new GnnDataIterable(
                this,
                manager,
                sampler,
                dataBatchifier,
                labelBatchifier,
                pipeline,
                targetPipeline,
                executorService,
                prefetchNumber,
                device);
    }

    public Record getBatch(NDManager manager, List<Long> indice) throws IOException {
        List<Node> batchNodes = new ArrayList<>(indice.size());
        for (Long id : indice) {
            Node node = nodes.get(id.intValue());
            batchNodes.add(node);
        }
        return sampleBatch(manager, batchNodes);
    }

    protected abstract Record sampleBatch(NDManager manager, List<Node> batchNodes);

    public static final class Builder extends BaseBuilder<Builder> {

        protected int seed;
        protected Transaction tx;
        protected List<Node> nodes;
        protected GnnModelConfig config;

        public Builder(int seed) {
            this.seed = seed;
            this.prefetchNumber = 1;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public GnnDataset build(boolean supervised) {
            if (supervised) {
                throw new UnsupportedOperationException("Supervised datasets are not supported yet.");
            }
            return new UnsupervisedDataset(this);
        }

        public Builder setTransaction(Transaction tx) {
            this.tx = tx;
            return self();
        }

        public Builder setNodes(List<Node> nodes) {
            this.nodes = nodes;
            return self();
        }

        public Builder setConfig(GnnModelConfig config) {
            this.config = config;
            return self();
        }

        /**
         * Sets the {@link Sampler} with the given batch size.
         *
         * @param batchSize the batch size
         * @param random    whether the sampling has to be random
         * @return this {@code BaseBuilder}
         */
        public Builder setSampling(int batchSize, boolean random) {
            return setSampling(batchSize, random, false);
        }

        /**
         * Sets the {@link Sampler} with the given batch size.
         *
         * @param batchSize the batch size
         * @param random    whether the sampling has to be random
         * @param dropLast  whether to drop the last incomplete batch
         * @param seed      the random seed for sampling
         * @return this {@code BaseBuilder}
         */
        public Builder setSampling(int batchSize, boolean random, boolean dropLast) {
            if (random) {
                sampler = new BatchSampler(new RandomSampler(seed), batchSize, dropLast);
            } else {
                sampler = new BatchSampler(new SequenceSampler(), batchSize, dropLast);
            }
            return self();
        }
    }

}
