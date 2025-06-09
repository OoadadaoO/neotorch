package app.adada.neo4j;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.BatchSampler;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.training.dataset.Sampler;
import ai.djl.training.dataset.SequenceSampler;
import app.adada.neo4j.gnn.dataset.RandomSampler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class GnnDataset extends RandomAccessDataset {

    @Override
    public void prepare(ai.djl.util.Progress progress) throws IOException {
        // No preparation needed for this dataset
    }

    protected GnnDataset(Builder builder) {
        super(builder);
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    protected long availableSize() {
        return 10; // 只產出 1 筆樣本
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        // // 6 nodes, 8-dim features
        // float[][] xData = {
        // { 1.0f, 2.0f, 1.1f, 2.1f, 1.2f, 2.2f, 1.3f, 2.3f }, // node 0
        // { 3.0f, 4.0f, 3.1f, 4.1f, 3.2f, 4.2f, 3.3f, 4.3f }, // node 1
        // { 5.0f, 6.0f, 5.1f, 6.1f, 5.2f, 6.2f, 5.3f, 6.3f }, // node 2
        // { 7.0f, 8.0f, 7.1f, 8.1f, 7.2f, 8.2f, 7.3f, 8.3f }, // node 3
        // { 9.0f, 1.0f, 9.1f, 1.1f, 9.2f, 1.2f, 9.3f, 1.3f }, // node 4
        // { 2.0f, 3.0f, 2.1f, 3.1f, 2.2f, 3.2f, 2.3f, 3.3f }, // node 5
        // };
        // NDArray x = manager.create(xData);

        // // 8 edges: shape [2, 8]
        // long[][] edgeIndexData = {
        // { 0, 1, 2, 3, 4, 5, 1, 3, 5, 0, 2, 4 }, // source
        // { 1, 2, 3, 4, 5, 0, 0, 2, 4, 1, 2, 3 } // target
        // };
        // NDArray edgeIndex = manager.create(edgeIndexData);

        // // 5 classes
        // long[] labelList = { 0, 0, 0, 1, 1, 2 }; // node labels
        // NDArray label = manager.create(labelList);

        int numNodes = 1000;
        int featureDim = 100;
        int numEdges = 10000;
        int numClasses = 128;

        // 生成節點特徵
        float[][] xData = new float[numNodes][featureDim];
        for (int i = 0; i < numNodes; i++) {
            for (int j = 0; j < featureDim; j++) {
                xData[i][j] = (float) Math.random(); // 或使用高斯亂數
            }
        }
        NDArray x = manager.create(xData);

        // 生成邊
        Set<String> edgeSet = new HashSet<>();
        List<Long> sources = new ArrayList<>();
        List<Long> targets = new ArrayList<>();

        while (edgeSet.size() < numEdges) {
            long src = (long) (Math.random() * numNodes);
            long dst = (long) (Math.random() * numNodes);
            if (src != dst && edgeSet.add(src + "_" + dst)) {
                sources.add(src);
                targets.add(dst);
            }
        }
        NDArray edgeIndex = manager.create(new long[][] {
                sources.stream().mapToLong(Long::longValue).toArray(),
                targets.stream().mapToLong(Long::longValue).toArray()
        });

        // 生成標籤
        long[] labelList = new long[numNodes];
        for (int i = 0; i < numNodes; i++) {
            labelList[i] = (long) (Math.random() * numClasses);
        }
        NDArray label = manager.create(labelList);

        NDList data = new NDList(x, edgeIndex);
        NDList labels = new NDList(label);
        return new Record(data, labels); // 無 label
    }

    public static final class Builder extends BaseBuilder<Builder> {

        public Builder() {
            this.dataBatchifier = new GnnBatchifier();
            this.labelBatchifier = new GnnBatchifier();
        }

        @Override
        protected Builder self() {
            return this;
        }

        public GnnDataset build() {
            return new GnnDataset(this);
        }

        /**
         * Sets the {@link Sampler} with the given batch size.
         *
         * @param batchSize the batch size
         * @param random    whether the sampling has to be random
         * @return this {@code BaseBuilder}
         */
        public Builder setSampling(int batchSize, boolean random) {
            return setSampling(batchSize, random, false, new Random().nextInt());
        }

        /**
         * Sets the {@link Sampler} with the given batch size.
         *
         * @param batchSize the batch size
         * @param random    whether the sampling has to be random
         * @param dropLast  whether to drop the last incomplete batch
         * @return this {@code BaseBuilder}
         */
        public Builder setSampling(int batchSize, boolean random, boolean dropLast) {
            return setSampling(batchSize, random, dropLast, new Random().nextInt());
        }

        /**
         * Sets the {@link Sampler} with the given batch size.
         * 
         * @param batchSize the batch size
         * @param random    whether the sampling has to be random
         * @param seed      the random seed for sampling
         */
        public Builder setSampling(int batchSize, boolean random, int seed) {
            return setSampling(batchSize, random, false, seed);
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
        public Builder setSampling(int batchSize, boolean random, boolean dropLast, int seed) {
            if (random) {
                sampler = new BatchSampler(new RandomSampler(seed), batchSize, dropLast);
            } else {
                sampler = new BatchSampler(new SequenceSampler(), batchSize, dropLast);
            }
            return self();
        }
    }
}
