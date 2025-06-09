package app.adada.neo4j.gnn.dataset;

import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Sampler;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.stream.LongStream;

/**
 * {@code RandomSampler} is an implementation of the {@link
 * ai.djl.training.dataset.Sampler.SubSampler} interface.
 *
 * <p>
 * {@code RandomSampler} samples the data from [0, dataset.size) randomly.
 */
public class RandomSampler implements Sampler.SubSampler {

    protected int seed;
    protected Random random;

    public RandomSampler(int seed) {
        // Default constructor
        this.seed = seed;
        this.random = new Random(seed);
    }

    private static void swap(long[] arr, int i, int j) {
        long tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    /** {@inheritDoc} */
    @Override
    public Iterator<Long> sample(RandomAccessDataset dataset) {
        return new Iterate(dataset, random);
    }

    static class Iterate implements Iterator<Long> {

        private long[] indices;
        private long current;

        Iterate(RandomAccessDataset dataset, Random random) {
            long size = dataset.size();
            current = 0;
            indices = LongStream.range(0, size).toArray();
            // java array didn't support index greater than max integer
            // so cast to int for now
            for (int i = Math.toIntExact(size) - 1; i > 0; --i) {
                swap(indices, i, random.nextInt(i));
            }
        }

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return current < indices.length;
        }

        /** {@inheritDoc} */
        @Override
        public Long next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            // java array didn't support index greater than max integer
            // so cast to int for now
            return indices[Math.toIntExact(current++)];
        }
    }
}
