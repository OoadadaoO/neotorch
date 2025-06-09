package app.adada.neo4j.gnn.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener.BatchData;
import ai.djl.translate.TranslateException;
import ai.djl.util.Preconditions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;

import org.neo4j.function.TriFunction;

/**
 * Helper for easy training of a whole model, a trainining batch, or a
 * validation batch.
 */
public final class GnnTrain {

    private GnnTrain() {
    }

    /**
     * Runs a basic epoch training experience with a given trainer.
     *
     * @param trainer         the trainer to train for
     * @param numEpoch        the number of epochs to train
     * @param trainingDataset the dataset to train on
     * @param validateDataset the dataset to validate against. Can be null for no
     *                        validation
     * @throws IOException        for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public static void fit(
            Trainer trainer, int numEpoch, Dataset trainingDataset, Dataset validateDataset,
            TriFunction<Trainer, NDList, NDList, NDArray> lossFunction // trainer, labels, predictions -> loss
    )
            throws IOException, TranslateException {

        // Deep learning is typically trained in epochs where each epoch trains the
        // model on each item in the dataset once
        try (Batch firstBatch = trainer.iterateDataset(trainingDataset).iterator().next()) {
            initialState(trainer, firstBatch);
        }

        long[] epochTimes = new long[numEpoch];
        for (int epoch = 0; epoch < numEpoch; epoch++) {
            long epochStart = System.nanoTime();

            // We iterate through the dataset once during each epoch
            for (Batch batch : trainer.iterateDataset(trainingDataset)) {

                // During trainBatch, we update the loss and evaluators with the results for the
                // training batch
                trainBatch(trainer, batch, lossFunction);

                // Now, we update the model parameters based on the results of the latest
                // trainBatch
                trainer.step();

                long epochEnd = System.nanoTime();
                epochTimes[epoch] = epochEnd - epochStart;

                // We must make sure to close the batch to ensure all the memory associated with
                // the batch is cleared.
                // If the memory isn't closed after each batch, you will very quickly run out of
                // memory on your GPU
                batch.close();
            }

            // After each epoch, test against the validation dataset if we have one
            evaluateDataset(trainer, validateDataset);

            // reset training and validation evaluators at end of epoch
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }
        // Calculate the average epoch time
        double avgEpochTimeMs = 0.0;
        for (long epochTime : epochTimes) {
            avgEpochTimeMs += epochTime / 1_000_000_000.0;
        }
        avgEpochTimeMs /= numEpoch;
        System.out.printf("Average epoch time: %.4f s%n", avgEpochTimeMs);
    }

    public static TriFunction<Trainer, NDList, NDList, NDArray> supervisedLossFunc() {
        return (Trainer trainer, NDList labels, NDList predictions) -> trainer.getLoss().evaluate(labels, predictions);
    }

    /**
     * Trains the model with one iteration of the given {@link Batch} of data.
     *
     * @param trainer the trainer to validate the batch with
     * @param batch   a {@link Batch} that contains data, and its respective labels
     * @throws IllegalArgumentException if the batch engine does not match the
     *                                  trainer engine
     */
    public static void trainBatch(Trainer trainer, Batch batch,
            TriFunction<Trainer, NDList, NDList, NDArray> lossFunction) {
        if (trainer.getManager().getEngine() != batch.getManager().getEngine()) {
            throw new IllegalArgumentException(
                    "The data must be on the same engine as the trainer. You may need to change one"
                            + " of your NDManagers.");
        }
        Batch[] splits = batch.split(trainer.getDevices(), false);
        BatchData batchData = new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        try (GradientCollector collector = trainer.newGradientCollector()) {

            if (splits.length > 1 && trainer.getExecutorService().isPresent()) {
                // multi-threaded
                ExecutorService executor = trainer.getExecutorService().get();
                List<CompletableFuture<Boolean>> futures = new ArrayList<>(splits.length);
                for (Batch split : splits) {
                    futures.add(
                            CompletableFuture.supplyAsync(
                                    () -> trainSplit(trainer, collector, batchData, split, lossFunction),
                                    executor));
                }
                CompletableFuture.allOf(futures.stream().toArray(CompletableFuture[]::new));
            } else {
                // sequence
                for (Batch split : splits) {
                    trainSplit(trainer, collector, batchData, split, lossFunction);
                }
            }
        }

        trainer.notifyListeners(listener -> listener.onTrainingBatch(trainer, batchData));
    }

    private static boolean trainSplit(
            Trainer trainer, GradientCollector collector, BatchData batchData, Batch split,
            TriFunction<Trainer, NDList, NDList, NDArray> lossFunction) {
        NDList data = split.getData();
        NDList labels = split.getLabels();
        NDList preds = trainer.forward(data, labels);
        long time = System.nanoTime();
        // NDArray lossValue = trainer.getLoss().evaluate(labels, preds);
        NDArray lossValue = lossFunction.apply(trainer, labels, preds);
        collector.backward(lossValue);
        trainer.addMetric("backward", time);
        time = System.nanoTime();
        batchData.getLabels().put(labels.get(0).getDevice(), labels);
        batchData.getPredictions().put(preds.get(0).getDevice(), preds);
        trainer.addMetric("training-metrics", time);
        return true;
    }

    /**
     * Validates the given batch of data.
     *
     * <p>
     * During validation, the evaluators and losses are computed, but gradients
     * aren't computed,
     * and parameters aren't updated.
     *
     * @param trainer the trainer to validate the batch with
     * @param batch   a {@link Batch} of data
     * @throws IllegalArgumentException if the batch engine does not match the
     *                                  trainer engine
     */
    public static void validateBatch(Trainer trainer, Batch batch) {
        Preconditions.checkArgument(
                trainer.getManager().getEngine() == batch.getManager().getEngine(),
                "The data must be on the same engine as the trainer. You may need to change one of"
                        + " your NDManagers.");
        Batch[] splits = batch.split(trainer.getDevices(), false);
        BatchData batchData = new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());

        if (splits.length > 1 && trainer.getExecutorService().isPresent()) {
            // multi-threaded
            ExecutorService executor = trainer.getExecutorService().get();
            List<CompletableFuture<Boolean>> futures = new ArrayList<>(splits.length);
            for (Batch split : splits) {
                futures.add(
                        CompletableFuture.supplyAsync(
                                () -> validateSplit(trainer, batchData, split), executor));
            }
            CompletableFuture.allOf(futures.stream().toArray(CompletableFuture[]::new));
        } else {
            // sequence
            for (Batch split : splits) {
                validateSplit(trainer, batchData, split);
            }
        }

        trainer.notifyListeners(listener -> listener.onValidationBatch(trainer, batchData));
    }

    /**
     * Initial the trainer state with the given batch of data.
     * 
     * <p>
     * Issue of Dropout not working if directly training with the batch.
     *
     * <p>
     * During initialization, the evaluators and losses are computed, but gradients
     * aren't computed,
     * and parameters aren't updated.
     *
     * @param trainer the trainer to validate the batch with
     * @param batch   a {@link Batch} of data
     * @throws IllegalArgumentException if the batch engine does not match the
     *                                  trainer engine
     */
    public static void initialState(Trainer trainer, Batch batch) {
        Preconditions.checkArgument(
                trainer.getManager().getEngine() == batch.getManager().getEngine(),
                "The data must be on the same engine as the trainer. You may need to change one of"
                        + " your NDManagers.");
        Batch[] splits = batch.split(trainer.getDevices(), false);
        BatchData batchData = new BatchData(batch, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());

        if (splits.length > 1 && trainer.getExecutorService().isPresent()) {
            // multi-threaded
            ExecutorService executor = trainer.getExecutorService().get();
            List<CompletableFuture<Boolean>> futures = new ArrayList<>(splits.length);
            for (Batch split : splits) {
                futures.add(
                        CompletableFuture.supplyAsync(
                                () -> validateSplit(trainer, batchData, split), executor));
            }
            CompletableFuture.allOf(futures.stream().toArray(CompletableFuture[]::new));
        } else {
            // sequence
            validateSplit(trainer, batchData, splits[0]);
        }
    }

    private static boolean validateSplit(Trainer trainer, BatchData batchData,
            Batch split) {
        NDList data = split.getData();
        NDList labels = split.getLabels();
        NDList preds = trainer.evaluate(data);
        batchData.getLabels().put(labels.get(0).getDevice(), labels);
        batchData.getPredictions().put(preds.get(0).getDevice(), preds);
        return true;
    }

    /**
     * Evaluates the test dataset.
     *
     * @param trainer     the trainer to evaluate on
     * @param testDataset the test dataset to evaluate
     * @throws IOException        for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public static void evaluateDataset(Trainer trainer, Dataset testDataset)
            throws IOException, TranslateException {

        if (testDataset != null) {
            for (Batch batch : trainer.iterateDataset(testDataset)) {
                validateBatch(trainer, batch);
                batch.close();
            }
        }
    }
}
