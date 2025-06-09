package app.adada.neo4j.util;

import java.util.*;

public class ReservoirSampling {

    // Generic method for reservoir sampling
    public static <T> List<T> reservoirSample(Iterator<T> stream, int k) {
        List<T> reservoir = new ArrayList<>(k);
        Random random = new Random();

        int i = 0;
        while (stream.hasNext()) {
            T item = stream.next();

            if (i < k) {
                // Fill the reservoir first
                reservoir.add(item);
            } else {
                // Replace elements with gradually decreasing probability
                int j = random.nextInt(i + 1);
                if (j < k) {
                    reservoir.set(j, item);
                }
            }
            i++;
        }

        return reservoir;
    }

    public static void main(String[] args) {
        // 模擬資料流
        List<Integer> dataStream = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            dataStream.add(i);
        }

        // 從資料流中抽取 10 個樣本
        List<Integer> sample = reservoirSample(dataStream.iterator(), 10);

        System.out.println("Sampling Result:");
        for (int num : sample) {
            System.out.print(num + " ");
        }
    }
}
