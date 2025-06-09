package app.adada.neo4j.algo;

public class DenseGraph {
    private final float[][] features;
    private final long[][] edges;

    public DenseGraph(float[][] features, long[][] edges) {
        this.features = features;
        this.edges = edges;
    }

    public float[][] getFeatures() {
        return features;
    }

    public long[][] getEdges() {
        return edges;
    }
}
