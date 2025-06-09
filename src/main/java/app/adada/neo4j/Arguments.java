package app.adada.neo4j;

import ai.djl.Device;
import ai.djl.engine.Engine;

import java.util.Map;

public class Arguments {

    protected int epoch = 5;
    protected int batchSize = 32;
    protected int maxGpus = 1;
    protected boolean preTrained = false;
    protected String outputDir = "build/model";
    protected long limit = Long.MAX_VALUE;
    protected String modelDir = null;
    protected Map<String, String> criteria = null;
    protected String engine = "PyTorch";

    protected void initialize() {
        epoch = 2;
        outputDir = "build/model";
        limit = Long.MAX_VALUE;
        modelDir = null;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getEpoch() {
        return epoch;
    }

    public Device[] getMaxGpus() {
        return Engine.getEngine(engine).getDevices(maxGpus);
    }

    public boolean isPreTrained() {
        return preTrained;
    }

    public String getModelDir() {
        return modelDir;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public long getLimit() {
        return limit;
    }

    public Map<String, String> getCriteria() {
        return criteria;
    }

    public String getEngine() {
        return engine;
    }
}