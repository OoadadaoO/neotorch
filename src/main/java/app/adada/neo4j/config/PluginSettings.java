package app.adada.neo4j.config;

import java.io.File;

// import org.neo4j.procedure.Context;
// import org.neo4j.configuration.Config;

public class PluginSettings {

    private static PluginSettings instance;

    public String engineName;
    public String neo4jHome;
    public String neotorchHome;
    public String modelHome;
    public String builderHome;
    public String pyInterpreter;

    private PluginSettings() {
        engineName = System.getenv("NEOTORCH_ENGINE_NAME");
        if (engineName == null) {
            engineName = "PyTorch";
        }

        neo4jHome = System.getenv("NEO4J_HOME");
        if (neo4jHome == null) {
            neo4jHome = "/var/lib/neo4j";
        }

        neotorchHome = System.getenv("NEOTORCH_HOME");
        if (neotorchHome == null) {
            neotorchHome = neo4jHome + "/plugins/neotorch";
        }

        modelHome = neotorchHome + "/models";
        File modelDir = new File(modelHome);
        if (!modelDir.exists()) {
            if (!modelDir.mkdirs()) {
                throw new IllegalStateException("Failed to create model directory: " + modelHome);
            }
        } else if (!modelDir.isDirectory()) {
            throw new IllegalStateException("Model directory is not a directory: " + modelHome);
        }

        builderHome = neotorchHome + "/builder";
        File builderDir = new File(builderHome);
        if (!builderDir.exists()) {
            if (!builderDir.mkdirs()) {
                throw new IllegalStateException("Failed to create builder directory: " + builderHome);
            }
        } else if (!builderDir.isDirectory()) {
            throw new IllegalStateException("Builder directory is not a directory: " + builderHome);
        }

        pyInterpreter = System.getenv("PYTHON_INTERPRETER");
        if (pyInterpreter == null) {
            pyInterpreter = neotorchHome + "/.venv/bin/python";
        }
        File interpreterFile = new File(pyInterpreter);
        if (!interpreterFile.exists() || !interpreterFile.canExecute()) {
            throw new IllegalStateException("Python interpreter not found or not executable: " + pyInterpreter);
        }
    }

    public static synchronized PluginSettings getInstance() {
        if (instance == null) {
            instance = new PluginSettings();
        }
        return instance;
    }
}
