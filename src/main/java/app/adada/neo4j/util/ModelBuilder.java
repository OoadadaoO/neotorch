package app.adada.neo4j.util;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

import app.adada.neo4j.config.PluginSettings;

public class ModelBuilder {
    public static void run(String name, List<String> command) throws IOException {
        PluginSettings settings = PluginSettings.getInstance();

        InputStream in = ModelBuilder.class.getResourceAsStream("/builder/" + name);
        if (in == null) {
            throw new FileNotFoundException("Cannot find " + name + " in resources");
        }

        Path tempScript = Files.createTempFile("script_" + System.nanoTime(), ".py");
        Files.copy(in, tempScript, StandardCopyOption.REPLACE_EXISTING);
        in.close();

        List<String> fullCommand = new ArrayList<>();
        fullCommand.add(settings.pyInterpreter);
        fullCommand.add(tempScript.toAbsolutePath().toString());
        fullCommand.addAll(command);
        ProcessBuilder pb = new ProcessBuilder(fullCommand);
        pb.inheritIO();
        Process process = pb.start();

        try {
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new RuntimeException("Python script execution failed with exit code: " + exitCode);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        tempScript.toFile().deleteOnExit();
    }
}
