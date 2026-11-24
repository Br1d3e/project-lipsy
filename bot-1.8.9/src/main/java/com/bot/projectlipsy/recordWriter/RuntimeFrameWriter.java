package com.bot.projectlipsy.recordWriter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.SimpleDateFormat;
import java.util.Date;

public class RuntimeFrameWriter {
    private static final String[] STATES = {"attack", "heal", "pearl", "speed", "eat", "refill"};
    private static final String[] AUXILIARIES = {"healAux", "pearlAux"};
    private BufferedWriter writer;
    private File file;
    private int pendingFrames;

    public RuntimeFrameWriter() {
        Runtime.getRuntime().addShutdownHook(new Thread(this::close, "lipsy-runtime-trace-close"));
    }

    public synchronized void start(File gameDirectory) throws IOException {
        if (writer != null) return;
        Path root = gameDirectory.toPath().resolve("runtime-traces");
        Files.createDirectories(root);
        String prefix = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss-").format(new Date());
        Path directory = Files.createTempDirectory(root, prefix);
        file = directory.resolve("frames.jsonl").toFile();
        writer = Files.newBufferedWriter(file.toPath(), StandardCharsets.UTF_8);
        pendingFrames = 0;
    }

    public synchronized boolean isRecording() {
        return writer != null;
    }

    public synchronized File getFile() {
        return file;
    }

    public synchronized void writeFrame(JsonLineBuilder frame) throws IOException {
        if (writer == null) return;
        writer.write(frame.finish());
        writer.newLine();
        if (++pendingFrames >= 20) {
            writer.flush();
            pendingFrames = 0;
        }
    }

    public static String modelState(float[] actions) {
        if (actions == null || actions.length < 7) return null;
        float state = actions[6];
        if (!Float.isFinite(state) || state != (int) state || state < 0 || state >= STATES.length) return null;
        return STATES[(int) state];
    }

    public static void appendModelOutput(JsonLineBuilder frame, float[] actions) {
        frame.field("modelState", modelState(actions));
        JsonLineBuilder probabilities = new JsonLineBuilder(192);
        for (int i = 0; i < STATES.length; i++) {
            probabilities.field(STATES[i], actions != null && actions.length >= 13 ? actions[7 + i] : null);
        }
        frame.field("modelStateProbabilities", new JsonLineBuilder.RawJson(probabilities.finish()));
        JsonLineBuilder auxProbs = new JsonLineBuilder(192);
        for (int i = 0; i < AUXILIARIES.length; i++) {
            auxProbs.field(AUXILIARIES[i], actions != null && actions.length >= 15 ? actions[13 + i] : null);
        }
        frame.field("modelAuxiliaryProbabilities", new JsonLineBuilder.RawJson(auxProbs.finish()));

    }

    public synchronized void close() {
        if (writer == null) return;
        try {
            writer.close();
        } catch (IOException e) {
            System.err.println("[Project Lipsy] Could not close runtime trace: " + e.getMessage());
        } finally {
            writer = null;
        }
    }
}
