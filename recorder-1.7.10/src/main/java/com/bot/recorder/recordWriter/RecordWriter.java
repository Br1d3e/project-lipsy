package com.bot.recorder.recordWriter;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

public class RecordWriter {
    public interface ErrorReporter {
        void report(String message);
    }

    private static final int WRITER_FLUSH_INTERVAL_FRAMES = 20;

    private final File rootDir;
    private final String recorderVersion;
    private final ErrorReporter errorReporter;
    private final Gson gson = new GsonBuilder().setPrettyPrinting().create();
    private final JsonLineBuilder eventBuilder = new JsonLineBuilder(4096);

    private File sessionDir;
    private String sessionId;
    private GZIPOutputStream framesWriter;
    private FileWriter eventsWriter;
    private long startTimeMs;
    private long endTimeMs;
    private long startWorldTick;
    private String serverIp;
    private int framesSinceFlush = 0;
    private int eventsSinceFlush = 0;

    public RecordWriter(File rootDir, String recorderVersion, ErrorReporter errorReporter) {
        this.rootDir = rootDir;
        this.recorderVersion = recorderVersion;
        this.errorReporter = errorReporter;
    }

    public void startSession(long startTimeMs, long startWorldTick, String serverIp) throws IOException {
        if (!rootDir.exists() && !rootDir.mkdirs()) {
            throw new IOException("Failed to create recordings root folder: " + rootDir.getAbsolutePath());
        }

        this.startTimeMs = startTimeMs;
        this.endTimeMs = 0;
        this.startWorldTick = startWorldTick;
        this.serverIp = serverIp;
        this.sessionId = createSessionId(startTimeMs);
        this.sessionDir = new File(rootDir, sessionId);
        this.framesSinceFlush = 0;
        this.eventsSinceFlush = 0;

        if (!sessionDir.exists() && !sessionDir.mkdirs()) {
            throw new IOException("Failed to create session folder: " + sessionDir.getAbsolutePath());
        }

        framesWriter = new GZIPOutputStream(new FileOutputStream(new File(sessionDir, "frames.jsonl.gz")));
        eventsWriter = new FileWriter(new File(sessionDir, "events.jsonl"));
        writeSessionMeta();
    }

    public void finishSession(long endTimeMs) {
        this.endTimeMs = endTimeMs;
        writeSessionMeta();
        closeFramesWriter();
        closeEventsWriter();
    }

    public void writeFrame(JsonLineBuilder frame) {
        if (framesWriter == null) return;

        try {
            framesWriter.write(frame.finish().getBytes(StandardCharsets.UTF_8));
            framesWriter.write('\n');
            framesSinceFlush++;
            if (framesSinceFlush >= WRITER_FLUSH_INTERVAL_FRAMES) {
                framesWriter.flush();
                framesSinceFlush = 0;
            }
        } catch (IOException ex) {
            ex.printStackTrace();
            report("Failed to write frame: " + ex.getMessage());
        }
    }

    public void writeEvent(String type, JsonLineBuilder details, int recordingTickIndex, long worldTick) {
        if (eventsWriter == null) return;

        JsonLineBuilder event = eventBuilder.reset();
        event.field("type", type);
        event.field("recording_tick_index", recordingTickIndex);
        event.field("world_tick", worldTick);
        event.field("time_ms", System.currentTimeMillis());
        event.field("details", new JsonLineBuilder.RawJson(details.finish()));
        writeEvent(event);
    }

    public JsonLineBuilder newFrameBuilder() {
        return new JsonLineBuilder(4096);
    }

    public JsonLineBuilder detailsBuilder() {
        return new JsonLineBuilder(512);
    }

    public String getSessionId() {
        return sessionId;
    }

    public File getSessionDir() {
        return sessionDir;
    }

    private void writeEvent(JsonLineBuilder event) {
        if (eventsWriter == null) return;

        try {
            eventsWriter.write(event.finish());
            eventsWriter.write("\n");
            eventsSinceFlush++;
            if (eventsSinceFlush >= WRITER_FLUSH_INTERVAL_FRAMES) {
                eventsWriter.flush();
                eventsSinceFlush = 0;
            }
        } catch (IOException ex) {
            ex.printStackTrace();
            report("Failed to write event: " + ex.getMessage());
        }
    }

    private void writeSessionMeta() {
        if (sessionDir == null) return;

        Map<String, Object> meta = new LinkedHashMap<String, Object>();
        meta.put("sessionId", sessionId);
        meta.put("recorderVersion", recorderVersion);
        meta.put("startTimeMs", startTimeMs);
        meta.put("endTimeMs", endTimeMs);
        meta.put("startWorldTick", startWorldTick);
        meta.put("serverIp", serverIp);

        FileWriter writer = null;
        try {
            writer = new FileWriter(new File(sessionDir, "session_meta.json"));
            gson.toJson(meta, writer);
        } catch (IOException ex) {
            ex.printStackTrace();
            report("Failed to write session_meta.json: " + ex.getMessage());
        } finally {
            if (writer != null) {
                try {
                    writer.close();
                } catch (IOException ignored) {
                }
            }
        }
    }

    private String createSessionId(long timeMs) {
        String base = "session_" + timeMs;
        String candidate = base;
        int suffix = 1;

        while (new File(rootDir, candidate).exists()) {
            candidate = base + "_" + suffix;
            suffix++;
        }

        return candidate;
    }

    private void closeFramesWriter() {
        if (framesWriter == null) return;

        try {
            framesWriter.flush();
            framesWriter.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        framesSinceFlush = 0;
        framesWriter = null;
    }

    private void closeEventsWriter() {
        if (eventsWriter == null) return;

        try {
            eventsWriter.flush();
            eventsWriter.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        eventsSinceFlush = 0;
        eventsWriter = null;
    }

    private void report(String message) {
        if (errorReporter != null) {
            errorReporter.report(message);
        }
    }
}
