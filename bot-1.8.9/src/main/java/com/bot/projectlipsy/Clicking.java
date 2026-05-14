package com.bot.projectlipsy;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;

public class Clicking {
    private final Minecraft mc;
    private final utils utils;
    private long lastClickTime = 0;
    private int index = 0;

    // Use the absolute path to the patterns directory
    private static final File PATTERN_DIR = new File("/Users/br1d3e/Library/CloudStorage/OneDrive-Personal/Project Lipsy - Continued/forge-1.8.9-11.15.1.2318-1.8.9-mdk/src/main/java/com/bot/projectlipsy/Click Patterns");
    private final Map<String, List<Long>> patterns;

    public Clicking(Minecraft mc) {
        this.mc = mc;
        this.utils = new utils(mc);
        this.patterns = loadPatterns();
        this.lastClickTime = System.currentTimeMillis();
    }

    private Map<String, List<Long>> loadPatterns() {
        Map<String, List<Long>> loadedPatterns = new java.util.HashMap<>();
        System.out.println("[Project Lipsy] Attempting to load click patterns from: " + PATTERN_DIR.getAbsolutePath());

        if (!PATTERN_DIR.exists() || !PATTERN_DIR.isDirectory()) {
            System.err.println("[Project Lipsy] Click pattern directory not found or is not a directory.");
            return loadedPatterns;
        }

        File[] patternFiles = PATTERN_DIR.listFiles();
        if (patternFiles == null || patternFiles.length == 0) {
            System.err.println("[Project Lipsy] No pattern files found in the directory.");
            return loadedPatterns;
        }

        System.out.println("[Project Lipsy] Found " + patternFiles.length + " files in pattern directory.");

        Gson gson = new Gson();
        Type type = new TypeToken<List<Long>>(){}.getType();

        for (File file : patternFiles) {
            if (file.getName().endsWith(".json")) {
                try (FileReader reader = new FileReader(file)) {
                    String patternName = file.getName().replace(".json", "");
                    List<Long> pattern = gson.fromJson(reader, type);
                    if (pattern != null && !pattern.isEmpty()) {
                        loadedPatterns.put(patternName, pattern);
                        System.out.println("[Project Lipsy] Successfully loaded click pattern: " + patternName);
                    } else {
                        System.err.println("[Project Lipsy] Failed to load or pattern is empty for: " + file.getName());
                    }
                } catch (IOException e) {
                    System.err.println("[Project Lipsy] Error reading pattern file: " + file.getName());
                    e.printStackTrace();
                }
            }
        }
        return loadedPatterns;
    }

    public void update(EntityPlayer target) {
        if (target == null) return;

        utils.switchItem("Diamond Sword");

        tick("butterfly1");
    }

    public void tick(String patternName) {
        List<Long> pattern = patterns.get(patternName);
        if (pattern == null || pattern.isEmpty()) {
            return;
        }

        long now = System.currentTimeMillis();
        if (now - lastClickTime >= pattern.get(index)) {
            KeyBinding.onTick(Minecraft.getMinecraft().gameSettings.keyBindAttack.getKeyCode()); // Correct way to simulate a click
            lastClickTime = now;
            index++;
            if (index >= pattern.size()) {
                index = 0; // Loop pattern
            }
        }
    }

}
