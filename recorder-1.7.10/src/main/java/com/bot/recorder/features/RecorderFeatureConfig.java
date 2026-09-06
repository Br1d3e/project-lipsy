package com.bot.recorder.features;

import java.io.File;
import net.minecraftforge.common.config.Configuration;

public class RecorderFeatureConfig {
    public double hitPressureDecay = 0.95D;
    public int combatHistoryTicks = 60;

    public void load(File file) {
        Configuration config = new Configuration(file);
        config.load();
        hitPressureDecay = config.getFloat("hitPressureDecay", "features", 0.95F, 0.0F, 1.0F,
                "Per recording tick hit-pressure multiplier. Applied before adding new hits.");
        if (Double.isNaN(hitPressureDecay)) hitPressureDecay = 0.95D;
        combatHistoryTicks = config.getInt("combatHistoryTicks", "features", 80, 1, 1200,
                "Rolling hit window K, including the current tick.");
        if (config.hasChanged()) config.save();
    }
}
