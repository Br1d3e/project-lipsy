package com.bot.projectlipsy;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.MathHelper;

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
    private int clickTicksRemaining = 0;
    private int index = 0;
    private String activePattern = "";
    private double comboScore = 0D;
    private double burstScore = 0D;
    private double spamScore = 0D;
    private double comboProb = 0D;
    private double burstProb = 0D;
    private double spamProb = 0D;

    // Use the absolute path to the patterns directory
    private static final File PATTERN_DIR = new File("/Users/br1d3e/Library/CloudStorage/GoogleDrive-wangwangjiman@gmail.com/My Drive/Project Lipsy - Continued/project-lipsy/bot-1.8.9/src/main/java/com/bot/projectlipsy/Click Patterns");
    private final Map<String, List<Long>> patterns;

    private enum CLICK_MODE {
        SPAM,
        BURST,
        COMBO
    }
    private CLICK_MODE clickMode = CLICK_MODE.SPAM;

    private enum TICK_STATE {
        NONE,
        CLICKING,
        FINISHED
    }
    private TICK_STATE tickState = TICK_STATE.NONE;

    // click mode heuristics
    private static final int MIN_DURATION = 10;
    private static final double MAX_BURST_DISTANCE = 1.8D;
    private static final double MIN_COMBO_DISTANCE = 2.1D;
    private static final int MIN_COMBO = 3;
    private static final double COMBO_TARGET_HP = 8D;
    private static final double MIN_HP_DIFF = 4D;
    private static final float TURN_YAW = 90F;

    private String lastMode = "";

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

        utils.updateCombos(target);
        utils.switchItem("Diamond Sword");

        double targetHp = target.getHealth();   // TODO: replace with target health estimator
        double hpDiff = mc.thePlayer.getHealth() - targetHp;

        double dx = mc.thePlayer.posX - target.posX;
        double dz = mc.thePlayer.posZ - target.posZ;
        float targetSupposedYaw = MathHelper.wrapAngleTo180_float((float) (Math.toDegrees(Math.atan2(dz, dx)) - 90F));

        boolean hasCombo = utils.getMyCombo() >= MIN_COMBO;
        boolean availableHp = targetHp <= COMBO_TARGET_HP && hpDiff >= MIN_HP_DIFF;
        boolean targetTurning = Math.abs(MathHelper.wrapAngleTo180_float(targetSupposedYaw - target.rotationYaw)) > TURN_YAW;

        boolean lowDistance = mc.thePlayer.getDistanceToEntity(target) <= MAX_BURST_DISTANCE;
        Entity crossHairEntity = mc.objectMouseOver == null ? null : mc.objectMouseOver.entityHit;
        boolean crossHairOnTarget = crossHairEntity == target;

        updateClickScores(target);
        computeProbabilities();

        if (clickTicksRemaining > 0)  {
            clickTicksRemaining--;
        } else {
            clickTicksRemaining = MIN_DURATION;
            if (comboProb >= 0.6) {
                setClickMode(CLICK_MODE.COMBO);
                if (!lastMode.equals("combo")) {
                    mc.thePlayer.sendChatMessage("[Project L1psy] Click: combo mode");
                    lastMode = "combo";
                }
                resetScores();
            } else if (burstProb >= 0.5) {
                setClickMode(CLICK_MODE.BURST);
                if (!lastMode.equals("burst")) {
                    mc.thePlayer.sendChatMessage("[Project L1psy] Click: burst mode");
                    lastMode = "burst";
                }
                resetScores();
            } else {
                setClickMode(CLICK_MODE.SPAM);
                if (!lastMode.equals("spam")) {
                    mc.thePlayer.sendChatMessage("[Project L1psy] Click: spam mode");
                    lastMode = "spam";
                }
                resetScores();
            }
        }

        switch (clickMode) {
            case COMBO:
                comboClick(target);
                break;
            case BURST:
                burstClick(target);
                break;
            case SPAM:
                spamClick();
                break;
        }

    }

    private void updateClickScores(EntityPlayer target) {
        if (target == null) return;

        double targetHp = target.getHealth();   // TODO: replace with target health estimator
        double hpDiff = mc.thePlayer.getHealth() - targetHp;

        double dx = mc.thePlayer.posX - target.posX;
        double dz = mc.thePlayer.posZ - target.posZ;
        float targetSupposedYaw = MathHelper.wrapAngleTo180_float((float) (Math.toDegrees(Math.atan2(dz, dx)) - 90F));

        boolean hasCombo = utils.getMyCombo() >= MIN_COMBO;
        boolean availableHp = targetHp <= COMBO_TARGET_HP && hpDiff >= MIN_HP_DIFF;
        boolean targetTurning = Math.abs(MathHelper.wrapAngleTo180_float(
                targetSupposedYaw - target.rotationYaw)) > TURN_YAW;
        boolean midDistance = mc.thePlayer.getDistanceToEntity(target) >= MIN_COMBO_DISTANCE;

        boolean lowDistance = mc.thePlayer.getDistanceToEntity(target) <= MAX_BURST_DISTANCE;
        Entity crossHairEntity = mc.objectMouseOver == null ? null : mc.objectMouseOver.entityHit;
        boolean crossHairOnTarget = crossHairEntity == target;

        if (hasCombo) {
            comboScore += 0.25D;
        } else {
            burstScore += 0.15D;
            spamScore += 0.2D;
        }
        if (availableHp) {
            comboScore += 0.15D;
        }
        if (targetTurning) {
            comboScore += 0.12D;
        }
        if (midDistance) {
            comboScore += 0.2D;
            spamScore += 0.1D;
        } else if (lowDistance) {
            comboScore -= 0.18D;
            burstScore += 0.3D;
            spamScore += 0.1D;
        }
        if (crossHairOnTarget) {
            comboScore += 0.05D;
            burstScore += 0.2D;
        } else {
            spamScore += 0.2D;
        }
    }

    private void resetScores() {
        comboScore = 0;
        burstScore = 0;
        spamScore = 0;
    }

    private void computeProbabilities() {
        double expSum = Math.exp(comboScore) + Math.exp(burstScore) + Math.exp(spamScore);
        comboProb = Math.exp(comboScore) / expSum;
        burstProb = Math.exp(burstScore) / expSum;
        spamProb = Math.exp(spamScore) / expSum;
    }


    private void setClickMode(CLICK_MODE nextMode) {
        if (clickMode == nextMode) {
            return;
        }
        clickMode = nextMode;
        index = 0;
        tickState = TICK_STATE.NONE;
        activePattern = "";
        lastClickTime = System.currentTimeMillis();
    }

    public void burstClick(EntityPlayer target) {
        if (target == null || clickMode != CLICK_MODE.BURST) return;

        if (tickState == TICK_STATE.FINISHED) {
            if (target.hurtTime <= 1 && mc.thePlayer.hurtTime > 0) {
                resetPattern("burst_click");
            } else {
                return;
            }
        }
        tick("burst_click", false);
    }

    public void spamClick() {
        if (clickMode != CLICK_MODE.SPAM) return;
        tick("butterfly_3");
    }

    public void comboClick(EntityPlayer target) {
        if (target == null || clickMode != CLICK_MODE.COMBO) return;

        if (tickState == TICK_STATE.FINISHED) {
            if (target.hurtTime <= 1) {
                resetPattern("triple_click");
            } else {
                return;
            }
        }
        tick("triple_click", false);
    }

    public void tick(String patternName) {
        tick(patternName, true);
    }
    public void tick(String patternName, boolean loop) {
        List<Long> pattern = patterns.get(patternName);
        if (pattern == null || pattern.isEmpty()) {
            return;
        }
        if (!patternName.equals(activePattern)) {
            resetPattern(patternName);
        }
        if (!loop && tickState == TICK_STATE.FINISHED) {
            return;
        }

        long now = System.currentTimeMillis();
        if (now - lastClickTime >= pattern.get(index)) {
            tickState = TICK_STATE.CLICKING;
            KeyBinding.onTick(Minecraft.getMinecraft().gameSettings.keyBindAttack.getKeyCode());
            lastClickTime = now;
            index++;
            if (index >= pattern.size()) {
                index = 0;
                tickState = TICK_STATE.FINISHED;
            }
        }
    }

    private void resetPattern(String patternName) {
        activePattern = patternName;
        index = 0;
        tickState = TICK_STATE.NONE;
        lastClickTime = System.currentTimeMillis();
    }

}
