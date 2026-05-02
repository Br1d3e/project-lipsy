package com.bot.recorder;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import cpw.mods.fml.client.registry.ClientRegistry;
import cpw.mods.fml.common.FMLCommonHandler;
import cpw.mods.fml.common.Mod;
import cpw.mods.fml.common.event.FMLInitializationEvent;
import cpw.mods.fml.common.eventhandler.SubscribeEvent;
import cpw.mods.fml.common.gameevent.InputEvent;
import cpw.mods.fml.common.gameevent.TickEvent;
import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.potion.Potion;
import net.minecraft.util.ChatComponentText;
import net.minecraft.util.MathHelper;
import net.minecraftforge.common.MinecraftForge;
import org.lwjgl.input.Keyboard;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

@Mod(modid = "projectlipsy", name = "Project Lipsy Recorder", version = "1.0")
public class Recorder {

    private final Minecraft mc = Minecraft.getMinecraft();
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    public static final File ROOT_DIR = new File(Minecraft.getMinecraft().mcDataDir, "ProjectLipsy/recordings");

    private KeyBinding recorderGuiKey;

    // === record state ===
    private boolean recordingAim = false;
    private boolean recordingWTap = false;
    private boolean recordingStrafe = false;
    private boolean recordingRetreat = false;
    private boolean recordingThrowPot = false;
    private boolean recordingTryhardPearl = false;
    private boolean recordingAggroPearl = false;
    private boolean recordingEscapePearl = false;

    // === record data lists ===
    private final List<Map<String, Object>> aimData = new ArrayList<>();
    private final List<Map<String, Object>> wtapData = new ArrayList<>();
    private final List<Map<String, Object>> strafeData = new ArrayList<>();
    private final List<Map<String, Object>> retreatData = new ArrayList<>();
    private final List<Map<String, Object>> throwPotData = new ArrayList<>();
    private final List<Map<String, Object>> tryhardPearlData = new ArrayList<>();
    private final List<Map<String, Object>> aggroPearlData = new ArrayList<>();
    private final List<Map<String, Object>> escapePearlData = new ArrayList<>();

    // === combo tracking ===
    private int myCombo = 0;
    private int targetCombo = 0;
    private int lastMyHurtTime = 0;
    private int lastTargetHurtTime = 0;

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) {
        System.out.println("[Lipsy] Recorder initialized.");
        if (!ROOT_DIR.exists()) ROOT_DIR.mkdirs();

        recorderGuiKey = new KeyBinding("Recorder GUI", Keyboard.KEY_RSHIFT, "Project Lipsy");
        ClientRegistry.registerKeyBinding(recorderGuiKey);
        MinecraftForge.EVENT_BUS.register(this);
        FMLCommonHandler.instance().bus().register(this);
    }

    @SubscribeEvent
    public void onKeyInput(InputEvent.KeyInputEvent event) {
        if (recorderGuiKey.isPressed()) {
            mc.displayGuiScreen(new RecorderGui(this));
        }
    }

    @SubscribeEvent
    public void onTick(TickEvent.ClientTickEvent e) {
        if (mc.thePlayer == null || findClosestPlayer() == null) return;

        EntityPlayer target = findClosestPlayer();
        if (target == null) return;

        updateCombos(target);

        long time = System.currentTimeMillis();
        double dist = mc.thePlayer.getDistanceToEntity(target);
        // Aim Offset
        double dx = target.posX - mc.thePlayer.posX;
        double dy = (target.posY + target.getEyeHeight()) - (mc.thePlayer.posY + mc.thePlayer.getEyeHeight());
        double dz = target.posZ - mc.thePlayer.posZ;
        double distanceXZ = Math.hypot(dx, dz);
        // Horizontal (yaw) offset
        double targetYaw = Math.toDegrees(Math.atan2(dz, dx)) - 90F;
        // Vertical (pitch) offset
        double targetPitch = -Math.toDegrees(Math.atan2(dy, distanceXZ));
        float yawDiff = MathHelper.wrapAngleTo180_float((float) targetYaw - mc.thePlayer.rotationYaw);
        float pitchDiff = MathHelper.wrapAngleTo180_float((float) targetPitch - mc.thePlayer.rotationPitch);

        double targetVelX = target.motionX;
        double targetVelY = target.motionY;
        double targetVelZ = target.motionZ;
        double targetMotion = Math.hypot(targetVelX, targetVelZ);


        if (recordingAim && dist <= 6) recordAimFrame(time, dist, dx, dy, dz, yawDiff, pitchDiff, targetVelX, targetVelY, targetVelZ);
        if (recordingWTap && dist <= 6) recordWTapFrame(time, dist, target, targetMotion);
        if (recordingStrafe && dist <= 6) recordStrafeFrame(time, dist, yawDiff, target);
        if (recordingRetreat) recordRetreatFrame(time, dist, target, targetCombo, yawDiff);
        if (recordingThrowPot && hasItem("Splash Potion of Healing")) recordThrowPotFrame(time, dist, target, yawDiff);
        if (recordingTryhardPearl && hasItem("Ender Pearl")) recordTryhardPearlFrame(time, dist, target, yawDiff, pitchDiff);
        if (recordingAggroPearl && hasItem("Ender Pearl")) recordAggroPearlFrame(time, dist, target, yawDiff, pitchDiff);
        if (recordingEscapePearl && hasItem("Ender Pearl")) recordEscapePearlFrame(time, dist, target, yawDiff, pitchDiff);
    }

    private void recordAimFrame(long time, double dist, double dx, double dy, double dz, float yawDiff, float pitchDiff, double targetVelX, double targetVelY, double targetVelZ) {
        Map<String, Object> frame = new HashMap<>();
        frame.put("time", time);
        frame.put("dx", dx);
        frame.put("dy", dy);
        frame.put("dz", dz);
        frame.put("dist", dist);
        frame.put("targetVelX", targetVelX);
        frame.put("targetVelY", targetVelY);
        frame.put("targetVelZ", targetVelZ);
        frame.put("yawDiffPrev", yawDiff);
        frame.put("PitchDiffPrev", pitchDiff);
        aimData.add(frame);
    }

    private void recordWTapFrame(long time, double dist, EntityPlayer target, double targetMotion) {
        Map<String, Object> frame = new HashMap<>();
        frame.put("time", time);
        frame.put("distance", dist);
        frame.put("targetHurtTime", target.hurtTime);
        frame.put("myHitCooldown", mc.thePlayer.swingProgressInt);
        frame.put("myMotion", Math.hypot(mc.thePlayer.motionX, mc.thePlayer.motionZ));
        frame.put("targetMotion", targetMotion);
        frame.put("targetHP", target.getHealth());
        frame.put("comboCount", myCombo);
        frame.put("pressW", key(mc.gameSettings.keyBindForward));
        frame.put("releaseW", !key(mc.gameSettings.keyBindForward));
        frame.put("pressS", key(mc.gameSettings.keyBindBack));
        wtapData.add(frame);
    }

    private void recordStrafeFrame(long time, double dist, float yawDiff, EntityPlayer target) {
        Map<String, Object> frame = new HashMap<>();
        frame.put("time", time);
        frame.put("distance", dist);
        frame.put("myMotion", Math.hypot(mc.thePlayer.motionX, mc.thePlayer.motionZ));
        frame.put("targetMotion", Math.hypot(target.motionX, target.motionZ));
        frame.put("yawDiff", yawDiff);
        frame.put("myYaw", mc.thePlayer.rotationYaw);
        frame.put("targetYaw", target.rotationYaw);
        frame.put("left", key(mc.gameSettings.keyBindLeft));
        frame.put("right", key(mc.gameSettings.keyBindRight));
        strafeData.add(frame);
    }

    private void recordRetreatFrame(long time, double dist, EntityPlayer target, long targetCombo, float yawDiff) {
        Map<String, Object> frame = new HashMap<>();
        frame.put("time", time);
        frame.put("distance", dist);
        frame.put("myHp", mc.thePlayer.getHealth());
        frame.put("targetHP", target.getHealth());
        frame.put("targetCombo", targetCombo);
        frame.put("targetMotion", Math.hypot(target.motionX, target.motionZ));
        frame.put("yawDiff", yawDiff);
        frame.put("isJumping", key(mc.gameSettings.keyBindJump));
        retreatData.add(frame);
    }

    private void recordThrowPotFrame(long time, double dist, EntityPlayer target, float yawDiff) {
        Map<String, Object> frame = new HashMap<>();
        frame.put("time", time);
        frame.put("myHP", mc.thePlayer.getHealth());
        frame.put("targetHP", target.getHealth());
        frame.put("distance", dist);
        frame.put("myHurtTime", mc.thePlayer.hurtTime);
        frame.put("targetHurtTime", target.hurtTime);
        frame.put("shouldPot", isHoldingItem("Splash Potion of Healing") && key(mc.gameSettings.keyBindUseItem));
        frame.put("aimYaw", mc.thePlayer.rotationYaw);
        frame.put("aimPitch", mc.thePlayer.rotationPitch);
        frame.put("yawDiff", yawDiff);
        frame.put("forward", key(mc.gameSettings.keyBindForward));
        frame.put("back", key(mc.gameSettings.keyBindBack));
        frame.put("jump", key(mc.gameSettings.keyBindJump));
        throwPotData.add(frame);
    }

    private void recordTryhardPearlFrame(long time, double dist, EntityPlayer target, float yawDiff, float pitchDiff) {
        Map<String, Object> frame = new HashMap<>();
        frame.put("time", time);
        frame.put("hpDiff", mc.thePlayer.getHealth() - target.getHealth());
        frame.put("distance", dist);
        frame.put("targetHP", target.getHealth());
        frame.put("myHP", mc.thePlayer.getHealth());
        frame.put("enemyMotion", Math.hypot(target.motionX, target.motionZ));
        frame.put("myMotion", Math.hypot(mc.thePlayer.motionX, target.motionZ));
        frame.put("pearlCooldown", mc.thePlayer.experience);
        frame.put("shouldPearl", isHoldingItem("Ender Pearl") && key(mc.gameSettings.keyBindUseItem));
        frame.put("aimOffsetYaw", yawDiff);
        frame.put("aimOffsetPitch", pitchDiff);
        tryhardPearlData.add(frame);
    }

    private void recordAggroPearlFrame(long time, double dist, EntityPlayer target, float yawDiff, float pitchDiff) {
        Map<String, Object> frame = new HashMap<>();
        frame.put("time", time);
        frame.put("distance", dist);
        frame.put("hpDiff", mc.thePlayer.getHealth() - target.getHealth());
        frame.put("targetHP", target.getHealth());
        frame.put("myHP", mc.thePlayer.getHealth());
        frame.put("enemySpeed", target.isPotionActive(Potion.moveSpeed) ? target.getActivePotionEffect(Potion.moveSpeed).getDuration() : 0);
        frame.put("enemyFood", target.getFoodStats().getFoodLevel());
        frame.put("mySpeed", mc.thePlayer.isPotionActive(Potion.moveSpeed));
        frame.put("myFood", mc.thePlayer.getFoodStats().getFoodLevel());
        frame.put("enemyMotion", Math.hypot(target.motionX, target.motionZ));
        frame.put("pearlCooldown", mc.thePlayer.experience);
        frame.put("shouldPearl", isHoldingItem("Ender Pearl") && key(mc.gameSettings.keyBindUseItem));
        frame.put("throwYawOffset",yawDiff);
        frame.put("throwPitchOffset", pitchDiff);
        aggroPearlData.add(frame);
    }

    private void recordEscapePearlFrame(long time, double dist, EntityPlayer target, float yawDiff, float pitchDiff) {
        Map<String, Object> frame = new HashMap<>();
        frame.put("time", time);
        frame.put("distance", dist);
        frame.put("targetHP", target.getHealth());
        frame.put("myHP", mc.thePlayer.getHealth());
        frame.put("hpDiff", mc.thePlayer.getHealth() - target.getHealth());
        frame.put("enemySpeed", target.isPotionActive(Potion.moveSpeed));
        frame.put("enemyFood", target.getFoodStats().getFoodLevel());
        frame.put("mySpeed", mc.thePlayer.isPotionActive(Potion.moveSpeed));
        frame.put("myFood", mc.thePlayer.getFoodStats().getFoodLevel());
        frame.put("enemyMotion", Math.hypot(target.motionX, target.motionZ));
        frame.put("pearlCooldown", mc.thePlayer.experience);
        frame.put("shouldPearl", isHoldingItem("Ender Pearl") && key(mc.gameSettings.keyBindUseItem));
        frame.put("throwYawOffset", yawDiff);
        frame.put("throwPitchOffset", pitchDiff);
        escapePearlData.add(frame);
    }

    private void updateCombos(EntityPlayer target) {
        if (target.hurtTime > lastTargetHurtTime) {
            myCombo++;
            targetCombo = 0;
        }
        lastTargetHurtTime = target.hurtTime;

        if (mc.thePlayer.hurtTime > lastMyHurtTime) {
            targetCombo++;
            myCombo = 0;
        }
        lastMyHurtTime = mc.thePlayer.hurtTime;
    }

    private EntityPlayer findClosestPlayer() {
        if (mc.theWorld == null) return null;
        EntityPlayer closestPlayer = null;
        double closestDistance = Double.MAX_VALUE;

        for (Object playerObject : mc.theWorld.playerEntities) {
            if (playerObject instanceof EntityPlayer) {
                EntityPlayer player = (EntityPlayer) playerObject;
                if (player != mc.thePlayer) {
                    double distance = mc.thePlayer.getDistanceToEntity(player);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestPlayer = player;
                    }
                }
            }
        }
        return closestPlayer;
    }

    private boolean isHoldingItem(String displayName) {
        if (mc.thePlayer.getHeldItem() != null) {
            return mc.thePlayer.getHeldItem().getDisplayName().contains(displayName);
        }
        return false;
    }

    private boolean hasItem(String displayName) {
        for (int i = 0; i < 36; i++) {
            if (mc.thePlayer.inventory.getStackInSlot(i) != null && mc.thePlayer.inventory.getStackInSlot(i).getDisplayName().contains(displayName)) {
                return true;
            }
        }
        return false;
    }

    private boolean key(KeyBinding key) {
        return key.getIsKeyPressed();
    }

    private void saveData(String name, List<Map<String, Object>> data) {
        if (data.isEmpty()) return;
        try {
            File dir = new File(ROOT_DIR, name);
            if (!dir.exists()) dir.mkdirs();
            File out = new File(dir, name + "_" + System.currentTimeMillis() + ".json");
            FileWriter w = new FileWriter(out);
            GSON.toJson(data, w);
            w.close();
            sendMessage("Saved " + data.size() + " frames to " + out.getName());
            data.clear();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void sendMessage(String message) {
        mc.thePlayer.addChatMessage(new ChatComponentText("§7[§bLipsy§7] §f" + message));
    }

    public void startAimRecording() {recordingAim = true; sendMessage("Aim recording started.");}
    public void stopAimRecording() {recordingAim = false; saveData("aim", aimData); sendMessage("Aim recording stopped."); }

    public void startWTapRecording() { recordingWTap = true; sendMessage("WTap recording started."); }
    public void stopWTapRecording() { recordingWTap = false; saveData("wtap", wtapData); sendMessage("WTap recording stopped."); }

    public void startStrafeRecording() { recordingStrafe = true; sendMessage("Strafe recording started."); }
    public void stopStrafeRecording() { recordingStrafe = false; saveData("strafe", strafeData); sendMessage("Strafe recording stopped."); }

    public void startRetreatRecording() { recordingRetreat = true; sendMessage("Retreat recording started."); }
    public void stopRetreatRecording() { recordingRetreat = false; saveData("retreat", retreatData); sendMessage("Retreat recording stopped."); }

    public void startThrowPotRecording() { recordingThrowPot = true; sendMessage("ThrowPot recording started."); }
    public void stopThrowPotRecording() { recordingThrowPot = false; saveData("throwpot", throwPotData); sendMessage("ThrowPot recording stopped."); }

    public void startTryhardPearlRecording() { recordingTryhardPearl = true; sendMessage("Tryhard Pearl recording started."); }
    public void stopTryhardPearlRecording() { recordingTryhardPearl = false; saveData("strategic_pearl", tryhardPearlData); sendMessage("Tryhard Pearl recording stopped."); }

    public void startAggroPearlRecording() { recordingAggroPearl = true; sendMessage("Aggro Pearl recording started."); }
    public void stopAggroPearlRecording() { recordingAggroPearl = false; saveData("aggro_pearl", aggroPearlData); sendMessage("Aggro Pearl recording stopped."); }

    public void startEscapePearlRecording() { recordingEscapePearl = true; sendMessage("Escape Pearl recording started."); }
    public void stopEscapePearlRecording() { recordingEscapePearl = false; saveData("escape_pearl", escapePearlData); sendMessage("Escape Pearl recording stopped."); }

    public boolean isRecordingAim() {return recordingAim;}
    public boolean isRecordingWTap() { return recordingWTap; }
    public boolean isRecordingStrafe() { return recordingStrafe; }
    public boolean isRecordingRetreat() { return recordingRetreat; }
    public boolean isRecordingThrowPot() { return recordingThrowPot; }
    public boolean isRecordingTryhardPearl() { return recordingTryhardPearl; }
    public boolean isRecordingAggroPearl() { return recordingAggroPearl; }
    public boolean isRecordingEscapePearl() { return recordingEscapePearl; }
}
