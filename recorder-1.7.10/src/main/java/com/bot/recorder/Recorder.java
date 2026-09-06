package com.bot.recorder;

import com.bot.recorder.features.CombatHistoryTracker;
import com.bot.recorder.features.TargetMovementTracker;
import com.bot.recorder.features.TargetPlaystyleTracker;
import com.bot.recorder.features.RecorderFeatureConfig;
import com.bot.recorder.recordWriter.JsonLineBuilder;
import com.bot.recorder.recordWriter.RecordWriter;
import com.bot.recorder.targetTracker.TargetFoodTracker;
import com.bot.recorder.targetTracker.TargetHpTracker;
import com.bot.recorder.targetTracker.TargetHitTracker;
import com.bot.recorder.targetTracker.PearlCooldownTracker;
import com.bot.recorder.targetTracker.TargetInventoryTracker;
import cpw.mods.fml.client.registry.ClientRegistry;
import cpw.mods.fml.common.FMLCommonHandler;
import cpw.mods.fml.common.Mod;
import cpw.mods.fml.common.event.FMLInitializationEvent;
import cpw.mods.fml.common.event.FMLPreInitializationEvent;
import cpw.mods.fml.common.eventhandler.SubscribeEvent;
import cpw.mods.fml.common.gameevent.InputEvent;
import cpw.mods.fml.common.gameevent.TickEvent;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.inventory.GuiInventory;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.block.Block;
import net.minecraft.entity.Entity;
import net.minecraft.entity.item.EntityEnderPearl;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemStack;
import net.minecraft.network.Packet;
import net.minecraft.network.play.client.C02PacketUseEntity;
import net.minecraft.network.play.client.C03PacketPlayer;
import net.minecraft.network.play.client.C08PacketPlayerBlockPlacement;
import net.minecraft.network.play.client.C09PacketHeldItemChange;
import net.minecraft.network.play.server.S12PacketEntityVelocity;
import net.minecraft.network.play.server.S08PacketPlayerPosLook;
import net.minecraft.network.play.server.S18PacketEntityTeleport;
import net.minecraft.network.play.server.S13PacketDestroyEntities;
import net.minecraft.network.play.server.S28PacketEffect;
import net.minecraft.potion.Potion;
import net.minecraft.util.ChatComponentText;
import net.minecraft.util.MathHelper;
import net.minecraftforge.common.MinecraftForge;
import org.lwjgl.input.Keyboard;
import java.io.File;
import java.io.IOException;
import java.util.*;

@Mod(modid = "projectlipsy", name = "Project Lipsy Recorder", version = "3.0")
public class Recorder {

    private final Minecraft mc = Minecraft.getMinecraft();
    private static final String RECORDER_VERSION = "v5";
    private static final int PEARL_COOLDOWN_TICKS = 320;
    private static final int WALL_CHECK_DISTANCE = 5;
    private static final int INVENTORY_SAMPLE_INTERVAL_TICKS = 5;
    private static final int SPATIAL_SAMPLE_INTERVAL_TICKS = 2;
    private static final int PACKET_HOOK_CHECK_INTERVAL_TICKS = 20;
    private static final String PACKET_HOOK_NAME = "lipsy_recorder_packet_hook";
    public static final File ROOT_DIR = new File(Minecraft.getMinecraft().mcDataDir, "ProjectLipsy/recordings");
    private final RecordWriter recordWriter = new RecordWriter(ROOT_DIR, RECORDER_VERSION, new RecordWriter.ErrorReporter() {
        public void report(String message) {
            sendMessage(message);
        }
    });
    private final ItemsHandler itemsHandler = new ItemsHandler(mc);
    private final RecorderFeatureConfig featureConfig = new RecorderFeatureConfig();
    private CombatHistoryTracker combatHistory = new CombatHistoryTracker(0.95D, 60);
    private final TargetMovementTracker targetMovement = new TargetMovementTracker();
    private final TargetPlaystyleTracker targetPlaystyle = new TargetPlaystyleTracker();
    private Float previousSelfHp;
    private final TargetHitTracker targetHitTracker = new TargetHitTracker();
    private final TargetHpTracker targetHpTracker = new TargetHpTracker(mc, itemsHandler, new TargetHpTracker.EventSink() {
        public void onTargetHealed(int targetEntityId) {
            targetHitTracker.recordHeal(targetEntityId);
            combatHistory.targetHealed(recordingTickIndex);
        }

        public void writeEvent(String type, JsonLineBuilder details) {
            Recorder.this.writeEvent(type, details);
        }
    });
    private final TargetFoodTracker targetFoodTracker = new TargetFoodTracker(new TargetFoodTracker.EventSink() {
        public void writeEvent(String type, JsonLineBuilder details) {
            Recorder.this.writeEvent(type, details);
        }
    });
    private final TargetInventoryTracker targetInventoryTracker = new TargetInventoryTracker(mc, itemsHandler);
    private final PacketHandler packetHandler = new PacketHandler(mc, PACKET_HOOK_NAME, PACKET_HOOK_CHECK_INTERVAL_TICKS, new PacketHandler.Listener() {
        public boolean isRecording() {
            return Recorder.this.isRecording();
        }

        public int getRecordingTickIndex() {
            return Recorder.this.getRecordingTickIndex();
        }

        public void onPacketHookStatus(String type, String message) {
            Recorder.this.writeEvent(type, detailsBuilder()
                    .field("message", message)
                    .field("hasNetworkManager", mc.getNetHandler() != null && mc.getNetHandler().getNetworkManager() != null));
        }

        public void onOutgoingPacket(Packet packet, int tick) {
            Recorder.this.handleOutgoingPacket(packet, tick);
        }

        public void onIncomingPacket(Packet packet, int tick) {
            Recorder.this.handleIncomingPacket(packet, tick);
        }
    });
    private KeyBinding recorderGuiKey;
    private KeyBinding nextRoundKey;
    private int recorderRoundId = 0;
    private UUID roundTargetUuid;
    private final PearlCooldownTracker selfPearlTracker = new PearlCooldownTracker();
    private boolean pendingSelfPearlUse;
    private RecorderGui recorderOverlay;

    // === record state ===
    private boolean recording = false;
    private String sessionId;
    private long startTimeMs;
    private long endTimeMs;
    private long startWorldTick;
    private String serverIp = "unknown";
    private int recordingTickIndex = 0;
    private Float previousMyYaw = null;
    private Float previousMyPitch = null;
    private final Map<Integer, Float> previousTargetYawById = new HashMap<Integer, Float>();
    private final Map<Integer, Float> previousTargetPitchById = new HashMap<Integer, Float>();
    private int pearlCooldownTicksRemaining = 0;
    private int enemyPearlCooldownTicksRemaining = 0;
    private boolean previousUseItemPressed = false;
    private boolean previousAttackPressed = false;
    private int lastAttackTick = -1;
    private int lastEatTick = -1;
    private int lastSpeedTick = -1;
    private int lastDamageTakenTick = -1;
    private int lastHitLandedTick = -1;
    private int lastPearlTick = -1;
    private int lastEnemyPearlTick = -1;
    private int lastPotTick = -1;
    private int lastBlockTick = -1;
    private int lastUseItemPacketTick = -1;
    private int lastVelocityEventTick = -1;
    private int lastVelocityEntityId = -1;
    private int lastVelocityMotionXRaw = 0;
    private int lastVelocityMotionYRaw = 0;
    private int lastVelocityMotionZRaw = 0;
    private int lastCriticalMovementPacketTick = -1;
    private int lastCriticalAttackTick = -1;
    private int lastCriticalAttackTargetEntityId = -1;
    private boolean lastCriticalAttackWasCritical = false;
    private int lastInventorySampleTick = -1;
    private List<String> cachedHotbar = new ArrayList<String>();
    private int cachedHotbarPotions = 0;
    private int cachedInventoryFood = 0;
    private int cachedInventorySpeedPotions = 0;
    private int cachedInventoryHealthPotions = 0;
    private int cachedInventoryPearls = 0;
    private int lastSpatialSampleTick = -1;
    private Integer cachedBlockDistanceFront = null;
    private Integer cachedBlockDistanceBack = null;
    private Integer cachedBlockDistanceLeft = null;
    private Integer cachedBlockDistanceRight = null;
    private boolean cachedWallFront = false;
    private boolean cachedWallBehind = false;
    private boolean cachedWallLeft = false;
    private boolean cachedWallRight = false;
    private final JsonLineBuilder frameBuilder = new JsonLineBuilder(4096);
    private final Set<Integer> seenEnemyPearlEntityIds = new HashSet<Integer>();
    private final Set<Integer> activeEnemyPearlEntityIds = new HashSet<Integer>();

    // === combo tracking ===
    private int myCombo = 0;
    private int targetCombo = 0;
    private int lastMyHurtTime = 0;
    private int lastTargetHurtTime = 0;
    private int comboTrackedTargetId = -1;

    @Mod.EventHandler
    public void preInit(FMLPreInitializationEvent event) {
        featureConfig.load(event.getSuggestedConfigurationFile());
        combatHistory = new CombatHistoryTracker(featureConfig.hitPressureDecay, featureConfig.combatHistoryTicks);
    }

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) {
        System.out.println("[Lipsy] Recorder initialized.");
        if (!ROOT_DIR.exists()) ROOT_DIR.mkdirs();

        recorderGuiKey = new KeyBinding("Recorder GUI", Keyboard.KEY_RSHIFT, "Project Lipsy");
        ClientRegistry.registerKeyBinding(recorderGuiKey);
        nextRoundKey = new KeyBinding("Next recorder round", Keyboard.KEY_BACK, "Project Lipsy");
        ClientRegistry.registerKeyBinding(nextRoundKey);
        recorderOverlay = new RecorderGui(this);
        MinecraftForge.EVENT_BUS.register(this);
        MinecraftForge.EVENT_BUS.register(recorderOverlay);
        FMLCommonHandler.instance().bus().register(this);
    }

    @SubscribeEvent
    public synchronized void onKeyInput(InputEvent.KeyInputEvent event) {
        if (nextRoundKey.isPressed() && recording) {
            recorderRoundId++;
            resetRoundState();
            writeEvent("ROUND_START", detailsBuilder().field("source", "hotkey"));
            sendMessage("Recorder round " + recorderRoundId + " started; tracking state cleared.");
        }
        if (recorderGuiKey.isPressed()) {
            mc.displayGuiScreen(new RecorderGui(this));
        }
    }

    @SubscribeEvent
    public synchronized void onTick(TickEvent.ClientTickEvent e) {
        if (e.phase != TickEvent.Phase.END) return;
        installPacketHook();
        if (!recording) return;
        if (mc.thePlayer == null) {
            clearTargetObservation();
            previousSelfHp = null;
            return;
        }

        EntityPlayer target = findClosestPlayer();
        int currentTick = recordingTickIndex;
        combatHistory.advance(currentTick);
        float selfHp = mc.thePlayer.getHealth();
        if (previousSelfHp != null && selfHp > previousSelfHp) combatHistory.selfHealed(currentTick);
        previousSelfHp = selfHp;
        if (target != null) {
            targetMovement.update(currentTick, target.getEntityId(), mc.thePlayer.posX, mc.thePlayer.posZ,
                    target.posX, target.posZ, target.rotationYaw);
        }
        targetHitTracker.updateTarget(target == null ? -1 : target.getEntityId());
        targetHpTracker.updateTargetState(target, currentTick);
        targetFoodTracker.updateTargetState(target, currentTick);
        updateComboTargetState(target);

        if (target != null) {
            updateCombos(target);
        } else {
            clearTargetObservation();
        }

        updateInventoryCacheIfNeeded(currentTick);
        updateSpatialCacheIfNeeded(currentTick);
        targetHpTracker.updateTargetPotionState(target, currentTick);
        if (targetInventoryTracker.trackHealthPotions(target)) {
            TargetPlaystyleTracker.HealEpisode episode = targetPlaystyle.healingPotionThrown(currentTick);
            if (episode != null) {
                writeEvent("TARGET_HEAL_EPISODE_SAMPLE", detailsBuilder()
                        .field("targetEntityId", target.getEntityId())
                        .field("hitsBeforeHealing", episode.hitsBeforeHealing)
                        .field("completeObservation", episode.hitsBeforeHealing != null)
                        .field("tick", currentTick));
            }
            targetHitTracker.recordHeal(target.getEntityId());
        }
        targetInventoryTracker.trackPearls(target);
        targetFoodTracker.updateMovementExhaustion(target);
        targetFoodTracker.updateFoodUse(target, target == null ? null : target.getHeldItem(),
                target != null && itemsHandler.isFood(target.getHeldItem()), currentTick);
        targetFoodTracker.finishFoodTick();
        if (target != null && targetHpTracker.updateNaturalRegen(currentTick, targetFoodTracker.getTargetHunger())) {
            targetFoodTracker.addNaturalRegenExhaustion();
        }
        boolean useItemPressed = key(mc.gameSettings.keyBindUseItem);
        boolean useItemStarted = useItemPressed && !previousUseItemPressed;
        boolean attackPressed = key(mc.gameSettings.keyBindAttack);
        boolean attackStarted = attackPressed && !previousAttackPressed;
        ItemStack heldItem = mc.thePlayer.getHeldItem();

        if (attackStarted) {
            lastAttackTick = currentTick;
        }

        if (useItemStarted && itemsHandler.isSplashHealthPotion(heldItem)) {
            lastPotTick = currentTick;
        }

        if (useItemStarted && itemsHandler.isFood(heldItem)) {
            lastEatTick = currentTick;
        }

        if (useItemStarted && itemsHandler.isSpeedPotion(heldItem)) {
            lastSpeedTick = currentTick;
        }

        boolean pearlUseAttempt = pendingSelfPearlUse || (itemsHandler.isPearl(heldItem) && useItemStarted);
        pendingSelfPearlUse = false;
        if (selfPearlTracker.update(pearlUseAttempt, itemsHandler.countPearls(), PEARL_COOLDOWN_TICKS)) {
            lastPearlTick = currentTick;
            writeEvent("PEARL_THROW", detailsBuilder().field("source", "use_or_inventory_decrease"));
        }
        pearlCooldownTicksRemaining = selfPearlTracker.getRemaining();

        if (itemsHandler.isSword(heldItem) && useItemStarted) {
            lastBlockTick = currentTick;
        }

        boolean targetPearling = updateEnemyPearlState(target, currentTick);
        previousUseItemPressed = useItemPressed;
        previousAttackPressed = attackPressed;

        long time = System.currentTimeMillis();
        Float myDeltaYaw = previousMyYaw == null ? null : MathHelper.wrapAngleTo180_float(mc.thePlayer.rotationYaw - previousMyYaw);
        Float myDeltaPitch = previousMyPitch == null ? null : mc.thePlayer.rotationPitch - previousMyPitch;
        previousMyYaw = mc.thePlayer.rotationYaw;
        previousMyPitch = mc.thePlayer.rotationPitch;

        boolean targetExists = target != null;
        int targetId = -1;
        Double targetRelX = null;
        Double targetRelY = null;
        Double targetRelZ = null;
        Double targetDistance = null;
        Double yawDiffToTarget = null;
        Double pitchDiffToTarget = null;
        Float targetYawVel = null;
        Float targetPitchVel = null;

        if (targetExists) {
            targetId = target.getEntityId();
            targetRelX = target.posX - mc.thePlayer.posX;
            targetRelY = (target.posY + target.getEyeHeight()) - (mc.thePlayer.posY + mc.thePlayer.getEyeHeight());
            targetRelZ = target.posZ - mc.thePlayer.posZ;
            targetDistance = (double) mc.thePlayer.getDistanceToEntity(target);
            double distanceXZ = Math.hypot(targetRelX, targetRelZ);
            double targetAimYaw = Math.toDegrees(Math.atan2(targetRelZ, targetRelX)) - 90F;
            double targetAimPitch = -Math.toDegrees(Math.atan2(targetRelY, distanceXZ));
            yawDiffToTarget = (double) MathHelper.wrapAngleTo180_float((float) targetAimYaw - mc.thePlayer.rotationYaw);
            pitchDiffToTarget = (double) MathHelper.wrapAngleTo180_float((float) targetAimPitch - mc.thePlayer.rotationPitch);

            Float previousTargetYaw = previousTargetYawById.get(targetId);
            Float previousTargetPitch = previousTargetPitchById.get(targetId);
            targetYawVel = previousTargetYaw == null ? null : MathHelper.wrapAngleTo180_float(target.rotationYaw - previousTargetYaw);
            targetPitchVel = previousTargetPitch == null ? null : target.rotationPitch - previousTargetPitch;
            previousTargetYawById.put(targetId, target.rotationYaw);
            previousTargetPitchById.put(targetId, target.rotationPitch);
        }

        Entity crosshairEntity = getMouseOverEntity();

        JsonLineBuilder frame = frameBuilder.reset();
        frame.field("recorderRoundId", recorderRoundId);
        frame.field("recordingTickIndex", recordingTickIndex++);
        frame.field("timeMs", time);
        frame.field("worldTick", getWorldTick());
        frame.field("myPosX", mc.thePlayer.posX);
        frame.field("myPosY", mc.thePlayer.posY);
        frame.field("myPosZ", mc.thePlayer.posZ);
        frame.field("myMotionX", mc.thePlayer.motionX);
        frame.field("myMotionY", mc.thePlayer.motionY);
        frame.field("myMotionZ", mc.thePlayer.motionZ);
        frame.field("myYaw", mc.thePlayer.rotationYaw);
        frame.field("myPitch", mc.thePlayer.rotationPitch);
        frame.field("myDeltaYaw", myDeltaYaw);
        frame.field("myDeltaPitch", myDeltaPitch);
        frame.field("myHp", mc.thePlayer.getHealth());
        frame.field("myHurtTime", mc.thePlayer.hurtTime);
        frame.field("myHurtResistantTime", mc.thePlayer.hurtResistantTime);
        frame.field("myOnGround", mc.thePlayer.onGround);
        frame.field("mySprinting", mc.thePlayer.isSprinting());
        frame.field("myMoveForward", mc.thePlayer.moveForward);
        frame.field("myMoveStrafing", mc.thePlayer.moveStrafing);
        frame.field("mySwingProgress", mc.thePlayer.swingProgress);
        frame.field("myUsingItem", mc.thePlayer.isUsingItem());
        frame.field("myItemUseTicks", mc.thePlayer.isUsingItem() ? mc.thePlayer.getItemInUseDuration() : 0);
        frame.field("myHeldItem", itemsHandler.getDisplayName(mc.thePlayer.getHeldItem()));
        frame.field("myHeldItemType", itemsHandler.getHeldItemType(mc.thePlayer));
        frame.field("myBlockingSword", itemsHandler.isSword(mc.thePlayer.getHeldItem()) && mc.thePlayer.isUsingItem());
        frame.field("myHasSpeedPotion", itemsHandler.hasPotion(mc.thePlayer, Potion.moveSpeed));
        frame.field("mySpeedDurationTicks", itemsHandler.getPotionDuration(mc.thePlayer, Potion.moveSpeed));
        frame.field("myFoodLevel", mc.thePlayer.getFoodStats().getFoodLevel());
        frame.field("myCombo", myCombo);
        frame.field("targetCombo", targetCombo);
        frame.field("pearlCooldownTicksRemaining", pearlCooldownTicksRemaining);
        frame.field("enemyPearlAvailable", enemyPearlCooldownTicksRemaining == 0);
        frame.field("enemyPearlCooldownTicksRemaining", enemyPearlCooldownTicksRemaining);
        frame.field("inInventory", mc.currentScreen instanceof GuiInventory);
        frame.field("selectedSlot", mc.thePlayer.inventory.currentItem);
        frame.field("hotbar", cachedHotbar);
        frame.field("hotbarHealthPotions", cachedHotbarPotions);
        frame.field("refillableHealthPotions", Math.max(0, cachedInventoryHealthPotions - cachedHotbarPotions));
        frame.field("inventoryFood", cachedInventoryFood);
        frame.field("inventorySpeedPotions", cachedInventorySpeedPotions);
        frame.field("inventoryHealthPotions", cachedInventoryHealthPotions);
        frame.field("inventoryPearls", cachedInventoryPearls);
        frame.field("crosshairTargetType", getCrosshairTargetType());
        frame.field("crosshairTargetId", crosshairEntity == null ? null : crosshairEntity.getEntityId());
        frame.field("crosshairOnTarget", targetExists && crosshairEntity == target);
        frame.field("crosshairDistance", crosshairEntity == null ? null : (double) mc.thePlayer.getDistanceToEntity(crosshairEntity));
        frame.field("ticksSinceLastAttack", ticksSince(lastAttackTick, currentTick));
        frame.field("ticksSinceLastBlock", ticksSince(lastBlockTick, currentTick));
        frame.field("ticksSinceLastPot", ticksSince(lastPotTick, currentTick));
        frame.field("ticksSinceLastEat", ticksSince(lastEatTick, currentTick));
        frame.field("ticksSinceLastSpeed", ticksSince(lastSpeedTick, currentTick));
        frame.field("ticksSinceLastDamageTaken", ticksSince(lastDamageTakenTick, currentTick));
        frame.field("ticksSinceLastHitLanded", ticksSince(lastHitLandedTick, currentTick));
        frame.field("ticksSinceLastPearl", ticksSince(lastPearlTick, currentTick));
        frame.field("ticksSinceLastEnemyPearl", ticksSince(lastEnemyPearlTick, currentTick));
        frame.field("ticksSinceLastUseItemPacket", ticksSince(lastUseItemPacketTick, currentTick));
        frame.field("blockDistanceFront", cachedBlockDistanceFront);
        frame.field("blockDistanceBack", cachedBlockDistanceBack);
        frame.field("blockDistanceLeft", cachedBlockDistanceLeft);
        frame.field("blockDistanceRight", cachedBlockDistanceRight);
        frame.field("aForward", actionForward());
        frame.field("aStrafe", actionStrafe());
        frame.field("aJump", key(mc.gameSettings.keyBindJump) ? 1 : 0);
        frame.field("aAttack", attackPressed || lastAttackTick == currentTick ? 1 : 0);
        frame.field("aUseItem", useItemPressed || lastUseItemPacketTick == currentTick ? 1 : 0);
        frame.field("targetExists", targetExists);
        frame.field("targetId", targetExists ? targetId : null);
        frame.field("targetRelX", targetRelX);
        frame.field("targetRelY", targetRelY);
        frame.field("targetRelZ", targetRelZ);
        frame.field("targetDistance", targetDistance);
        frame.field("targetMotionX", targetExists ? target.motionX : null);
        frame.field("targetMotionY", targetExists ? target.motionY : null);
        frame.field("targetMotionZ", targetExists ? target.motionZ : null);
        frame.field("targetYaw", targetExists ? target.rotationYaw : null);
        frame.field("targetPitch", targetExists ? target.rotationPitch : null);
        frame.field("targetDeltaYaw", targetYawVel);
        frame.field("targetDeltaPitch", targetPitchVel);
        frame.field("targetHpRaw", targetExists ? target.getHealth() : null);
        frame.field("hitsBeforeTargetHeals", targetExists ? targetHitTracker.getHits() : null);
        frame.field("meanHitsBeforeTargetHeals", targetPlaystyle.getMeanHealHits());
        frame.field("medianHitsBeforeTargetHeals", targetPlaystyle.getMedianHealHits());
        frame.field("targetHealEpisodeSampleCount", targetPlaystyle.getHealSampleCount());
        frame.field("meanTargetPearlGapTicks", targetPlaystyle.getMeanPearlGap());
        frame.field("medianTargetPearlGapTicks", targetPlaystyle.getMedianPearlGap());
        frame.field("targetPearlIntervalSampleCount", targetPlaystyle.getPearlSampleCount());
        frame.field("myHitPressure", combatHistory.getMyPressure());
        frame.field("targetHitPressure", combatHistory.getTargetPressure());
        frame.field("hitsLandedLastK", combatHistory.getHitsLanded());
        frame.field("hitsTakenLastK", combatHistory.getHitsTaken());
        frame.field("netHitsLastK", combatHistory.getNetHits());
        frame.field("ticksSinceCombat", combatHistory.getTicksSinceCombat(currentTick));
        frame.field("targetRadialVelocity", targetMovement.getRadialVelocity());
        frame.field("targetOutwardVelocity", targetMovement.getOutwardVelocity());
        frame.field("targetFacingMeCos", targetMovement.getFacingMeCos());
        frame.field("distanceDelta5", targetMovement.getDistanceDelta5());
        frame.field("distanceDelta20", targetMovement.getDistanceDelta20());
        frame.field("targetEstimatedHp", targetExists ? targetHpTracker.getEstimatedTargetHp() : null);
        frame.field("targetEstimatedPotions", targetExists ? targetInventoryTracker.getTargetHealthPotions() : null);
        frame.field("targetEstimatedPearls", targetExists ? targetInventoryTracker.getTargetPearls() : null);
        frame.field("targetHurtTime", targetExists ? target.hurtTime : null);
        frame.field("targetHurtResistantTime", targetExists ? target.hurtResistantTime : null);
        frame.field("targetOnGround", targetExists ? target.onGround : null);
        frame.field("targetSprinting", targetExists ? target.isSprinting() : null);
        frame.field("targetSwingProgress", targetExists ? target.swingProgress : null);
        frame.field("targetUsingItem", targetExists ? target.isUsingItem() : null);
        frame.field("targetHeldItemType", targetExists ? itemsHandler.getHeldItemType(target) : null);
        frame.field("targetHasSpeedPotion", targetExists ? itemsHandler.hasPotion(target, Potion.moveSpeed) : null);
        frame.field("targetSpeedDurationTicks", targetExists ? itemsHandler.getPotionDuration(target, Potion.moveSpeed): null);
        frame.field("targetFood", targetExists ? targetFoodTracker.getTargetHunger() : null);
        frame.field("targetFoodRaw", targetExists ? target.getFoodStats().getFoodLevel() : null);
        Boolean targetHoldingFood = targetExists ? itemsHandler.isFood(target.getHeldItem()) : null;
        Boolean targetHoldingSpeedPotion = targetExists ? itemsHandler.isSpeedPotion(target.getHeldItem()) : null;
        Boolean targetHoldingHealPotion = targetExists ? itemsHandler.isSplashHealthPotion(target.getHeldItem()) : null;
        frame.field("targetHoldingSword", targetExists ? itemsHandler.isSword(target.getHeldItem()) : null);
        frame.field("targetHoldingFood", targetHoldingFood);
        frame.field("targetHoldingSpeedPotion", targetHoldingSpeedPotion);
        frame.field("targetPearling", targetExists ? targetPearling : null);
        frame.field("targetHoldingHealPotion", targetHoldingHealPotion);
        // Compatibility aliases now describe holding, not active item use.
        frame.field("targetEating", targetHoldingFood);
        frame.field("targetSpeeding", targetHoldingSpeedPotion);
        boolean targetPotting = combatHistory.consumeTargetPotting();
        frame.field("targetPotting", targetExists ? targetPotting : null);
        frame.field("yawDiffToTarget", yawDiffToTarget);
        frame.field("pitchDiffToTarget", pitchDiffToTarget);
        frame.field("hpDiff", targetExists ? targetHpTracker.getEstimatedTargetHp() - mc.thePlayer.getHealth() : null);
        writeFrame(frame);

    }



    private void updateCombos(EntityPlayer target) {
        if (target.hurtTime > lastTargetHurtTime) {
            if (targetHitTracker.recordHurt(target.getEntityId(), recordingTickIndex)) {
                combatHistory.landedHit(recordingTickIndex);
                targetPlaystyle.landedHit();
            }
            myCombo++;
            targetCombo = 0;
            lastHitLandedTick = recordingTickIndex;
            targetHpTracker.applyEstimatedTargetDamage(recordingTickIndex, target, mc.thePlayer == null ? null : mc.thePlayer.getHeldItem(),
                    wasLastLandedHitCritical(target));
            targetFoodTracker.addTakingDamageExhaustion();
        }
        lastTargetHurtTime = target.hurtTime;

        if (mc.thePlayer.hurtTime > lastMyHurtTime) {
            combatHistory.tookHit(recordingTickIndex);
            targetCombo++;
            myCombo = 0;
            lastDamageTakenTick = recordingTickIndex;
            targetFoodTracker.addSuccessfulMeleeAttackExhaustion();
            writeEvent("DAMAGE_TAKEN", detailsBuilder()
                    .field("hurtTime", mc.thePlayer.hurtTime)
                    .field("hurtResistantTime", mc.thePlayer.hurtResistantTime)
                    .field("health", mc.thePlayer.getHealth()));
        }
        lastMyHurtTime = mc.thePlayer.hurtTime;
    }

    private boolean wasLastLandedHitCritical(EntityPlayer target) {
        if (!lastCriticalAttackWasCritical) return false;
        if (lastCriticalAttackTick < 0 || recordingTickIndex - lastCriticalAttackTick > 3) return false;

        return target != null && target.getEntityId() == lastCriticalAttackTargetEntityId;
    }

    private void updateComboTargetState(EntityPlayer target) {
        int targetId = target == null ? -1 : target.getEntityId();
        if (targetId == comboTrackedTargetId) return;

        comboTrackedTargetId = targetId;
        lastTargetHurtTime = target == null ? 0 : target.hurtTime;
        lastMyHurtTime = mc.thePlayer == null ? 0 : mc.thePlayer.hurtTime;
    }

    private EntityPlayer findClosestPlayer() {
        if (mc.theWorld == null || mc.thePlayer == null) return null;
        EntityPlayer closestPlayer = null;
        double closestDistance = Double.MAX_VALUE;

        for (Object playerObject : mc.theWorld.playerEntities) {
            if (playerObject instanceof EntityPlayer) {
                EntityPlayer player = (EntityPlayer) playerObject;
                if (player != mc.thePlayer && !player.isInvisible() && player.getHealth() > 0 && !player.isDead) {
                    if (roundTargetUuid != null && !roundTargetUuid.equals(player.getUniqueID())) continue;
                    double distance = mc.thePlayer.getDistanceToEntity(player);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestPlayer = player;
                    }
                }
            }
        }
        if (closestPlayer != null && roundTargetUuid == null) roundTargetUuid = closestPlayer.getUniqueID();
        return closestPlayer;
    }

    private void clearTargetObservation() {
        targetPlaystyle.observationLost();
        targetMovement.reset();
        comboTrackedTargetId = -1;
        previousTargetYawById.clear();
        previousTargetPitchById.clear();
        targetHitTracker.updateTarget(-1);
        targetHpTracker.updateTargetState(null, recordingTickIndex);
        targetHpTracker.updateTargetPotionState(null, recordingTickIndex);
        targetFoodTracker.updateTargetState(null, recordingTickIndex);
        activeEnemyPearlEntityIds.clear();
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

    private void updateInventoryCacheIfNeeded(int currentTick) {
        if (lastInventorySampleTick >= 0 && currentTick - lastInventorySampleTick < INVENTORY_SAMPLE_INTERVAL_TICKS) {
            return;
        }

        lastInventorySampleTick = currentTick;
        cachedHotbar = itemsHandler.getHotbarSnapshot();
        cachedHotbarPotions = itemsHandler.countHotbarHealthPotions();
        cachedInventoryFood = itemsHandler.countFoodItems();
        cachedInventorySpeedPotions = itemsHandler.countSpeedPotions();
        cachedInventoryHealthPotions = itemsHandler.countSplashHealthPotions();
        cachedInventoryPearls = itemsHandler.countPearls();
    }

    private void updateSpatialCacheIfNeeded(int currentTick) {
        if (lastSpatialSampleTick >= 0 && currentTick - lastSpatialSampleTick < SPATIAL_SAMPLE_INTERVAL_TICKS) {
            return;
        }

        lastSpatialSampleTick = currentTick;
        cachedBlockDistanceFront = blockDistanceInDirection(0.0F);
        cachedBlockDistanceBack = blockDistanceInDirection(180.0F);
        cachedBlockDistanceLeft = blockDistanceInDirection(-90.0F);
        cachedBlockDistanceRight = blockDistanceInDirection(90.0F);
        cachedWallFront = cachedBlockDistanceFront != null && cachedBlockDistanceFront <= WALL_CHECK_DISTANCE;
        cachedWallBehind = cachedBlockDistanceBack != null && cachedBlockDistanceBack <= WALL_CHECK_DISTANCE;
        cachedWallLeft = cachedBlockDistanceLeft != null && cachedBlockDistanceLeft <= WALL_CHECK_DISTANCE;
        cachedWallRight = cachedBlockDistanceRight != null && cachedBlockDistanceRight <= WALL_CHECK_DISTANCE;
    }

    private boolean updateEnemyPearlState(EntityPlayer target, int currentTick) {
        if (enemyPearlCooldownTicksRemaining > 0) {
            enemyPearlCooldownTicksRemaining--;
        }

        if (target == null || mc.theWorld == null) {
            activeEnemyPearlEntityIds.clear();
            return false;
        }

        boolean detectedThrow = false;
        Set<Integer> visibleEnemyPearlIds = new HashSet<Integer>();
        for (Object entityObject : mc.theWorld.loadedEntityList) {
            if (!(entityObject instanceof EntityEnderPearl)) continue;

            EntityEnderPearl pearl = (EntityEnderPearl) entityObject;
            int pearlId = pearl.getEntityId();
            if (activeEnemyPearlEntityIds.contains(pearlId)) {
                visibleEnemyPearlIds.add(pearlId);
                continue;
            }

            if (seenEnemyPearlEntityIds.contains(pearlId)) continue;

            if (!isEnemyPearlFromTarget(pearl, target)) continue;

            detectedThrow = true;
            seenEnemyPearlEntityIds.add(pearlId);
            activeEnemyPearlEntityIds.add(pearlId);
            visibleEnemyPearlIds.add(pearlId);
            Integer pearlGap = targetPlaystyle.pearlThrown(currentTick);
            if (pearlGap != null) {
                writeEvent("TARGET_PEARL_INTERVAL_SAMPLE", detailsBuilder()
                        .field("targetEntityId", target.getEntityId())
                        .field("pearlEntityId", pearlId)
                        .field("gapTicks", pearlGap)
                        .field("tick", currentTick));
            }
            lastEnemyPearlTick = currentTick;
            enemyPearlCooldownTicksRemaining = PEARL_COOLDOWN_TICKS;
            writeEvent("ENEMY_PEARL_THROW", detailsBuilder()
                    .field("targetEntityId", target.getEntityId())
                    .field("pearlEntityId", pearlId)
                    .field("pearlTicksExisted", pearl.ticksExisted)
                    .field("distanceToTarget", (double) target.getDistanceToEntity(pearl)));
        }

        applyEnemyPearlDamageForLandedPearls(target, visibleEnemyPearlIds, currentTick);
        return detectedThrow;
    }

    private void applyEnemyPearlDamageForLandedPearls(EntityPlayer target, Set<Integer> visibleEnemyPearlIds, int currentTick) {
        Iterator<Integer> iterator = activeEnemyPearlEntityIds.iterator();
        while (iterator.hasNext()) {
            int pearlId = iterator.next();
            if (visibleEnemyPearlIds.contains(pearlId)) continue;

            targetHpTracker.applyEstimatedTargetPearlDamage(currentTick, target, pearlId);
            iterator.remove();
        }
    }

    private boolean isEnemyPearlFromTarget(EntityEnderPearl pearl, EntityPlayer target) {
        Entity thrower = pearl.getThrower();
        if (thrower == mc.thePlayer) return false;
        if (thrower == target) return true;
        if (thrower != null) return false;

        if (pearl.ticksExisted > 5) return false;

        double targetDistance = target.getDistanceToEntity(pearl);
        if (targetDistance > 4.0D) return false;

        if (mc.thePlayer != null) {
            double selfDistance = mc.thePlayer.getDistanceToEntity(pearl);
            if (selfDistance + 0.5D < targetDistance) return false;
        }

        return true;
    }

    private void installPacketHook() {
        packetHandler.install();
    }

    private synchronized void handleOutgoingPacket(Packet packet, int tick) {
        if (!recording) return;

        if (packet instanceof C02PacketUseEntity) {
            handleUseEntityPacket((C02PacketUseEntity) packet, tick);
        } else if (packet instanceof C03PacketPlayer) {
            handlePlayerMovementPacket((C03PacketPlayer) packet, tick);
        } else if (packet instanceof C08PacketPlayerBlockPlacement) {
            handleUseItemPacket((C08PacketPlayerBlockPlacement) packet, tick);
        } else if (packet instanceof C09PacketHeldItemChange) {
            handleHeldItemChangePacket((C09PacketHeldItemChange) packet, tick);
        }
    }

    private void handleUseEntityPacket(C02PacketUseEntity packet, int tick) {
        Entity entity = mc.theWorld == null ? null : packet.getEntityFromWorld(mc.theWorld);
        JsonLineBuilder details = detailsBuilder()
                .field("action", packet.getAction() == null ? null : packet.getAction().name())
                .field("targetEntityId", entity == null ? null : entity.getEntityId())
                .field("targetEntityType", entity == null ? null : entity.getClass().getSimpleName());

        if (packet.getAction() == C02PacketUseEntity.Action.ATTACK) {
            targetHitTracker.recordAttack(entity == null ? -1 : entity.getEntityId(), tick);
            boolean criticalHit = isCriticalAttackPacket(tick);
            lastAttackTick = tick;
            lastCriticalAttackTick = tick;
            lastCriticalAttackTargetEntityId = entity == null ? -1 : entity.getEntityId();
            lastCriticalAttackWasCritical = criticalHit;
            details.field("criticalHit", criticalHit)
                    .field("lastCriticalMovementPacketTick", lastCriticalMovementPacketTick < 0 ? null : lastCriticalMovementPacketTick);
            writeEvent("ATTACK_PACKET_SENT", details);
        }
    }

    private void handlePlayerMovementPacket(C03PacketPlayer packet, int tick) {
        if (!packet.func_149465_i()) {
            lastCriticalMovementPacketTick = tick;
        }
    }

    private boolean isCriticalAttackPacket(int tick) {
        if (lastCriticalMovementPacketTick >= 0 && tick - lastCriticalMovementPacketTick <= 1) {
            return true;
        }

        if (mc.thePlayer == null) return false;
        if (mc.thePlayer.onGround) return false;
        if (mc.thePlayer.fallDistance <= 0.0F) return false;
        if (mc.thePlayer.isOnLadder()) return false;
        if (mc.thePlayer.isInWater()) return false;
        if (mc.thePlayer.isPotionActive(Potion.blindness)) return false;
        return mc.thePlayer.ridingEntity == null;
    }

    private void handleUseItemPacket(C08PacketPlayerBlockPlacement packet, int tick) {
        ItemStack stack = packet.getStack();
        if (stack == null && mc.thePlayer != null) {
            stack = mc.thePlayer.getHeldItem();
        }

        lastUseItemPacketTick = tick;
        writeEvent("USE_ITEM_PACKET_SENT", detailsBuilder()
                .field("itemType", itemsHandler.getItemType(stack))
                .field("displayName", itemsHandler.getDisplayName(stack))
                .field("placedBlockX", packet.getPlacedBlockX())
                .field("placedBlockY", packet.getPlacedBlockY())
                .field("placedBlockZ", packet.getPlacedBlockZ())
                .field("placedBlockDirection", packet.getPlacedBlockDirection()));

        if (itemsHandler.isSplashHealthPotion(stack)) {
            lastPotTick = tick;
            writeEvent("POTION_USE", itemDetails(stack).field("source", "C08PacketPlayerBlockPlacement"));
        }

        if (itemsHandler.isFood(stack)) {
            lastEatTick = tick;
            writeEvent("FOOD_EAT", itemDetails(stack).field("source", "C08PacketPlayerBlockPlacement"));
        }

        if (itemsHandler.isSpeedPotion(stack)) {
            lastSpeedTick = tick;
            writeEvent("SPEED_USE", itemDetails(stack).field("source", "C08PacketPlayerBlockPlacement"));
        }

        if (itemsHandler.isPearl(stack)) {
            pendingSelfPearlUse = true;
        }
    }

    private void handleHeldItemChangePacket(C09PacketHeldItemChange packet, int tick) {
        writeEvent("SLOT_CHANGED", detailsBuilder()
                .field("slot", packet.getSlotId()));
    }

    private synchronized void handleIncomingPacket(Packet packet, int tick) {
        if (!recording) return;

        if (packet instanceof S08PacketPlayerPosLook) {
            targetMovement.reset();
        } else if (packet instanceof S18PacketEntityTeleport) {
            EntityPlayer target = findClosestPlayer();
            if (target != null && target.getEntityId() == ((S18PacketEntityTeleport) packet).func_149451_c()) {
                targetMovement.reset();
            }
        } else if (packet instanceof S12PacketEntityVelocity) {
            handleEntityVelocityPacket((S12PacketEntityVelocity) packet, tick);
        } else if (packet instanceof S28PacketEffect) {
            handleEffectPacket((S28PacketEffect) packet, tick);
        } else if (packet instanceof S13PacketDestroyEntities) {
            targetHpTracker.handleDestroyedEntities(((S13PacketDestroyEntities) packet).func_149098_c(), tick);
        }
    }

    private void handleEffectPacket(S28PacketEffect packet, int tick) {
        targetHpTracker.handlePotionImpactEffect(findClosestPlayer(),
                packet.getSoundType(),
                packet.getSoundData(),
                packet.getPosX(),
                packet.getPosY(),
                packet.getPosZ(),
                tick);
    }

    private void handleEntityVelocityPacket(S12PacketEntityVelocity packet, int tick) {
        int entityId = packet.func_149412_c();
        boolean isSelf = mc.thePlayer != null && entityId == mc.thePlayer.getEntityId();
        if (!isSelf) return;

        int motionXRaw = packet.func_149411_d();
        int motionYRaw = packet.func_149410_e();
        int motionZRaw = packet.func_149409_f();
        if (lastVelocityEventTick == tick
                && lastVelocityEntityId == entityId
                && lastVelocityMotionXRaw == motionXRaw
                && lastVelocityMotionYRaw == motionYRaw
                && lastVelocityMotionZRaw == motionZRaw) {
            return;
        }

        lastVelocityEventTick = tick;
        lastVelocityEntityId = entityId;
        lastVelocityMotionXRaw = motionXRaw;
        lastVelocityMotionYRaw = motionYRaw;
        lastVelocityMotionZRaw = motionZRaw;

        writeEvent("VELOCITY_PACKET_RECEIVED", detailsBuilder()
                .field("entityId", entityId)
                .field("isSelf", isSelf)
                .field("motionXRaw", motionXRaw)
                .field("motionYRaw", motionYRaw)
                .field("motionZRaw", motionZRaw)
                .field("motionX", motionXRaw / 8000.0D)
                .field("motionY", motionYRaw / 8000.0D)
                .field("motionZ", motionZRaw / 8000.0D));
    }

    private Integer blockDistanceInDirection(float yawOffset) {
        if (mc.theWorld == null || mc.thePlayer == null) return null;

        double yawRadians = Math.toRadians(mc.thePlayer.rotationYaw + yawOffset);
        double dirX = -Math.sin(yawRadians);
        double dirZ = Math.cos(yawRadians);

        for (int distance = 1; distance <= WALL_CHECK_DISTANCE; distance++) {
            int x = MathHelper.floor_double(mc.thePlayer.posX + dirX * distance);
            int y = MathHelper.floor_double(mc.thePlayer.posY + 1.0D);
            int z = MathHelper.floor_double(mc.thePlayer.posZ + dirZ * distance);
            Block block = mc.theWorld.getBlock(x, y, z);

            if (block != null && block.getMaterial().isSolid()) return distance;
        }

        return null;
    }

    private Entity getMouseOverEntity() {
        if (mc.objectMouseOver == null) return null;

        return mc.objectMouseOver.entityHit;
    }

    private String getCrosshairTargetType() {
        if (mc.objectMouseOver == null || mc.objectMouseOver.typeOfHit == null) return null;

        return mc.objectMouseOver.typeOfHit.name();
    }

    private Integer nullableTick(int tick) {
        return tick < 0 ? null : tick;
    }

    private Integer ticksSince(int tick, int currentTick) {
        return tick < 0 ? null : currentTick - tick;
    }

    private int actionForward() {
        int forward = key(mc.gameSettings.keyBindForward) ? 1 : 0;
        int back = key(mc.gameSettings.keyBindBack) ? 1 : 0;

        return forward - back;
    }

    private int actionStrafe() {
        int right = key(mc.gameSettings.keyBindRight) ? 1 : 0;
        int left = key(mc.gameSettings.keyBindLeft) ? 1 : 0;

        return right - left;
    }

    private void writeFrame(JsonLineBuilder frame) {
        recordWriter.writeFrame(frame);
    }

    private void writeEvent(String type, JsonLineBuilder details) {
        recordWriter.writeEvent(type, details.field("recorderRoundId", recorderRoundId), recordingTickIndex, getWorldTick());
    }

    private JsonLineBuilder detailsBuilder() {
        return recordWriter.detailsBuilder();
    }

    private JsonLineBuilder itemDetails(ItemStack stack) {
        return detailsBuilder()
                .field("itemType", itemsHandler.getItemType(stack))
                .field("displayName", itemsHandler.getDisplayName(stack));
    }

    private long getWorldTick() {
        return mc.theWorld == null ? 0 : mc.theWorld.getTotalWorldTime();
    }

    private String getServerIp() {
        if (mc.getCurrentServerData() != null && mc.getCurrentServerData().serverIP != null) {
            return mc.getCurrentServerData().serverIP;
        }

        return "singleplayer";
    }

    public void sendMessage(String message) {
        if (mc.thePlayer != null) {
            mc.thePlayer.addChatMessage(new ChatComponentText("§7[§bL1psy§7] §f" + message));
        }
    }

    private void resetRoundState() {
        targetPlaystyle.reset();
        combatHistory.reset();
        targetMovement.reset();
        previousSelfHp = null;
        roundTargetUuid = null;
        pendingSelfPearlUse = false;
        selfPearlTracker.reset(mc.thePlayer == null ? -1 : itemsHandler.countPearls());
        lastInventorySampleTick = -1;
        cachedHotbar = new ArrayList<String>();
        cachedHotbarPotions = 0;
        cachedInventoryFood = 0;
        cachedInventorySpeedPotions = 0;
        cachedInventoryHealthPotions = 0;
        cachedInventoryPearls = 0;
        lastSpatialSampleTick = -1;
        cachedWallBehind = false;
        cachedWallLeft = false;
        cachedWallRight = false;
        cachedWallFront = false;
        cachedBlockDistanceFront = null;
        cachedBlockDistanceBack = null;
        cachedBlockDistanceLeft = null;
        cachedBlockDistanceRight = null;
        previousMyYaw = null;
        previousMyPitch = null;
        previousTargetYawById.clear();
        previousTargetPitchById.clear();
        pearlCooldownTicksRemaining = 0;
        enemyPearlCooldownTicksRemaining = 0;
        seenEnemyPearlEntityIds.clear();
        activeEnemyPearlEntityIds.clear();
        previousUseItemPressed = false;
        previousAttackPressed = false;
        lastAttackTick = -1;
        lastEatTick = -1;
        lastSpeedTick = -1;
        lastDamageTakenTick = -1;
        lastHitLandedTick = -1;
        lastPearlTick = -1;
        lastEnemyPearlTick = -1;
        lastPotTick = -1;
        lastBlockTick = -1;
        lastUseItemPacketTick = -1;
        lastVelocityEventTick = -1;
        lastVelocityEntityId = -1;
        lastVelocityMotionXRaw = 0;
        lastVelocityMotionYRaw = 0;
        lastVelocityMotionZRaw = 0;
        lastCriticalMovementPacketTick = -1;
        lastCriticalAttackTick = -1;
        lastCriticalAttackTargetEntityId = -1;
        lastCriticalAttackWasCritical = false;
        myCombo = 0;
        targetCombo = 0;
        lastMyHurtTime = 0;
        lastTargetHurtTime = 0;
        comboTrackedTargetId = -1;
        targetHitTracker.reset();
        targetHpTracker.reset();
        targetFoodTracker.reset();
        targetInventoryTracker.reset();
    }

    public synchronized void startRecording() {
        if (recording) return;

        startTimeMs = System.currentTimeMillis();
        endTimeMs = 0;
        startWorldTick = getWorldTick();
        serverIp = getServerIp();
        recordingTickIndex = 0;
        recorderRoundId = 1;
        resetRoundState();
        packetHandler.resetSession();

        try {
            recordWriter.startSession(startTimeMs, startWorldTick, serverIp);
            sessionId = recordWriter.getSessionId();
            recording = true;
            writeEvent("RECORD_START", detailsBuilder()
                    .field("hitPressureDecay", featureConfig.hitPressureDecay)
                    .field("combatHistoryTicks", featureConfig.combatHistoryTicks)
                    .field("movementVelocityWindowTicks", 5)
                    .field("healEpisodeGapTicks", TargetPlaystyleTracker.HEAL_EPISODE_GAP_TICKS)
                    .field("sessionId", sessionId)
                    .field("serverIp", serverIp)
                    .field("startTimeMs", startTimeMs)
                    .field("startWorldTick", startWorldTick));
            writeEvent("ROUND_START", detailsBuilder().field("source", "record_start"));
            sendMessage("Recording started: " + sessionId + " (round 1; Backspace starts next round)");
        } catch (IOException ex) {
            ex.printStackTrace();
            sendMessage("Failed to start recording: " + ex.getMessage());
        }
    }

    public synchronized void stopRecording() {
        if (!recording) return;

        recording = false;
        endTimeMs = System.currentTimeMillis();
        writeEvent("RECORD_STOP", detailsBuilder()
                .field("sessionId", sessionId)
                .field("endTimeMs", endTimeMs)
                .field("endWorldTick", getWorldTick()));
        recordWriter.finishSession(endTimeMs);
        sendMessage("Recording stopped: " + sessionId);
    }


    public boolean isRecording() {return recording;}

    public String getSessionId() {return sessionId;}

    public int getRecordingTickIndex() {return recordingTickIndex;}

    public long getCurrentTimeMs() {return System.currentTimeMillis();}

    public int getRecordingRoundId() {return recorderRoundId;}
}
