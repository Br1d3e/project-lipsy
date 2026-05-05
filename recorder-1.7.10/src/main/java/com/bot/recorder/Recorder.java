package com.bot.recorder;

import com.bot.recorder.recordWriter.JsonLineBuilder;
import com.bot.recorder.recordWriter.RecordWriter;
import com.bot.recorder.targetTracker.TargetFoodTracker;
import com.bot.recorder.targetTracker.TargetHpTracker;
import cpw.mods.fml.client.registry.ClientRegistry;
import cpw.mods.fml.common.FMLCommonHandler;
import cpw.mods.fml.common.Mod;
import cpw.mods.fml.common.event.FMLInitializationEvent;
import cpw.mods.fml.common.eventhandler.SubscribeEvent;
import cpw.mods.fml.common.gameevent.InputEvent;
import cpw.mods.fml.common.gameevent.TickEvent;
import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.block.Block;
import net.minecraft.entity.Entity;
import net.minecraft.entity.item.EntityEnderPearl;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemStack;
import net.minecraft.network.Packet;
import net.minecraft.network.play.client.C02PacketUseEntity;
import net.minecraft.network.play.client.C08PacketPlayerBlockPlacement;
import net.minecraft.network.play.client.C09PacketHeldItemChange;
import net.minecraft.network.play.server.S12PacketEntityVelocity;
import net.minecraft.potion.Potion;
import net.minecraft.util.ChatComponentText;
import net.minecraft.util.MathHelper;
import net.minecraftforge.common.MinecraftForge;
import org.lwjgl.input.Keyboard;
import java.io.File;
import java.io.IOException;
import java.util.*;

@Mod(modid = "projectlipsy", name = "Project Lipsy Recorder", version = "2.0")
public class Recorder {

    private final Minecraft mc = Minecraft.getMinecraft();
    private static final String RECORDER_VERSION = "v2";
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
    private final TargetHpTracker targetHpTracker = new TargetHpTracker(mc, new TargetHpTracker.EventSink() {
        public void writeEvent(String type, JsonLineBuilder details) {
            Recorder.this.writeEvent(type, details);
        }
    });
    private final TargetFoodTracker targetFoodTracker = new TargetFoodTracker(new TargetFoodTracker.EventSink() {
        public void writeEvent(String type, JsonLineBuilder details) {
            Recorder.this.writeEvent(type, details);
        }
    });
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
    private final ItemsHandler itemsHandler = new ItemsHandler(mc);

    private KeyBinding recorderGuiKey;
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
    private int lastUseItemPacketTick = -1;
    private int lastVelocityEventTick = -1;
    private int lastVelocityEntityId = -1;
    private int lastVelocityMotionXRaw = 0;
    private int lastVelocityMotionYRaw = 0;
    private int lastVelocityMotionZRaw = 0;
    private int lastInventorySampleTick = -1;
    private List<String> cachedHotbar = new ArrayList<String>();
    private int cachedInventoryFood = 0;
    private int cachedInventorySpeedPotions = 0;
    private int cachedInventoryHealthPotions = 0;
    private int cachedInventoryPearls = 0;
    private int lastSpatialSampleTick = -1;
    private boolean cachedWallBehind = false;
    private boolean cachedWallLeft = false;
    private boolean cachedWallRight = false;
    private boolean cachedWallFront = false;
    private Integer cachedBlockDistanceFront = null;
    private Integer cachedBlockDistanceBack = null;
    private Integer cachedBlockDistanceLeft = null;
    private Integer cachedBlockDistanceRight = null;
    private final JsonLineBuilder frameBuilder = new JsonLineBuilder(4096);
    private final Set<Integer> seenEnemyPearlEntityIds = new HashSet<Integer>();

    // === combo tracking ===
    private int myCombo = 0;
    private int targetCombo = 0;
    private int lastMyHurtTime = 0;
    private int lastTargetHurtTime = 0;
    private int comboTrackedTargetId = -1;

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) {
        System.out.println("[Lipsy] Recorder initialized.");
        if (!ROOT_DIR.exists()) ROOT_DIR.mkdirs();

        recorderGuiKey = new KeyBinding("Recorder GUI", Keyboard.KEY_RSHIFT, "Project Lipsy");
        ClientRegistry.registerKeyBinding(recorderGuiKey);
        recorderOverlay = new RecorderGui(this);
        MinecraftForge.EVENT_BUS.register(this);
        MinecraftForge.EVENT_BUS.register(recorderOverlay);
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
        if (e.phase != TickEvent.Phase.END) return;
        installPacketHook();
        if (!recording) return;
        if (mc.thePlayer == null) return;

        EntityPlayer target = findClosestPlayer();
        int currentTick = recordingTickIndex;
        targetHpTracker.updateTargetState(target, currentTick);
        targetFoodTracker.updateTargetState(target, currentTick);
        updateComboTargetState(target);

        if (target != null) {
            updateCombos(target);
        }

        updateInventoryCacheIfNeeded(currentTick);
        updateSpatialCacheIfNeeded(currentTick);
        targetHpTracker.updateTargetPotionState(target, currentTick);
        targetFoodTracker.updateMovementExhaustion(target);
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

        if (itemsHandler.isPearl(heldItem) && useItemStarted && pearlCooldownTicksRemaining <= 0) {
            lastPearlTick = currentTick;
            pearlCooldownTicksRemaining = PEARL_COOLDOWN_TICKS;
        } else if (pearlCooldownTicksRemaining > 0) {
            pearlCooldownTicksRemaining--;
        }

        boolean enemyPearl = updateEnemyPearlState(target, currentTick);
        previousUseItemPressed = useItemPressed;
        previousAttackPressed = attackPressed;

        long time = System.currentTimeMillis();
        Float myDeltaYaw = previousMyYaw == null ? null : MathHelper.wrapAngleTo180_float(mc.thePlayer.rotationYaw - previousMyYaw);
        Float myDeltaPitch = previousMyPitch == null ? null : mc.thePlayer.rotationPitch - previousMyPitch;
        previousMyYaw = mc.thePlayer.rotationYaw;
        previousMyPitch = mc.thePlayer.rotationPitch;

        boolean targetExists = target != null;
        Double targetRelX = null;
        Double targetRelY = null;
        Double targetRelZ = null;
        Double targetDistance = null;
        Double yawDiffToTarget = null;
        Double pitchDiffToTarget = null;
        Double hpDiff = null;
        Float targetYawVel = null;
        Float targetPitchVel = null;

        if (targetExists) {
            targetRelX = target.posX - mc.thePlayer.posX;
            targetRelY = (target.posY + target.getEyeHeight()) - (mc.thePlayer.posY + mc.thePlayer.getEyeHeight());
            targetRelZ = target.posZ - mc.thePlayer.posZ;
            targetDistance = (double) mc.thePlayer.getDistanceToEntity(target);
            double distanceXZ = Math.hypot(targetRelX, targetRelZ);
            double targetAimYaw = Math.toDegrees(Math.atan2(targetRelZ, targetRelX)) - 90F;
            double targetAimPitch = -Math.toDegrees(Math.atan2(targetRelY, distanceXZ));
            yawDiffToTarget = (double) MathHelper.wrapAngleTo180_float((float) targetAimYaw - mc.thePlayer.rotationYaw);
            pitchDiffToTarget = (double) MathHelper.wrapAngleTo180_float((float) targetAimPitch - mc.thePlayer.rotationPitch);
            hpDiff = (double) (mc.thePlayer.getHealth() - target.getHealth());

            int targetId = target.getEntityId();
            Float previousTargetYaw = previousTargetYawById.get(targetId);
            Float previousTargetPitch = previousTargetPitchById.get(targetId);
            targetYawVel = previousTargetYaw == null ? null : MathHelper.wrapAngleTo180_float(target.rotationYaw - previousTargetYaw);
            targetPitchVel = previousTargetPitch == null ? null : target.rotationPitch - previousTargetPitch;
            previousTargetYawById.put(targetId, target.rotationYaw);
            previousTargetPitchById.put(targetId, target.rotationPitch);
        }

        Entity crosshairEntity = getMouseOverEntity();

        JsonLineBuilder frame = frameBuilder.reset();
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
        frame.field("myHeldItemType", itemsHandler.getHeldItemType(mc.thePlayer));
        frame.field("myHasSpeedPotion", itemsHandler.hasPotion(mc.thePlayer, Potion.moveSpeed));
        frame.field("mySpeedDurationTicks", itemsHandler.getPotionDuration(mc.thePlayer, Potion.moveSpeed));
        frame.field("myFoodLevel", mc.thePlayer.getFoodStats().getFoodLevel());
        frame.field("myCombo", myCombo);
        frame.field("targetCombo", targetCombo);
        frame.field("pearlCooldownTicksRemaining", pearlCooldownTicksRemaining);
        frame.field("pearlCooldownSecondsRemaining", pearlCooldownTicksRemaining / 20.0F);
        frame.field("enemyPearl", enemyPearl);
        frame.field("enemyPearlCooldownTicksRemaining", enemyPearlCooldownTicksRemaining);
        frame.field("enemyPearlCooldownSecondsRemaining", enemyPearlCooldownTicksRemaining / 20.0F);
        frame.field("selectedSlot", mc.thePlayer.inventory.currentItem);
        frame.field("hotbar", cachedHotbar);
        frame.field("inventoryFood", cachedInventoryFood);
        frame.field("inventorySpeedPotions", cachedInventorySpeedPotions);
        frame.field("inventoryHealthPotions", cachedInventoryHealthPotions);
        frame.field("inventoryPearls", cachedInventoryPearls);
        frame.field("crosshairTargetType", getCrosshairTargetType());
        frame.field("crosshairTargetId", crosshairEntity == null ? null : crosshairEntity.getEntityId());
        frame.field("crosshairOnTarget", targetExists && crosshairEntity == target);
        frame.field("crosshairDistance", crosshairEntity == null ? null : (double) mc.thePlayer.getDistanceToEntity(crosshairEntity));
        frame.field("ticksSinceLastAttack", ticksSince(lastAttackTick, currentTick));
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
        frame.field("targetHp", targetExists ? target.getHealth() : null);
        frame.field("targetEstimatedHp", targetExists ? targetHpTracker.getEstimatedTargetHp() : null);
        frame.field("targetEstimatedHpDefault", targetHpTracker.getTargetHpDefault());
        frame.field("targetEstimatedLastDamage", targetExists ? targetHpTracker.getLastEstimatedTargetDamage() : null);
        frame.field("targetEstimatedLastHeal", targetExists ? targetHpTracker.getLastEstimatedTargetHeal() : null);
        frame.field("targetHurtTime", targetExists ? target.hurtTime : null);
        frame.field("targetHurtResistantTime", targetExists ? target.hurtResistantTime : null);
        frame.field("targetOnGround", targetExists ? target.onGround : null);
        frame.field("targetSprinting", targetExists ? target.isSprinting() : null);
        frame.field("targetSwingProgress", targetExists ? target.swingProgress : null);
        frame.field("targetUsingItem", targetExists ? target.isUsingItem() : null);
        frame.field("targetHeldItemType", targetExists ? itemsHandler.getHeldItemType(target) : null);
        frame.field("targetSpeedPotion", targetExists ? itemsHandler.hasPotion(target, Potion.moveSpeed) : null);
        frame.field("targetSpeedDurationTicks", targetExists ? itemsHandler.getPotionDuration(target, Potion.moveSpeed): null);
        frame.field("targetEstimatedSaturation", targetExists ? targetFoodTracker.getTargetSaturation() : null);
        frame.field("targetEstimatedExhaustion", targetExists ? targetFoodTracker.getTargetExhaustion() : null);
        frame.field("targetEstimatedFoodTimer", targetExists ? targetHpTracker.getNaturalRegenTimer() : null);
        frame.field("targetFood", targetExists ? targetFoodTracker.getTargetHunger() : null);
        frame.field("targetFoodLevel", targetExists ? targetFoodTracker.getTargetHunger() : null);
        frame.field("targetFoodRaw", targetExists ? target.getFoodStats().getFoodLevel() : null);
        frame.field("lineOfSight", targetExists && crosshairEntity == target);
        frame.field("yawDiffToTarget", yawDiffToTarget);
        frame.field("pitchDiffToTarget", pitchDiffToTarget);
        frame.field("hpDiff", hpDiff);
        writeFrame(frame);

    }



    private void updateCombos(EntityPlayer target) {
        if (target.hurtTime > lastTargetHurtTime) {
            myCombo++;
            targetCombo = 0;
            lastHitLandedTick = recordingTickIndex;
            targetHpTracker.applyEstimatedTargetDamage(recordingTickIndex);
            targetFoodTracker.addTakingDamageExhaustion();
        }
        lastTargetHurtTime = target.hurtTime;

        if (mc.thePlayer.hurtTime > lastMyHurtTime) {
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

    private void updateComboTargetState(EntityPlayer target) {
        int targetId = target == null ? -1 : target.getEntityId();
        if (targetId == comboTrackedTargetId) return;

        comboTrackedTargetId = targetId;
        lastTargetHurtTime = target == null ? 0 : target.hurtTime;
    }

    private EntityPlayer findClosestPlayer() {
        if (mc.theWorld == null) return null;
        EntityPlayer closestPlayer = null;
        double closestDistance = Double.MAX_VALUE;

        for (Object playerObject : mc.theWorld.playerEntities) {
            if (playerObject instanceof EntityPlayer) {
                EntityPlayer player = (EntityPlayer) playerObject;
                if (player != mc.thePlayer && !player.isInvisible() && player.getHealth() > 0 && !player.isDead) {
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

    private void updateInventoryCacheIfNeeded(int currentTick) {
        if (lastInventorySampleTick >= 0 && currentTick - lastInventorySampleTick < INVENTORY_SAMPLE_INTERVAL_TICKS) {
            return;
        }

        lastInventorySampleTick = currentTick;
        cachedHotbar = itemsHandler.getHotbarSnapshot();
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
            return false;
        }

        for (Object entityObject : mc.theWorld.loadedEntityList) {
            if (!(entityObject instanceof EntityEnderPearl)) continue;

            EntityEnderPearl pearl = (EntityEnderPearl) entityObject;
            int pearlId = pearl.getEntityId();
            if (seenEnemyPearlEntityIds.contains(pearlId)) continue;

            if (!isEnemyPearlFromTarget(pearl, target)) continue;

            seenEnemyPearlEntityIds.add(pearlId);
            lastEnemyPearlTick = currentTick;
            enemyPearlCooldownTicksRemaining = PEARL_COOLDOWN_TICKS;
            writeEvent("ENEMY_PEARL_THROW", detailsBuilder()
                    .field("targetEntityId", target.getEntityId())
                    .field("pearlEntityId", pearlId)
                    .field("pearlTicksExisted", pearl.ticksExisted)
                    .field("distanceToTarget", (double) target.getDistanceToEntity(pearl)));
            return true;
        }

        return false;
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

    private void handleOutgoingPacket(Packet packet, int tick) {
        if (!recording) return;

        if (packet instanceof C02PacketUseEntity) {
            handleUseEntityPacket((C02PacketUseEntity) packet, tick);
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
            lastAttackTick = tick;
            lastHitLandedTick = tick;
            writeEvent("ATTACK_PACKET_SENT", details);
        }
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
            lastPearlTick = tick;
            pearlCooldownTicksRemaining = PEARL_COOLDOWN_TICKS;
            writeEvent("PEARL_THROW", itemDetails(stack).field("source", "C08PacketPlayerBlockPlacement"));
        }
    }

    private void handleHeldItemChangePacket(C09PacketHeldItemChange packet, int tick) {
        writeEvent("SLOT_CHANGED", detailsBuilder()
                .field("slot", packet.getSlotId()));
    }

    private void handleIncomingPacket(Packet packet, int tick) {
        if (!recording) return;

        if (packet instanceof S12PacketEntityVelocity) {
            handleEntityVelocityPacket((S12PacketEntityVelocity) packet, tick);
        }
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
        recordWriter.writeEvent(type, details, recordingTickIndex, getWorldTick());
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
            mc.thePlayer.addChatMessage(new ChatComponentText("\u00A77[\u00A7bLipsy\u00A77] \u00A7f" + message));
        }
    }

    public void startRecording() {
        if (recording) return;

        startTimeMs = System.currentTimeMillis();
        endTimeMs = 0;
        startWorldTick = getWorldTick();
        serverIp = getServerIp();
        recordingTickIndex = 0;
        lastInventorySampleTick = -1;
        cachedHotbar = new ArrayList<String>();
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
        lastUseItemPacketTick = -1;
        lastVelocityEventTick = -1;
        lastVelocityEntityId = -1;
        lastVelocityMotionXRaw = 0;
        lastVelocityMotionYRaw = 0;
        lastVelocityMotionZRaw = 0;
        packetHandler.resetSession();
        myCombo = 0;
        targetCombo = 0;
        lastMyHurtTime = 0;
        lastTargetHurtTime = 0;
        comboTrackedTargetId = -1;
        targetHpTracker.reset();
        targetFoodTracker.reset();

        try {
            recordWriter.startSession(startTimeMs, startWorldTick, serverIp);
            sessionId = recordWriter.getSessionId();
            recording = true;
            writeEvent("RECORD_START", detailsBuilder()
                    .field("sessionId", sessionId)
                    .field("serverIp", serverIp)
                    .field("startTimeMs", startTimeMs)
                    .field("startWorldTick", startWorldTick));
            sendMessage("Recording started: " + sessionId);
        } catch (IOException ex) {
            ex.printStackTrace();
            sendMessage("Failed to start recording: " + ex.getMessage());
        }
    }

    public void stopRecording() {
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

}
