package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemStack;
import net.minecraft.potion.Potion;
import net.minecraft.util.MathHelper;
import net.minecraft.util.MovingObjectPosition;

import java.io.IOException;
import java.util.Locale;

public class ModelConnect {
    private static final String MODEL_HOST = "127.0.0.1";
    private static final int MODEL_PORT = 33333;
    public static final int WTAP_ACTION_INDEX = 8;
    private static final long RETRY_DELAY_MS = 2000L;
    private static final int FAR_TICKS = 100000;
    private static final int PEARL_COOLDOWN_TICKS = 320;

    private static final int myMotionX = 0;
    private static final int myMotionY = 1;
    private static final int myMotionZ = 2;
    private static final int myYaw = 3;
    private static final int myPitch = 4;
    private static final int myDeltaYaw = 5;
    private static final int myDeltaPitch = 6;
    private static final int myHp = 7;
    private static final int myHurtTime = 8;
    private static final int myMoveForward = 9;
    private static final int myMoveStrafing = 10;
    private static final int mySwingProgress = 11;
    private static final int myUsingItem = 12;
    private static final int myHoldingSword = 13;
    private static final int myBlockingSword = 14;
    private static final int myHasSpeedPotion = 15;
    private static final int mySpeedDurationTicks = 16;
    private static final int myFoodLevel = 17;
    private static final int myCombo = 18;
    private static final int targetCombo = 19;
    private static final int pearlCooldownTicksRemaining = 20;
    private static final int enemyPearlAvailable = 21;
    private static final int enemyPearlCooldownTicksRemaining = 22;
    private static final int hotbarHealthPotions = 23;
    private static final int inventoryFood = 24;
    private static final int inventorySpeedPotions = 25;
    private static final int inventoryHealthPotions = 26;
    private static final int inventoryPearls = 27;
    private static final int crosshairOnTarget = 28;
    private static final int ticksSinceLastAttack = 29;
    private static final int ticksSinceLastBlock = 30;
    private static final int ticksSinceLastPot = 31;
    private static final int ticksSinceLastEat = 32;
    private static final int ticksSinceLastSpeed = 33;
    private static final int ticksSinceLastDamageTaken = 34;
    private static final int ticksSinceLastHitLanded = 35;
    private static final int ticksSinceLastPearl = 36;
    private static final int ticksSinceLastEnemyPearl = 37;
    private static final int blockDistanceFront = 38;
    private static final int blockDistanceBack = 39;
    private static final int blockDistanceLeft = 40;
    private static final int blockDistanceRight = 41;
    private static final int aStrafe = 42;
    private static final int aAttack = 43;
    private static final int aUseItem = 44;
    private static final int targetExists = 45;
    private static final int targetRelX = 46;
    private static final int targetRelY = 47;
    private static final int targetRelZ = 48;
    private static final int targetDistance = 49;
    private static final int targetMotionX = 50;
    private static final int targetMotionY = 51;
    private static final int targetMotionZ = 52;
    private static final int targetYaw = 53;
    private static final int targetPitch = 54;
    private static final int targetDeltaYaw = 55;
    private static final int targetDeltaPitch = 56;
    private static final int targetEstimatedHp = 57;
    private static final int targetEstimatedPotions = 58;
    private static final int targetEstimatedPearls = 59;
    private static final int targetHurtTime = 60;
    private static final int targetSwingProgress = 61;
    private static final int targetUsingItem = 62;
    private static final int targetHasSpeedPotion = 63;
    private static final int targetSpeedDurationTicks = 64;
    private static final int targetFood = 65;
    private static final int targetEating = 66;
    private static final int targetSpeeding = 67;
    private static final int targetPearling = 68;
    private static final int targetPotting = 69;
    private static final int yawDiffToTarget = 70;
    private static final int pitchDiffToTarget = 71;
    private static final int hpDiff = 72;
    private static final int canHeal = 73;
    private static final int canPearl = 74;
    private static final int canSpeed = 75;
    private static final int canEat = 76;
    private static final int s_attack = 77;
    private static final int s_heal = 78;
    private static final int s_pearl = 79;
    private static final int s_speed = 80;
    private static final int s_eat = 81;
    private static final int s_refill = 82;
    private static final int BATCH_SIZE = 83;

    private final Minecraft mc;
    private final utils utils;
    private final float[] batch = new float[BATCH_SIZE];
    private float[] actions = null;
    private ModelClient modelClient;
    private long lastConnectAttemptMs = 0L;
    private boolean printedOfflineMessage = false;
    private Float previousMyYaw = null;
    private Float previousMyPitch = null;
    private Float previousTargetYaw = null;
    private Float previousTargetPitch = null;
    private int tickIndex = 0;
    private int lastAttackTick = -FAR_TICKS;
    private int lastBlockTick = -FAR_TICKS;
    private int lastPotTick = -FAR_TICKS;
    private int lastEatTick = -FAR_TICKS;
    private int lastSpeedTick = -FAR_TICKS;
    private int lastDamageTakenTick = -FAR_TICKS;
    private int lastHitLandedTick = -FAR_TICKS;
    private int lastPearlTick = -FAR_TICKS;
    private int lastEnemyPearlTick = -FAR_TICKS;
    private float previousSwingProgress = 0.0F;
    private int previousTargetHurtTime = 0;
    private String decisionState = "attack";

    public ModelConnect(Minecraft mc) {
        this.mc = mc;
        this.utils = new utils(mc);
    }

    public void setDecisionState(String state) {
        if (state == null) return;
        this.decisionState = state.toLowerCase(Locale.US);
    }

    public void sendObs(EntityPlayer target) {
        if (mc.thePlayer == null || target == null) {
            actions = null;
            return;
        }
        if (!ensureConnected()) {
            actions = null;
            return;
        }
        tickIndex++;

        double relX = target.posX - mc.thePlayer.posX;
        double relY = target.posY + target.getEyeHeight() - (mc.thePlayer.posY + mc.thePlayer.getEyeHeight());
        double relZ = target.posZ - mc.thePlayer.posZ;
        double distance = mc.thePlayer.getDistanceToEntity(target);
        double distanceXZ = Math.hypot(relX, relZ);
        double aimYaw = Math.toDegrees(Math.atan2(relZ, relX)) - 90F;
        double aimPitch = -Math.toDegrees(Math.atan2(relY, distanceXZ));
        double yawDiff = MathHelper.wrapAngleTo180_float((float) aimYaw - mc.thePlayer.rotationYaw);
        double pitchDiff = MathHelper.wrapAngleTo180_float((float) aimPitch - mc.thePlayer.rotationPitch);

        float deltaYaw = previousMyYaw == null ? 0.0F : MathHelper.wrapAngleTo180_float(mc.thePlayer.rotationYaw - previousMyYaw);
        float deltaPitch = previousMyPitch == null ? 0.0F : mc.thePlayer.rotationPitch - previousMyPitch;
        previousMyYaw = mc.thePlayer.rotationYaw;
        previousMyPitch = mc.thePlayer.rotationPitch;

        float targetYawDelta = previousTargetYaw == null ? 0.0F : MathHelper.wrapAngleTo180_float(target.rotationYaw - previousTargetYaw);
        float targetPitchDelta = previousTargetPitch == null ? 0.0F : target.rotationPitch - previousTargetPitch;
        previousTargetYaw = target.rotationYaw;
        previousTargetPitch = target.rotationPitch;

        boolean attackDown = mc.gameSettings.keyBindAttack.isKeyDown();
        boolean swingStarted = mc.thePlayer.swingProgress > 0.0F && previousSwingProgress <= 0.0F;
        boolean attackAction = attackDown || swingStarted;
        if (attackAction) {
            lastAttackTick = tickIndex;
        }
        previousSwingProgress = mc.thePlayer.swingProgress;

        if (mc.thePlayer.hurtTime == 10 || mc.thePlayer.hurtTime > 0 && ticksSince(lastDamageTakenTick) > 10) {
            lastDamageTakenTick = tickIndex;
        }
        if (target.hurtTime > previousTargetHurtTime) {
            lastHitLandedTick = tickIndex;
        }
        previousTargetHurtTime = target.hurtTime;

        boolean useItemDown = mc.gameSettings.keyBindUseItem.isKeyDown();
        boolean holdingSword = utils.isHoldingSword();
        boolean useItemAction = useItemDown || mc.thePlayer.isUsingItem();
        ItemStack held = mc.thePlayer.getHeldItem();
        utils.ItemSnapshot inventory = snapshotInventory();
        updateUseTimers(held, useItemDown, holdingSword);
        updateEnemyUseTimers(target);

        int strafeAction = mc.gameSettings.keyBindRight.isKeyDown() ? 1 : (mc.gameSettings.keyBindLeft.isKeyDown() ? -1 : 0);
        boolean crosshairIsTarget = mc.objectMouseOver != null
                && mc.objectMouseOver.typeOfHit == MovingObjectPosition.MovingObjectType.ENTITY
                && mc.objectMouseOver.entityHit == target;

        int speedDuration = utils.potionDuration(mc.thePlayer, Potion.moveSpeed);
        int targetSpeedDuration = utils.potionDuration(target, Potion.moveSpeed);
        float targetHp = target.getHealth();
        int pearlCooldown = Math.max(0, PEARL_COOLDOWN_TICKS - ticksSince(lastPearlTick));
        int enemyPearlCooldown = Math.max(0, PEARL_COOLDOWN_TICKS - ticksSince(lastEnemyPearlTick));

        put(myMotionX, mc.thePlayer.motionX);
        put(myMotionY, mc.thePlayer.motionY);
        put(myMotionZ, mc.thePlayer.motionZ);
        put(myYaw, mc.thePlayer.rotationYaw);
        put(myPitch, mc.thePlayer.rotationPitch);
        put(myDeltaYaw, deltaYaw);
        put(myDeltaPitch, deltaPitch);
        put(myHp, mc.thePlayer.getHealth());
        put(myHurtTime, mc.thePlayer.hurtTime);
        put(myMoveForward, mc.thePlayer.moveForward);
        put(myMoveStrafing, mc.thePlayer.moveStrafing);
        put(mySwingProgress, mc.thePlayer.swingProgress);
        put(myUsingItem, useItemAction);
        put(myHoldingSword, holdingSword);
        put(myBlockingSword, holdingSword && useItemDown);
        put(myHasSpeedPotion, speedDuration > 0);
        put(mySpeedDurationTicks, speedDuration);
        put(myFoodLevel, mc.thePlayer.getFoodStats().getFoodLevel());
        put(myCombo, utils.getMyCombo());
        put(targetCombo, utils.getTargetCombo());
        put(pearlCooldownTicksRemaining, pearlCooldown);
        put(enemyPearlAvailable, utils.isPearl(target.getHeldItem()) && enemyPearlCooldown == 0);
        put(enemyPearlCooldownTicksRemaining, enemyPearlCooldown);
        put(hotbarHealthPotions, inventory.hotbarHealthPotions);
        put(inventoryFood, inventory.food);
        put(inventorySpeedPotions, inventory.speedPotions);
        put(inventoryHealthPotions, inventory.healthPotions);
        put(inventoryPearls, inventory.pearls);
        put(crosshairOnTarget, crosshairIsTarget);
        put(ticksSinceLastAttack, ticksSince(lastAttackTick));
        put(ticksSinceLastBlock, ticksSince(lastBlockTick));
        put(ticksSinceLastPot, ticksSince(lastPotTick));
        put(ticksSinceLastEat, ticksSince(lastEatTick));
        put(ticksSinceLastSpeed, ticksSince(lastSpeedTick));
        put(ticksSinceLastDamageTaken, ticksSince(lastDamageTakenTick));
        put(ticksSinceLastHitLanded, ticksSince(lastHitLandedTick));
        put(ticksSinceLastPearl, ticksSince(lastPearlTick));
        put(ticksSinceLastEnemyPearl, ticksSince(lastEnemyPearlTick));
        put(blockDistanceFront, utils.blockDistanceAtYaw(mc.thePlayer.rotationYaw));
        put(blockDistanceBack, utils.blockDistanceAtYaw(mc.thePlayer.rotationYaw + 180.0F));
        put(blockDistanceLeft, utils.blockDistanceAtYaw(mc.thePlayer.rotationYaw - 90.0F));
        put(blockDistanceRight, utils.blockDistanceAtYaw(mc.thePlayer.rotationYaw + 90.0F));
        put(aStrafe, strafeAction);
        put(aAttack, attackAction);
        put(aUseItem, useItemAction);
        put(targetExists, true);
        put(targetRelX, relX);
        put(targetRelY, relY);
        put(targetRelZ, relZ);
        put(targetDistance, distance);
        put(targetMotionX, target.motionX);
        put(targetMotionY, target.motionY);
        put(targetMotionZ, target.motionZ);
        put(targetYaw, target.rotationYaw);
        put(targetPitch, target.rotationPitch);
        put(targetDeltaYaw, targetYawDelta);
        put(targetDeltaPitch, targetPitchDelta);
        put(targetEstimatedHp, targetHp);
        put(targetEstimatedPotions, 29);
        put(targetEstimatedPearls, 16);
        put(targetHurtTime, target.hurtTime);
        put(targetSwingProgress, target.swingProgress);
        put(targetUsingItem, target.isUsingItem());
        put(targetHasSpeedPotion, targetSpeedDuration > 0);
        put(targetSpeedDurationTicks, targetSpeedDuration);
        put(targetFood, target.getFoodStats().getFoodLevel());
        put(targetEating, target.isUsingItem() && utils.isFood(target.getHeldItem()));
        put(targetSpeeding, target.isUsingItem() && utils.isSpeedPotion(target.getHeldItem()));
        put(targetPearling, utils.isPearl(target.getHeldItem()));
        put(targetPotting, target.isUsingItem() && utils.isHealthPotion(target.getHeldItem()));
        put(yawDiffToTarget, yawDiff);
        put(pitchDiffToTarget, pitchDiff);
        put(hpDiff, targetHp - mc.thePlayer.getHealth());
        put(canHeal, inventory.healthPotions > 0 && ticksSince(lastPotTick) > 12);
        put(canPearl, inventory.pearls > 0 && pearlCooldown == 0);
        put(canSpeed, inventory.speedPotions > 0 && speedDuration < 20 * 20);
        put(canEat, inventory.food > 0 && mc.thePlayer.getFoodStats().getFoodLevel() < 20);
        putStateFeatures();

        try {
            modelClient.sendPacket(batch);
            actions = modelClient.readPacket();
        } catch (IOException e) {
            actions = null;
            closeClient();
            System.out.println("[Project Lipsy] Model connection lost: " + e.getMessage());
        }
    }

    public float[] getActions() {
        return actions;
    }

    private void putStateFeatures() {
        put(s_attack, "attack".equals(decisionState));
        put(s_heal, "heal".equals(decisionState));
        put(s_pearl, "pearl".equals(decisionState));
        put(s_speed, "speed".equals(decisionState));
        put(s_eat, "eat".equals(decisionState));
        put(s_refill, "refill".equals(decisionState));
    }

    private void updateUseTimers(ItemStack held, boolean useItemDown, boolean holdingSword) {
        if (useItemDown && holdingSword) {
            lastBlockTick = tickIndex;
            return;
        }
        if (!useItemDown) return;

        if (utils.isPearl(held)) {
            lastPearlTick = tickIndex;
        } else if (utils.isHealthPotion(held)) {
            lastPotTick = tickIndex;
        } else if (utils.isSpeedPotion(held)) {
            lastSpeedTick = tickIndex;
        } else if (utils.isFood(held)) {
            lastEatTick = tickIndex;
        }
    }

    private void updateEnemyUseTimers(EntityPlayer target) {
        if (utils.isPearl(target.getHeldItem())) {
            lastEnemyPearlTick = tickIndex;
        }
    }

    private boolean ensureConnected() {
        if (modelClient != null) return true;

        long now = System.currentTimeMillis();
        if (now - lastConnectAttemptMs < RETRY_DELAY_MS) return false;

        lastConnectAttemptMs = now;
        try {
            modelClient = new ModelClient(MODEL_HOST, MODEL_PORT);
            printedOfflineMessage = false;
            return true;
        } catch (IOException e) {
            if (!printedOfflineMessage) {
                System.out.println("[Project Lipsy] Model server is offline; gameplay will continue without model actions.");
                printedOfflineMessage = true;
            }
            return false;
        }
    }

    private void closeClient() {
        if (modelClient == null) return;

        try {
            modelClient.stopConn();
        } catch (IOException ignored) {
        }
        modelClient = null;
    }

    private utils.ItemSnapshot snapshotInventory() {
        utils.ItemSnapshot snapshot = new utils.ItemSnapshot();
        for (int i = 0; i < 36; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            int count = utils.itemCount(stack);
            if (count <= 0) continue;

            if (utils.isHealthPotion(stack)) {
                snapshot.healthPotions += count;
                if (i < 9) {
                    snapshot.hotbarHealthPotions += count;
                }
            } else if (utils.isSpeedPotion(stack)) {
                snapshot.speedPotions += count;
            } else if (utils.isPearl(stack)) {
                snapshot.pearls += count;
            } else if (utils.isFood(stack)) {
                snapshot.food += count;
            }
        }
        return snapshot;
    }

    private void put(int index, double value) {
        batch[index] = (float) value;
    }

    private void put(int index, boolean value) {
        batch[index] = value ? 1.0F : 0.0F;
    }

    private int ticksSince(int tick) {
        return tickIndex - tick;
    }
}
