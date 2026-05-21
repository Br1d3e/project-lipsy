package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.network.NetworkPlayerInfo;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemStack;
import net.minecraft.item.ItemSword;
import net.minecraft.util.MathHelper;

import java.util.Random;

public class Blockhit {
    public enum Mode {
        MACRO,
        PREDICTION
    }

    public enum State {
        IDLE,
        BLOCKING
    }

    private static final double START_MIN_DISTANCE = 1.0D;
    private static final double START_MAX_DISTANCE = 2.5D;
    private static final double CANCEL_MAX_DISTANCE = 3.0D;
    private static final float START_MAX_YAW_DIFF = 20.0F;
    private static final float CANCEL_MAX_YAW_DIFF = 50.0F;

    private final Minecraft mc;
    private final Random random = new Random();
    private Mode mode;
    private State state = State.IDLE;
    private int blockTicksLeft = 0;
    private int tickIndex = 0;
    private int lastAttackTick = -100000;
    private int lastDamageTick = -100000;
    private int observedSwordBlockTicks = 0;
    private float previousSwingProgress = 0.0F;

    public Blockhit(Minecraft mc) {
        this(mc, Mode.MACRO);
    }

    public Blockhit(Minecraft mc, Mode mode) {
        this.mc = mc;
        this.mode = mode;
    }

    public void update(EntityPlayer target, boolean utilityActive) {
        update(target, utilityActive, null);
    }

    public void update(EntityPlayer target, boolean utilityActive, float[] actions) {
        tickIndex++;

        if (mc.thePlayer == null) {
            stopBlocking("no player", true);
            return;
        }

        updateRecentSignals();
        updateObservedBlockTicks();

        if (target == null) {
            stopBlocking("target missing", true);
            return;
        }

        double distance = mc.thePlayer.getDistanceToEntity(target);
        float yawDiffToTarget = getYawDiffToTarget(target);
        boolean holdingSword = isHoldingSword();

        if (state == State.IDLE && isUseItemDown() && holdingSword) {
            stopBlocking("stale sword block", true);
        }

        if (mode == Mode.PREDICTION) {
            updatePredictionMode(actions, distance, yawDiffToTarget, holdingSword, utilityActive);
            return;
        }

        if (state == State.BLOCKING) {
            if (!holdingSword) {
                stopBlocking("not holding sword", true);
                return;
            }
            if (utilityActive || isUsingUtilityItem()) {
                stopBlocking("utility item", true);
                return;
            }
            if (distance > CANCEL_MAX_DISTANCE) {
                stopBlocking("distance", true);
                return;
            }
            if (Math.abs(yawDiffToTarget) > CANCEL_MAX_YAW_DIFF) {
                stopBlocking("aim lost", true);
                return;
            }

            holdUseItem(true);
            mc.thePlayer.sendChatMessage("[Project Lipsy] Blockhitting");
            blockTicksLeft--;
            if (blockTicksLeft <= 0) {
                releaseBlocking();
            }
            return;
        }

        if (shouldStartBlockhit(target, distance, yawDiffToTarget, holdingSword, utilityActive)) {
            String reason = chooseStartReason(target);
            startBlocking(chooseBlockTicks(target, distance), reason);
        }
    }

    public void setMode(Mode mode) {
        if (this.mode == mode) return;
        this.mode = mode;
        stopBlocking("mode switch", true);
    }

    public Mode getMode() {
        return mode;
    }

    public boolean isBlocking() {
        return state == State.BLOCKING;
    }

    public void cancel(String reason) {
        stopBlocking(reason, true);
    }

    public int getObservedSwordBlockTicks() {
        return observedSwordBlockTicks;
    }

    private void updatePredictionMode(float[] actions, double distance, float yawDiffToTarget,
                                      boolean holdingSword, boolean utilityActive) {
        if (state == State.BLOCKING && shouldCancelPredictionBlock(distance, yawDiffToTarget, holdingSword, utilityActive)) {
            return;
        }

        boolean predictedBlock = actions != null && actions.length >= 5 && actions[4] >= 0.5F;
        if (!predictedBlock) {
            if (state == State.BLOCKING) {
                releaseBlocking();
            }
            return;
        }

        if (!canPredictionBlock(distance, yawDiffToTarget, holdingSword, utilityActive)) {
            if (state == State.BLOCKING) {
                stopBlocking("prediction safety", true);
            }
            return;
        }

        if (state != State.BLOCKING) {
            startBlocking(1, "prediction");
        } else {
            holdUseItem(true);
        }
        blockTicksLeft = 1;
    }

    private boolean shouldCancelPredictionBlock(double distance, float yawDiffToTarget,
                                                boolean holdingSword, boolean utilityActive) {
        if (!holdingSword) {
            stopBlocking("not holding sword", true);
            return true;
        }
        if (utilityActive || isUsingUtilityItem()) {
            stopBlocking("utility item", true);
            return true;
        }
        if (distance > CANCEL_MAX_DISTANCE) {
            stopBlocking("distance", true);
            return true;
        }
        if (Math.abs(yawDiffToTarget) > CANCEL_MAX_YAW_DIFF) {
            stopBlocking("aim lost", true);
            return true;
        }
        return false;
    }

    private boolean canPredictionBlock(double distance, float yawDiffToTarget,
                                       boolean holdingSword, boolean utilityActive) {
        if (!holdingSword) return false;
        if (utilityActive || isUsingUtilityItem()) return false;
        if (distance > CANCEL_MAX_DISTANCE) return false;
        return Math.abs(yawDiffToTarget) <= CANCEL_MAX_YAW_DIFF;
    }

    private boolean shouldStartBlockhit(EntityPlayer target, double distance, float yawDiffToTarget,
                                       boolean holdingSword, boolean utilityActive) {
        if (!holdingSword) return false;
        if (utilityActive || isUsingUtilityItem()) return false;
        if (distance < START_MIN_DISTANCE || distance > START_MAX_DISTANCE) return false;
        if (Math.abs(yawDiffToTarget) > START_MAX_YAW_DIFF) return false;

        boolean botRecentlyAttacked = ticksSince(lastAttackTick) <= 6 || mc.thePlayer.swingProgress > 0.0F;
        boolean targetRecentlyHurt = target.hurtTime > 0 && target.hurtTime < 10;
        boolean targetSwinging = target.swingProgress > 0.0F;
        boolean recentlyTookDamage = ticksSince(lastDamageTick) <= 10 || mc.thePlayer.hurtTime > 0;
        boolean latencyConfirmed = isLatencyConfirmedHit(target);

        int score = 0;
        if (botRecentlyAttacked) score += 2;
        if (latencyConfirmed) score += 2;
        if (targetRecentlyHurt) score += 2;
        if (targetSwinging) score += 1;
        if (recentlyTookDamage) score += 1;

        return botRecentlyAttacked && score >= 5;
    }

    private boolean isLatencyConfirmedHit(EntityPlayer target) {
        int latencyTicks = getLatencyTicks();
        int sinceAttack = ticksSince(lastAttackTick);
        return (sinceAttack >= latencyTicks && sinceAttack <= latencyTicks + 8)
                || (target.hurtTime > 0 && target.hurtTime < 10);
    }

    private int chooseBlockTicks(EntityPlayer target, double distance) {
        boolean pressure = mc.thePlayer.hurtTime > 0 || target.swingProgress > 0.0F || distance <= 1.8D;
        int roll = random.nextInt(100);

        if (pressure) {
            if (roll < 40) return randomBetween(3, 5);
            if (roll < 80) return randomBetween(6, 10);
            return randomBetween(11, 16);
        }

        if (roll < 50) return randomBetween(2, 4);
        if (roll < 85) return randomBetween(5, 8);
        return randomBetween(9, 14);
    }

    private void startBlocking(int ticks, String reason) {
        state = State.BLOCKING;
        blockTicksLeft = ticks;
        holdUseItem(true);
        System.out.println("[Project Lipsy] BlockHit START ticks=" + ticks + " reason=" + reason);
    }

    private void releaseBlocking() {
        state = State.IDLE;
        blockTicksLeft = 0;
        observedSwordBlockTicks = 0;
        holdUseItem(false);
        System.out.println("[Project Lipsy] BlockHit RELEASE");
    }

    private void stopBlocking(String reason, boolean forceReleaseUseItem) {
        boolean wasBlocking = state == State.BLOCKING;
        boolean shouldReleaseStaleSwordBlock = forceReleaseUseItem
                && isUseItemDown()
                && (mc.thePlayer == null || isHoldingSword());
        if (!wasBlocking && !shouldReleaseStaleSwordBlock) return;

        state = State.IDLE;
        blockTicksLeft = 0;
        observedSwordBlockTicks = 0;
        if (forceReleaseUseItem) {
            holdUseItem(false);
        }
        System.out.println("[Project Lipsy] BlockHit CANCEL reason=" + reason);
    }

    private void holdUseItem(boolean pressed) {
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), pressed);
    }

    private void updateRecentSignals() {
        // TODO: replace this with exact C02 attack packet or Clicking-owned attack tick if exposed later.
        if (mc.thePlayer.swingProgress > 0.0F && previousSwingProgress <= 0.0F) {
            lastAttackTick = tickIndex;
        }
        previousSwingProgress = mc.thePlayer.swingProgress;

        if (mc.thePlayer.hurtTime == 10 || mc.thePlayer.hurtTime > 0 && ticksSince(lastDamageTick) > 10) {
            lastDamageTick = tickIndex;
        }
    }

    private void updateObservedBlockTicks() {
        if (isUseItemDown() && isHoldingSword()) {
            observedSwordBlockTicks++;
        } else {
            observedSwordBlockTicks = 0;
        }
    }

    private boolean isHoldingSword() {
        ItemStack stack = mc.thePlayer.getHeldItem();
        return stack != null && stack.getItem() instanceof ItemSword;
    }

    private boolean isUsingUtilityItem() {
        if (!mc.thePlayer.isUsingItem()) return false;
        return !isHoldingSword();
    }

    private boolean isUseItemDown() {
        return mc.gameSettings.keyBindUseItem.isKeyDown();
    }

    private String chooseStartReason(EntityPlayer target) {
        if (mc.thePlayer.hurtTime > 0 || ticksSince(lastDamageTick) <= 10) return "damage";
        if (target.swingProgress > 0.0F) return "target_swing";
        if (target.hurtTime > 0 && target.hurtTime < 10) return "target_hurt";
        return "attack_window";
    }

    private float getYawDiffToTarget(EntityPlayer target) {
        double dx = target.posX - mc.thePlayer.posX;
        double dz = target.posZ - mc.thePlayer.posZ;
        float targetYaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0F);
        return MathHelper.wrapAngleTo180_float(targetYaw - mc.thePlayer.rotationYaw);
    }

    private int getLatencyTicks() {
        int serverMs = 0;
        try {
            NetworkPlayerInfo info = mc.getNetHandler().getPlayerInfo(mc.thePlayer.getUniqueID());
            if (info != null) {
                serverMs = Math.max(0, info.getResponseTime());
            }
        } catch (Exception ignored) {
            // TODO: expose server ping from the connection layer if this is noisy on some servers.
        }
        return (int) Math.ceil(serverMs / 50.0D) + 1;
    }

    private int ticksSince(int tick) {
        return tickIndex - tick;
    }

    private int randomBetween(int minInclusive, int maxInclusive) {
        return minInclusive + random.nextInt(maxInclusive - minInclusive + 1);
    }
}
