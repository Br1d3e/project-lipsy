package com.bot.recorder.targetTracker;

import com.bot.recorder.recordWriter.JsonLineBuilder;
import net.minecraft.entity.player.EntityPlayer;

public class TargetFoodTracker {
    public interface EventSink {
        void writeEvent(String type, JsonLineBuilder details);
    }

    private static final int DEFAULT_HUNGER = 20;
    private static final float DEFAULT_SATURATION = 5.0F;
    private static final float DEFAULT_EXHAUSTION = 0.0F;
    private static final float MAX_EXHAUSTION = 40.0F;
    private static final float NORMAL_JUMP_EXHAUSTION = 0.2F;
    private static final float SPRINT_JUMP_EXHAUSTION = 0.8F;
    private static final float GROUND_WALK_EXHAUSTION = 0.01F;
    private static final float GROUND_SPRINT_EXHAUSTION = 0.099999994F;
    private static final float SUCCESSFUL_MELEE_ATTACK_EXHAUSTION = 0.3F;
    private static final float TAKING_DAMAGE_EXHAUSTION = 0.3F;
    private static final float NATURAL_REGEN_EXHAUSTION = 3.0F;

    private final EventSink eventSink;
    private int trackedTargetId = -1;
    private int targetHunger = DEFAULT_HUNGER;
    private float targetSaturation = DEFAULT_SATURATION;
    private float targetExhaustion = DEFAULT_EXHAUSTION;
    private int foodTimer = 0;
    private double previousPosX;
    private double previousPosY;
    private double previousPosZ;
    private boolean previousOnGround;
    private boolean hasPreviousTargetPosition = false;

    public TargetFoodTracker(EventSink eventSink) {
        this.eventSink = eventSink;
    }

    public void reset() {
        trackedTargetId = -1;
        targetHunger = DEFAULT_HUNGER;
        targetSaturation = DEFAULT_SATURATION;
        targetExhaustion = DEFAULT_EXHAUSTION;
        foodTimer = 0;
        hasPreviousTargetPosition = false;
    }

    public void updateTargetState(EntityPlayer target, int currentTick) {
        int currentTargetId = target == null ? -1 : target.getEntityId();
        if (currentTargetId == trackedTargetId) return;

        trackedTargetId = currentTargetId;
        targetHunger = DEFAULT_HUNGER;
        targetSaturation = DEFAULT_SATURATION;
        targetExhaustion = DEFAULT_EXHAUSTION;
        foodTimer = 0;
        hasPreviousTargetPosition = false;

        if (target != null) {
            rememberTargetPosition(target);
        }

        writeEvent("TARGET_FOOD_RESET", detailsBuilder()
                .field("targetEntityId", currentTargetId < 0 ? null : currentTargetId)
                .field("targetHunger", currentTargetId < 0 ? null : targetHunger)
                .field("targetSaturation", currentTargetId < 0 ? null : targetSaturation)
                .field("targetExhaustion", currentTargetId < 0 ? null : targetExhaustion)
                .field("tick", currentTick));
    }

    public void updateMovementExhaustion(EntityPlayer target) {
        if (target == null) return;

        if (!hasPreviousTargetPosition) {
            rememberTargetPosition(target);
            return;
        }

        if (previousOnGround && !target.onGround && target.motionY > 0.0D) {
            addExhaustion(target.isSprinting() ? SPRINT_JUMP_EXHAUSTION : NORMAL_JUMP_EXHAUSTION);
        }

        double dx = target.posX - previousPosX;
        double dz = target.posZ - previousPosZ;
        double horizontalDistance = Math.sqrt(dx * dx + dz * dz);
        int distanceCentimeters = Math.round((float) (horizontalDistance * 100.0D));

        if (target.onGround && distanceCentimeters > 0) {
            float multiplier = target.isSprinting() ? GROUND_SPRINT_EXHAUSTION : GROUND_WALK_EXHAUSTION;
            addExhaustion(multiplier * (float) distanceCentimeters * 0.01F);
        }

        rememberTargetPosition(target);
    }

    public void addSuccessfulMeleeAttackExhaustion() {
        addExhaustion(SUCCESSFUL_MELEE_ATTACK_EXHAUSTION);
    }

    public void addTakingDamageExhaustion() {
        addExhaustion(TAKING_DAMAGE_EXHAUSTION);
    }

    public void addNaturalRegenExhaustion() {
        addExhaustion(NATURAL_REGEN_EXHAUSTION);
    }

    public void finishFoodTick() {
        if (targetExhaustion > 4.0F) {
            targetExhaustion -= 4.0F;

            if (targetSaturation > 0.0F) {
                targetSaturation = Math.max(targetSaturation - 1.0F, 0.0F);
            } else {
                targetHunger = Math.max(targetHunger - 1, 0);
            }
        }
    }

    public int getTargetHunger() {
        return targetHunger;
    }

    public float getTargetSaturation() {
        return targetSaturation;
    }

    public float getTargetExhaustion() {
        return targetExhaustion;
    }

    public int getFoodTimer() {
        return foodTimer;
    }

    public void setFoodTimer(int foodTimer) {
        this.foodTimer = foodTimer;
    }

    private void addExhaustion(float exhaustion) {
        targetExhaustion = Math.min(targetExhaustion + exhaustion, MAX_EXHAUSTION);
    }

    private void rememberTargetPosition(EntityPlayer target) {
        previousPosX = target.posX;
        previousPosY = target.posY;
        previousPosZ = target.posZ;
        previousOnGround = target.onGround;
        hasPreviousTargetPosition = true;
    }

    private JsonLineBuilder detailsBuilder() {
        return new JsonLineBuilder(512);
    }

    private void writeEvent(String type, JsonLineBuilder details) {
        if (eventSink != null) {
            eventSink.writeEvent(type, details);
        }
    }
}
