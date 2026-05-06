package com.bot.recorder.targetTracker;

import com.bot.recorder.recordWriter.JsonLineBuilder;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemFood;
import net.minecraft.item.ItemStack;

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
    private boolean targetWasUsingFood = false;
    private int targetFoodUseStartTick = -1;
    private ItemStack targetFoodUseStack = null;

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
        targetWasUsingFood = false;
        targetFoodUseStartTick = -1;
        targetFoodUseStack = null;
    }

    public void updateTargetState(EntityPlayer target, int currentTick) {
        int currentTargetId = target == null ? -1 : target.getEntityId();
        if (currentTargetId == trackedTargetId) return;

        trackedTargetId = currentTargetId;
        targetHunger = getInitialTargetHunger(target);
        String targetHungerSource = hasFetchableTargetHunger(target) ? "client_food_stats" : "default";
        targetSaturation = DEFAULT_SATURATION;
        targetExhaustion = DEFAULT_EXHAUSTION;
        foodTimer = 0;
        hasPreviousTargetPosition = false;
        targetWasUsingFood = false;
        targetFoodUseStartTick = -1;
        targetFoodUseStack = null;

        if (target != null) {
            rememberTargetPosition(target);
        }

        writeEvent("TARGET_FOOD_RESET", detailsBuilder()
                .field("targetEntityId", currentTargetId < 0 ? null : currentTargetId)
                .field("targetHunger", currentTargetId < 0 ? null : targetHunger)
                .field("targetHungerSource", currentTargetId < 0 ? null : targetHungerSource)
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

    public void updateFoodUse(EntityPlayer target, ItemStack heldItem, boolean heldItemIsFood, int currentTick) {
        if (target == null) {
            targetWasUsingFood = false;
            targetFoodUseStartTick = -1;
            targetFoodUseStack = null;
            return;
        }

        boolean usingFood = target.isUsingItem() && heldItemIsFood && heldItem != null && heldItem.getItem() instanceof ItemFood;
        if (usingFood) {
            if (!targetWasUsingFood) {
                targetFoodUseStartTick = currentTick;
                targetFoodUseStack = heldItem.copy();
            } else if (targetFoodUseStack == null) {
                targetFoodUseStack = heldItem.copy();
            }

            targetWasUsingFood = true;
            return;
        }

        if (targetWasUsingFood) {
            int useTicks = targetFoodUseStartTick < 0 ? 0 : currentTick - targetFoodUseStartTick;
            if (targetFoodUseStack != null && useTicks >= 24) {
                applyFoodEaten(target, targetFoodUseStack, useTicks, currentTick);
            }
        }

        targetWasUsingFood = false;
        targetFoodUseStartTick = -1;
        targetFoodUseStack = null;
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

    private int getInitialTargetHunger(EntityPlayer target) {
        if (!hasFetchableTargetHunger(target)) return DEFAULT_HUNGER;

        int foodLevel = target.getFoodStats().getFoodLevel();
        return foodLevel;
    }

    private boolean hasFetchableTargetHunger(EntityPlayer target) {
        if (target == null || target.getFoodStats() == null) return false;

        int foodLevel = target.getFoodStats().getFoodLevel();
        return foodLevel > 0 && foodLevel <= DEFAULT_HUNGER;
    }

    private void applyFoodEaten(EntityPlayer target, ItemStack foodStack, int useTicks, int currentTick) {
        if (!(foodStack.getItem() instanceof ItemFood)) return;

        ItemFood food = (ItemFood) foodStack.getItem();
        int hungerBefore = targetHunger;
        float saturationBefore = targetSaturation;
        int healAmount = food.getHealAmount(foodStack);
        float saturationModifier = food.getSaturationModifier(foodStack);

        targetHunger = Math.min(targetHunger + healAmount, DEFAULT_HUNGER);
        targetSaturation = Math.min(targetSaturation + (float) healAmount * saturationModifier * 2.0F, (float) targetHunger);

        writeEvent("TARGET_FOOD_EAT", detailsBuilder()
                .field("targetEntityId", target.getEntityId())
                .field("itemType", foodStack.getItem().getUnlocalizedName())
                .field("displayName", foodStack.getDisplayName())
                .field("useTicks", useTicks)
                .field("hungerAdded", healAmount)
                .field("saturationModifier", saturationModifier)
                .field("targetHungerBefore", hungerBefore)
                .field("targetHunger", targetHunger)
                .field("targetSaturationBefore", saturationBefore)
                .field("targetSaturation", targetSaturation)
                .field("tick", currentTick));
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
