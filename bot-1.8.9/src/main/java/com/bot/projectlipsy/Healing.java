package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.projectile.EntityPotion;
import net.minecraft.potion.Potion;
import net.minecraft.item.ItemStack;
import net.minecraft.util.MathHelper;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class Healing {

    private final Minecraft mc;
    private final utils utils;
    private final Movement movement;
    private final Retreat retreat;
    private long lastPotTime = 0;
    private static final int MAX_UTILITY_USE_TICKS = 50;
    private static final int MAX_HEAL_SEQUENCE_TICKS = 40;

    public enum HealingState {
        NONE,
        TURNING,
        POTTING
    }

    public enum HealingMode {
        TURN,
        FRONT,
        FAST
    }

    private HealingState potState = HealingState.NONE;
    private HealingMode potMode = HealingMode.TURN;
    private int potTicks = 0;
    private int speedTicks = 0;
    private int eatTicks = 0;
    private boolean waitingForCurrentPotExplosion = false;
    private boolean currentPotExploded = false;
    private final Set<Integer> activeBotPotionIds = new HashSet<Integer>();
    private final Set<Integer> seenBotPotionIds = new HashSet<Integer>();
    private final Set<Integer> currentPotEntityIds = new HashSet<Integer>();

    private double frontScore = 0D;
    private double turnScore = 0D;
    private double fastScore = 0D;
    private double frontProb = 0D;
    private double turnProb = 0D;
    private double fastProb = 0D;

    public Healing(Minecraft mc, Movement movement, Retreat retreat) {
        this.mc = mc;
        this.utils = new utils(mc);
        this.movement = movement;
        this.retreat = retreat;
    }

    public HealingState getState() {
        return potState;
    }

    public boolean isPotting() {
        return potState != HealingState.NONE;
    }

    public void startHealingSequence() {
        if (canPot() && !isPotting()) {
            stopUtilityUse();
            potState = HealingState.TURNING;
            potTicks = 0;
            resetScores();
            waitingForCurrentPotExplosion = false;
            currentPotExploded = false;
            currentPotEntityIds.clear();
        }
    }

    public void advanceState() {
        if (potState == HealingState.TURNING) {
            potState = HealingState.POTTING;
            potTicks = 0;
        }
    }

    private void resetScores() {
        frontScore = 0D;
        turnScore = 0D;
        fastScore = 0D;
    }

    private void updateHealScores(EntityPlayer target) {
        if (target == null) return;

        double targetDistance = mc.thePlayer.getDistanceToEntity(target);
        double targetHp = target.getHealth();
        double hpDiff = mc.thePlayer.getHealth() - targetHp;
        int myCombo = utils.getMyCombo();
        int targetCombo = utils.getTargetCombo();

        double dy = mc.thePlayer.posY - target.posY;
        double dx = mc.thePlayer.posX - target.posX;
        double dz = mc.thePlayer.posZ - target.posZ;
        float targetSupposedYaw = MathHelper.wrapAngleTo180_float((float) (Math.toDegrees(Math.atan2(dz, dx)) - 90F));
        boolean targetTurning = Math.abs(MathHelper.wrapAngleTo180_float(
                targetSupposedYaw - target.rotationYaw)) > 120;
        boolean myOnGround = mc.thePlayer.onGround;

        if (targetDistance <= 1.5D) {
            frontScore = Math.max(frontScore - 0.3D, 0);
            turnScore += 0.1D;
            fastScore += 0.15D;
        } else if (targetDistance <= 4D) {
            frontScore = Math.max(frontScore - 0.2D, 0);
            turnScore += 0.15D;
            fastScore += 0.1D;
        } else if (targetDistance > 6D) {
            fastScore -= 0.3D;
            turnScore -= 0.05D;
            frontScore += 0.4D;
        }
        if (myCombo >= 3 || targetTurning) {
            frontScore += myCombo * 0.05 + 0.15D;
            turnScore = Math.max(turnScore - myCombo * 0.3, turnScore);
            fastScore -= 0.3D;
        }
        if (!myOnGround || dy > 0.3D) {
            fastScore += 0.15D;
            turnScore += 0.05D;
            frontScore -= 0.2D;
        }
        if (targetCombo >= 2) {
            turnScore += targetCombo * 0.05 + 0.15D;
            fastScore += 0.05D;
            frontScore = Math.max(frontScore - targetCombo * 0.3, frontScore);
        }
        if (targetHp < 8D || hpDiff >= 5D) {
            frontScore += 0.1D;
            fastScore += 0.2D;
            turnScore -= 0.1D;
        }
    }

    private void computeProbabilities() {
        double expSum = Math.exp(frontScore) + Math.exp(turnScore) + Math.exp(fastScore);
        frontProb = Math.exp(frontScore) / expSum;
        turnProb = Math.exp(turnScore) / expSum;
        fastProb = Math.exp(fastScore) / expSum;
    }

    public void updatePotMacro(EntityPlayer target) {
        if (potState == HealingState.NONE) return;
        if (target == null || mc.thePlayer == null || mc.theWorld == null) {
            stopPotting();
            return;
        }

        updateBotPotionHistory();
        potTicks++;
        if (potTicks >= MAX_HEAL_SEQUENCE_TICKS
                || (potState == HealingState.TURNING && !hasHotbarHealingPotion())) {
            // A throw can be rejected, or its entity can spawn and disappear between client ticks.
            lastPotTime = System.currentTimeMillis();
            stopPotting();
            return;
        }

        if (potTicks == 1) {
            utils.updateCombos(target);
            updateHealScores(target);
            computeProbabilities();

            mc.thePlayer.sendChatMessage(
                    "turnProb: " + String.format("%.2f", turnProb) +
                            "frontProb: " + String.format("%.2f", frontProb) +
                            "fastProb: " + String.format("%.2f", fastProb)
            );

            if (fastProb >= 0.40D) {
                potMode = HealingMode.FAST;
                resetScores();
            } else if (frontProb >= 0.38D) {
                potMode = HealingMode.FRONT;
                resetScores();
            } else {
                potMode = HealingMode.TURN;
                resetScores();
            }
        }

        if (potMode == HealingMode.TURN) {
            turnPot(target);
        } else if (potMode == HealingMode.FRONT) {
            frontPot(target);
        } else {
            fastPot(target);
        }

        if (currentPotExploded) {
            stopPotting();
        }
    }

    private void frontPot(EntityPlayer target) {
        if (potState == HealingState.NONE || potMode != HealingMode.FRONT || target == null || mc.thePlayer == null) {
            return;
        }

        double yawOffset = utils.getRandomOffset(2F);
        float lookYaw = utils.getTargetYaw(target) + (float) yawOffset;
        movement.sprint();
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), false);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), false);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), false);

        if (potTicks == 1) {
            mc.thePlayer.rotationYaw = lookYaw;
            mc.thePlayer.rotationPitch = 8F;
            mc.thePlayer.sendChatMessage("Front Pot");
        } else if (potTicks == 2) {
            mc.thePlayer.rotationPitch = 20F;
        } else if (potTicks == 5) {
            potState = HealingState.POTTING;
            mc.thePlayer.rotationPitch = 25F;
            utils.switchHotbarItem("Splash Potion of Healing");
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), true);
            KeyBinding.onTick(mc.gameSettings.keyBindUseItem.getKeyCode());
            waitingForCurrentPotExplosion = true;
            currentPotExploded = false;
            currentPotEntityIds.clear();
        } else if (potTicks == 6) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
            lastPotTime = System.currentTimeMillis();
        }
    }

    private void turnPot(EntityPlayer target) {
        if (potState == HealingState.NONE || potMode != HealingMode.TURN || target == null || mc.thePlayer == null) {
            return;
        }

        movement.sprint();
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), false);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), false);

        float retreatYaw = retreat.getRetreatYaw(target);
        boolean isLeft = Math.random() < 0.5;
        if (potTicks == 1) {
            mc.thePlayer.rotationYaw = isLeft ? retreatYaw + 150F : retreatYaw - 150F;
            potState = HealingState.TURNING;
            mc.thePlayer.sendChatMessage("Turn Pot " + (isLeft ? "left" : "right"));
        } else if (potTicks == 2) {
            mc.thePlayer.rotationYaw = isLeft ? retreatYaw + 80F : retreatYaw - 80F;
            mc.thePlayer.rotationPitch = 10F;
        } else if (potTicks == 3) {
            mc.thePlayer.rotationYaw = isLeft ? retreatYaw + 20F : retreatYaw - 20F;
            mc.thePlayer.rotationPitch = 20F;
        } else if (potTicks == 4) {
            mc.thePlayer.rotationYaw = retreatYaw;
            mc.thePlayer.rotationPitch = mc.thePlayer.onGround ? 25F : 30F;
        } else if (potTicks == 6) {
            utils.switchHotbarItem("Splash Potion of Healing");
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), true);
            KeyBinding.onTick(mc.gameSettings.keyBindUseItem.getKeyCode());
            potState = HealingState.POTTING;
            waitingForCurrentPotExplosion = true;
            currentPotExploded = false;
            currentPotEntityIds.clear();
        } else if (potTicks == 7) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
            lastPotTime = System.currentTimeMillis();
        }
    }

    private void fastPot(EntityPlayer target) {
        if (potState == HealingState.NONE || potMode != HealingMode.FAST || target == null || mc.thePlayer == null) {
            return;
        }

        double yawOffset = utils.getRandomOffset(2F);
        float targetYaw = utils.getTargetYaw(target) + (float) yawOffset;
        float retreatYaw = retreat.getRetreatYaw(target);

        if (potTicks == 1) {
            mc.thePlayer.rotationYaw = targetYaw;
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), true);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindJump.getKeyCode(), true);
            mc.thePlayer.sendChatMessage("Fast Pot");
        } else if (potTicks == 2) {
            potState = HealingState.TURNING;
            mc.thePlayer.rotationYaw = retreatYaw + 90F;
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), false);
            mc.thePlayer.rotationPitch = 20F;
        } else if (potTicks == 3) {
            mc.thePlayer.rotationYaw = retreatYaw;
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindJump.getKeyCode(), false);
            movement.sprint();
            mc.thePlayer.rotationPitch = 45F;
        } else if (potTicks == 6) {
            utils.switchHotbarItem("Splash Potion of Healing");
            movement.sprint();
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), true);
            KeyBinding.onTick(mc.gameSettings.keyBindUseItem.getKeyCode());
            potState = HealingState.POTTING;
            waitingForCurrentPotExplosion = true;
            currentPotExploded = false;
            currentPotEntityIds.clear();
        } else if (potTicks == 7) {
            movement.sprint();
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
            lastPotTime = System.currentTimeMillis();
        } else if (potTicks == 8) {
//            movement.sprint();
            mc.thePlayer.rotationYaw = targetYaw;
        }
    }

    public void pot(EntityPlayer target) {
        updatePotMacro(target);
    }

    public void stopPotting() {
        potState = HealingState.NONE;
        potTicks = 0;
        waitingForCurrentPotExplosion = false;
        currentPotExploded = false;
        currentPotEntityIds.clear();
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), false);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindJump.getKeyCode(), false);
    }

    private boolean hasHotbarHealingPotion() {
        if (mc.thePlayer == null) return false;
        for (int slot = 0; slot < 9; slot++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(slot);
            if (stack != null && stack.getDisplayName().contains("Splash Potion of Healing")) return true;
        }
        return false;
    }

    public boolean canPot() {
        long currentTime = System.currentTimeMillis();
        long potInterval = 600;
        return hasHotbarHealingPotion() && currentTime - lastPotTime >= potInterval;
    }

    public boolean canEat() {
        return mc.thePlayer.getFoodStats().getFoodLevel() < 20 &&
                (utils.hasItem("Golden Carrot") || utils.hasItem("Steak"));
    }

    public boolean canSpeed() {
        return (
                !mc.thePlayer.isPotionActive(Potion.moveSpeed) || mc.thePlayer.getActivePotionEffect
                (Potion.moveSpeed).getDuration() < 20 * 20) &&
                utils.hasItem("Potion of Swiftness"
        );     // 20s
    }

    public boolean speedActive() {
        return (
                mc.thePlayer.isPotionActive(Potion.moveSpeed) &&
                        mc.thePlayer.getActivePotionEffect(Potion.moveSpeed).getDuration() >= 80 * 20
        );
    }

    public boolean foodActive() {
        return mc.thePlayer.getFoodStats().getFoodLevel() == 20;
    }

    public void speed() {
        eatTicks = 0;

        if (!canSpeed()) {
            stopUtilityUse();
            return;
        }

        utils.switchItem("Potion of Swiftness");
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), true);
        speedTicks++;

        if (speedActive() || !utils.isSpeedPotion(mc.thePlayer.getHeldItem())) {
            stopUtilityUse();
        }
    }

    public void eat() {
        speedTicks = 0;

        if (!canEat()) {
            stopUtilityUse();
            return;
        }

        if (utils.hasItem("Golden Carrot")) {
            utils.switchItem("Golden Carrot");
        } else if (utils.hasItem("Steak")) {
            utils.switchItem("Steak");
        }

        KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), true);
        eatTicks++;

        if (foodActive() || !utils.isFood(mc.thePlayer.getHeldItem())) {
            stopUtilityUse();
        }
    }

    public void stopUtilityUse() {
        speedTicks = 0;
        eatTicks = 0;
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
    }

    private void updateBotPotionHistory() {
        if (mc.theWorld == null || mc.thePlayer == null) {
            activeBotPotionIds.clear();
            return;
        }

        Set<Integer> visibleBotPotionIds = new HashSet<Integer>();

        for (Object entityObject : mc.theWorld.loadedEntityList) {
            if (!(entityObject instanceof EntityPotion)) continue;

            EntityPotion potion = (EntityPotion) entityObject;
            if (!isPotionThrownByBot(potion)) continue;

            int potionId = potion.getEntityId();
            visibleBotPotionIds.add(potionId);

            if (!seenBotPotionIds.contains(potionId)) {
                seenBotPotionIds.add(potionId);
                activeBotPotionIds.add(potionId);
                if (waitingForCurrentPotExplosion) {
                    currentPotEntityIds.add(potionId);
                }
            }
        }

        Iterator<Integer> iterator = activeBotPotionIds.iterator();
        while (iterator.hasNext()) {
            int potionId = iterator.next();
            if (visibleBotPotionIds.contains(potionId)) continue;

            iterator.remove();
            if (waitingForCurrentPotExplosion && currentPotEntityIds.remove(potionId)) {
                currentPotExploded = true;
            }
        }
    }

    private boolean isPotionThrownByBot(EntityPotion potion) {
        Entity thrower = potion.getThrower();
        if (thrower == mc.thePlayer) return true;
        if (thrower != null) return false;

        return waitingForCurrentPotExplosion
                && potion.ticksExisted <= 5
                && mc.thePlayer.getDistanceToEntity(potion) <= 4.0F;
    }
}
