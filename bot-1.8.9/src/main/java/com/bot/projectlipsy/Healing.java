package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.projectile.EntityPotion;
import net.minecraft.potion.Potion;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class Healing {

    private final Minecraft mc;
    private final utils utils;
    private final Movement movement;
    private long lastPotTime = 0;
    private static final int MAX_UTILITY_USE_TICKS = 40;

    public enum HealingState {
        NONE,
        TURNING,
        POTTING
    }

    private HealingState potState = HealingState.NONE;
    private int potTicks = 0;
    private int speedTicks = 0;
    private int eatTicks = 0;
    private boolean waitingForCurrentPotExplosion = false;
    private boolean currentPotExploded = false;
    private final Set<Integer> activeBotPotionIds = new HashSet<Integer>();
    private final Set<Integer> seenBotPotionIds = new HashSet<Integer>();
    private final Set<Integer> currentPotEntityIds = new HashSet<Integer>();

    public Healing(Minecraft mc, Movement movement) {
        this.mc = mc;
        this.utils = new utils(mc);
        this.movement = movement;
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
        }
    }

    public void advanceState() {
        if (potState == HealingState.TURNING) {
            potState = HealingState.POTTING;
            potTicks = 0;
        }
    }

    public void updatePotMacro(EntityPlayer target) {
        if (potState == HealingState.NONE || target == null || mc.thePlayer == null) return;

        updateBotPotionHistory();
        potTicks ++;

        float retreatYaw = movement.getRetreatYaw(target);
        if (potTicks == 1) {
            mc.thePlayer.rotationYaw = retreatYaw - 150F;
            potState = HealingState.TURNING;
        } else if (potTicks == 2) {
            mc.thePlayer.rotationYaw = retreatYaw - 80F;
            mc.thePlayer.rotationPitch = 10F;
        } else if (potTicks == 3) {
            mc.thePlayer.rotationYaw = retreatYaw - 20F;
            mc.thePlayer.rotationPitch = 20F;
        } else if (potTicks == 4) {
            mc.thePlayer.rotationYaw = retreatYaw;
            mc.thePlayer.rotationPitch = 30F;
        } else if (potTicks == 5) {
            utils.switchItem("Splash Potion of Healing");
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), true);
            KeyBinding.onTick(mc.gameSettings.keyBindUseItem.getKeyCode());
            potState = HealingState.POTTING;
            waitingForCurrentPotExplosion = true;
            currentPotExploded = false;
            currentPotEntityIds.clear();
        } else if (potTicks == 6) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
            lastPotTime = System.currentTimeMillis();
        } else if (currentPotExploded) {
            stopPotting();
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
    }

    public boolean canPot() {
        long currentTime = System.currentTimeMillis();
        long potInterval = 600;
        return utils.hasItem("Splash Potion of Healing") && currentTime - lastPotTime >= potInterval;
    }

    public boolean canEat() {
        return this.mc.thePlayer.getFoodStats().getFoodLevel() < 20 && (utils.hasItem("Golden Carrot") || utils.hasItem("Steak"));
    }

    public boolean canSpeed() {
        return (!this.mc.thePlayer.isPotionActive(Potion.moveSpeed) || this.mc.thePlayer.getActivePotionEffect
                (Potion.moveSpeed).getDuration() < 20 * 20) && utils.hasItem("Potion of Swiftness");     // 20s
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

        if (speedTicks > MAX_UTILITY_USE_TICKS) {
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

        if (eatTicks > MAX_UTILITY_USE_TICKS) {
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
