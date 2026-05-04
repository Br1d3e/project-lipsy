package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.potion.Potion;

public class Healing {

    private final Minecraft mc;
    private final utils utils;
    private long lastPotTime = 0;

    public enum HealingState {
        NONE,
        TURNING,
        POTTING
    }

    private HealingState potState = HealingState.NONE;
    private int potTicks = 0;

    public Healing(Minecraft mc) {
        this.mc = mc;
        this.utils = new utils(mc);
    }

    public HealingState getState() {
        return potState;
    }

    public boolean isPotting() {
        return potState != HealingState.NONE;
    }

    public void startHealingSequence() {
        if (canPot() && !isPotting()) {
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

    public void pot() {
        if (potState != HealingState.POTTING) return;

        potTicks++;

        if (potTicks == 1) {
            mc.thePlayer.rotationPitch = 20F; // Look down
            utils.switchItem("Splash Potion of Healing");
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), true);
        } else if (potTicks == 3) { // Hold for 2 ticks
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
            lastPotTime = System.currentTimeMillis();
        } else if (potTicks > 15) { // Wait for pot to land
            stopPotting();
        }
    }

    public void stopPotting() {
        potState = HealingState.NONE;
        potTicks = 0;
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
    }

    public boolean canPot() {
        long currentTime = System.currentTimeMillis();
        long potInterval = 600;
        return utils.hasItem("Splash Potion of Healing") && currentTime - lastPotTime >= potInterval;
    }

    public boolean canEat() {
        return this.mc.thePlayer.getFoodStats().getFoodLevel() < 20 && utils.hasItem("Golden Carrot");
    }

    public boolean canSpeed() {
        return (!this.mc.thePlayer.isPotionActive(Potion.moveSpeed) || this.mc.thePlayer.getActivePotionEffect
                (Potion.moveSpeed).getDuration() < 20 * 20) && utils.hasItem("Potion of Swiftness");     // 20s
    }

    public void speed() {
        if (utils.hasItem("Potion of Swiftness")) {
            utils.switchItem("Potion of Swiftness");
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), true);
        }
    }

    public void eat() {
        if (utils.hasItem("Golden Carrot")) {
            utils.switchItem("Golden Carrot");
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), true);
        }
    }
}