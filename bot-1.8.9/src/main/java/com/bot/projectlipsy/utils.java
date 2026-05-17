package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.Item;

public class utils {
    private final Minecraft mc;
    private int myCombo = 0;
    private int targetCombo = 0;
    private int lastMyHurtTime = 0;
    private int lastTargetHurtTime = 0;

    public utils(Minecraft mc) {this.mc = mc; }

    public void updateCombos(EntityPlayer target) {
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

    public int getMyCombo() {
        return myCombo;
    }

    public int getTargetCombo() {
        return targetCombo;
    }

    public boolean hasItem(String displayName) {
        for (int i = 0; i < 36; i++) {
            if (mc.thePlayer.inventory.getStackInSlot(i) != null && mc.thePlayer.inventory.getStackInSlot(i).getDisplayName().contains(displayName)) {
                return true;
            }
        }
        return false;
    }

    public void switchItem(String displayName) {
        if (!hasItem(displayName)) return;
        for (int i = 0; i < 9; i++) {
            if (mc.thePlayer.inventory.getStackInSlot(i) != null && mc.thePlayer.inventory.getStackInSlot(i).getDisplayName().contains(displayName)) {
                mc.thePlayer.inventory.currentItem = i;
                return;
            }
        }

        for (int i = 9; i < 36; i++) {
            if (mc.thePlayer.inventory.getStackInSlot(i) != null && mc.thePlayer.inventory.getStackInSlot(i).getDisplayName().contains(displayName)) {
                int emptySlot = -1;
                for (int j = 0; j < 9; j++) {
                    if (mc.thePlayer.inventory.getStackInSlot(j) == null) {
                        emptySlot = j;
                        break;
                    }
                }

                if (emptySlot != -1) {
                    mc.playerController.windowClick(mc.thePlayer.inventoryContainer.windowId, i, emptySlot, 2, mc.thePlayer);
                    mc.thePlayer.inventory.currentItem = emptySlot;
                }
                return;
            }
        }
    }

    public void refillPotions(int amount) {
        int potionsRefilled = 0;
        String potionName = "Splash Potion of Healing";

        if (amount <= 0) {
            return;
        }

        // Find empty hotbar slots and fill them
        for (int j = 0; j < 9; j++) { // Iterate through hotbar
            if (mc.thePlayer.inventory.getStackInSlot(j) == null) {
                // This hotbar slot is empty, let's find a potion for it
                for (int i = 9; i < 36; i++) { // Iterate through inventory
                    if (mc.thePlayer.inventory.getStackInSlot(i) != null && mc.thePlayer.inventory.getStackInSlot(i).getDisplayName().contains(potionName)) {
                        // Move potion
                        mc.playerController.windowClick(mc.thePlayer.inventoryContainer.windowId, i, j, 2, mc.thePlayer);
                        potionsRefilled++;
                        
                        if (potionsRefilled >= amount) {
                            return; // Refilled requested amount
                        }
                        // Potion moved, so break from inner loop to fill next empty hotbar slot
                        break;
                    }
                }
            }
        }
    }

    public float getTargetYaw(EntityPlayer target) {
        double dx = target.posX - mc.thePlayer.posX;
        double dz = target.posZ - mc.thePlayer.posZ;
        return (float) Math.toDegrees(Math.atan2(dz, dx) - 90F);
    }

    public double getRandomOffset(float range){
        return (Math.random() * range * 2) - range;
    }
}
