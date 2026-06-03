package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.inventory.GuiInventory;

public class Refill {
    private final Minecraft mc;
    private final utils utils;

    private final String POTION_NAME = "Splash Potion of Healing";

    public Refill(Minecraft mc) {
        this.mc = mc;
        this.utils = new utils(mc);
    }

    public void resetGuiScreen() {
        if (mc.currentScreen instanceof GuiInventory) {
            mc.displayGuiScreen(null);
            mc.setIngameFocus();
        }
    }

    public int hotbarPotions() {
        int count = 0;
        for (int i = 0; i < 9; i++) {
            if (mc.thePlayer.inventory.getStackInSlot(i).getDisplayName().contains(POTION_NAME)) {
                count++;
            }
        }
        return count;
    }

    public void refillPotions(int amount) {
        if (mc.thePlayer == null) return;

        int potionsRefilled = 0;

        if (amount <= 0) {
            return;
        }

        if ( mc.currentScreen == null) {
            mc.displayGuiScreen(new GuiInventory(mc.thePlayer));
        }
        boolean inventoryOpen = mc.currentScreen instanceof GuiInventory;

        if (!inventoryOpen) {
            mc.thePlayer.sendChatMessage("[Project L1psy] Refill: unable to open inventory");
        }
        for (int j = 0; j < 9; j++) {
            if (mc.thePlayer.inventory.getStackInSlot(j) == null) {
                for (int i = 9; i < 36; i++) {
                    if (mc.thePlayer.inventory.getStackInSlot(i) != null &&
                            mc.thePlayer.inventory.getStackInSlot(i).getDisplayName().contains(POTION_NAME)) {
                        // Move potion
                        mc.playerController.windowClick(mc.thePlayer.inventoryContainer.windowId, i, j, 2, mc.thePlayer);
                        potionsRefilled++;

                        if (potionsRefilled >= amount) {
                            resetGuiScreen();
                            return;
                        }
                        // Potion moved, so break from inner loop to fill next empty hotbar slot
                        break;
                    }
                }
            }
        }
    }

    public void panicRefill(int amount) {

    }
}
