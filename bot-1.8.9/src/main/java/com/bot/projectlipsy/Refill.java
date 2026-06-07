package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.inventory.GuiInventory;
import net.minecraft.item.ItemStack;

public class Refill {
    private final Minecraft mc;
    private final utils utils;

    private int refillTicks;
    private boolean refilling = false;

    private final String POTION_NAME = "Splash Potion of Healing";
    private final String BOTTLE_NAME = "Glass Bottle";
    private final int FULL_HOTBAR_POTIONS = 6;
    private final int FOOD_SLOT = 8;
    private final int[] SPEED_SLOTS = new int[] {2, 7};

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
        if (mc.thePlayer == null) return 0;

        int count = 0;
        for (int i = 0; i < 9; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            if (stack != null && stack.getDisplayName() != null && stack.getDisplayName().contains(POTION_NAME)) {
                count++;
            }
        }
        return count;
    }

    public int refillableHotbarSlots() {
        if (mc.thePlayer == null) return 0;

        int count = 0;
        for (int i = 0; i < 9; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            if (stack == null || stack.getDisplayName() != null && stack.getDisplayName().equals(BOTTLE_NAME)) {
                count++;
            }
        }
        return count;
    }

    public boolean isRefilling() {
        return refilling;
    }

    public boolean refillPotions(int amount) {
        if (mc.thePlayer == null || amount <= 0) {
            stopRefilling();
            return true;
        }
        if (!refilling) {
            if (!hasInventoryPotion()) return true;
            refilling = true;
        }
        if (!hasInventoryPotion() || getRefillAmount() <= 0) {
            stopRefilling();
            return true;
        }

        if (refillTicks > 0) {
            refillTicks--;
            if (refillTicks > 0) {
                return false;
            }
        }

        if (!openInventory()) {
            return false;
        }
        boolean inventoryOpen = mc.currentScreen instanceof GuiInventory;
        mc.thePlayer.sendChatMessage(String.format("[Project L1psy] Refill: inventoryOpen=%s", inventoryOpen));
        if (!inventoryOpen) {
            mc.thePlayer.sendChatMessage("[Project L1psy] Refill: unable to open inventory");
            return false;
        } else {
            for (int j = 0; j < 9; j++) {
                ItemStack hotbarStack = mc.thePlayer.inventory.getStackInSlot(j);
                if (hotbarStack == null) {
                    for (int i = 9; i < 36; i++) {
                        ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
                        if (stack != null && stack.getDisplayName() != null && stack.getDisplayName().contains(POTION_NAME)) {
                            // Move potion
                            mc.playerController.windowClick(mc.thePlayer.inventoryContainer.windowId, i, j, 2, mc.thePlayer);
                            refillTicks = nextRefillTicks();
                            if (amount <= 1 || getRefillAmount() <= 0) {
                                stopRefilling();
                                return true;
                            }
                            return false;
                        }
                    }
                } else if (hotbarStack.getDisplayName() != null && hotbarStack.getDisplayName().equals(BOTTLE_NAME)) {
                    mc.thePlayer.inventory.currentItem = j;
                    mc.playerController.updateController();
                    mc.thePlayer.dropOneItem(false);
                    refillTicks = nextRefillTicks();
                    return false;
                }

            }
        }
        stopRefilling();
        return true;
    }

    private void stopRefilling() {
        refilling = false;
    }

    public boolean refillSpeed() {
        if (mc.thePlayer == null) return false;
        if (hasHotbarSpeed()) {
            resetGuiScreen();
            return true;
        }
        if (!hasInventorySpeed()) {
            resetGuiScreen();
            return false;
        }
        if (!tickRefillDelay()) return false;

        if (!openInventory()) return false;

        int targetSlot = findSpeedRefillSlot();
        int sourceSlot = findInventorySpeedSlot();
        if (targetSlot < 0 || sourceSlot < 0) return false;

        mc.playerController.windowClick(mc.thePlayer.inventoryContainer.windowId, sourceSlot, targetSlot, 2, mc.thePlayer);
        refillTicks = nextRefillTicks();
        resetGuiScreen();
        return true;
    }

    public boolean refillFood() {
        if (mc.thePlayer == null) return false;
        if (hasHotbarFoodInFoodSlot()) {
            resetGuiScreen();
            return true;
        }
        if (!hasInventoryFood()) {
            resetGuiScreen();
            return false;
        }
        if (!tickRefillDelay()) return false;

        if (!openInventory()) return false;

        ItemStack targetStack = mc.thePlayer.inventory.getStackInSlot(FOOD_SLOT);
        int sourceSlot = findInventoryFoodSlot();
        if (!canReplaceWithUtility(targetStack) || sourceSlot < 0) return false;

        mc.playerController.windowClick(mc.thePlayer.inventoryContainer.windowId, sourceSlot, FOOD_SLOT, 2, mc.thePlayer);
        refillTicks = nextRefillTicks();
        resetGuiScreen();
        return true;
    }

    private int nextRefillTicks() {
        return (int) Math.floor(Math.random() * 3.5D) + 3;
    }

    private boolean tickRefillDelay() {
        if (refillTicks <= 0) return true;

        refillTicks--;
        return refillTicks <= 0;
    }

    private boolean openInventory() {
        if (mc.currentScreen == null) {
            mc.displayGuiScreen(new GuiInventory(mc.thePlayer));
            return false;
        }

        boolean inventoryOpen = mc.currentScreen instanceof GuiInventory;
        if (!inventoryOpen) {
            mc.thePlayer.sendChatMessage("[Project L1psy] Refill: unable to open inventory");
        }
        return inventoryOpen;
    }

    private int findSpeedRefillSlot() {
        for (int slot : SPEED_SLOTS) {
            if (canReplaceWithUtility(mc.thePlayer.inventory.getStackInSlot(slot))) {
                return slot;
            }
        }
        for (int slot = 0; slot < 9; slot++) {
            if (slot == FOOD_SLOT) continue;
            if (canReplaceWithUtility(mc.thePlayer.inventory.getStackInSlot(slot))) {
                return slot;
            }
        }
        return -1;
    }

    private boolean canReplaceWithUtility(ItemStack stack) {
        return stack == null || utils.isHealthPotion(stack);
    }

    private int findInventorySpeedSlot() {
        if (mc.thePlayer == null) return -1;
        for (int i = 9; i < 36; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            if (utils.isSpeedPotion(stack)) {
                return i;
            }
        }
        return -1;
    }

    private int findInventoryFoodSlot() {
        if (mc.thePlayer == null) return -1;
        for (int i = 9; i < 36; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            if (utils.isFood(stack)) {
                return i;
            }
        }
        return -1;
    }

    public boolean hasHotbarSpeed() {
        if (mc.thePlayer == null) return false;
        for (int i = 0; i < 9; i++) {
            if (utils.isSpeedPotion(mc.thePlayer.inventory.getStackInSlot(i))) {
                return true;
            }
        }
        return false;
    }

    public boolean hasHotbarFoodInFoodSlot() {
        if (mc.thePlayer == null) return false;
        return utils.isFood(mc.thePlayer.inventory.getStackInSlot(FOOD_SLOT));
    }

    public boolean hasInventorySpeed() {
        return findInventorySpeedSlot() >= 0;
    }

    public boolean hasInventoryFood() {
        return findInventoryFoodSlot() >= 0;
    }

    public int getRefillAmount() {
        if (!hasInventoryPotion()) return 0;
        return Math.max(0, Math.min(refillableHotbarSlots(), FULL_HOTBAR_POTIONS - hotbarPotions()));
    }

    public boolean hasPotion() {
        return utils.hasItem(POTION_NAME);
    }

    public boolean hasInventoryPotion() {
        if (mc.thePlayer == null) return false;

        for (int i = 9; i < 36; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            if (stack != null && stack.getDisplayName() != null && stack.getDisplayName().contains(POTION_NAME)) {
                return true;
            }
        }
        return false;
    }

    public void panicRefill(int amount) {

    }
}
