package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.item.ItemSword;
import net.minecraft.potion.Potion;
import net.minecraft.potion.PotionEffect;
import net.minecraft.util.BlockPos;

import java.util.Locale;

public class utils {
    private final Minecraft mc;
    private int myCombo = 0;
    private int targetCombo = 0;
    private int lastMyHurtTime = 0;
    private int lastTargetHurtTime = 0;
    private static final int WALL_CHECK_MAX_DISTANCE = 5;
    private static final float NO_WALL_DISTANCE = 100.0F;


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

    public int itemCount(ItemStack stack) {
        return stack == null ? 0 : Math.max(1, stack.stackSize);
    }

    public boolean isHoldingSword() {
        ItemStack stack = mc.thePlayer.getHeldItem();
        return stack != null && stack.getItem() instanceof ItemSword;
    }

    public boolean isHealthPotion(ItemStack stack) {
        String text = itemText(stack);
        return text.contains("potion") && (text.contains("splash") || text.contains("healing") || text.contains("instant health"));
    }

    public boolean isSpeedPotion(ItemStack stack) {
        String text = itemText(stack);
        return text.contains("potion") && (text.contains("swiftness") || text.contains("speed"));
    }

    public boolean isPearl(ItemStack stack) {
        String text = itemText(stack);
        return text.contains("pearl");
    }

    public boolean isFood(ItemStack stack) {
        String text = itemText(stack);
        return text.contains("carrot") || text.contains("steak") || text.contains("beef") || text.contains("food");
    }

    public String itemText(ItemStack stack) {
        if (stack == null || stack.getItem() == null) return "";
        String display = stack.getDisplayName() == null ? "" : stack.getDisplayName();
        String itemName = stack.getItem().getUnlocalizedName() == null ? "" : stack.getItem().getUnlocalizedName();
        return (display + " " + itemName).toLowerCase(Locale.US);
    }

    public int potionDuration(EntityPlayer player, Potion potion) {
        if (player == null || !player.isPotionActive(potion)) return 0;
        PotionEffect effect = player.getActivePotionEffect(potion);
        return effect == null ? 0 : effect.getDuration();
    }

    public float blockDistanceAtYaw(float yaw) {
        if (mc.theWorld == null || mc.thePlayer == null) return NO_WALL_DISTANCE;
        double radians = Math.toRadians(yaw);
        double dx = -Math.sin(radians);
        double dz = Math.cos(radians);
        double baseX = mc.thePlayer.posX;
        double baseY = mc.thePlayer.posY;
        double baseZ = mc.thePlayer.posZ;

        for (int distance = 1; distance <= WALL_CHECK_MAX_DISTANCE; distance++) {
            BlockPos body = new BlockPos(baseX + dx * distance, baseY, baseZ + dz * distance);
            BlockPos head = new BlockPos(baseX + dx * distance, baseY + 1.0D, baseZ + dz * distance);
            if (!mc.theWorld.isAirBlock(body) || !mc.theWorld.isAirBlock(head)) {
                return distance;
            }
        }
        return NO_WALL_DISTANCE;
    }

    public static class ItemSnapshot {
        int hotbarHealthPotions = 0;
        int food = 0;
        int speedPotions = 0;
        int healthPotions = 0;
        int pearls = 0;
    }
}
