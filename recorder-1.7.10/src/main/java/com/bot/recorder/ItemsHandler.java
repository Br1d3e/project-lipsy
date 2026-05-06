package com.bot.recorder;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemStack;
import net.minecraft.potion.Potion;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ItemsHandler {
    private final Minecraft mc;

    public ItemsHandler(Minecraft mc) {
        this.mc = mc;
    }

    public String getHeldItemType(EntityPlayer player) {
        ItemStack stack = player.getHeldItem();
        if (stack == null || stack.getItem() == null) return null;

        return stack.getItem().getUnlocalizedName();
    }

    public Integer getPotionDuration(EntityPlayer player, Potion potion) {
        if (!player.isPotionActive(potion)) return null;

        return player.getActivePotionEffect(potion).getDuration();
    }

    public boolean hasPotion(EntityPlayer player, Potion potion) {
        return player.isPotionActive(potion);
    }

    public List<String> getHotbarSnapshot() {
        List<String> hotbar = new ArrayList<String>();

        for (int i = 0; i < 9; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            hotbar.add(getItemType(stack));
        }

        return hotbar;
    }

    public int countFoodItems() {
        return countInventoryItems(new ItemMatcher() {
            public boolean matches(ItemStack stack) {
                return isFood(stack);
            }
        });
    }

    public int countSpeedPotions() {
        return countInventoryItems(new ItemMatcher() {
            public boolean matches(ItemStack stack) {
                return isSpeedPotion(stack);
            }
        });
    }

    public int countSplashHealthPotions() {
        return countInventoryItems(new ItemMatcher() {
            public boolean matches(ItemStack stack) {
                return isSplashHealthPotion(stack);
            }
        });
    }

    public int countPearls() {
        return countInventoryItems(new ItemMatcher() {
            public boolean matches(ItemStack stack) {
                return isPearl(stack);
            }
        });
    }

    public int countInventoryItems(ItemMatcher matcher) {
        int count = 0;

        for (int i = 0; i < 36; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            if (stack != null && matcher.matches(stack)) {
                count += stack.stackSize;
            }
        }

        return count;
    }

    public String getItemType(ItemStack stack) {
        if (stack == null || stack.getItem() == null) return null;

        return stack.getItem().getUnlocalizedName();
    }

    public String getDisplayNameLower(ItemStack stack) {
        return stack == null ? "" : stack.getDisplayName().toLowerCase(Locale.US);
    }

    public String getDisplayName(ItemStack stack) {
        return stack == null ? null : stack.getDisplayName();
    }

    public String getItemTypeLower(ItemStack stack) {
        String type = getItemType(stack);

        return type == null ? "" : type.toLowerCase(Locale.US);
    }

    public boolean isFood(ItemStack stack) {
        String type = getItemTypeLower(stack);

        return type.contains("beefcooked") || type.contains("carrotgolden");
    }

    public boolean isSpeedPotion(ItemStack stack) {
        String name = getDisplayNameLower(stack);
        String type = getItemTypeLower(stack);

        return name.contains("swiftness") && type.contains("potion");
    }

    public boolean isSplashHealthPotion(ItemStack stack) {
        String name = getDisplayNameLower(stack);
        String type = getItemTypeLower(stack);

        return type.contains("potion") && name.contains("splash") && (name.contains("instant health") || name.contains("healing"));
    }

    public boolean isPearl(ItemStack stack) {
        return getItemTypeLower(stack).contains("enderpearl");
    }

    public boolean isSword(ItemStack stack) {
        String type = getItemTypeLower(stack);

        return type.contains("sword");
    }

    public String heldSwordType(ItemStack stack) {
        String type = getItemTypeLower(stack);

        if (!isSword(stack)) return "";

        if (type.contains("diamond")) {
            return "diamond";
        } else if (type.contains("iron")) {
            return "iron";
        } else if (type.contains("stone")) {
            return "stone";
        } else if (type.contains("gold") || type.contains("golden")) {
            return "gold";
        } else if (type.contains("wood")) {
            return "wood";
        }
        return "";
    }

    private interface ItemMatcher {
        boolean matches(ItemStack stack);
    }



}
