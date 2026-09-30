package com.bot.projectlipsy.features;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.projectile.EntityPotion;
import net.minecraft.init.Items;
import net.minecraft.item.ItemPotion;
import net.minecraft.item.ItemStack;
import net.minecraft.potion.Potion;
import net.minecraft.potion.PotionEffect;
import net.minecraft.util.AxisAlignedBB;
import java.util.*;

/** Recorder v3 potion attribution and splash geometry, without the HP damage estimator. */
final class TargetHealingObserver {
    private final Minecraft mc;
    private final Map<Integer, double[]> potions = new HashMap<Integer, double[]>();
    private final Set<Integer> processed = new HashSet<Integer>();
    private final Set<Integer> throwsSeen = new HashSet<Integer>();
    private final Deque<AxisAlignedBB> hitboxes = new ArrayDeque<AxisAlignedBB>();
    private int targetId = -1;

    TargetHealingObserver(Minecraft mc) { this.mc = mc; }

    void reset() {
        observationLost();
        processed.clear();
        throwsSeen.clear();
        targetId = -1;
    }

    void observationLost() { potions.clear(); hitboxes.clear(); }

    void rememberTarget(EntityPlayer target) {
        if (target.getEntityId() != targetId) {
            observationLost();
            targetId = target.getEntityId();
        }
        hitboxes.addLast(target.getEntityBoundingBox().expand(0.3D, 0.3D, 0.3D));
        while (hitboxes.size() > 8) hitboxes.removeFirst();
    }

    boolean scan(EntityPlayer target) {
        boolean thrown = false;
        Set<Integer> visible = new HashSet<Integer>();
        for (Entity entity : mc.theWorld.loadedEntityList) {
            if (!(entity instanceof EntityPotion)) continue;
            EntityPotion potion = (EntityPotion) entity;
            if (!isHealingSplash(potion.getPotionDamage())) continue;
            int id = potion.getEntityId();
            if (belongsToTarget(potion, target) && throwsSeen.add(id)) thrown = true;
            // Our own splashes can also heal the enemy.
            if (processed.contains(id) || target.getDistanceToEntity(potion) > 4.0D) continue;
            visible.add(id);
            potions.put(id, new double[] {potion.posX, potion.posY, potion.posZ});
        }
        Iterator<Integer> iterator = potions.keySet().iterator();
        while (iterator.hasNext()) {
            int id = iterator.next();
            if (!visible.contains(id)) { processed.add(id); iterator.remove(); }
        }
        return thrown;
    }

    boolean impact(EntityPlayer target, int type, int damage, int x, int y, int z) {
        if (type != 2002 || !isHealingSplash(damage)) return false;
        Integer best = null;
        double distance = Double.MAX_VALUE;
        for (Map.Entry<Integer, double[]> entry : potions.entrySet()) {
            double[] p = entry.getValue();
            double d = square(p[0] - x) + square(p[1] - y) + square(p[2] - z);
            if (d < distance) { best = entry.getKey(); distance = d; }
        }
        if (best == null || distance > 16.0D) return false;
        double[] p = potions.remove(best);
        processed.add(best);
        // As in TargetHpTracker, use the last projectile position, not the block-coordinate effect.
        double boxDistance = Double.POSITIVE_INFINITY;
        for (AxisAlignedBB box : hitboxes) boxDistance = Math.min(boxDistance, distanceToBox(p, box));
        if (Double.isInfinite(boxDistance)) {
            boxDistance = distanceToBox(p, target.getEntityBoundingBox().expand(0.3D, 0.3D, 0.3D));
        }
        float effectiveness = boxDistance <= 1.44D ? 1.0F : Math.max(0.0F, 1.0F - (float) Math.sqrt(boxDistance) / 4.0F);
        return Math.floor(8.0F * effectiveness + 0.5F) > 0;
    }

    void destroyed(int[] ids) {
        for (int id : ids) if (potions.remove(id) != null) processed.add(id);
    }

    private boolean belongsToTarget(EntityPotion potion, EntityPlayer target) {
        Entity owner = potion.getThrower();
        if (owner == mc.thePlayer) return false;
        if (owner == target) return true;
        if (owner != null || potion.ticksExisted > 5) return false;
        double distance = target.getDistanceToEntity(potion);
        return distance <= 4.0D && mc.thePlayer.getDistanceToEntity(potion) + 0.5D >= distance;
    }

    private static boolean isHealingSplash(int damage) {
        if (!ItemPotion.isSplash(damage)) return false;
        List<PotionEffect> effects = Items.potionitem.getEffects(damage);
        if (effects != null) for (PotionEffect effect : effects) {
            if (effect.getPotionID() == Potion.heal.id && effect.getAmplifier() >= 1) return true;
        }
        return false;
    }

    // Match ItemsHandler's recorded holding labels exactly, rather than utils' broader heuristics.
    static boolean holdingFood(ItemStack stack) {
        String type = type(stack);
        return type.contains("beefcooked") || type.contains("carrotgolden");
    }
    static boolean holdingSpeed(ItemStack stack) {
        return type(stack).contains("potion") && name(stack).contains("swiftness");
    }
    static boolean holdingHeal(ItemStack stack) {
        String name = name(stack);
        return type(stack).contains("potion") && name.contains("splash")
                && (name.contains("instant health") || name.contains("healing"));
    }
    private static String type(ItemStack stack) {
        return stack == null ? "" : stack.getItem().getUnlocalizedName().toLowerCase(Locale.US);
    }
    private static String name(ItemStack stack) {
        return stack == null ? "" : stack.getDisplayName().toLowerCase(Locale.US);
    }
    private static double square(double x) { return x * x; }
    private static double axis(double x, double min, double max) {
        return x < min ? min - x : x > max ? x - max : 0;
    }
    private static double distanceToBox(double[] p, AxisAlignedBB box) {
        return square(axis(p[0], box.minX, box.maxX)) + square(axis(p[1], box.minY, box.maxY))
                + square(axis(p[2], box.minZ, box.maxZ));
    }
}
