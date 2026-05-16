package com.bot.recorder.targetTracker;

import com.bot.recorder.ItemsHandler;
import net.minecraft.client.Minecraft;
import net.minecraft.entity.Entity;
import net.minecraft.entity.item.EntityEnderPearl;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.projectile.EntityPotion;
import net.minecraft.init.Items;
import net.minecraft.item.ItemPotion;
import net.minecraft.potion.Potion;
import net.minecraft.potion.PotionEffect;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class TargetInventoryTracker {

    private final Minecraft mc;
    private final ItemsHandler itemsHandler;

    private final int TARGET_POTIONS_DEFAULT = 29;
    private final int TARGET_PEARLS_DEFAULT = 16;

    private int targetPotions;
    private int targetPearls;
    private int trackedTargetId = -1;
    private final Set<Integer> seenTargetPotionEntityIds = new HashSet<Integer>();
    private final Set<Integer> seenTargetPearlEntityIds = new HashSet<Integer>();

    public TargetInventoryTracker(Minecraft mc, ItemsHandler itemsHandler) {
        this.mc = mc;
        this.itemsHandler = itemsHandler;
        reset();
    }

    public void reset() {
        targetPotions = TARGET_POTIONS_DEFAULT;
        targetPearls = TARGET_PEARLS_DEFAULT;
        trackedTargetId = -1;
        seenTargetPotionEntityIds.clear();
        seenTargetPearlEntityIds.clear();
    }

    public void trackHealthPotions(EntityPlayer target) {
        if (!ensureTarget(target) || mc.theWorld == null) return;

        for (Object entityObject : mc.theWorld.loadedEntityList) {
            if (!(entityObject instanceof EntityPotion)) continue;

            EntityPotion potion = (EntityPotion) entityObject;
            int potionId = potion.getEntityId();
            if (seenTargetPotionEntityIds.contains(potionId)) continue;
            if (!isTargetHealingPotion(potion, target)) continue;

            seenTargetPotionEntityIds.add(potionId);
            if (targetPotions > 0) {
                targetPotions--;
            }
        }
    }

    public void trackPearls(EntityPlayer target) {
        if (!ensureTarget(target) || mc.theWorld == null) return;

        for (Object entityObject : mc.theWorld.loadedEntityList) {
            if (!(entityObject instanceof EntityEnderPearl)) continue;

            EntityEnderPearl pearl = (EntityEnderPearl) entityObject;
            int pearlId = pearl.getEntityId();
            if (seenTargetPearlEntityIds.contains(pearlId)) continue;
            if (!isTargetPearl(pearl, target)) continue;

            seenTargetPearlEntityIds.add(pearlId);
            if (targetPearls > 0) {
                targetPearls--;
            }
        }
    }

    public int getTargetHealthPotions() {
        return targetPotions;
    }

    public int getTargetPearls() {
        return targetPearls;
    }

    private boolean ensureTarget(EntityPlayer target) {
        if (target == null) {
            trackedTargetId = -1;
            seenTargetPotionEntityIds.clear();
            seenTargetPearlEntityIds.clear();
            return false;
        }

        int targetId = target.getEntityId();
        if (targetId != trackedTargetId) {
            targetPotions = TARGET_POTIONS_DEFAULT;
            targetPearls = TARGET_PEARLS_DEFAULT;
            trackedTargetId = targetId;
            seenTargetPotionEntityIds.clear();
            seenTargetPearlEntityIds.clear();
        }

        return true;
    }

    private boolean isTargetHealingPotion(EntityPotion potion, EntityPlayer target) {
        if (!isInstantHealthIISplashPotion(potion)) return false;

        Entity thrower = potion.getThrower();
        if (thrower == mc.thePlayer) return false;
        if (thrower == target) return true;
        if (thrower != null) return false;

        return potion.ticksExisted <= 5 && target.getDistanceToEntity(potion) <= 4.0F;
    }

    private boolean isTargetPearl(EntityEnderPearl pearl, EntityPlayer target) {
        Entity thrower = pearl.getThrower();
        if (thrower == mc.thePlayer) return false;
        if (thrower == target) return true;
        if (thrower != null) return false;

        if (pearl.ticksExisted > 5) return false;

        double targetDistance = target.getDistanceToEntity(pearl);
        if (targetDistance > 4.0D) return false;

        if (mc.thePlayer != null) {
            double selfDistance = mc.thePlayer.getDistanceToEntity(pearl);
            if (selfDistance + 0.5D < targetDistance) return false;
        }

        return true;
    }

    private boolean isInstantHealthIISplashPotion(EntityPotion potion) {
        return isInstantHealthIISplashPotion(potion.getPotionDamage());
    }

    private boolean isInstantHealthIISplashPotion(int potionDamage) {
        if (!ItemPotion.isSplash(potionDamage)) return false;

        List effects = Items.potionitem.getEffects(potionDamage);
        if (effects == null) return false;

        for (Object effectObject : effects) {
            if (!(effectObject instanceof PotionEffect)) continue;

            PotionEffect effect = (PotionEffect) effectObject;
            if (effect.getPotionID() == Potion.heal.id && effect.getAmplifier() >= 1) {
                return true;
            }
        }

        return false;
    }
}
