package com.bot.recorder.targetTracker;

import com.bot.recorder.ItemsHandler;
import com.bot.recorder.recordWriter.JsonLineBuilder;

import net.minecraft.client.Minecraft;
import net.minecraft.enchantment.Enchantment;
import net.minecraft.enchantment.EnchantmentHelper;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.projectile.EntityPotion;
import net.minecraft.init.Items;
import net.minecraft.item.ItemPotion;
import net.minecraft.item.ItemStack;
import net.minecraft.potion.Potion;
import net.minecraft.potion.PotionEffect;
import net.minecraft.util.AxisAlignedBB;
import net.minecraft.util.MathHelper;

import java.util.*;

public class TargetHpTracker {
    public interface EventSink {
        void writeEvent(String type, JsonLineBuilder details);
    }

    private final Minecraft mc;
    private final EventSink eventSink;
    private final ItemsHandler itemsHandler;
    private static final float TARGET_HP_DEFAULT = 20.0F;
    private static final float CRIT_MULTIPLIER = 1.5F;
    private static final int DEFAULT_SHARPNESS_LEVEL = 4;
    private static final int DEFAULT_TARGET_ARMOR_POINTS = 20;
    private static final int DEFAULT_TARGET_PROTECTION_LEVEL_PER_PIECE = 2;
    private static final int DEFAULT_TARGET_FEATHER_FALLING_LEVEL = 0;
    private static final float PEARL_BASE_DAMAGE_HP = 5.0F;
    private static final float INSTANT_HEALTH_II_BASE_HEAL_HP = 8.0F;
    private static final double POTION_DIRECT_HIT_BOX_EXPAND = 0.3D;
    private static final int NATURAL_REGEN_INTERVAL_TICKS = 80;
    private static final float NATURAL_REGEN_HEAL_HP = 1.0F;
    private int trackedTargetId = -1;
    private float estimatedTargetHp = TARGET_HP_DEFAULT;
    private float lastEstimatedTargetDamage = 0.0F;
    private float lastEstimatedTargetHeal = 0.0F;
    private int naturalRegenTimer = 0;
    private int targetArmorPoints = DEFAULT_TARGET_ARMOR_POINTS;
    private int targetProtectionEpf;
    private int targetFeatherFallingLevel = DEFAULT_TARGET_FEATHER_FALLING_LEVEL;
    private final Map<Integer, TrackedPotion> targetPotions = new HashMap<Integer, TrackedPotion>();
    private final Set<Integer> processedTargetPotionEntityIds = new HashSet<Integer>();

    public TargetHpTracker(Minecraft mc, ItemsHandler itemsHandler, EventSink eventSink) {
        this.mc = mc;
        this.eventSink = eventSink;
        this.itemsHandler = itemsHandler;
    }

    public void reset() {
        trackedTargetId = -1;
        estimatedTargetHp = TARGET_HP_DEFAULT;
        lastEstimatedTargetDamage = 0.0F;
        lastEstimatedTargetHeal = 0.0F;
        naturalRegenTimer = 0;
        targetArmorPoints = DEFAULT_TARGET_ARMOR_POINTS;
        targetProtectionEpf = calculateDefaultProtectionEpf();
        targetFeatherFallingLevel = DEFAULT_TARGET_FEATHER_FALLING_LEVEL;
        targetPotions.clear();
        processedTargetPotionEntityIds.clear();
    }

    public void updateTargetState(EntityPlayer target, int currentTick) {
        int currentTargetId = target == null ? -1 : target.getEntityId();
        if (currentTargetId == trackedTargetId) {
            updateTargetArmorState(target);
            return;
        }

        int previousTargetId = trackedTargetId;
        trackedTargetId = currentTargetId;
        estimatedTargetHp = TARGET_HP_DEFAULT;
        lastEstimatedTargetDamage = 0.0F;
        lastEstimatedTargetHeal = 0.0F;
        naturalRegenTimer = 0;
        targetPotions.clear();
        processedTargetPotionEntityIds.clear();
        updateTargetArmorState(target);

        writeEvent("TARGET_CHANGED", detailsBuilder()
                .field("previousTargetId", previousTargetId < 0 ? null : previousTargetId)
                .field("targetEntityId", currentTargetId < 0 ? null : currentTargetId)
                .field("estimatedTargetHp", currentTargetId < 0 ? null : estimatedTargetHp)
                .field("tick", currentTick));
    }

    public float getEstimatedTargetHp() {
        return estimatedTargetHp;
    }

    public float getLastEstimatedTargetDamage() {
        return lastEstimatedTargetDamage;
    }

    public float getLastEstimatedTargetHeal() {
        return lastEstimatedTargetHeal;
    }

    public float getTargetHpDefault() {
        return TARGET_HP_DEFAULT;
    }

    public int getNaturalRegenTimer() {
        return naturalRegenTimer;
    }

    public int getTargetProtectionEpf() {
        return targetProtectionEpf;
    }

    public int getTargetFeatherFallingLevel() {
        return targetFeatherFallingLevel;
    }

    public int getTargetPearlDamageEpf() {
        return calculatePearlDamageEpf();
    }

    public void applyEstimatedTargetDamage(int currentTick, EntityPlayer target, ItemStack attackerHeldItem, boolean criticalHit) {
        updateTargetArmorState(target);
        int baseDamage = getSwordBaseDamage(itemsHandler.heldSwordType(attackerHeldItem));
        int sharpnessLevel = getSharpnessLevel(attackerHeldItem);
        float damage = calculateEstimatedSwordDamage(baseDamage, sharpnessLevel, criticalHit);
        estimatedTargetHp = clampFloat(estimatedTargetHp - damage, 0.0F, TARGET_HP_DEFAULT);
        lastEstimatedTargetDamage = damage;
        lastEstimatedTargetHeal = 0.0F;
        writeEvent("TARGET_ESTIMATED_DAMAGE", detailsBuilder()
                .field("targetEntityId", trackedTargetId < 0 ? null : trackedTargetId)
                .field("damageHp", damage)
                .field("baseDamage", baseDamage)
                .field("sharpnessLevel", sharpnessLevel)
                .field("criticalHit", criticalHit)
                .field("critMultiplier", criticalHit ? CRIT_MULTIPLIER : 1.0F)
                .field("targetArmorPoints", targetArmorPoints)
                .field("targetProtectionEpf", targetProtectionEpf)
                .field("estimatedTargetHp", estimatedTargetHp)
                .field("tick", currentTick));
    }

    public float calculateEstimatedSwordDamage(int baseDamage, int sharpnessLevel, boolean criticalHit) {
        float critMultiplier = criticalHit ? CRIT_MULTIPLIER : 1.0F;
        float preArmorDamage = baseDamage * critMultiplier + 1.25F * sharpnessLevel;
        float armorMultiplier = (25.0F - targetArmorPoints) / 25.0F;
        float epf = Math.min(targetProtectionEpf, 20.0F);
        float protectionMultiplier = (25.0F - epf) / 25.0F;

        return preArmorDamage * armorMultiplier * protectionMultiplier;
    }

    public void applyEstimatedTargetPearlDamage(int currentTick, EntityPlayer target, int pearlEntityId) {
        updateTargetArmorState(target);
        int totalEpf = getTargetPearlDamageEpf();
        float damage = calculateEstimatedPearlDamage(totalEpf);
        estimatedTargetHp = clampFloat(estimatedTargetHp - damage, 0.0F, TARGET_HP_DEFAULT);
        lastEstimatedTargetDamage = damage;
        lastEstimatedTargetHeal = 0.0F;
        writeEvent("TARGET_ESTIMATED_PEARL_DAMAGE", detailsBuilder()
                .field("targetEntityId", target == null ? null : target.getEntityId())
                .field("pearlEntityId", pearlEntityId)
                .field("damageHp", damage)
                .field("targetProtectionEpf", targetProtectionEpf)
                .field("totalEpf", totalEpf)
                .field("estimatedTargetHp", estimatedTargetHp)
                .field("tick", currentTick));
    }

    public float calculateEstimatedPearlDamage(int totalEpf) {
        float clampedEpf = Math.min(totalEpf, 20);
        return PEARL_BASE_DAMAGE_HP * (25.0F - clampedEpf) / 25.0F;
    }

    public boolean updateNaturalRegen(int currentTick, int targetFoodLevel) {
        if (targetFoodLevel >= 18 && shouldHeal()) {
            naturalRegenTimer++;
            if (naturalRegenTimer >= NATURAL_REGEN_INTERVAL_TICKS) {
                estimatedTargetHp = clampFloat(estimatedTargetHp + NATURAL_REGEN_HEAL_HP, 0.0F, TARGET_HP_DEFAULT);
                lastEstimatedTargetHeal = NATURAL_REGEN_HEAL_HP;
                lastEstimatedTargetDamage = 0.0F;
                naturalRegenTimer = 0;
                writeEvent("TARGET_ESTIMATED_NATURAL_REGEN", detailsBuilder()
                        .field("targetEntityId", trackedTargetId < 0 ? null : trackedTargetId)
                        .field("healHp", NATURAL_REGEN_HEAL_HP)
                        .field("estimatedTargetHp", estimatedTargetHp)
                        .field("targetFoodLevel", targetFoodLevel)
                        .field("tick", currentTick));
                return true;
            }
        } else {
            naturalRegenTimer = 0;
        }

        return false;
    }

    public void updateTargetPotionState(EntityPlayer target, int currentTick) {
        if (target == null || mc.theWorld == null) {
            targetPotions.clear();
            return;
        }

        Set<Integer> visiblePotionIds = new HashSet<Integer>();

        for (Object entityObject : mc.theWorld.loadedEntityList) {
            if (!(entityObject instanceof EntityPotion)) continue;

            EntityPotion potion = (EntityPotion) entityObject;
            int potionId = potion.getEntityId();
            if (processedTargetPotionEntityIds.contains(potionId)) continue;
            if (!isTargetHealingSplashPotion(potion, target)) continue;

            visiblePotionIds.add(potionId);
            TrackedPotion tracked = targetPotions.get(potionId);
            if (tracked == null) {
                tracked = new TrackedPotion(potionId, currentTick);
                targetPotions.put(potionId, tracked);
            }
            tracked.update(potion.posX, potion.posY, potion.posZ, potion.ticksExisted);
        }

        Iterator<Map.Entry<Integer, TrackedPotion>> iterator = targetPotions.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Integer, TrackedPotion> entry = iterator.next();
            if (visiblePotionIds.contains(entry.getKey())) continue;

            processedTargetPotionEntityIds.add(entry.getKey());
            iterator.remove();
        }
    }

    public void handlePotionImpactEffect(EntityPlayer target, int effectType, int potionDamage, int packetX, int packetY, int packetZ, int currentTick) {
        if (target == null || effectType != 2002 || !isInstantHealthIISplashPotion(potionDamage)) return;

        TrackedPotion tracked = findTrackedPotionForImpact(packetX, packetY, packetZ);
        if (tracked == null) {
            writeEvent("TARGET_POTION_IMPACT_UNMATCHED", detailsBuilder()
                    .field("targetEntityId", target.getEntityId())
                    .field("effectType", effectType)
                    .field("potionDamage", potionDamage)
                    .field("impactX", packetX)
                    .field("impactY", packetY)
                    .field("impactZ", packetZ)
                    .field("tick", currentTick));
            return;
        }

        tracked.markImpact(packetX, packetY, packetZ, currentTick);
        applyEstimatedTargetPotionHeal(target, tracked, currentTick);
        processedTargetPotionEntityIds.add(tracked.entityId);
        targetPotions.remove(tracked.entityId);
    }

    public void handleDestroyedEntities(int[] entityIds, int currentTick) {
        if (entityIds == null) return;

        for (int entityId : entityIds) {
            TrackedPotion tracked = targetPotions.remove(entityId);
            if (tracked == null) continue;

            processedTargetPotionEntityIds.add(entityId);
            writeEvent("TARGET_POTION_DESTROY_NO_IMPACT", detailsBuilder()
                    .field("potionEntityId", entityId)
                    .field("lastVisibleX", tracked.lastX)
                    .field("lastVisibleY", tracked.lastY)
                    .field("lastVisibleZ", tracked.lastZ)
                    .field("tick", currentTick));
        }
    }

    private boolean isTargetHealingSplashPotion(EntityPotion potion, EntityPlayer target) {
        if (!isInstantHealthIISplashPotion(potion)) return false;

        Entity thrower = potion.getThrower();
        if (thrower == mc.thePlayer) return false;
        if (thrower == target) return true;
        if (thrower != null) return false;

        return target.getDistanceToEntity(potion) <= 4.0F;
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

    public void applyEstimatedTargetPotionHeal(EntityPlayer target, TrackedPotion potion, int currentTick) {
        double impactX = potion.impactDetected ? potion.impactX : potion.lastX;
        double impactY = potion.impactDetected ? potion.impactY : potion.lastY;
        double impactZ = potion.impactDetected ? potion.impactZ : potion.lastZ;
        double distanceSq = target.getDistanceSq(impactX, impactY, impactZ);
        float effectiveness = 0.0F;
        boolean directHitEstimated = isImpactCenterInTargetHitBox(target, impactX, impactY, impactZ);

        if (directHitEstimated) {
            effectiveness = 1.0F;
        } else if (distanceSq < 16.0D) {
            effectiveness = clampFloat(1.0F - ((float) Math.sqrt(distanceSq) / 4.0F), 0.0F, 1.0F);
        }

        float healHp = (float) Math.floor(INSTANT_HEALTH_II_BASE_HEAL_HP * effectiveness + 0.5F);
        if (healHp <= 0.0F) return;

        estimatedTargetHp = clampFloat(estimatedTargetHp + healHp, 0.0F, TARGET_HP_DEFAULT);
        lastEstimatedTargetHeal = healHp;
        lastEstimatedTargetDamage = 0.0F;
        writeEvent("TARGET_ESTIMATED_HEAL", detailsBuilder()
                .field("targetEntityId", target.getEntityId())
                .field("potionEntityId", potion.entityId)
                .field("healHp", healHp)
                .field("effectiveness", effectiveness)
                .field("directHitEstimated", directHitEstimated)
                .field("impactDetected", potion.impactDetected)
                .field("impactTick", potion.impactTick < 0 ? null : potion.impactTick)
                .field("impactDistanceSq", distanceSq)
                .field("impactX", impactX)
                .field("impactY", impactY)
                .field("impactZ", impactZ)
                .field("lastVisibleX", potion.lastX)
                .field("lastVisibleY", potion.lastY)
                .field("lastVisibleZ", potion.lastZ)
                .field("estimatedTargetHp", estimatedTargetHp)
                .field("tick", currentTick));
    }

    private boolean isImpactCenterInTargetHitBox(EntityPlayer target, double impactX, double impactY, double impactZ) {
        if (target == null || target.boundingBox == null) return false;

        AxisAlignedBB hitBox = target.boundingBox.expand(
                POTION_DIRECT_HIT_BOX_EXPAND,
                POTION_DIRECT_HIT_BOX_EXPAND,
                POTION_DIRECT_HIT_BOX_EXPAND);

        AxisAlignedBB potionHitbox = AxisAlignedBB.getBoundingBox(impactX, impactY, impactZ,
                impactX + 1.0D, impactY + 1.0D, impactZ + 1.0D);

        boolean intersects = hitBox.intersectsWith(potionHitbox);
        return intersects;
    }

    private TrackedPotion findTrackedPotionForImpact(int impactX, int impactY, int impactZ) {
        TrackedPotion best = null;
        double bestDistanceSq = Double.MAX_VALUE;

        for (TrackedPotion potion : targetPotions.values()) {
            double dx = potion.lastX - impactX;
            double dy = potion.lastY - impactY;
            double dz = potion.lastZ - impactZ;
            double distanceSq = dx * dx + dy * dy + dz * dz;
            if (distanceSq < bestDistanceSq) {
                bestDistanceSq = distanceSq;
                best = potion;
            }
        }

        return bestDistanceSq <= 16.0D ? best : null;
    }

    private int getTargetArmor(EntityPlayer target) {
        if (target == null) return 0;
        return target.getTotalArmorValue();
    }

    private void updateTargetArmorState(EntityPlayer target) {
        if (target == null) {
            targetArmorPoints = DEFAULT_TARGET_ARMOR_POINTS;
            targetProtectionEpf = calculateDefaultProtectionEpf();
            targetFeatherFallingLevel = DEFAULT_TARGET_FEATHER_FALLING_LEVEL;
            return;
        }

        targetArmorPoints = getTargetArmor(target);
        targetProtectionEpf = getProtectionEpf(target);
        targetFeatherFallingLevel = getFeatherFallingLevel(target);
    }

    private int getFeatherFallingLevel(EntityPlayer target) {
        if (target == null) return 0;

        ItemStack boots = target.getCurrentArmor(0);
        if (boots == null) return 0;

        return EnchantmentHelper.getEnchantmentLevel(Enchantment.featherFalling.effectId, boots);
    }

    private int getProtectionEpf(EntityPlayer target) {
        if (target == null) return calculateDefaultProtectionEpf();

        int protectionEpf = 0;

        for (int slot = 0; slot < 4; slot++) {
            ItemStack armor = target.getCurrentArmor(slot);
            if (armor == null) continue;

            int protectionLevel = EnchantmentHelper.getEnchantmentLevel(Enchantment.protection.effectId, armor);
            protectionEpf += calculateProtectionEpf(protectionLevel);
        }

        return protectionEpf;
    }

    private int calculateProtectionEpf(int protectionLevel) {
        if (protectionLevel <= 0) return 0;

        return MathHelper.floor_float(((6.0F + protectionLevel * protectionLevel) / 3.0F) * 0.75F);
    }

    private int calculateDefaultProtectionEpf() {
        return calculateProtectionEpf(DEFAULT_TARGET_PROTECTION_LEVEL_PER_PIECE) * 4;
    }

    private int calculatePearlDamageEpf() {
        int featherFallingEpf = MathHelper.floor_float(((6.0F + targetFeatherFallingLevel * targetFeatherFallingLevel) / 3.0F) * 2.5F);
        return targetProtectionEpf + featherFallingEpf;
    }

    private int getSwordBaseDamage(String swordType) {
        if ("diamond".equals(swordType)) {
            return 7;
        } else if ("iron".equals(swordType)) {
            return 6;
        } else if ("stone".equals(swordType)) {
            return 5;
        } else if ("gold".equals(swordType) || "wood".equals(swordType)) {
            return 4;
        }

        return 1;
    }

    private int getSharpnessLevel(ItemStack stack) {
        if (stack == null) return DEFAULT_SHARPNESS_LEVEL;
        if (!itemsHandler.isSword(stack)) return 0;

        return EnchantmentHelper.getEnchantmentLevel(Enchantment.sharpness.effectId, stack);
    }

    private float clampFloat(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private boolean shouldHeal() {
        return estimatedTargetHp > 0.0F && estimatedTargetHp < TARGET_HP_DEFAULT;
    }

    private JsonLineBuilder detailsBuilder() {
        return new JsonLineBuilder(512);
    }

    private void writeEvent(String type, JsonLineBuilder details) {
        if (eventSink != null) {
            eventSink.writeEvent(type, details);
        }
    }

}
