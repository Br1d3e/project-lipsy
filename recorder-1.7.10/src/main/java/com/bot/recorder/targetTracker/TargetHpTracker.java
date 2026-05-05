package com.bot.recorder.targetTracker;

import com.bot.recorder.recordWriter.JsonLineBuilder;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.entity.projectile.EntityPotion;
import net.minecraft.init.Items;
import net.minecraft.item.ItemPotion;
import net.minecraft.potion.Potion;
import net.minecraft.potion.PotionEffect;
import net.minecraft.util.AxisAlignedBB;
import net.minecraft.util.Vec3;

import java.util.*;

public class TargetHpTracker {
    public interface EventSink {
        void writeEvent(String type, JsonLineBuilder details);
    }

    private final Minecraft mc;
    private final EventSink eventSink;
    private static final float TARGET_HP_DEFAULT = 20.0F;
    private static final float SWORD_BASE_DAMAGE = 7.0F;
    private static final float CRIT_MULTIPLIER = 1.5F;
    private static final int SHARPNESS_LEVEL = 4;
    private static final int ARMOR_POINTS = 20;
    private static final int PROTECTION_LEVEL = 2;
    private static final float INSTANT_HEALTH_II_BASE_HEAL_HP = 8.0F;
    private static final double POTION_DIRECT_HIT_BOX_EXPAND = 0.1D;
    private static final int NATURAL_REGEN_INTERVAL_TICKS = 80;
    private static final float NATURAL_REGEN_HEAL_HP = 1.0F;
    private int trackedTargetId = -1;
    private float estimatedTargetHp = TARGET_HP_DEFAULT;
    private float lastEstimatedTargetDamage = 0.0F;
    private float lastEstimatedTargetHeal = 0.0F;
    private int naturalRegenTimer = 0;
    private final Map<Integer, TrackedPotion> targetPotions = new HashMap<Integer, TrackedPotion>();
    private final Set<Integer> processedTargetPotionEntityIds = new HashSet<Integer>();

    public TargetHpTracker(Minecraft mc, EventSink eventSink) {
        this.mc = mc;
        this.eventSink = eventSink;
    }

    public void reset() {
        trackedTargetId = -1;
        estimatedTargetHp = TARGET_HP_DEFAULT;
        lastEstimatedTargetDamage = 0.0F;
        lastEstimatedTargetHeal = 0.0F;
        naturalRegenTimer = 0;
        targetPotions.clear();
        processedTargetPotionEntityIds.clear();
    }

    public void updateTargetState(EntityPlayer target, int currentTick) {
        int currentTargetId = target == null ? -1 : target.getEntityId();
        if (currentTargetId == trackedTargetId) return;

        int previousTargetId = trackedTargetId;
        trackedTargetId = currentTargetId;
        estimatedTargetHp = TARGET_HP_DEFAULT;
        lastEstimatedTargetDamage = 0.0F;
        lastEstimatedTargetHeal = 0.0F;
        naturalRegenTimer = 0;
        targetPotions.clear();
        processedTargetPotionEntityIds.clear();

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

    public void applyEstimatedTargetDamage(int currentTick) {
        float damage = calculateEstimatedSwordDamage();
        estimatedTargetHp = clampFloat(estimatedTargetHp - damage, 0.0F, TARGET_HP_DEFAULT);
        lastEstimatedTargetDamage = damage;
        lastEstimatedTargetHeal = 0.0F;
        writeEvent("TARGET_ESTIMATED_DAMAGE", detailsBuilder()
                .field("targetEntityId", trackedTargetId < 0 ? null : trackedTargetId)
                .field("damageHp", damage)
                .field("estimatedTargetHp", estimatedTargetHp)
                .field("tick", currentTick));
    }

    public float calculateEstimatedSwordDamage() {
        float preArmorDamage = SWORD_BASE_DAMAGE * CRIT_MULTIPLIER + 1.25F * SHARPNESS_LEVEL;
        float armorMultiplier = (25.0F - ARMOR_POINTS) / 25.0F;
        float epf = Math.min(PROTECTION_LEVEL * 4.0F, 20.0F);
        float protectionMultiplier = (25.0F - epf) / 25.0F;

        return preArmorDamage * armorMultiplier * protectionMultiplier;
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

            applyEstimatedTargetPotionHeal(target, entry.getValue(), currentTick);
            processedTargetPotionEntityIds.add(entry.getKey());
            iterator.remove();
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
        int potionDamage = potion.getPotionDamage();
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
        double distanceSq = target.getDistanceSq(potion.lastX, potion.lastY, potion.lastZ);
        float effectiveness = 0.0F;
        boolean directHitEstimated = isPotionDirectHit(target, potion);

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
                .field("impactDistanceSq", distanceSq)
                .field("segmentStartX", potion.previousX)
                .field("segmentStartY", potion.previousY)
                .field("segmentStartZ", potion.previousZ)
                .field("segmentEndX", potion.lastX)
                .field("segmentEndY", potion.lastY)
                .field("segmentEndZ", potion.lastZ)
                .field("estimatedTargetHp", estimatedTargetHp)
                .field("tick", currentTick));
    }

    private boolean isPotionDirectHit(EntityPlayer target, TrackedPotion potion) {
        if (target == null || target.boundingBox == null) return false;

        AxisAlignedBB hitBox = target.boundingBox.expand(
                POTION_DIRECT_HIT_BOX_EXPAND,
                POTION_DIRECT_HIT_BOX_EXPAND,
                POTION_DIRECT_HIT_BOX_EXPAND);
        Vec3 segmentStart = Vec3.createVectorHelper(potion.previousX, potion.previousY, potion.previousZ);
        Vec3 segmentEnd = Vec3.createVectorHelper(potion.lastX, potion.lastY, potion.lastZ);

        return hitBox.isVecInside(segmentStart)
                || hitBox.isVecInside(segmentEnd)
                || hitBox.calculateIntercept(segmentStart, segmentEnd) != null;
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
