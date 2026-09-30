package com.bot.projectlipsy.features;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.network.play.client.C02PacketUseEntity;
import net.minecraft.network.play.server.S13PacketDestroyEntities;
import net.minecraft.network.play.server.S28PacketEffect;
import net.minecraft.world.World;
import java.util.UUID;

/** Client-tick feature collection, independent of model socket availability. */
public final class HLFeatureTracker {
    private static final double HIT_PRESSURE_DECAY = 0.95F;
    private static final int COMBAT_HISTORY_TICKS = 80;
    private static final int PEARL_COOLDOWN_TICKS = 320;
    public final CombatHistoryTracker combat;
    public final TargetMovementTracker movement = new TargetMovementTracker();
    private final TargetInventoryTracker inventory;
    private final PearlCooldownTracker enemyPearlCooldown = new PearlCooldownTracker();
    private Integer lastEnemyPearlTick;
    private final TargetHitTracker hits = new TargetHitTracker();
    private final TargetHealingObserver healing;
    private final FeaturePacketObserver packets;
    private final Minecraft mc;
    private World world;
    private UUID opponent;
    private int observedTargetId = -1;
    private int lastTargetHurt, lastSelfHurt;
    private Float previousHp;
    private String state = "attack";
    private int stateStartTick = -1;
    private boolean targetPotting;
    private boolean targetPearling;

    public HLFeatureTracker(Minecraft mc) {
        this.mc = mc;
        inventory = new TargetInventoryTracker(mc);
        combat = new CombatHistoryTracker(HIT_PRESSURE_DECAY, COMBAT_HISTORY_TICKS);
        healing = new TargetHealingObserver(mc);
        packets = new FeaturePacketObserver(mc);
    }

    public void setDecisionState(String next, int tick) {
        if (!next.equals(state)) { state = next; stateStartTick = tick + 1; }
    }

    public int getTicksInCurrentState(int tick) { return Math.max(0, tick - stateStartTick); }
    public int getHitsBeforeTargetHeals() { Integer value = hits.getHits(); return value == null ? 0 : value; }
    public boolean isTargetPotting() { return targetPotting; }
    public boolean isTargetPearling() { return targetPearling; }
    public int getTargetEstimatedPotions() { return inventory.getTargetHealthPotions(); }
    public int getTargetEstimatedPearls() { return inventory.getTargetPearls(); }
    public int getEnemyPearlCooldown() { return enemyPearlCooldown.getRemaining(); }
    public Integer getTicksSinceLastEnemyPearl(int tick) {
        return lastEnemyPearlTick == null ? null : tick - lastEnemyPearlTick;
    }
    public boolean holdingFood(EntityPlayer target) { return TargetHealingObserver.holdingFood(target.getHeldItem()); }
    public boolean holdingSpeed(EntityPlayer target) { return TargetHealingObserver.holdingSpeed(target.getHeldItem()); }
    public boolean holdingHeal(EntityPlayer target) { return TargetHealingObserver.holdingHeal(target.getHeldItem()); }

    /** A missing/invisible target ends the current target history. */
    public boolean update(EntityPlayer target, int tick) {
        if (target != null && target.isInvisible()) target = null;
        boolean reset = world != mc.theWorld
                || (opponent != null && (target == null || mc.thePlayer == null));
        if (target != null && opponent != null && !opponent.equals(target.getUniqueID())) reset = true;
        if (reset) resetRound(tick);
        world = mc.theWorld;
        packets.install(tick);
        if (stateStartTick < 0) stateStartTick = tick;
        combat.advance(tick);
        if (mc.thePlayer != null) {
            float hp = mc.thePlayer.getHealth();
            if (previousHp != null && hp > previousHp) combat.selfHealed(tick);
            previousHp = hp;
        } else previousHp = null;
        targetPotting = false;
        targetPearling = false;
        if (target == null || mc.thePlayer == null) {
            movement.reset();
            healing.observationLost();
            hits.updateTarget(-1);
            packets.clear();
            observedTargetId = -1;
            return reset;
        }
        opponent = target.getUniqueID();
        inventory.trackHealthPotions(target);
        updateEnemyPearlState(target, tick);
        hits.updateTarget(target.getEntityId());
        if (observedTargetId != target.getEntityId()) {
            lastTargetHurt = target.hurtTime;
            lastSelfHurt = mc.thePlayer.hurtTime;
            observedTargetId = target.getEntityId();
        }
        movement.update(tick, mc.thePlayer.posX, mc.thePlayer.posZ, target.posX, target.posZ, target.rotationYaw);
        healing.rememberTarget(target);
        FeaturePacketObserver.ObservedPacket event;
        while ((event = packets.poll()) != null) {
            if (event.packet instanceof C02PacketUseEntity) {
                C02PacketUseEntity packet = (C02PacketUseEntity) event.packet;
                Entity attacked = packet.getEntityFromWorld(mc.theWorld);
                if (packet.getAction() == C02PacketUseEntity.Action.ATTACK && attacked != null) {
                    hits.recordAttack(attacked.getEntityId(), event.tick);
                }
            } else if (event.packet instanceof S28PacketEffect) {
                S28PacketEffect packet = (S28PacketEffect) event.packet;
                if (healing.impact(target, packet.getSoundType(), packet.getSoundData(),
                        packet.getSoundPos().getX(), packet.getSoundPos().getY(), packet.getSoundPos().getZ())) {
                    hits.recordHeal(target.getEntityId());
                    combat.targetHealed(tick);
                }
            } else if (event.packet instanceof S13PacketDestroyEntities) {
                healing.destroyed(((S13PacketDestroyEntities) event.packet).getEntityIDs());
            }
        }
        if (target.hurtTime > lastTargetHurt && hits.recordHurt(target.getEntityId(), tick)) combat.landedHit(tick);
        if (mc.thePlayer.hurtTime > lastSelfHurt) combat.tookHit(tick);
        lastTargetHurt = target.hurtTime;
        lastSelfHurt = mc.thePlayer.hurtTime;
        if (healing.scan(target)) hits.recordHeal(target.getEntityId());
        targetPotting = combat.consumeTargetPotting();
        return reset;
    }

    private void updateEnemyPearlState(EntityPlayer target, int tick) {
        // Inventory tracking uses the recorder's thrower/proximity attribution and entity-ID deduplication.
        targetPearling = inventory.trackPearls(target);
        enemyPearlCooldown.updateEnemy(targetPearling, PEARL_COOLDOWN_TICKS);
        if (targetPearling) lastEnemyPearlTick = tick;
    }

    private void resetRound(int tick) {
        combat.reset(); movement.reset(); hits.reset(); healing.reset(); packets.clear();
        inventory.reset();
        enemyPearlCooldown.reset(-1);
        lastEnemyPearlTick = null;
        targetPearling = targetPotting = false;
        opponent = null;
        observedTargetId = -1;
        previousHp = null;
        state = "attack";
        stateStartTick = tick;
    }
}
