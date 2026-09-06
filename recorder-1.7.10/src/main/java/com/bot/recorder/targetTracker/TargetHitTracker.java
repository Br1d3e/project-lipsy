package com.bot.recorder.targetTracker;

/** Counts inferred melee hits since an observed heal or healing-potion throw. */
public class TargetHitTracker {
    // Client damage notifications have no attacker; correlate with our attack packet.
    private static final int ATTACK_CONFIRMATION_WINDOW_TICKS = 5;
    private int targetId = -1;
    private int hits = 0;
    private int pendingAttackTick = -1;

    public synchronized void reset() {
        targetId = -1;
        hits = 0;
        pendingAttackTick = -1;
    }

    public synchronized void updateTarget(int entityId) {
        if (entityId < 0) {
            pendingAttackTick = -1;
            return;
        }
        if (entityId == targetId) return;
        targetId = entityId;
        pendingAttackTick = -1;
    }

    public synchronized void recordAttack(int entityId, int tick) {
        if (entityId >= 0 && entityId == targetId) pendingAttackTick = tick;
    }

    public synchronized boolean recordHurt(int entityId, int tick) {
        if (entityId < 0 || entityId != targetId || pendingAttackTick < 0) return false;
        int elapsed = tick - pendingAttackTick;
        if (elapsed < 0) return false;
        pendingAttackTick = -1;
        if (elapsed > ATTACK_CONFIRMATION_WINDOW_TICKS) return false;
        hits++;
        return true;
    }

    public synchronized void recordHeal(int entityId) {
        if (entityId < 0 || entityId != targetId) return;
        hits = 0;
        pendingAttackTick = -1;
    }

    public synchronized Integer getHits() {
        return targetId < 0 ? null : hits;
    }
}
