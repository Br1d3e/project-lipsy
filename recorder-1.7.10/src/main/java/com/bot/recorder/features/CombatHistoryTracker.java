package com.bot.recorder.features;

import java.util.ArrayDeque;
import java.util.Deque;

/** Round-scoped event history. Healing resets pressure, not the rolling hit history. */
public class CombatHistoryTracker {
    private final double decay;
    private final int windowTicks;
    private final Deque<Integer> landed = new ArrayDeque<Integer>();
    private final Deque<Integer> taken = new ArrayDeque<Integer>();
    private double myPressure;
    private double targetPressure;
    private int lastTick = -1;
    private int lastCombatTick = -1;
    private boolean targetPotting;

    public CombatHistoryTracker(double decay, int windowTicks) {
        if (Double.isNaN(decay) || decay < 0 || decay > 1 || windowTicks < 1) {
            throw new IllegalArgumentException("Invalid combat history configuration");
        }
        this.decay = decay;
        this.windowTicks = windowTicks;
    }

    public void reset() {
        landed.clear();
        taken.clear();
        myPressure = targetPressure = 0;
        lastTick = lastCombatTick = -1;
        targetPotting = false;
    }

    public void advance(int tick) {
        if (lastTick >= 0 && tick > lastTick) {
            double factor = Math.pow(decay, tick - lastTick);
            myPressure *= factor;
            targetPressure *= factor;
        }
        lastTick = tick;
        expire(landed, tick);
        expire(taken, tick);
    }

    private void expire(Deque<Integer> hits, int tick) {
        while (!hits.isEmpty() && hits.peekFirst() <= tick - windowTicks) hits.removeFirst();
    }

    public void landedHit(int tick) {
        advance(tick);
        landed.addLast(tick);
        myPressure += 1;
        lastCombatTick = tick;
    }

    public void tookHit(int tick) {
        advance(tick);
        taken.addLast(tick);
        targetPressure += 1;
        lastCombatTick = tick;
    }

    public void targetHealed(int tick) {
        advance(tick);
        myPressure = 0;
        targetPotting = true;
    }

    public void selfHealed(int tick) {
        advance(tick);
        targetPressure = 0;
    }

    public double getMyPressure() { return myPressure; }
    public double getTargetPressure() { return targetPressure; }
    public int getHitsLanded() { return landed.size(); }
    public int getHitsTaken() { return taken.size(); }
    public int getNetHits() { return landed.size() - taken.size(); }
    public Integer getTicksSinceCombat(int tick) { return lastCombatTick < 0 ? null : tick - lastCombatTick; }

    /** Consume once per recorded frame so an impact between ticks is not lost. */
    public boolean consumeTargetPotting() {
        boolean value = targetPotting;
        targetPotting = false;
        return value;
    }
}
