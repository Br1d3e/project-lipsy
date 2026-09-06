package com.bot.recorder.features;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Samples observed hit cycles and throw intervals, never future or unfinished intervals. */
public class TargetPlaystyleTracker {
    public static final int HEAL_EPISODE_GAP_TICKS = 20;
    private int hits;
    private boolean completeHitCycle = true;
    private int lastHealThrowTick = -1;
    private int lastPearlThrowTick = -1;
    private final Samples healHits = new Samples();
    private final Samples pearlGaps = new Samples();

    public void reset() {
        hits = 0;
        completeHitCycle = true;
        lastHealThrowTick = lastPearlThrowTick = -1;
        healHits.clear();
        pearlGaps.clear();
    }

    public void observationLost() {
        hits = 0;
        completeHitCycle = false;
        lastHealThrowTick = lastPearlThrowTick = -1;
    }

    public void landedHit() { hits++; }

    /** Null means another throw in the same episode. A null sample means incomplete observation. */
    public HealEpisode healingPotionThrown(int tick) {
        boolean newEpisode = lastHealThrowTick < 0 || tick - lastHealThrowTick > HEAL_EPISODE_GAP_TICKS;
        Integer sample = completeHitCycle ? hits : null;
        lastHealThrowTick = tick;
        hits = 0;
        completeHitCycle = true;
        if (!newEpisode) return null;
        if (sample != null) healHits.add(sample);
        return new HealEpisode(sample);
    }

    /** Returns only completed positive intervals, including cooldown time. */
    public Integer pearlThrown(int tick) {
        Integer gap = lastPearlThrowTick >= 0 && tick > lastPearlThrowTick ? tick - lastPearlThrowTick : null;
        lastPearlThrowTick = tick;
        if (gap != null) pearlGaps.add(gap);
        return gap;
    }

    public int getHealSampleCount() { return healHits.size(); }
    public Double getMeanHealHits() { return healHits.mean(); }
    public Double getMedianHealHits() { return healHits.median(); }
    public int getPearlSampleCount() { return pearlGaps.size(); }
    public Double getMeanPearlGap() { return pearlGaps.mean(); }
    public Double getMedianPearlGap() { return pearlGaps.median(); }

    public static class HealEpisode {
        public final Integer hitsBeforeHealing;
        private HealEpisode(Integer hitsBeforeHealing) { this.hitsBeforeHealing = hitsBeforeHealing; }
    }

    private static class Samples {
        private final List<Integer> values = new ArrayList<Integer>();
        private long sum;
        void add(int value) {
            int index = Collections.binarySearch(values, value);
            values.add(index < 0 ? -index - 1 : index, value);
            sum += value;
        }
        void clear() { values.clear(); sum = 0; }
        int size() { return values.size(); }
        Double mean() { return values.isEmpty() ? null : (double) sum / values.size(); }
        Double median() {
            if (values.isEmpty()) return null;
            int mid = values.size() / 2;
            return values.size() % 2 == 1 ? (double) values.get(mid)
                    : ((double) values.get(mid - 1) + values.get(mid)) / 2.0D;
        }
    }
}
