package com.bot.recorder.features;

public class TargetPlaystyleTrackerTest {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }
    public static void main(String[] args) {
        TargetPlaystyleTracker tracker = new TargetPlaystyleTracker();
        check(tracker.getMeanHealHits() == null && tracker.getMedianPearlGap() == null, "no observations is unknown");
        tracker.landedHit(); tracker.landedHit(); tracker.landedHit();
        check(tracker.healingPotionThrown(10).hitsBeforeHealing == 3, "capture pre-throw hits");
        check(tracker.getHealSampleCount() == 1 && tracker.getMeanHealHits() == 3, "first completed sample");
        tracker.landedHit();
        check(tracker.healingPotionThrown(30) == null, "20-tick burst boundary is same episode");
        check(tracker.getHealSampleCount() == 1, "burst does not add zero sample");
        tracker.landedHit();
        check(tracker.healingPotionThrown(51).hitsBeforeHealing == 1, "new episode after 21 ticks");
        check(tracker.getMeanHealHits() == 2 && tracker.getMedianHealHits() == 2, "even median and mean");
        check(tracker.healingPotionThrown(72).hitsBeforeHealing == 0, "genuine observed zero-hit episode retained");
        check(tracker.getMedianHealHits() == 1, "odd median");
        tracker.landedHit(); tracker.observationLost(); tracker.landedHit();
        check(tracker.healingPotionThrown(90).hitsBeforeHealing == null, "gap censors hit cycle");
        check(tracker.getHealSampleCount() == 3, "invalid cycle excluded from aggregate");
        tracker.landedHit(); tracker.landedHit();
        check(tracker.healingPotionThrown(111).hitsBeforeHealing == 2, "first observed throw restores baseline");
        check(tracker.getHealSampleCount() == 4, "valid post-gap sample");
        check(tracker.pearlThrown(100) == null, "first pearl is baseline only");
        check(tracker.pearlThrown(420) == 320, "interval includes cooldown");
        check(tracker.pearlThrown(420) == null, "same-tick detections have no positive interval");
        tracker.observationLost();
        check(tracker.pearlThrown(900) == null, "no interval across visibility gap");
        check(tracker.pearlThrown(1400) == 500, "fresh post-gap interval");
        check(tracker.getMeanPearlGap() == 410 && tracker.getMedianPearlGap() == 410, "pearl aggregates");
        check(tracker.getPearlSampleCount() == 2, "only completed intervals sampled");
        tracker.reset();
        check(tracker.getHealSampleCount() == 0 && tracker.getPearlSampleCount() == 0, "round clears samples");
        check(tracker.getMeanHealHits() == null && tracker.getMedianHealHits() == null, "round clears aggregates");
        check(tracker.pearlThrown(2000) == null, "round clears unfinished pearl interval");
        System.out.println("Target playstyle episode and interval scenarios passed");
    }
}
