package com.bot.recorder.features;

public class CombatHistoryTrackerTest {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }
    private static void near(double value, double expected) {
        check(Math.abs(value - expected) < 1E-9, "expected " + expected + ", got " + value);
    }
    public static void main(String[] args) {
        CombatHistoryTracker tracker = new CombatHistoryTracker(0.95, 60);
        tracker.advance(0);
        check(tracker.getTicksSinceCombat(0) == null, "no combat is unknown");
        tracker.landedHit(0);
        tracker.tookHit(0);
        near(tracker.getMyPressure(), 1);
        near(tracker.getTargetPressure(), 1);
        check(tracker.getNetHits() == 0 && tracker.getHitsLanded() == 1, "equal trade is not inactivity");
        tracker.advance(10);
        near(tracker.getMyPressure(), Math.pow(0.95, 10));
        check(tracker.getTicksSinceCombat(10) == 10, "combat recency");
        tracker.landedHit(10);
        near(tracker.getMyPressure(), 1 + Math.pow(0.95, 10));
        tracker.targetHealed(10);
        near(tracker.getMyPressure(), 0);
        near(tracker.getTargetPressure(), Math.pow(0.95, 10));
        check(tracker.getHitsLanded() == 2, "healing preserves hit history");
        check(tracker.consumeTargetPotting() && !tracker.consumeTargetPotting(), "one frame per splash pulse");
        tracker.landedHit(11);
        tracker.selfHealed(11);
        near(tracker.getTargetPressure(), 0);
        near(tracker.getMyPressure(), 1);
        tracker.advance(59);
        check(tracker.getHitsLanded() == 3 && tracker.getHitsTaken() == 1, "oldest hit still in window");
        tracker.advance(60);
        check(tracker.getHitsLanded() == 2 && tracker.getHitsTaken() == 0, "exclusive lower window boundary");
        tracker.advance(71);
        check(tracker.getHitsLanded() == 0 && tracker.getTicksSinceCombat(71) == 60, "history expiry preserves last combat");
        tracker.targetHealed(72);
        tracker.reset();
        check(!tracker.consumeTargetPotting() && tracker.getTicksSinceCombat(0) == null, "round reset clears events");
        near(tracker.getMyPressure(), 0);
        CombatHistoryTracker custom = new CombatHistoryTracker(0.5, 80);
        custom.landedHit(0); custom.advance(1); near(custom.getMyPressure(), 0.5);
        custom.advance(79); check(custom.getHitsLanded() == 1, "custom window");
        custom.advance(80); check(custom.getHitsLanded() == 0, "custom window expiry");
        System.out.println("Combat history scenarios passed");
    }
}
