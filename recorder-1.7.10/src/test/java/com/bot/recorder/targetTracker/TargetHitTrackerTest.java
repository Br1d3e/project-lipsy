package com.bot.recorder.targetTracker;

/** Standalone regression checks; no Minecraft client or test framework required. */
public class TargetHitTrackerTest {
    private static void expect(TargetHitTracker tracker, Integer expected, String scenario) {
        Integer actual = tracker.getHits();
        if (expected == null ? actual != null : !expected.equals(actual)) {
            throw new AssertionError(scenario + ": expected " + expected + ", got " + actual);
        }
    }

    public static void main(String[] args) {
        TargetHitTracker tracker = new TargetHitTracker();
        expect(tracker, null, "no target");
        tracker.updateTarget(7);
        expect(tracker, 0, "new target");
        tracker.recordHurt(7, 10);
        expect(tracker, 0, "unattributed damage");
        tracker.recordAttack(8, 11);
        tracker.recordHurt(7, 12);
        expect(tracker, 0, "attack on another player");
        tracker.recordAttack(7, 20);
        expect(tracker, 0, "attack attempt alone");
        tracker.recordHurt(7, 25);
        expect(tracker, 1, "hit at confirmation window boundary");
        tracker.recordHurt(7, 26);
        expect(tracker, 1, "one attack cannot count twice");
        tracker.recordAttack(7, 30);
        tracker.recordHurt(7, 36);
        expect(tracker, 1, "expired attack");
        tracker.recordAttack(7, 40);
        tracker.recordHurt(7, 41);
        tracker.updateTarget(7);
        expect(tracker, 2, "ongoing hits survive ordinary ticks");
        tracker.recordHeal(8);
        expect(tracker, 2, "another player healing");
        tracker.recordAttack(7, 42);
        tracker.recordHeal(7);
        tracker.recordHurt(7, 43);
        expect(tracker, 0, "heal clears count and pending attack");
        tracker.recordAttack(7, 50);
        tracker.recordHurt(7, 51);
        expect(tracker, 1, "new sequence after healing");
        tracker.recordAttack(7, 52);
        tracker.updateTarget(8);
        tracker.recordHurt(8, 53);
        expect(tracker, 1, "round opponent entity rebind preserves hits, clears pending attack");
        tracker.updateTarget(-1);
        tracker.recordAttack(-1, 60);
        tracker.recordHurt(-1, 61);
        expect(tracker, 1, "visibility loss preserves round hits");
        tracker.updateTarget(7);
        expect(tracker, 1, "reappearance preserves round hits");
        tracker.recordAttack(7, 70);
        tracker.reset();
        tracker.updateTarget(7);
        tracker.recordHurt(7, 71);
        expect(tracker, 0, "session reset");
        System.out.println("TargetHitTracker: 16 regression checks passed");
    }
}
