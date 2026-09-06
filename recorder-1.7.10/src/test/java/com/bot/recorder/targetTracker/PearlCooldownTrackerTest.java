package com.bot.recorder.targetTracker;

public class PearlCooldownTrackerTest {
    private static void check(boolean condition, String scenario) {
        if (!condition) throw new AssertionError(scenario);
    }

    public static void main(String[] args) {
        PearlCooldownTracker tracker = new PearlCooldownTracker();
        tracker.reset(16);
        check(tracker.getRemaining() == 0, "round starts available");
        check(tracker.update(true, 16, 320), "use at zero starts cooldown");
        check(tracker.getRemaining() == 320, "full duration at throw");
        check(!tracker.update(true, 16, 320), "rejected use must not restart cooldown");
        check(tracker.getRemaining() == 319, "rejected use still counts down");
        check(tracker.update(false, 15, 320), "one consumed pearl confirms throw during cooldown");
        check(tracker.getRemaining() == 320, "consumption restarts full duration");
        check(!tracker.update(true, 15, 320), "same consumption cannot confirm twice");
        check(!tracker.update(false, 16, 320), "inventory restoration does not restart");
        check(!tracker.update(false, 14, 320), "loss of two is not one-pearl confirmation");
        for (int i = 0; i < 320; i++) tracker.update(false, 14, 320);
        check(tracker.getRemaining() == 0, "countdown expires without underflow");
        check(tracker.update(true, 14, 320), "next use after expiry accepted");
        tracker.reset(14);
        check(tracker.getRemaining() == 0, "new round clears old cooldown");
        check(!tracker.update(false, 14, 320), "reset refreshes inventory baseline");
        tracker.reset(-1);
        check(!tracker.update(false, 10, 320), "first inventory sample is not consumption");
        System.out.println("PearlCooldownTracker: 15 regression checks passed");
    }
}
