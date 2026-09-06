package com.bot.recorder.features;

public class TargetMovementTrackerTest {
    private static void check(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }
    private static void near(Double value, double expected) {
        check(value != null && Math.abs(value - expected) < 1E-9, "expected " + expected + ", got " + value);
    }
    public static void main(String[] args) {
        TargetMovementTracker tracker = new TargetMovementTracker();
        tracker.update(0, 1, 0, 0, 10, 0, 90);
        near(tracker.getFacingMeCos(), 1);
        check(tracker.getRadialVelocity() == null, "no velocity without history");
        for (int t = 1; t <= 20; t++) tracker.update(t, 1, 0, 0, 10 + t * 0.2, 0, 90);
        near(tracker.getRadialVelocity(), 0.2);
        near(tracker.getOutwardVelocity(), 0.2);
        near(tracker.getDistanceDelta5(), 1);
        near(tracker.getDistanceDelta20(), 4);
        tracker.reset();
        for (int t = 0; t <= 5; t++) tracker.update(t, 1, -t * 0.1, 0, 10, 0, -90);
        near(tracker.getRadialVelocity(), 0.1);
        near(tracker.getOutwardVelocity(), 0);
        near(tracker.getFacingMeCos(), -1);
        tracker.reset();
        for (int t = 0; t <= 5; t++) tracker.update(t, 1, t * 0.2, 0, 10 + t * 0.2, 0, 0);
        near(tracker.getRadialVelocity(), 0);
        near(tracker.getOutwardVelocity(), 0.2);
        near(tracker.getFacingMeCos(), 0);
        tracker.reset();
        for (int t = 0; t <= 5; t++) tracker.update(t, 1, 0, 0, 10 - t * 0.2, 0, 90);
        near(tracker.getRadialVelocity(), -0.2);
        tracker.update(9, 1, 0, 0, 9, 0, 90);
        check(tracker.getRadialVelocity() == null && tracker.getDistanceDelta5() == null, "tick gap invalidates history");
        tracker.update(10, 2, 0, 0, 9, 0, 90);
        check(tracker.getRadialVelocity() == null, "entity rebind invalidates motion history");
        tracker.update(11, 2, 0, 0, 100, 0, 90);
        check(tracker.getDistanceDelta20() == null, "large displacement invalidates history");
        tracker.reset();
        tracker.update(0, 2, 0, 0, 0, 0, 90);
        check(tracker.getFacingMeCos() == null && tracker.getRadialVelocity() == null, "zero separation has no direction");
        tracker.reset();
        check(tracker.getFacingMeCos() == null && tracker.getDistanceDelta5() == null, "round or visibility reset");
        System.out.println("Target movement scenarios passed");
    }
}
