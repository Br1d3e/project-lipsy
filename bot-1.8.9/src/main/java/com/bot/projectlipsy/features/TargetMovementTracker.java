package com.bot.projectlipsy.features;

import java.util.ArrayDeque;
import java.util.Deque;

/** Horizontal movement in blocks per tick. The recorder resets history on round or observation changes. */
public class TargetMovementTracker {
    private static final int VELOCITY_WINDOW = 5;
    private static final int LONG_DISTANCE_WINDOW = 20;
    private final Deque<Sample> history = new ArrayDeque<Sample>();
    private Double radialVelocity;
    private Double outwardVelocity;
    private Double facingMeCos;
    private Double distanceDelta5;
    private Double distanceDelta20;

    public void reset() {
        history.clear();
        radialVelocity = outwardVelocity = facingMeCos = distanceDelta5 = distanceDelta20 = null;
    }

    public void update(int tick, double myX, double myZ, double targetX, double targetZ, float targetYaw) {
        Sample current = new Sample(tick, myX, myZ, targetX, targetZ);
        radialVelocity = outwardVelocity = distanceDelta5 = distanceDelta20 = null;
        double dx = targetX - myX;
        double dz = targetZ - myZ;
        double distance = current.distance;
        facingMeCos = null;
        if (distance > 1.0E-6D) {
            double ux = dx / distance;
            double uz = dz / distance;
            double yaw = Math.toRadians(targetYaw);
            facingMeCos = Math.max(-1.0D, Math.min(1.0D, Math.sin(yaw) * ux - Math.cos(yaw) * uz));
            for (Sample old : history) {
                if (tick - old.tick == VELOCITY_WINDOW) {
                    double vx = (targetX - old.targetX) / VELOCITY_WINDOW;
                    double vz = (targetZ - old.targetZ) / VELOCITY_WINDOW;
                    outwardVelocity = vx * ux + vz * uz;
                    radialVelocity = (vx - (myX - old.myX) / VELOCITY_WINDOW) * ux
                            + (vz - (myZ - old.myZ) / VELOCITY_WINDOW) * uz;
                }
            }
        }
        for (Sample old : history) {
            if (tick - old.tick == 5) distanceDelta5 = distance - old.distance;
            if (tick - old.tick == 20) distanceDelta20 = distance - old.distance;
        }
        history.addLast(current);
        while (history.peekFirst().tick < tick - LONG_DISTANCE_WINDOW) history.removeFirst();
    }

    private static double squared(double x, double z) { return x * x + z * z; }
    public Double getRadialVelocity() { return radialVelocity; }
    public Double getOutwardVelocity() { return outwardVelocity; }
    public Double getFacingMeCos() { return facingMeCos; }
    public Double getDistanceDelta5() { return distanceDelta5; }
    public Double getDistanceDelta20() { return distanceDelta20; }

    private static class Sample {
        final int tick;
        final double myX, myZ, targetX, targetZ, distance;
        Sample(int tick, double myX, double myZ, double targetX, double targetZ) {
            this.tick = tick;
            this.myX = myX;
            this.myZ = myZ;
            this.targetX = targetX;
            this.targetZ = targetZ;
            this.distance = Math.sqrt(squared(targetX - myX, targetZ - myZ));
        }
    }
}
