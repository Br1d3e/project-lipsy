package com.bot.recorder.targetTracker;

public class TrackedPotion {
    public final int entityId;
    public final int firstSeenTick;
    public double previousX;
    public double previousY;
    public double previousZ;
    public double lastX;
    public double lastY;
    public double lastZ;
    public int lastTicksExisted;
    public boolean hasPosition;

    public TrackedPotion(int entityId, int firstSeenTick) {
        this.entityId = entityId;
        this.firstSeenTick = firstSeenTick;
    }

    public void update(double x, double y, double z, int ticksExisted) {
        if (hasPosition) {
            this.previousX = this.lastX;
            this.previousY = this.lastY;
            this.previousZ = this.lastZ;
        } else {
            this.previousX = x;
            this.previousY = y;
            this.previousZ = z;
            this.hasPosition = true;
        }

        this.lastX = x;
        this.lastY = y;
        this.lastZ = z;
        this.lastTicksExisted = ticksExisted;
    }
}
