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
//    public double motionX;
//    public double motionY;
//    public double motionZ;
    public int lastTicksExisted;
    public boolean hasPosition;
    public boolean impactDetected;
    public int impactTick = -1;
    public int s28ImpactBlockX;
    public int s28ImpactBlockY;
    public int s28ImpactBlockZ;

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
//        this.motionX = motionX;
//        this.motionY = motionY;
//        this.motionZ = motionZ;
        this.lastTicksExisted = ticksExisted;
    }

    public void markImpact(int s28BlockX, int s28BlockY, int s28BlockZ, int currentTick) {
        impactDetected = true;
        impactTick = currentTick;
        s28ImpactBlockX = s28BlockX;
        s28ImpactBlockY = s28BlockY;
        s28ImpactBlockZ = s28BlockZ;
    }
}
