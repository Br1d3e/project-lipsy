package com.bot.recorder.targetTracker;

import net.minecraft.util.AxisAlignedBB;
import net.minecraft.util.Vec3;

public class TrackedPotion {
    public final int entityId;
    public final int firstSeenTick;
    public double previousX;
    public double previousY;
    public double previousZ;
    public double lastX;
    public double lastY;
    public double lastZ;
    public double motionX;
    public double motionY;
    public double motionZ;
    public int lastTicksExisted;
    public boolean hasPosition;
    public boolean everDirectHit;
    public int firstDirectHitTick = -1;
    public double directHitSegmentStartX;
    public double directHitSegmentStartY;
    public double directHitSegmentStartZ;
    public double directHitSegmentEndX;
    public double directHitSegmentEndY;
    public double directHitSegmentEndZ;

    public TrackedPotion(int entityId, int firstSeenTick) {
        this.entityId = entityId;
        this.firstSeenTick = firstSeenTick;
    }

    public void update(double x, double y, double z, double motionX, double motionY, double motionZ, int ticksExisted) {
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
        this.motionX = motionX;
        this.motionY = motionY;
        this.motionZ = motionZ;
        this.lastTicksExisted = ticksExisted;
    }

    public void updateDirectHit(AxisAlignedBB expandedTargetHitBox, int currentTick) {
        if (everDirectHit || expandedTargetHitBox == null || !hasPosition) return;

        Vec3 segmentStart = Vec3.createVectorHelper(lastX, lastY, lastZ);
        Vec3 segmentEnd = Vec3.createVectorHelper(lastX + motionX, lastY + motionY, lastZ + motionZ);
        boolean directHit = expandedTargetHitBox.isVecInside(segmentStart)
                || expandedTargetHitBox.isVecInside(segmentEnd)
                || expandedTargetHitBox.calculateIntercept(segmentStart, segmentEnd) != null;
        if (!directHit) return;

        everDirectHit = true;
        firstDirectHitTick = currentTick;
        directHitSegmentStartX = segmentStart.xCoord;
        directHitSegmentStartY = segmentStart.yCoord;
        directHitSegmentStartZ = segmentStart.zCoord;
        directHitSegmentEndX = segmentEnd.xCoord;
        directHitSegmentEndY = segmentEnd.yCoord;
        directHitSegmentEndZ = segmentEnd.zCoord;
    }
}
