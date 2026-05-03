package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.Entity;
import net.minecraft.util.MathHelper;

public class Aiming {
    private final Minecraft mc;
    public Aiming(Minecraft mc) { this.mc = mc; }

    public void update(Entity target, float[] actions) {
        if (target == null) return;

        double xOffset = getRandomOffset(0.05f);
        double yOffset = getRandomOffset(0.05f);
        double zOffset = getRandomOffset(0.05f);

        double dx = (target.posX + xOffset) - mc.thePlayer.posX;
        double dy = (target.posY + yOffset) - (mc.thePlayer.posY) - 0.6;
        double dz = (target.posZ + zOffset) - mc.thePlayer.posZ;
        double distance = Math.sqrt(dx * dx + dz * dz);

        float yaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0F);
        float pitch = (float) -Math.toDegrees(Math.atan2(dy, distance));

        if (true || actions == null) {
            mc.thePlayer.rotationYaw = updateRotation(mc.thePlayer.rotationYaw, yaw, 30f);
            mc.thePlayer.rotationPitch = updateRotation(mc.thePlayer.rotationPitch, pitch, 30f);
        } else {
            mc.thePlayer.rotationYaw = mc.thePlayer.rotationYaw + actions[0];
            mc.thePlayer.rotationPitch = updateRotation(mc.thePlayer.rotationPitch, pitch, 30f);
        }
    }

    private float updateRotation(float current, float target, float maxIncrement) {
        float delta = MathHelper.wrapAngleTo180_float(target - current);
        if (delta > maxIncrement) {
            delta = maxIncrement;
        }
        if (delta < -maxIncrement) {
            delta = -maxIncrement;
        }
        return current + delta;
    }

    private double getRandomOffset(float range){
        return (Math.random() * range * 2) - range;
    }
}