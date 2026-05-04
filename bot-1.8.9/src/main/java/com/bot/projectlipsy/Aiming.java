package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.Entity;
import net.minecraft.util.MathHelper;

public class Aiming {
    private final Minecraft mc;
    double lastMsgMs = System.currentTimeMillis();
    public Aiming(Minecraft mc) { this.mc = mc; }

    public void update(Entity target, float[] actions) {
        if (target == null) return;

        double xOffset = getRandomOffset(0.05f);
        double yOffset = getRandomOffset(0.05f);
        double zOffset = getRandomOffset(0.05f);

        double dx = (target.posX + xOffset) - mc.thePlayer.posX;
        double dy = (target.posY + yOffset) - (mc.thePlayer.posY) - 0.6F;
        double dz = (target.posZ + zOffset) - mc.thePlayer.posZ;
        double distance = Math.sqrt(dx * dx + dz * dz);

        float yaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0F);
        float pitch = (float) -Math.toDegrees(Math.atan2(dy, distance));

        float yawDiff = MathHelper.wrapAngleTo180_float(yaw - mc.thePlayer.rotationYaw);
        float pitchDiff = pitch - mc.thePlayer.rotationPitch;

        if (actions == null || actions.length < 2 || Math.abs(yawDiff) > 50.0F || pitchDiff < -20.0F || pitchDiff > 10.0F) {
            mc.thePlayer.rotationYaw = MathHelper.wrapAngleTo180_float(updateRotation(mc.thePlayer.rotationYaw, yaw, 90f));
            mc.thePlayer.rotationPitch = clampPitch(updateRotation(mc.thePlayer.rotationPitch, pitch, 30f));

            if (System.currentTimeMillis() - lastMsgMs >= 250) {
                String testMsg = "[Project Lipsy] Aiming w/o model. yaw: " + yaw + " pitch:" + pitch +
                        " yawDiff: " + yawDiff + " pitchDiff: " + pitchDiff;
                mc.thePlayer.sendChatMessage(testMsg);
                lastMsgMs = System.currentTimeMillis();
            }

        } else {
            float nextYawDiff = actions[0];
            float nextPitchDiff = actions[1];
            float desiredYaw = mc.thePlayer.rotationYaw + yawDiff - nextYawDiff;
            float desiredPitch = mc.thePlayer.rotationPitch + pitchDiff - nextPitchDiff;

            if (System.currentTimeMillis() - lastMsgMs >= 250) {
                String testMsg = "[Project Lipsy] Aiming w/ model. desiredYaw: " + desiredYaw + " desiredPitch: " + desiredPitch
                        + " currentYawDiff: " + yawDiff + " currentPitchDiff: " + pitchDiff
                        + " nextYawDiff: " + nextYawDiff + " nextPitchDiff: " + nextPitchDiff;
                mc.thePlayer.sendChatMessage(testMsg);
                lastMsgMs = System.currentTimeMillis();
            }

            mc.thePlayer.rotationYaw = MathHelper.wrapAngleTo180_float(updateRotation(mc.thePlayer.rotationYaw, desiredYaw, 30f));
            mc.thePlayer.rotationPitch = clampPitch(updateRotation(mc.thePlayer.rotationPitch, desiredPitch, 30f));
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

    private float clampPitch(float pitch) {
        if (pitch > 90.0F) return 90.0F;
        if (pitch < -90.0F) return -90.0F;
        return pitch;
    }

    private double getRandomOffset(float range){
        return (Math.random() * range * 2) - range;
    }
}
