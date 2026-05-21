package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.Entity;
import net.minecraft.util.MathHelper;

public class Aiming {
    private final Minecraft mc;
    private final utils utils;

    double lastMsgMs = System.currentTimeMillis();
    String lastMode = "";
    private boolean modelLocked = false;
    private int recoverTicksLeft = 0;

    private static final float SENSITIVITY = 1.0F;

    private static final float MODEL_ENTER_YAW = 40.0F;
    private static final float MODEL_EXIT_YAW = 50.0F;
    private static final float MODEL_ENTER_PITCH = 20.0F;
    private static final float MODEL_EXIT_PITCH = 30.0F;

    private static final double MODEL_MIN_DISTANCE = 0.5;
    private static final int MIN_RECOVER_TICKS = 2;
    public Aiming(Minecraft mc) { this.mc = mc; this.utils = new utils(mc); }

    public void update(Entity target, float[] actions) {
        if (mc.thePlayer == null || target == null) return;

        double xOffset = utils.getRandomOffset(0.08f);
        double yOffset = utils.getRandomOffset(0.05f);
        double zOffset = utils.getRandomOffset(0.08f);

        double dx = (target.posX + xOffset) - mc.thePlayer.posX;
        double dy = (target.posY + yOffset) - (mc.thePlayer.posY) - 0.6F;
        double dz = (target.posZ + zOffset) - mc.thePlayer.posZ;
        double distance = Math.sqrt(dx * dx + dz * dz);

        float yaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0F);
        float pitch = (float) -Math.toDegrees(Math.atan2(dy, distance));

        float yawDiff = MathHelper.wrapAngleTo180_float(yaw - mc.thePlayer.rotationYaw);
        float pitchDiff = pitch - mc.thePlayer.rotationPitch;

        boolean hasModel = actions != null && actions.length >= 2;

        boolean forceRecover =
                !hasModel
                        || distance < MODEL_MIN_DISTANCE
                        || Math.abs(yawDiff) > MODEL_EXIT_YAW
                        || Math.abs(pitchDiff) > MODEL_EXIT_PITCH;

        boolean canEnterModel =
                hasModel
                        && distance >= MODEL_MIN_DISTANCE
                        && Math.abs(yawDiff) < MODEL_ENTER_YAW
                        && Math.abs(pitchDiff) < MODEL_ENTER_PITCH;

        if (forceRecover) {
            modelLocked = false;
            recoverTicksLeft = MIN_RECOVER_TICKS;
        }

        if (!modelLocked) {
            if (recoverTicksLeft > 0) {
                recoverTicksLeft--;
            } else if (canEnterModel) {
                modelLocked = true;
            }
        }

        // Recovery mode
        if (!modelLocked || !hasModel) {
            mc.thePlayer.rotationYaw = MathHelper.wrapAngleTo180_float(updateRotation(mc.thePlayer.rotationYaw, yaw, 100f));
            mc.thePlayer.rotationPitch = clampPitch(updateRotation(mc.thePlayer.rotationPitch, pitch, 30f));

//            if (System.currentTimeMillis() - lastMsgMs >= 250) {
//            if (!lastMode.equals("w/o")) {
//                String testMsg = "[Project Lipsy] Aiming w/o model. yaw: " + yaw + " pitch:" + pitch +
//                        " yawDiff: " + yawDiff + " pitchDiff: " + pitchDiff;
//                mc.thePlayer.sendChatMessage(testMsg);
//                lastMsgMs = System.currentTimeMillis();
//                lastMode = "w/o";
//            }

        // aim with model
        } else {
            float predictedDeltaYaw = actions[0] * SENSITIVITY;
            float predictedDeltaPitch = actions[1];
            predictedDeltaYaw = clampFloat(predictedDeltaYaw, -50F, 50F);
            predictedDeltaPitch = clampFloat(predictedDeltaPitch, -15F, 15F);
            float desiredYaw = mc.thePlayer.rotationYaw + predictedDeltaYaw;
            float desiredPitch = mc.thePlayer.rotationPitch + predictedDeltaPitch;

//            if (System.currentTimeMillis() - lastMsgMs >= 250) {
//            if (!lastMode.equals("/w")) {
//                String testMsg = "[Project Lipsy] Aiming w/ model. desiredYaw: " + desiredYaw + " desiredPitch: " + desiredPitch
//                        + " currentYawDiff: " + yawDiff + " currentPitchDiff: " + pitchDiff
//                        + " predictedDeltaYaw: " + predictedDeltaYaw + " predictedDeltaPitch: " + predictedDeltaPitch;
//                mc.thePlayer.sendChatMessage(testMsg);
//                lastMsgMs = System.currentTimeMillis();
//                lastMode = "/w";
//            }

            float desiredYawDiff = yaw - desiredYaw;
            if (Math.abs(MathHelper.wrapAngleTo180_float(desiredYawDiff)) <= MODEL_EXIT_YAW) {
                mc.thePlayer.rotationYaw = MathHelper.wrapAngleTo180_float(updateRotation(mc.thePlayer.rotationYaw, desiredYaw, 50f));
                mc.thePlayer.rotationPitch = clampPitch(updateRotation(mc.thePlayer.rotationPitch, desiredPitch, 15f));
            } else {
//                mc.thePlayer.sendChatMessage("[Project L1psy] rejected desiredYawDiff " + desiredYawDiff);
            }
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

    private float clampFloat(float value, float min, float max) {
        return Math.min(Math.max(value, min), max);
    }

    private float clampPitch(float pitch) {
        if (pitch > 90.0F) return 90.0F;
        if (pitch < -90.0F) return -90.0F;
        return pitch;
    }

}
