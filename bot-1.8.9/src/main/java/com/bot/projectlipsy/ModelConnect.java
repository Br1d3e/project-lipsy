package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.MathHelper;

import java.io.IOException;
public class ModelConnect {
    // Real-time data logger, sending information to model
    private static final String MODEL_HOST = "127.0.0.1";
    private static final int MODEL_PORT = 33333;
    private static final long RETRY_DELAY_MS = 2000L;

    private final Minecraft mc;
    private final int batchSize = 20;
    float[] batch = new float[batchSize];
    float[] actions = null;
    private ModelClient modelClient;
    private long lastConnectAttemptMs = 0L;
    private boolean printedOfflineMessage = false;
    private Float previousMyYaw = null;
    private Float previousMyPitch = null;
    private Float previousTargetYaw = null;
    private Float previousTargetPitch = null;

    public ModelConnect(Minecraft mc) {
        this.mc = mc;
    }

    public void sendObs(EntityPlayer target) {
        if (mc.thePlayer == null || target == null) return;
        if (!ensureConnected()) return;

        // Whole list of player data from gameplay
        double myMotionX = mc.thePlayer.motionX;
        double myMotionY = mc.thePlayer.motionY;
        double myMotionZ = mc.thePlayer.motionZ;
        double myYaw = mc.thePlayer.rotationYaw;
        double myPitch = mc.thePlayer.rotationPitch;
        Float myDeltaYaw = previousMyYaw == null ? null : MathHelper.wrapAngleTo180_float(mc.thePlayer.rotationYaw - previousMyYaw);
        Float myDeltaPitch = previousMyPitch == null ? null : mc.thePlayer.rotationPitch - previousMyPitch;
        previousMyYaw = mc.thePlayer.rotationYaw;
        previousMyPitch = mc.thePlayer.rotationPitch;
        double myMoveForward = mc.thePlayer.moveForward;
        double myMoveStrafing = mc.thePlayer.moveStrafing;

        double targetRelX = target.posX - mc.thePlayer.posX;
        double targetRelY = target.posY + target.getEyeHeight() - (mc.thePlayer.posY + mc.thePlayer.getEyeHeight());
        double targetRelZ = target.posZ - mc.thePlayer.posZ;
        double targetDistance = mc.thePlayer.getDistanceToEntity(target);
        double targetMotionX = target.motionX;
        double targetMotionY = target.motionY;
        double targetMotionZ = target.motionZ;
        double targetYaw = target.rotationYaw;
        double targetPitch = target.rotationPitch;
        float targetDeltaYaw = previousTargetYaw == null ? 0 : MathHelper.wrapAngleTo180_float(target.rotationYaw - previousTargetYaw);
        float targetDeltaPitch = previousTargetPitch == null ? 0 : MathHelper.wrapAngleTo180_float(target.rotationPitch - previousTargetPitch);
        previousTargetYaw = target.rotationYaw;
        previousTargetPitch = target.rotationPitch;

        double distanceXZ = Math.hypot(targetRelX, targetRelZ);
        double targetAimYaw = Math.toDegrees(Math.atan2(targetRelZ, targetRelX)) - 90F;
        double targetAimPitch = -Math.toDegrees(Math.atan2(targetRelY, distanceXZ));
        double yawDiffToTarget = (double) MathHelper.wrapAngleTo180_float((float) targetAimYaw - mc.thePlayer.rotationYaw);
        double pitchDiffToTarget = (double) MathHelper.wrapAngleTo180_float((float) targetAimPitch - mc.thePlayer.rotationPitch);

        batch[0] = (float) myMotionX;
        batch[1] = (float) myMotionY;
        batch[2] = (float) myMotionZ;
        batch[3] = (float) myYaw;
        batch[4] = (float) myPitch;
        batch[5] = (float) myMoveForward;
        batch[6] = (float) myMoveStrafing;
        batch[7] = (float) targetRelX;
        batch[8] = (float) targetRelY;
        batch[9] = (float) targetRelZ;
        batch[10] = (float) targetDistance;
        batch[11] = (float) targetMotionX;
        batch[12] = (float) targetMotionY;
        batch[13] = (float) targetMotionZ;
        batch[14] = (float) targetYaw;
        batch[15] = (float) targetPitch;
        batch[16] = targetDeltaYaw;
        batch[17] = targetDeltaPitch;
        batch[18] = (float) yawDiffToTarget;
        batch[19] = (float) pitchDiffToTarget;



        try {
            modelClient.sendPacket(batch);

            actions = modelClient.readPacket();
//            mc.thePlayer.sendChatMessage(Arrays.toString(actions));
        } catch (IOException e) {
            closeClient();
            System.out.println("[Project Lipsy] Model connection lost: " + e.getMessage());
        }
//        modelClient.stopConn();
    }

    public float[] getActions() {
        return actions;
    }

    private boolean ensureConnected() {
        if (modelClient != null) return true;

        long now = System.currentTimeMillis();
        if (now - lastConnectAttemptMs < RETRY_DELAY_MS) return false;

        lastConnectAttemptMs = now;
        try {
            modelClient = new ModelClient(MODEL_HOST, MODEL_PORT);
            printedOfflineMessage = false;
            return true;
        } catch (IOException e) {
            if (!printedOfflineMessage) {
                System.out.println("[Project Lipsy] Model server is offline; gameplay will continue without model actions.");
                printedOfflineMessage = true;
            }
            return false;
        }
    }

    private void closeClient() {
        if (modelClient == null) return;

        try {
            modelClient.stopConn();
        } catch (IOException ignored) {
        }
        modelClient = null;
    }
}
