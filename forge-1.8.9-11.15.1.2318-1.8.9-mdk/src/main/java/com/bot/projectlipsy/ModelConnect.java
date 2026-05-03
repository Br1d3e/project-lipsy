package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.MathHelper;

import java.io.IOException;
import java.util.Arrays;


public class ModelConnect {
    // Real-time data logger, sending information to model
    private static final String MODEL_HOST = "127.0.0.1";
    private static final int MODEL_PORT = 33333;
    private static final long RETRY_DELAY_MS = 2000L;

    private final Minecraft mc;
    private final int batchSize = 9;
    float[] batch = new float[batchSize];
    float[] actions = new float[2];
    private ModelClient modelClient;
    private long lastConnectAttemptMs = 0L;
    private boolean printedOfflineMessage = false;

    public ModelConnect(Minecraft mc) {
        this.mc = mc;
    }

    public void sendObs(EntityPlayer target) {
        if (mc.thePlayer == null || target == null) return;
        if (!ensureConnected()) return;

        // Whole list of player data from gameplay
        double dx = mc.thePlayer.posX - target.posX;
        double dy = mc.thePlayer.posY - target.posY;
        double dz = mc.thePlayer.posZ - target.posZ;
        double dist = mc.thePlayer.getDistanceToEntity(target);
        double distanceXZ = Math.hypot(dx, dz);
        double targetYaw = Math.toDegrees(Math.atan2(dz, dx)) - 90F;
        // Vertical (pitch) offset
        double targetPitch = -Math.toDegrees(Math.atan2(dy, distanceXZ));
        float yawDiff = MathHelper.wrapAngleTo180_float((float) targetYaw - mc.thePlayer.rotationYaw);
        float pitchDiff = MathHelper.wrapAngleTo180_float((float) targetPitch - mc.thePlayer.rotationPitch);
        double targetVelX = target.motionX;
        double targetVelY = target.motionY;
        double targetVelZ = target.motionZ;
        double targetMotion = Math.hypot(targetVelX, targetVelZ);

        batch[0] = (float) dx;
        batch[1] = (float) dy;
        batch[2] = (float) dz;
        batch[3] = (float) dist;
        batch[4] = (float) targetVelX;
        batch[5] = (float) targetVelY;
        batch[6] = (float) targetVelZ;
        batch[7] = yawDiff;
        batch[8] = pitchDiff;

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
            System.out.println("[Project Lipsy] Connected to model server at " + MODEL_HOST + ":" + MODEL_PORT);
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
