package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemStack;
import net.minecraft.item.ItemSword;
import net.minecraft.util.MathHelper;
import net.minecraft.util.MovingObjectPosition;

import java.io.IOException;
public class ModelConnect {
    // Real-time data logger, sending information to model
    private static final String MODEL_HOST = "127.0.0.1";
    private static final int MODEL_PORT = 33333;
    private static final long RETRY_DELAY_MS = 2000L;

    private final Minecraft mc;
    private final int batchSize = 28;
    float[] batch = new float[batchSize];
    float[] actions = null;
    private ModelClient modelClient;
    private long lastConnectAttemptMs = 0L;
    private boolean printedOfflineMessage = false;
    private Float previousMyYaw = null;
    private Float previousMyPitch = null;
    private Float previousTargetYaw = null;
    private Float previousTargetPitch = null;
    private int tickIndex = 0;
    private int lastAttackTick = -100000;
    private int lastDamageTick = -100000;
    private float previousSwingProgress = 0.0F;

    public ModelConnect(Minecraft mc) {
        this.mc = mc;
    }

    public void sendObs(EntityPlayer target) {
        if (mc.thePlayer == null || target == null) {
            actions = null;
            return;
        }
        if (!ensureConnected()) {
            actions = null;
            return;
        }
        tickIndex++;

        // Whole list of player data from gameplay
        double myMotionX = mc.thePlayer.motionX;
        double myMotionY = mc.thePlayer.motionY;
        double myMotionZ = mc.thePlayer.motionZ;
        double myYaw = mc.thePlayer.rotationYaw;
        double myPitch = mc.thePlayer.rotationPitch;
        float myDeltaYaw = previousMyYaw == null ? 0.0F : MathHelper.wrapAngleTo180_float(mc.thePlayer.rotationYaw - previousMyYaw);
        float myDeltaPitch = previousMyPitch == null ? 0.0F : mc.thePlayer.rotationPitch - previousMyPitch;
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
        boolean attackDown = mc.gameSettings.keyBindAttack.isKeyDown();
        boolean swingStarted = mc.thePlayer.swingProgress > 0.0F && previousSwingProgress <= 0.0F;
        boolean aAttack = attackDown || swingStarted;
        if (aAttack) {
            lastAttackTick = tickIndex;
        }
        previousSwingProgress = mc.thePlayer.swingProgress;

        if (mc.thePlayer.hurtTime == 10 || mc.thePlayer.hurtTime > 0 && ticksSince(lastDamageTick) > 10) {
            lastDamageTick = tickIndex;
        }

        int aStrafe = mc.gameSettings.keyBindRight.isKeyDown() ? 1 : (mc.gameSettings.keyBindLeft.isKeyDown() ? -1 : 0);
        boolean holdingSword = isHoldingSword();
        boolean myUsingItem = mc.thePlayer.isUsingItem();
        boolean crosshairOnTarget = mc.objectMouseOver != null
                && mc.objectMouseOver.typeOfHit == MovingObjectPosition.MovingObjectType.ENTITY
                && mc.objectMouseOver.entityHit == target;

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
        batch[14] = (float) yawDiffToTarget;
        batch[15] = (float) pitchDiffToTarget;
        batch[16] = myDeltaYaw;
        batch[17] = aAttack ? 1.0F : 0.0F;
        batch[18] = (float) aStrafe;
        batch[19] = (float) target.hurtTime;
        batch[20] = (float) mc.thePlayer.hurtTime;
        batch[21] = (float) ticksSince(lastDamageTick);
        batch[22] = (float) ticksSince(lastAttackTick);
        batch[23] = target.swingProgress;
        batch[24] = mc.thePlayer.swingProgress;
        batch[25] = holdingSword ? 1.0F : 0.0F;
        batch[26] = myUsingItem ? 1.0F : 0.0F;
        batch[27] = crosshairOnTarget ? 1.0F : 0.0F;



        try {
            modelClient.sendPacket(batch);

            actions = modelClient.readPacket();
//            mc.thePlayer.sendChatMessage(Arrays.toString(actions));
        } catch (IOException e) {
            actions = null;
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

    private boolean isHoldingSword() {
        ItemStack stack = mc.thePlayer.getHeldItem();
        return stack != null && stack.getItem() instanceof ItemSword;
    }

    private int ticksSince(int tick) {
        return tickIndex - tick;
    }
}
