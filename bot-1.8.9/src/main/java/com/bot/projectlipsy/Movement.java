package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;

public class Movement {
    private final Minecraft mc;
    String lastMsg = "";
    private final float CONF_THRESHOLD = 0.6F;
    private int currentStrafe = 0;
    private int strafeHoldTicks = 0;

    public Movement(Minecraft mc) {
        this.mc = mc;
    }

    public void update(EntityPlayer target, float[] actions) {
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), true);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);

        wtap(target);
        predictStrafe(actions);
    }

    private void wtap(EntityPlayer target) {
        double dist = mc.thePlayer.getDistanceToEntity(target);
        if (target.hurtTime > 0){
            if (dist > 2.4F && dist <= 3.0F) {
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), true);
            } else if (dist <= 2.4F) {
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), false);
            }
        }
    }

    private void predictStrafe(float[] actions) {
        if (actions == null || actions.length < 4) return;

        int strafeAxis = (int) actions[2];
        float confidence = actions[3];

        if (currentStrafe != strafeAxis) {
            if (confidence >= CONF_THRESHOLD) {
                currentStrafe = strafeAxis;
                strafeHoldTicks = 0;
            } else {
                if (strafeHoldTicks > 0) strafeHoldTicks--;
            }
        }
        applyStrafe(currentStrafe);
    }

    private void applyStrafe(int strafeAxis) {
        if (strafeAxis == 0) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), false);
            testMsg("[Project Lipsy] straight");
        } else if (strafeAxis == -1) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), true);
            testMsg("[Project Lipsy] left");
        } else if (strafeAxis == 1) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), true);
            testMsg("[Project Lipsy] right");
        }
    }

    public float getRetreatYaw(EntityPlayer target) {
        // Always run directly away from the target for stability
        double dx = mc.thePlayer.posX - target.posX;
        double dz = mc.thePlayer.posZ - target.posZ;
        return (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90);
    }

    public void retreat() {
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), true);
    }

    private void testMsg(String msg) {
        if (!lastMsg.equals(msg)) {
            mc.thePlayer.sendChatMessage(msg);
            lastMsg = msg;
        }
    }
}
