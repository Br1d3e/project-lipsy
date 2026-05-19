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
    private int MIN_HOLD_TICKS = 3;

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
            if (strafeHoldTicks > 0 || confidence < CONF_THRESHOLD) {
                strafeAxis = currentStrafe;
            } else {
                currentStrafe = strafeAxis;
                strafeHoldTicks = MIN_HOLD_TICKS;
            }
        } else {
            if (strafeHoldTicks > 0) strafeHoldTicks--;
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

    public void sprint() {
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), true);
    }

    public void jump(){
        jump(2);
    }

    public void jump(int durationTicks) {
        long startTime = System.currentTimeMillis();
        int durationTime = durationTicks * 20;
        KeyBinding.onTick(mc.gameSettings.keyBindJump.getKeyCode());
        if (System.currentTimeMillis() - startTime > durationTime) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindJump.getKeyCode(), false);
        }
    }

    private void testMsg(String msg) {
        if (!lastMsg.equals(msg)) {
            mc.thePlayer.sendChatMessage(msg);
            lastMsg = msg;
        }
    }
}
