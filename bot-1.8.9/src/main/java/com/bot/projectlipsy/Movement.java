package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;

public class Movement {
    private final Minecraft mc;
    private final utils utils;
    String lastMsg = "";
    private final float CONF_THRESHOLD = 0.6F;
    private int currentStrafe = 0;
    private int strafeHoldTicks = 0;
    private int MIN_HOLD_TICKS = 3;
    private int ticksLastJump = 0;
    private int jumpTicksLeft = 0;

    public Movement(Minecraft mc) {
        this.mc = mc;
        this.utils = new utils(mc);
    }

    public void update(EntityPlayer target, float[] actions) {
        tickJump();
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), true);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), false);

        utils.updateCombos(target);
//        if (mc.thePlayer.getDistanceToEntity(target) <= 6 || !predictWtap(actions)) {
        if (mc.thePlayer.getDistanceToEntity(target) <= 6) {
            wtap(target);
        }
        predictStrafe(actions);
        offCombo(target);
    }

    private boolean predictWtap(float[] actions) {
        if (actions == null || actions.length <= ModelConnect.WTAP_ACTION_INDEX) return false;

        int forwardAxis = Math.round(actions[ModelConnect.WTAP_ACTION_INDEX]);
        if (forwardAxis > 0) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), false);
            testMsg("[Project L1psy] w");
        } else if (forwardAxis < 0) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), true);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), false);
            testMsg("[Project L1psy] s");
        } else {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), false);
            testMsg("[Project L1psy] w released");
        }
        return true;
    }

    private void wtap(EntityPlayer target) {
        double dist = mc.thePlayer.getDistanceToEntity(target);
        if (target.hurtTime > 0){
            if (dist > 2.5F && dist <= 3.0F) {
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), true);
            } else if (dist <= 2.5F) {
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
//            testMsg("[Project Lipsy] straight");
        } else if (strafeAxis == -1) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), true);
//            testMsg("[Project Lipsy] left");
        } else if (strafeAxis == 1) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindLeft.getKeyCode(), false);
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindRight.getKeyCode(), true);
//            testMsg("[Project Lipsy] right");
        }
    }

    private void offCombo(EntityPlayer target) {
        int targetCombo = utils.getTargetCombo();
        ticksLastJump++;
        double probOffCombo = 1 - Math.exp(-0.35 * targetCombo);
        if (ticksLastJump > 10 && targetCombo >= 2 && mc.thePlayer.getDistanceToEntity(target) < 3 && Math.random() < probOffCombo) {
            jump();
            ticksLastJump = 0;
            mc.thePlayer.sendChatMessage("[Project L1psy] attempted to jump");
        }
    }

    public void sprint() {
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), true);
    }

    public void jump(){
        jump(10);
    }

    public void jump(int durationTicks) {
        jumpTicksLeft = Math.max(jumpTicksLeft, durationTicks);
        tickJump();
    }

    public void stopJump() {
        jumpTicksLeft = 0;
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindJump.getKeyCode(), false);
    }

    private void tickJump() {
        if (jumpTicksLeft > 0) {
            KeyBinding.setKeyBindState(mc.gameSettings.keyBindJump.getKeyCode(), true);
            jumpTicksLeft--;
        } else {
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
