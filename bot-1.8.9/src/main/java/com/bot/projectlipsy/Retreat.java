package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;

public class Retreat {
    private final Minecraft mc;
    private final Movement movement;
    private final utils utils;

    private int backTicks = 0;
    private int backMaxTicks = (int) Math.floor(Math.random() * 8) + 2;

    public Retreat(Minecraft mc, Movement movement) {
        this.mc = mc;
        this.movement = movement;
        this.utils = new utils(mc);
    }

    public float getRetreatYaw(EntityPlayer target) {
        // Always run directly away from the target for stability
        double dx = mc.thePlayer.posX - target.posX;
        double dz = mc.thePlayer.posZ - target.posZ;
        return (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90);
    }

    public void turn(EntityPlayer target) {
        mc.thePlayer.rotationYaw = getRetreatYaw(target);
        movement.sprint();
    }

    public void back(EntityPlayer target) {
        boolean isJump = Math.random() < 0.75;
        back(target, isJump);
    }
    public void back(EntityPlayer target, boolean isJump) {
        backTicks++;

        if (backTicks % 4 == 1) {
            double yawOffset = utils.getRandomOffset(5F);
            float lookYaw = utils.getTargetYaw(target) + (float) yawOffset + 5F;
            mc.thePlayer.rotationYaw = lookYaw;
        }

        if (backTicks > backMaxTicks) {
           backTicks = 0;
           backMaxTicks = (int) Math.floor(Math.random() * 8) + 2;
//           return;

           KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), false);
           turn(target);    // switch to turn if backTicks reaches limit
        }

        KeyBinding.setKeyBindState(mc.gameSettings.keyBindBack.getKeyCode(), true);
        if (isJump) {
            movement.jump();
        }
    }
}
