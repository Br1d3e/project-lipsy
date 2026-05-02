package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;

public class Pearl {

    private final Minecraft mc;
    private final utils utils;
    private long lastPearlTime = 0;

    public Pearl(Minecraft mc) {
        this.mc = mc;
        this.utils = new utils(mc);
    }

    public boolean canThrowPearl() {
        long currentTime = System.currentTimeMillis();
        long pearlCooldown = 16000; // 16 second cooldown
        return currentTime - lastPearlTime >= pearlCooldown;
    }

    public void tryhardPearl(EntityPlayer target) {
        if (utils.hasItem("Ender Pearl") && canThrowPearl()) {
            utils.switchItem("Ender Pearl");

            double dx = target.posX - mc.thePlayer.posX;
            double dy = target.posY + target.getEyeHeight() - (mc.thePlayer.posY + mc.thePlayer.getEyeHeight());
            double dz = target.posZ - mc.thePlayer.posZ;
            double distance = Math.sqrt(dx * dx + dz * dz);

            float yaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0F);
            float pitch = (float) -Math.toDegrees(Math.atan2(dy, distance));

            mc.thePlayer.rotationYaw = yaw;
            mc.thePlayer.rotationPitch = pitch;

            KeyBinding.onTick(mc.gameSettings.keyBindUseItem.getKeyCode());
            lastPearlTime = System.currentTimeMillis();
//            KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
        }
    }

    public void sidePearl(EntityPlayer target) {
        if (utils.hasItem("Ender Pearl") && canThrowPearl()) {
            utils.switchItem("Ender Pearl");

            // Calculate a position 2 blocks to the side of the target
            double sideOffset = 1.5 + Math.random() * 1.5;
            double targetYaw = Math.toRadians(target.rotationYaw);
            double sideX = target.posX + Math.cos(targetYaw) * sideOffset;
            double sideZ = target.posZ + Math.sin(targetYaw) * sideOffset;

            double dx = sideX - mc.thePlayer.posX;
            double dy = target.posY - mc.thePlayer.posY;
            double dz = sideZ - mc.thePlayer.posZ;
            double distance = Math.sqrt(dx * dx + dz * dz);

            float yaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0F);
            float pitch = (float) -Math.toDegrees(Math.atan2(dy, distance));

            mc.thePlayer.rotationYaw = yaw;
            mc.thePlayer.rotationPitch = pitch;

            KeyBinding.onTick(mc.gameSettings.keyBindUseItem.getKeyCode());
            lastPearlTime = System.currentTimeMillis();
        }
    }

    public void escapePearl(EntityPlayer target) {
        if (utils.hasItem("Ender Pearl") && canThrowPearl()) {
            utils.switchItem("Ender Pearl");

            // Run away from the target
            double dx = mc.thePlayer.posX - target.posX;
            double dz = mc.thePlayer.posZ - target.posZ;
            float yaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0F);

            mc.thePlayer.rotationYaw = yaw;
            mc.thePlayer.rotationPitch = -45; // Aim high for max distance

            KeyBinding.onTick(mc.gameSettings.keyBindUseItem.getKeyCode());
            lastPearlTime = System.currentTimeMillis();
        }
    }
}
