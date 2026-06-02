package com.bot.projectlipsy;

import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemStack;
import net.minecraft.util.MathHelper;

import java.util.Locale;

public class Pearl {

    private final Minecraft mc;
    private final utils utils;
    private final long PEARL_COOLDOWN = 16000L;
    private long lastPearlTime = 0;
    private int pearlTicks = 0;
    private boolean pearling = false;
    private double tryhardScore = 0D;
    private double sideScore = 0D;
    private double tryhardProb = 0D;
    private double sideProb = 0D;
    private boolean sideTargetSet = false;
    private double sideTargetX = 0D;
    private double sideTargetY = 0D;
    private double sideTargetZ = 0D;

    private static final double SIDE_MIN_OFFSET = 1.5D;
    private static final double SIDE_MAX_OFFSET = 2.7D;
    private static final float SIDE_MAX_YAW_FLICK = 45.0F;

    private enum PearlMode {
        TRYHARD,
        SIDE
    }

    private PearlMode pearlMode = PearlMode.TRYHARD;

    public Pearl(Minecraft mc) {
        this.mc = mc;
        this.utils = new utils(mc);
    }

    public boolean canThrowPearl() {
        long currentTime = System.currentTimeMillis();
        return currentTime - lastPearlTime >= PEARL_COOLDOWN;
    }

    public boolean canUsePearl() {
        return utils.hasItem("Ender Pearl") && canThrowPearl();
    }

    public long msSinceLastPearl() {
        return System.currentTimeMillis() - lastPearlTime;
    }

    public int ticksSinceLastPearl() {
        return (int) msSinceLastPearl() / 1000 * 20;
    }

    public void startPearlSequence() {
        startPearlSequence(null);
    }

    public void startPearlSequence(EntityPlayer target) {
        if (!canUsePearl()) return;
        pearlTicks = 0;
        pearling = true;
        sideTargetSet = false;
        choosePearlMode(target);
    }

    public boolean isPearling() {
        return pearling;
    }

    public void updatePearl(EntityPlayer target) {
        if (target == null) {
            stopPearling();
            return;
        }

        if (!pearling) {
            startPearlSequence(target);
        }
        if (!pearling) {
            return;
        }

        if (pearlMode == PearlMode.SIDE) {
            sidePearl(target);
        } else {
            tryhardPearl(target);
        }
    }

    public void tryhardPearl(EntityPlayer target) {
        if (target == null || mc.thePlayer == null) {
            stopPearling();
            return;
        }

        double dx = target.posX - mc.thePlayer.posX;
        double dy = target.posY + target.getEyeHeight() - (mc.thePlayer.posY + mc.thePlayer.getEyeHeight());
        double dz = target.posZ - mc.thePlayer.posZ;
        double distance = Math.sqrt(dx * dx + dz * dz);

        float yaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0F);
        float pitch = (float) -Math.toDegrees(Math.atan2(dy, distance)) + (float) Math.random() * 5;

        tickPearlThrow(yaw, pitch);
    }

    public void sidePearl(EntityPlayer target) {
        if (target == null || mc.thePlayer == null) {
            stopPearling();
            return;
        }

        if (!sideTargetSet) {
            setSideTarget(target);
        }

        double dx = sideTargetX - mc.thePlayer.posX;
        double dy = sideTargetY - (mc.thePlayer.posY + mc.thePlayer.getEyeHeight());
        double dz = sideTargetZ - mc.thePlayer.posZ;
        double distance = Math.sqrt(dx * dx + dz * dz);

        float targetYaw = utils.getTargetYaw(target);
        float sideYaw = (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90.0F);
        float yawOffset = MathHelper.wrapAngleTo180_float(sideYaw - targetYaw);
        yawOffset = MathHelper.clamp_float(yawOffset, -SIDE_MAX_YAW_FLICK, SIDE_MAX_YAW_FLICK);
        float yaw = MathHelper.wrapAngleTo180_float(targetYaw + yawOffset);
        float pitch = sidePearlPitch(distance, dy);

        tickPearlThrow(yaw, pitch);
    }

    private void setSideTarget(EntityPlayer target) {
        double toTargetX = target.posX - mc.thePlayer.posX;
        double toTargetZ = target.posZ - mc.thePlayer.posZ;
        double length = Math.sqrt(toTargetX * toTargetX + toTargetZ * toTargetZ);
        if (length < 0.001D) {
            toTargetX = -Math.sin(Math.toRadians(mc.thePlayer.rotationYaw));
            toTargetZ = Math.cos(Math.toRadians(mc.thePlayer.rotationYaw));
            length = 1.0D;
        }

        double sideX = -toTargetZ / length;
        double sideZ = toTargetX / length;
        double sideSign = Math.random() < 0.5D ? -1.0D : 1.0D;
        double sideOffset = SIDE_MIN_OFFSET + Math.random() * (SIDE_MAX_OFFSET - SIDE_MIN_OFFSET);

        sideTargetX = target.posX + sideX * sideSign * sideOffset;
        sideTargetY = target.posY + 0.15D;
        sideTargetZ = target.posZ + sideZ * sideSign * sideOffset;
        sideTargetSet = true;
    }

    private float sidePearlPitch(double horizontalDistance, double dy) {
        float directPitch = (float) -Math.toDegrees(Math.atan2(dy, horizontalDistance));
        float dropPitch = 12.0F + (float) Math.min(horizontalDistance * 2.4D, 18.0D);
        return MathHelper.clamp_float(Math.max(directPitch, dropPitch), 12.0F, 34.0F);
    }

    private void tickPearlThrow(float yaw, float pitch) {
        pearlTicks++;

        if (pearlTicks <= 3) {
            mc.thePlayer.rotationYaw = yaw;
            mc.thePlayer.rotationPitch = pitch;
        }

        if (pearlTicks == 1) {
            utils.switchItem("Ender Pearl");
        } else if (pearlTicks == 3) {
            ItemStack held = mc.thePlayer.getHeldItem();
            if (utils.isPearl(held)) {
                KeyBinding.onTick(mc.gameSettings.keyBindUseItem.getKeyCode());
                lastPearlTime = System.currentTimeMillis();
            } else {
                stopPearling();
            }
        } else if (pearlTicks >= 4) {
            stopPearling();
        }
    }

    private void stopPearling() {
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindUseItem.getKeyCode(), false);
        pearlTicks = 0;
        pearling = false;
        sideTargetSet = false;
    }

    private void choosePearlMode(EntityPlayer target) {
        resetScores();
        updatePearlScores(target);
        computeProbabilities();

        pearlMode = tryhardProb > sideProb ? PearlMode.TRYHARD : PearlMode.SIDE;
        if (mc.thePlayer != null) {
            mc.thePlayer.sendChatMessage(String.format(
                    Locale.US,
                    "[Project L1psy] Pearl mode=%s tryhard=%.2f side=%.2f",
                    pearlMode == PearlMode.TRYHARD ? "tryhard" : "side",
                    tryhardProb,
                    sideProb
            ));
        }
        resetScores();
    }

    private void updatePearlScores(EntityPlayer target) {
        if (mc.thePlayer == null || target == null) {
            tryhardScore += 0.55D;
            sideScore += 0.45D;
            return;
        }

        double myHp = mc.thePlayer.getHealth();
        double targetHp = target.getHealth();
        double hpDiff = myHp - targetHp;
        double targetDistance = mc.thePlayer.getDistanceToEntity(target);
        double dy = target.posY - mc.thePlayer.posY;
        int myCombo = utils.getMyCombo();
        int targetCombo = utils.getTargetCombo();

        double dx = mc.thePlayer.posX - target.posX;
        double dz = mc.thePlayer.posZ - target.posZ;
        float targetYawToPlayer = MathHelper.wrapAngleTo180_float(
                (float) (Math.toDegrees(Math.atan2(dz, dx)) - 90F));
        boolean targetTurning = Math.abs(MathHelper.wrapAngleTo180_float(targetYawToPlayer - target.rotationYaw)) > 100F;

        boolean myOnGround = mc.thePlayer.onGround;
        boolean targetOnGround = target.onGround;
        boolean targetPotting = target.isUsingItem() && utils.isHealthPotion(target.getHeldItem());
        boolean targetSpeeding = target.isUsingItem() && utils.isSpeedPotion(target.getHeldItem());
        boolean targetEating = target.isUsingItem() && utils.isFood(target.getHeldItem());

        tryhardScore += 0.05D;

        if (hpDiff >= 4D || targetHp <= 8D) {
            tryhardScore += 0.35D;
            sideScore += 0.25D;
        }
        if (hpDiff >= 8D || targetHp <= 5D) {
            tryhardScore += 0.25D;
            sideScore += 0.10D;
        }

        if (targetDistance >= 6D) {
            tryhardScore += 0.55D;
        } else if (targetDistance >= 3.5D) {
            tryhardScore += 0.25D;
            sideScore += 0.10D;
        } else if (targetDistance <= 2.2D) {
            sideScore += 0.20D;
        }

        if (targetTurning) {
            tryhardScore += 0.35D;
            sideScore += 0.30D;
        }

        if (myCombo >= 3) {
            sideScore += 0.25D + Math.min(myCombo, 8) * 0.05D;
        }
        if (targetCombo >= 2) {
            sideScore += 0.30D + Math.min(targetCombo, 8) * 0.05D;
        }

        if (myOnGround && (!targetOnGround || dy > 0.25D)) {
            tryhardScore += 0.30D;
        } else if (!myOnGround && targetOnGround || dy < -0.25D) {
            sideScore += 0.30D;
        }

        if (targetPotting) {
            tryhardScore += 0.20D;
            sideScore += 0.35D;
        }
        if (targetSpeeding) {
            sideScore += 0.30D;
        }
        if (targetEating) {
            sideScore += 0.35D;
        }
    }

    private void computeProbabilities() {
        double maxScore = Math.max(tryhardScore, sideScore);
        double tryhardExp = Math.exp(tryhardScore - maxScore);
        double sideExp = Math.exp(sideScore - maxScore);
        double expSum = tryhardExp + sideExp;
        tryhardProb = tryhardExp / expSum;
        sideProb = sideExp / expSum;
    }

    private void resetScores() {
        tryhardScore = 0D;
        sideScore = 0D;
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
