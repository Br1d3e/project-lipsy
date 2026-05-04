package com.bot.projectlipsy;

import com.sun.org.apache.xpath.internal.operations.Bool;
import net.minecraft.client.Minecraft;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.entity.player.EntityPlayer;
import scala.tools.nsc.doc.base.comment.Bold;

public class Movement {
    private final Minecraft mc;

    public Movement(Minecraft mc) {
        this.mc = mc;
    }

    public void update(EntityPlayer target) {
        double dist = mc.thePlayer.getDistanceToEntity(target);

        KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), true);
        KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);

        wtap(target, dist);
    }

    private void wtap(EntityPlayer target, double dist) {
        if (target.hurtTime > 0){
            if (dist > 2.4F && dist <= 3.0F) {
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), true);
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindSprint.getKeyCode(), true);
            } else if (dist <= 2.4F) {
                KeyBinding.setKeyBindState(mc.gameSettings.keyBindForward.getKeyCode(), false);
            }
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
}
