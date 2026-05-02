package com.bot.projectlipsy;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.minecraft.client.Minecraft;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.MathHelper;

import java.util.*;


public class ModelConnect {
    // Real-time data logger, sending information to model
    // Request predicted data from model
    private final Minecraft mc;
    Map<String, List<Double>> batch = new HashMap<>();
    Gson gson = new GsonBuilder().create();
    private String json;

    public ModelConnect(Minecraft mc) {
        this.mc = mc;
    }

    public void initBatch() {
        batch.put("dx", new ArrayList<>());
        batch.put("dy", new ArrayList<>());
        batch.put("dz", new ArrayList<>());
        batch.put("dist", new ArrayList<>());
        batch.put("targetVelX", new ArrayList<>());
        batch.put("targetVelY", new ArrayList<>());
        batch.put("targetVelZ", new ArrayList<>());
        batch.put("yawDiff", new ArrayList<>());
        batch.put("PitchDiff", new ArrayList<>());

    }

    public void sendPacket(EntityPlayer target) {
        if (mc.thePlayer == null || target == null) return;

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

        // Compiling the data via array
        batch.get("dx").add(dx);
        batch.get("dy").add(dy);
        batch.get("dz").add(dz);
        batch.get("dist").add(dist);
        batch.get("targetVelX").add(targetVelX);
        batch.get("targetVelY").add(targetVelY);
        batch.get("targetVelZ").add(targetVelZ);
        batch.get("yawDiff").add((double) yawDiff);
        batch.get("PitchDiff").add((double) pitchDiff);

        // Send json
        json = gson.toJson(batch);
        socketOut.write(json + "\n");
        socketOut.flush();
    }

}
