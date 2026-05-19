package com.bot.projectlipsy;

import net.minecraft.entity.player.EntityPlayer;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;
import net.minecraft.client.Minecraft;

import java.util.Comparator;

@Mod(modid = ProjectLipsy.MODID, version = ProjectLipsy.VERSION, name = ProjectLipsy.NAME)
public class ProjectLipsy {
    public static final String MODID = "projectlipsy";
    public static final String VERSION = "1.0";
    public static final String NAME = "Project Lipsy";

    private final Minecraft mc = Minecraft.getMinecraft();
    private Movement movement;
    private Aiming aiming;
    private Clicking clicking;
    private Healing healing;
    private utils utils;
    private Pearl pearl;
    private ModelConnect models;
    private Blockhit blockhit;

    private enum Status{
        ATTACK,
        RETREAT,
        HEAL,
        SPEED,
        EAT,
        PEARL
    }

    private Status status = Status.ATTACK;
    private Status prevStatus = Status.ATTACK;

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) throws Exception {
        // This runs when the mod loads
        System.out.println("[Project Lipsy] Mod initialized!");
        movement = new Movement(mc);
        aiming = new Aiming(mc);
        clicking = new Clicking(mc);
        healing = new Healing(mc, movement);
        utils = new utils(mc);
        pearl = new Pearl(mc);
        models = new ModelConnect(mc);
        blockhit = new Blockhit(mc, Blockhit.Mode.PREDICTION);
        // Register this class to receive tick events
        net.minecraftforge.common.MinecraftForge.EVENT_BUS.register(this);
    }

    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) throws Exception {
        if (mc.thePlayer == null) {
            if (blockhit != null) blockhit.cancel("player missing");
            if (healing != null) healing.stopUtilityUse();
            return;
        }

        EntityPlayer target = findClosestPlayer();
        if (target == null) {
            blockhit.cancel("target missing");
            healing.stopUtilityUse();
            return;
        }

        utils.updateCombos(target);

        prevStatus = status;
        statusController(target);

        if ((prevStatus == Status.EAT || prevStatus == Status.SPEED) && status != prevStatus) {
            healing.stopUtilityUse();
        }

        if (status != Status.ATTACK) {
            blockhit.cancel("status " + status);
        }

        switch (status) {
            case ATTACK:
                models.sendObs(target);
                float[] actions = models.getActions();
                movement.update(target, actions);
                aiming.update(target, actions);
                clicking.update(target);
                blockhit.update(target, false, actions);
                break;

            case RETREAT:
                mc.thePlayer.rotationYaw = movement.getRetreatYaw(target);
                movement.sprint();
                break;

            case HEAL:
                if (!utils.hasItem("Splash Potion of Healing")){
                    healing.stopPotting();
                    status = Status.ATTACK;
                    break;
                }
                healing.updatePotMacro(target);
//                movement.sprint(); // Always retreat while healing
                break;

            case SPEED:
                healing.speed();
                break;

            case EAT:
                healing.eat();
                break;

            case PEARL:
                pearl.tryhardPearl(target);
                break;

            default:
                break;

        }

    }

    private void statusController(EntityPlayer target) {
        // If we are in the middle of potting, don't change status
        if (healing.isPotting()) {
            status = Status.HEAL;
            return;
        }

        // Health check for retreating/healing
        if (this.mc.thePlayer.getHealth() <= 10) {
            if (healing.canPot() && utils.hasItem("Splash Potion of Healing")) {
                healing.startHealingSequence();
                status = Status.HEAL;
            } else {
                status = Status.ATTACK;
            }
            return;
        }

        double dist = mc.thePlayer.getDistanceToEntity(target);
        boolean canEat = healing.canEat();
        boolean canSpeed = healing.canSpeed();

        // If far away, prioritize buffs
        if (dist > 6) {
            if (canSpeed) {
                status = Status.SPEED;
                return;
            }
            if (canEat) {
                status = Status.EAT;
                return;
            }
        }

        // If close or buffs are not needed, attack
        status = Status.ATTACK;
        if (pearl.canThrowPearl() && ((this.mc.thePlayer.getHealth() - target.getHealth() >= 8) || (target.getHealth() < 5 && this.mc.thePlayer.getHealth() > 5))){
            status = Status.PEARL;
        }
    }

    private EntityPlayer findClosestPlayer() {
//        if (mc.theWorld == null) return null;
        return mc.theWorld.playerEntities.stream()
                .filter(p -> p != mc.thePlayer)
                .min(Comparator.comparingDouble(mc.thePlayer::getDistanceToEntity))
                .orElse(null);
    }
}
