package com.bot.projectlipsy;

import net.minecraft.entity.player.EntityPlayer;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.common.event.FMLInitializationEvent;
import net.minecraftforge.fml.common.eventhandler.SubscribeEvent;
import net.minecraftforge.fml.common.gameevent.TickEvent;
import net.minecraft.client.Minecraft;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Locale;

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
    private Retreat retreat;
    private float[] actions;

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

    private int pearlCooldownTicksRemaining = 0;

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) throws Exception {
        // This runs when the mod loads
        System.out.println("[Project Lipsy] Mod initialized!");
        movement = new Movement(mc);
        aiming = new Aiming(mc);
        clicking = new Clicking(mc);
        retreat = new Retreat(mc, movement);
        healing = new Healing(mc, movement, retreat);
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

        models.sendObs(target);
        actions = models.getActions();

        utils.updateCombos(target);

        if (prevStatus != status) {
            mc.thePlayer.sendChatMessage("[Project L1psy] Status: " + status);
        }

        prevStatus = status;
        statusController(target);
        models.setDecisionState(status.name());

        if ((prevStatus == Status.EAT || prevStatus == Status.SPEED) && status != prevStatus) {
            healing.stopUtilityUse();
        }

        if (status != Status.ATTACK) {
            blockhit.cancel("status " + status);
        }

        pearlCooldownTicksRemaining = Math.max(pearlCooldownTicksRemaining - 1, 0);

        switch (status) {
            case ATTACK:
                movement.update(target, actions);
                aiming.update(target, actions);
                clicking.update(target);
                blockhit.update(target, false, actions);
                break;

            case RETREAT:
                retreat.turn(target);
                break;

            case HEAL:
                if (!utils.hasItem("Splash Potion of Healing")){
                    healing.stopPotting();
                    status = Status.ATTACK;
                    break;
                }
                healing.updatePotMacro(target);
                break;

            case SPEED:
                healing.speed();
                break;

            case EAT:
                healing.eat();
                break;

            case PEARL:
                pearlCooldownTicksRemaining = 16 * 20;
                pearl.updatePearl(target);
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
        if (pearl.isPearling()) {
            status = Status.PEARL;
            return;
        }

        // heal prediction from actions vector
        boolean actionsAvailable = actions != null && actions.length >= 6;
        boolean predPot = actionsAvailable && actions[5] == 1.0F;
        boolean canHeal = healing.canPot() && utils.hasItem("Splash Potion of Healing");
        boolean shouldHeal = mc.thePlayer.getHealth() <= 14;
        boolean ruleShouldHeal = mc.thePlayer.getHealth() <= 12;
        boolean mustHeal = mc.thePlayer.getHealth() <= 4;

        if (mustHeal && canHeal) {
            healing.startHealingSequence();
            status = Status.HEAL;
            mc.thePlayer.sendChatMessage("[Project L1psy] Must heal. HP: " + mc.thePlayer.getHealth());
            return;
        } else if (actionsAvailable) {
            if (predPot && canHeal && shouldHeal) {
                healing.startHealingSequence();
                status = Status.HEAL;
                return;
            }
        } else {        // rule-based fallback
            if (canHeal & ruleShouldHeal) {
                healing.startHealingSequence();
                status = Status.HEAL;
            } else {
                status = Status.ATTACK;
            }
            return;
        }


        boolean allowPearl = pearl.canUsePearl() && status == Status.ATTACK && utils.isHoldingSword();
        boolean strongPearlRule = target.getHealth() < 8 || (
                mc.thePlayer.getHealth() - target.getHealth() >= 4 & mc.thePlayer.getHealth() > 5)
                ;
        if (actions != null && actions.length >= 8) {
            boolean predPearl = actions[6] == 1.0F;
            float pearlProb = actions[7];
//            if (predPearl) {
//                if (allowPearl) {
//                    status = Status.PEARL;
//                    mc.thePlayer.sendChatMessage(String.format(
//                            Locale.US,
//                            "[Project L1psy] Pearl accepted prob=%.2f",
//                            pearlProb
//                    ));
//                    return;
//                } else {
//                    mc.thePlayer.sendChatMessage(String.format(
//                            Locale.US,
//                            "[Project L1psy] Pearl rejected prob=%.2f",
//                            pearlProb
//                    ));
//                }
//            }

            if (allowPearl) {
//                if (strongPearlRule) {
//                    if (pearlProb >= 0.25) {
//                        status = Status.PEARL;
//                        pearl.startPearlSequence();
//                        mc.thePlayer.sendChatMessage(String.format(
//                                Locale.US,
//                                "[Project L1psy] Pearl accepted strong rule: prob=%.2f",
//                                pearlProb
//                        ));
//                        return;
//                    }
//                }
                if (pearlProb >= 0.38) {
                    status = Status.PEARL;
                    pearl.startPearlSequence(target);
                    mc.thePlayer.sendChatMessage(String.format(
                            Locale.US,
                            "[Project L1psy] Pearl accepted prob=%.2f",
                            pearlProb
                    ));
                    return;
                } else {
                    mc.thePlayer.sendChatMessage(String.format(
                            Locale.US,
                            "[Project L1psy] Pearl rejected prob=%.2f",
                            pearlProb
                    ));
                }
            }
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
    }

    private EntityPlayer findClosestPlayer() {
//        if (mc.theWorld == null) return null;
        return mc.theWorld.playerEntities.stream()
                .filter(p -> p != mc.thePlayer)
                .min(Comparator.comparingDouble(mc.thePlayer::getDistanceToEntity))
                .orElse(null);
    }
}
