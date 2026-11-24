package com.bot.projectlipsy;

import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.util.ChatComponentText;
import com.bot.projectlipsy.recordWriter.JsonLineBuilder;
import com.bot.projectlipsy.recordWriter.RuntimeFrameWriter;
import java.io.IOException;
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
    private Refill refill;
    private ModelConnect models;
    private Blockhit blockhit;
    private Retreat retreat;
    private float[] actions;
    private final RuntimeFrameWriter runtimeLog = new RuntimeFrameWriter();
    private boolean runtimeLogFailed;
    private String overrideRule;

    private enum State {
        ATTACK,
        HEAL,
        SPEED,
        EAT,
        PEARL,
        REFILL,
        RETREAT,
    }

    private State state = State.ATTACK;
    private State prevState = State.ATTACK;

    private int refillAmount;

    public int tickIndex = 0;
    private int lastPearlTick = -1;

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
        refill = new Refill(mc);
        refillAmount = 0;
        // Register this class to receive tick events
        net.minecraftforge.common.MinecraftForge.EVENT_BUS.register(this);
    }

    @SubscribeEvent
    public void onClientTick(TickEvent.ClientTickEvent event) throws Exception {
        if (event.phase != TickEvent.Phase.END) return;
        if (mc.thePlayer == null) {
            runtimeLog.close();
            runtimeLogFailed = false;
            models.sendObs(null);
            if (blockhit != null) blockhit.cancel("player missing");
            if (healing != null) { healing.stopPotting(); healing.stopUtilityUse(); }
            return;
        }

        EntityPlayer target = findClosestPlayer();
        if (target == null) {
            models.sendObs(null);
            blockhit.cancel("target missing");
            healing.stopPotting();
            healing.stopUtilityUse();
            recordRuntimeFrame(null, null, "target_missing");
            return;
        }

        if (!runtimeLog.isRecording() && !runtimeLogFailed) {
            try {
                runtimeLog.start(mc.mcDataDir);
                mc.thePlayer.addChatMessage(new ChatComponentText(
                        "[Project Lipsy] Runtime trace started: " + runtimeLog.getFile()));
            } catch (IOException e) {
                failRuntimeLog(e);
            }
        }
        models.sendObs(target);
        actions = models.getActions();

        utils.updateCombos(target);

        if (prevState != state) {
            mc.thePlayer.sendChatMessage("[Project L1psy] State: " + state);
        }

        prevState = state;
        overrideRule = null;
        stateController(target);
        models.setDecisionState(state.name());

        if ((prevState == State.EAT || prevState == State.SPEED) && state != prevState) {
            healing.stopUtilityUse();
        }

        if (state != State.ATTACK) {
            blockhit.cancel("state " + state);
        }

        if (state != State.REFILL && state != State.SPEED && state != State.EAT) {
            refill.resetGuiScreen();
        }

        String selectedState = state.name().toLowerCase(java.util.Locale.US);
        if (selectedState.equals(RuntimeFrameWriter.modelState(actions))) overrideRule = null;
        String executedState = selectedState;
        switch (state) {
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
                if (!healing.isPotting() && !healing.canPot()) {
                    state = refill.hotbarPotions() == 0 && refill.hasInventoryPotion()
                            ? State.REFILL : State.ATTACK;
                    executedState = null;
                    overrideRule = "heal_unavailable_at_execution";
                    break;
                }
                healing.startHealingSequence();
                healing.updatePotMacro(target);
                if (!healing.isPotting()) state = State.ATTACK;
                break;

            case SPEED:
                if (!refill.refillSpeed() || mc.thePlayer.getDistanceToEntity(target) < 2.5D) {
                    executedState = null;
                    overrideRule = "speed_item_unavailable_or_target_close";
                    break;
                }
                healing.speed();
                break;

            case EAT:
                if (!refill.refillFood() || mc.thePlayer.getDistanceToEntity(target) < 2.5D) {
                    executedState = null;
                    overrideRule = "eat_item_unavailable_or_target_close";
                    break;
                }
                healing.eat();
                break;

            case PEARL:
                pearl.updatePearl(target);
                break;

            case REFILL:
                if (!refill.isRefilling() && (!refill.hasInventoryPotion() || refill.getRefillAmount() <= 0)) {
                    state = State.ATTACK;
                    executedState = null;
                    overrideRule = "refill_unavailable_at_execution";
                    break;
                }

                double distance = mc.thePlayer.getDistanceToEntity(target);
                double myHp = mc.thePlayer.getHealth();
                boolean safe = distance >= 2.5D && myHp >= 6;

                if (!safe) {
                    refill.stopRefilling();
                    executedState = null;
                    overrideRule = "refill_unsafe";
                    if (Math.random() < 0.3) {
                        retreat.back(target);
                        executedState = "retreat";
                    }
                    state = State.ATTACK;
                    break;
                }
                refillAmount = refill.getRefillAmount();
                if (refill.refillPotions(refillAmount)) {
                    state = State.ATTACK;
                }
                break;

            default:
                break;

        }
        recordRuntimeFrame(selectedState, executedState, overrideRule);
        tickIndex++;
    }

    private void recordRuntimeFrame(String selectedState, String executedState, String rule) {
        if (!runtimeLog.isRecording()) return;
        JsonLineBuilder frame = new JsonLineBuilder(4096).field("tick", models.getObservationTick());
        models.appendRawFeatures(frame);
        RuntimeFrameWriter.appendModelOutput(frame, models.getActions());
        frame.field("selectedState", selectedState);
        frame.field("executedState", executedState);
        frame.field("stateAfterTick", state.name().toLowerCase(java.util.Locale.US));
        frame.field("overrideRule", rule);
        try {
            runtimeLog.writeFrame(frame);
        } catch (IOException e) {
            failRuntimeLog(e);
        }
    }

    private void failRuntimeLog(IOException e) {
        runtimeLog.close();
        runtimeLogFailed = true;
        mc.thePlayer.addChatMessage(new ChatComponentText(
                "[Project Lipsy] Runtime trace stopped: " + e.getMessage()));
    }

    private void stateController(EntityPlayer target) {
        // If we are in the middle of potting, don't change status
        if (healing.isPotting()) {
            overrideRule = "continue_healing";
            state = State.HEAL;
            return;
        }
        if (pearl.isPearling()) {
            overrideRule = "continue_pearling";
            state = State.PEARL;
            return;
        }
//        if (refill.isRefilling()) {
//            state = State.REFILL;
//            return;
//        }
        if (!refill.hasPotion()) {
            overrideRule = "no_healing_potions";
            state = State.ATTACK;
            return;
        }

//        // heal prediction from actions vector
//        boolean actionsAvailable = actions != null && actions.length >= 7;
//        boolean predPot = actionsAvailable && actions[7] == 1.0F;
        boolean canHeal = healing.canPot() && refill.hasPotion() && refill.hotbarPotions() > 0;
        boolean shouldHeal = mc.thePlayer.getHealth() <= 14;
//        boolean ruleShouldHeal = mc.thePlayer.getHealth() <= 12;
        boolean mustHeal = mc.thePlayer.getHealth() <= 4;

        boolean mustRefill = refill.hotbarPotions() <= 0;
        boolean needRefill = refill.hotbarPotions() < 6;

//
//        if (!mustHeal && tickIndex - lastPearlTick <= 80) {
//            status = Status.ATTACK;
//            return;
//        }
//
//        if (mustHeal && canHeal) {
//            healing.startHealingSequence();
//            status = Status.HEAL;
//            mc.thePlayer.sendChatMessage("[Project L1psy] Must heal. HP: " + mc.thePlayer.getHealth());
//            return;
//        } else if (actionsAvailable) {
//            if (predPot && canHeal && shouldHeal) {
//                healing.startHealingSequence();
//                status = Status.HEAL;
//                return;
//            }
//        } else {        // rule-based fallback
//            if (canHeal & ruleShouldHeal) {
//                healing.startHealingSequence();
//                status = Status.HEAL;
//            } else {
//                status = Status.ATTACK;
//            }
//            return;
//        }
//
//
//        boolean allowPearl = pearl.canUsePearl() && status == Status.ATTACK && utils.isHoldingSword();
//        boolean strongPearlRule = target.getHealth() < 8 || (
//                mc.thePlayer.getHealth() - target.getHealth() >= 4 & mc.thePlayer.getHealth() > 5)
//                ;
//        if (actions != null && actions.length >= 8) {
//            boolean predPearl = actions[6] == 1.0F;
//            float pearlProb = actions[7];
////            if (predPearl) {
////                if (allowPearl) {
////                    status = Status.PEARL;
////                    mc.thePlayer.sendChatMessage(String.format(
////                            Locale.US,
////                            "[Project L1psy] Pearl accepted prob=%.2f",
////                            pearlProb
////                    ));
////                    return;
////                } else {
////                    mc.thePlayer.sendChatMessage(String.format(
////                            Locale.US,
////                            "[Project L1psy] Pearl rejected prob=%.2f",
////                            pearlProb
////                    ));
////                }
////            }
//
//            if (allowPearl) {
////                if (strongPearlRule) {
////                    if (pearlProb >= 0.25) {
////                        status = Status.PEARL;
////                        pearl.startPearlSequence();
////                        mc.thePlayer.sendChatMessage(String.format(
////                                Locale.US,
////                                "[Project L1psy] Pearl accepted strong rule: prob=%.2f",
////                                pearlProb
////                        ));
////                        return;
////                    }
////                }
//                if (predPearl) {
//                    status = Status.PEARL;
//                    pearl.startPearlSequence(target);
////                    mc.thePlayer.sendChatMessage(String.format(
////                            Locale.US,
////                            "[Project L1psy] Pearl accepted prob=%.2f",
////                            pearlProb
////                    ));
//                    lastPearlTick = tickIndex;
//                    return;
//                } else {
////                    mc.thePlayer.sendChatMessage(String.format(
////                            Locale.US,
////                            "[Project L1psy] Pearl rejected prob=%.2f",
////                            pearlProb
////                    ));
//                }
////                mc.thePlayer.sendChatMessage(
////                        String.format(
////                                Locale.US,
////                                "[Project L1psy] allowPearl prob=%.2f",
////                                pearlProb
////                        ));
//            } else {
////                mc.thePlayer.sendChatMessage(String.format(
////                        Locale.US,
////                        "[Project L1psy] !allowPearl rejected prob=%.2f",
////                        pearlProb
////                ));
//            }
//        }
//
        double dist = mc.thePlayer.getDistanceToEntity(target);
        boolean canEat = healing.canEat();
        boolean canSpeed = healing.canSpeed();

//
//        refillAmount = refill.getRefillAmount();
//        if (refillAmount > 0) {
//            status = Status.REFILL;
//            return;
//        }



        // Updated status switcher with HL Multihead support
        boolean actionsAvailable = actions != null && actions.length >= 7;
        if (!actionsAvailable) {
            overrideRule = "model_unavailable";
            mc.thePlayer.sendChatMessage("[Project L1psy] actions vector unavailable.");
            return;
        }

        int HLState = (int) actions[6];
        if (mustHeal && canHeal) {
            overrideRule = "must_heal";
            state = State.HEAL;
            mc.thePlayer.sendChatMessage("[Project L1psy] Must heal. prob: " + actions[8]);
            return;
        }

        float healProb = actions[13];
        float pearlProb = actions[14];
        if (healProb >= 0.5F) {
            state = State.HEAL;
            overrideRule = "aux_heal";
            mc.thePlayer.sendChatMessage("[Project L1psy] Aux: Heal. prob: " + healProb);
            return;
        }
        if (pearlProb >= 0.5F) {
            state = State.PEARL;
            overrideRule = "aux_pearl";
            mc.thePlayer.sendChatMessage("[Project L1psy] Aux: Pearl. prob: " + pearlProb);
        }

        switch (HLState) {
            case 0:
                if (mustRefill) {
                    overrideRule = "must_refill";
                    state = State.REFILL;
                    return;
                }
                // If far away, prioritize buffs
                if (dist > 8) {
                    if (canSpeed) {
                        overrideRule = "distant_speed";
                        state = State.SPEED;
                        mc.thePlayer.sendChatMessage("[Project L1psy] Rule: speed. prob: " + actions[10]);
                        return;
                    }
                    if (canEat) {
                        overrideRule = "distant_eat";
                        state = State.EAT;
                        mc.thePlayer.sendChatMessage("[Project L1psy] Rule: eat. prob: " + actions[11]);
                        return;
                    }
                }

                state = State.ATTACK;
//                mc.thePlayer.sendChatMessage("[Project L1psy] State: Attack. prob: " + actions[7]);
                return;
            case 1:
                if (!canHeal) {
                    overrideRule = "heal_unavailable";
                    state = State.ATTACK;
                    return;
                }
                state = State.HEAL;
                mc.thePlayer.sendChatMessage("[Project L1psy] State: Heal. prob: " + actions[8]);
                return;
            case 2:
                state = State.PEARL;
                mc.thePlayer.sendChatMessage("[Project L1psy] State: Pearl. prob: " + actions[9]);
                return;
            case 3:
                if (!canSpeed) {
                    overrideRule = "speed_unavailable";
                    state = State.ATTACK;
                    return;
                }
                state = State.SPEED;
                mc.thePlayer.sendChatMessage("[Project L1psy] State: Speed. prob: " + actions[10]);
                return;
            case 4:
                if (!canEat) {
                    overrideRule = "eat_unavailable";
                    state = State.ATTACK;
                    return;
                }
                state = State.EAT;
                mc.thePlayer.sendChatMessage("[Project L1psy] State: Eat. prob: " + actions[11]);
                return;
            case 5:
                if (!needRefill) {
                    overrideRule = "refill_not_needed";
                    state = State.ATTACK;
                    return;
                }
                state = State.REFILL;
                mc.thePlayer.sendChatMessage("[Project L1psy] State: Refill. prob: " + actions[12]);
                return;
            default:
                overrideRule = "invalid_model_state";
                state = State.ATTACK;
        }

        // If close or buffs are not needed, attack
//        status = Status.ATTACK;
    }

    private EntityPlayer findClosestPlayer() {
        if (mc.theWorld == null || mc.thePlayer == null) return null;
        return mc.theWorld.playerEntities.stream()
                .filter(p -> p != mc.thePlayer)
                .min(Comparator.comparingDouble(mc.thePlayer::getDistanceToEntity))
                .orElse(null);
    }
}
