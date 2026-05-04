package com.bot.recorder;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import cpw.mods.fml.client.registry.ClientRegistry;
import cpw.mods.fml.common.FMLCommonHandler;
import cpw.mods.fml.common.Mod;
import cpw.mods.fml.common.event.FMLInitializationEvent;
import cpw.mods.fml.common.eventhandler.SubscribeEvent;
import cpw.mods.fml.common.gameevent.InputEvent;
import cpw.mods.fml.common.gameevent.TickEvent;
import io.netty.channel.Channel;
import io.netty.channel.ChannelDuplexHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.ChannelPromise;
import net.minecraft.client.Minecraft;
import net.minecraft.client.network.NetHandlerPlayClient;
import net.minecraft.client.settings.KeyBinding;
import net.minecraft.block.Block;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.item.ItemStack;
import net.minecraft.network.NetworkManager;
import net.minecraft.network.Packet;
import net.minecraft.network.play.client.C02PacketUseEntity;
import net.minecraft.network.play.client.C08PacketPlayerBlockPlacement;
import net.minecraft.potion.Potion;
import net.minecraft.util.ChatComponentText;
import net.minecraft.util.MathHelper;
import net.minecraftforge.common.MinecraftForge;
import org.lwjgl.input.Keyboard;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.*;

@Mod(modid = "projectlipsy", name = "Project Lipsy Recorder", version = "2.0")
public class Recorder {

    private final Minecraft mc = Minecraft.getMinecraft();
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final Gson JSONL_GSON = new Gson();
    private static final String RECORDER_VERSION = "v2";
    private static final int PEARL_COOLDOWN_TICKS = 320;
    private static final int WALL_CHECK_DISTANCE = 5;
    private static final String PACKET_HOOK_NAME = "lipsy_recorder_packet_hook";
    public static final File ROOT_DIR = new File(Minecraft.getMinecraft().mcDataDir, "ProjectLipsy/recordings");

    private KeyBinding recorderGuiKey;

    // === record state ===
    private boolean recording = false;
    private String sessionId;
    private File sessionDir;
    private FileWriter framesWriter;
    private long startTimeMs;
    private long endTimeMs;
    private long startWorldTick;
    private String serverIp = "unknown";
    private int recordingTickIndex = 0;
    private Float previousMyYaw = null;
    private Float previousMyPitch = null;
    private final Map<Integer, Float> previousTargetYawById = new HashMap<Integer, Float>();
    private final Map<Integer, Float> previousTargetPitchById = new HashMap<Integer, Float>();
    private int pearlCooldownTicksRemaining = 0;
    private boolean previousUseItemPressed = false;
    private boolean previousAttackPressed = false;
    private int lastAttackTick = -1;
    private int lastEatTick = -1;
    private int lastSpeedTick = -1;
    private int lastDamageTakenTick = -1;
    private int lastHitLandedTick = -1;
    private int lastPearlTick = -1;
    private int lastPotTick = -1;
    private int lastUseItemPacketTick = -1;
    private NetworkManager hookedNetworkManager = null;
    private Channel hookedChannel = null;

    // === combo tracking ===
    private int myCombo = 0;
    private int targetCombo = 0;
    private int lastMyHurtTime = 0;
    private int lastTargetHurtTime = 0;

    @Mod.EventHandler
    public void init(FMLInitializationEvent event) {
        System.out.println("[Lipsy] Recorder initialized.");
        if (!ROOT_DIR.exists()) ROOT_DIR.mkdirs();

        recorderGuiKey = new KeyBinding("Recorder GUI", Keyboard.KEY_RSHIFT, "Project Lipsy");
        ClientRegistry.registerKeyBinding(recorderGuiKey);
        MinecraftForge.EVENT_BUS.register(this);
        FMLCommonHandler.instance().bus().register(this);
    }

    @SubscribeEvent
    public void onKeyInput(InputEvent.KeyInputEvent event) {
        if (recorderGuiKey.isPressed()) {
            mc.displayGuiScreen(new RecorderGui(this));
        }
    }

    @SubscribeEvent
    public void onTick(TickEvent.ClientTickEvent e) {
        if (e.phase != TickEvent.Phase.END) return;
        installPacketHook();
        if (!recording) return;
        if (mc.thePlayer == null) return;

        EntityPlayer target = findClosestPlayer();

        if (target != null) {
            updateCombos(target);
        }

        int currentTick = recordingTickIndex;
        boolean useItemPressed = key(mc.gameSettings.keyBindUseItem);
        boolean useItemStarted = useItemPressed && !previousUseItemPressed;
        boolean attackPressed = key(mc.gameSettings.keyBindAttack);
        boolean attackStarted = attackPressed && !previousAttackPressed;
        ItemStack heldItem = mc.thePlayer.getHeldItem();

        if (attackStarted) {
            lastAttackTick = currentTick;
        }

        if (useItemStarted && isSplashHealthPotion(heldItem)) {
            lastPotTick = currentTick;
        }

        if (useItemStarted && isFood(heldItem)) {
            lastEatTick = currentTick;
        }

        if (useItemStarted && isSpeedPotion(heldItem)) {
            lastSpeedTick = currentTick;
        }

        if (isPearl(heldItem) && useItemStarted && pearlCooldownTicksRemaining == 0) {
            lastPearlTick = currentTick;
            pearlCooldownTicksRemaining = PEARL_COOLDOWN_TICKS;
        } else if (pearlCooldownTicksRemaining > 0) {
            pearlCooldownTicksRemaining--;
        }
        previousUseItemPressed = useItemPressed;
        previousAttackPressed = attackPressed;

        long time = System.currentTimeMillis();
        Float myDeltaYaw = previousMyYaw == null ? null : MathHelper.wrapAngleTo180_float(mc.thePlayer.rotationYaw - previousMyYaw);
        Float myDeltaPitch = previousMyPitch == null ? null : mc.thePlayer.rotationPitch - previousMyPitch;
        previousMyYaw = mc.thePlayer.rotationYaw;
        previousMyPitch = mc.thePlayer.rotationPitch;

        boolean targetExists = target != null;
        Double targetRelX = null;
        Double targetRelY = null;
        Double targetRelZ = null;
        Double targetDistance = null;
        Double yawDiffToTarget = null;
        Double pitchDiffToTarget = null;
        Double hpDiff = null;
        Float targetYawVel = null;
        Float targetPitchVel = null;

        if (targetExists) {
            targetRelX = target.posX - mc.thePlayer.posX;
            targetRelY = (target.posY + target.getEyeHeight()) - (mc.thePlayer.posY + mc.thePlayer.getEyeHeight());
            targetRelZ = target.posZ - mc.thePlayer.posZ;
            targetDistance = (double) mc.thePlayer.getDistanceToEntity(target);
            double distanceXZ = Math.hypot(targetRelX, targetRelZ);
            double targetAimYaw = Math.toDegrees(Math.atan2(targetRelZ, targetRelX)) - 90F;
            double targetAimPitch = -Math.toDegrees(Math.atan2(targetRelY, distanceXZ));
            yawDiffToTarget = (double) MathHelper.wrapAngleTo180_float((float) targetAimYaw - mc.thePlayer.rotationYaw);
            pitchDiffToTarget = (double) MathHelper.wrapAngleTo180_float((float) targetAimPitch - mc.thePlayer.rotationPitch);
            hpDiff = (double) (mc.thePlayer.getHealth() - target.getHealth());

            int targetId = target.getEntityId();
            Float previousTargetYaw = previousTargetYawById.get(targetId);
            Float previousTargetPitch = previousTargetPitchById.get(targetId);
            targetYawVel = previousTargetYaw == null ? null : MathHelper.wrapAngleTo180_float(target.rotationYaw - previousTargetYaw);
            targetPitchVel = previousTargetPitch == null ? null : target.rotationPitch - previousTargetPitch;
            previousTargetYawById.put(targetId, target.rotationYaw);
            previousTargetPitchById.put(targetId, target.rotationPitch);
        }

        Entity crosshairEntity = getMouseOverEntity();

        Map<String, Object> frame = new LinkedHashMap<>();
        frame.put("recordingTickIndex", recordingTickIndex++);
        frame.put("timeMs", time);
        frame.put("ts", time);
        frame.put("worldTick", getWorldTick());
        frame.put("myPosX", mc.thePlayer.posX);
        frame.put("myPosY", mc.thePlayer.posY);
        frame.put("myPosZ", mc.thePlayer.posZ);
        frame.put("myMotionX", mc.thePlayer.motionX);
        frame.put("myMotionY", mc.thePlayer.motionY);
        frame.put("myMotionZ", mc.thePlayer.motionZ);
        frame.put("myYaw", mc.thePlayer.rotationYaw);
        frame.put("myPitch", mc.thePlayer.rotationPitch);
        frame.put("myDeltaYaw", myDeltaYaw);
        frame.put("myDeltaPitch", myDeltaPitch);
        frame.put("myHp", mc.thePlayer.getHealth());
        frame.put("myHurtTime", mc.thePlayer.hurtTime);
        frame.put("myHurtResistantTime", mc.thePlayer.hurtResistantTime);
        frame.put("myOnGround", mc.thePlayer.onGround);
        frame.put("mySprinting", mc.thePlayer.isSprinting());
        frame.put("myMoveForward", mc.thePlayer.moveForward);
        frame.put("myMoveStrafing", mc.thePlayer.moveStrafing);
        frame.put("mySwingProgress", mc.thePlayer.swingProgress);
        frame.put("myUsingItem", mc.thePlayer.isUsingItem());
        frame.put("myHeldItemType", getHeldItemType(mc.thePlayer));
        frame.put("myHasSpeedPotion", hasPotion(mc.thePlayer, Potion.moveSpeed));
        frame.put("mySpeedDurationTicks", getPotionDuration(mc.thePlayer, Potion.moveSpeed));
        frame.put("myFoodLevel", mc.thePlayer.getFoodStats().getFoodLevel());
        frame.put("myLineOfSight", getMouseOverEntity() != null);
        frame.put("myCombo", myCombo);
        frame.put("targetCombo", targetCombo);
        frame.put("wallBehind", wallInDirection(180.0F));
        frame.put("wallLeft", wallInDirection(-90.0F));
        frame.put("wallRight", wallInDirection(90.0F));
        frame.put("wallFront", wallInDirection(0.0F));
        frame.put("pearlCooldownTicksRemaining", pearlCooldownTicksRemaining);
        frame.put("pearlCooldownSecondsRemaining", pearlCooldownTicksRemaining / 20.0F);
        frame.put("selectedSlot", mc.thePlayer.inventory.currentItem);
        frame.put("hotbar", getHotbarSnapshot());
        frame.put("inventoryFood", countFoodItems());
        frame.put("inventorySpeedPotions", countSpeedPotions());
        frame.put("inventoryHealthPotions", countSplashHealthPotions());
        frame.put("inventoryPearls", countPearls());
        frame.put("crosshairTargetType", getCrosshairTargetType());
        frame.put("crosshairTargetId", crosshairEntity == null ? null : crosshairEntity.getEntityId());
        frame.put("crosshairOnTarget", targetExists && crosshairEntity == target);
        frame.put("crosshairDistance", crosshairEntity == null ? null : (double) mc.thePlayer.getDistanceToEntity(crosshairEntity));
        frame.put("ticksSinceLastAttack", ticksSince(lastAttackTick, currentTick));
        frame.put("ticksSinceLastPot", ticksSince(lastPotTick, currentTick));
        frame.put("ticksSinceLastEat", ticksSince(lastEatTick, currentTick));
        frame.put("ticksSinceLastSpeed", ticksSince(lastSpeedTick, currentTick));
        frame.put("ticksSinceLastDamageTaken", ticksSince(lastDamageTakenTick, currentTick));
        frame.put("ticksSinceLastHitLanded", ticksSince(lastHitLandedTick, currentTick));
        frame.put("ticksSinceLastPearl", ticksSince(lastPearlTick, currentTick));
        frame.put("ticksSinceLastUseItemPacket", ticksSince(lastUseItemPacketTick, currentTick));
        frame.put("lastAttackTick", nullableTick(lastAttackTick));
        frame.put("lastEatTick", nullableTick(lastEatTick));
        frame.put("lastSpeedTick", nullableTick(lastSpeedTick));
        frame.put("lastDamageTakenTick", nullableTick(lastDamageTakenTick));
        frame.put("lastHitLandedTick", nullableTick(lastHitLandedTick));
        frame.put("lastPearlTick", nullableTick(lastPearlTick));
        frame.put("lastPotTick", nullableTick(lastPotTick));
        frame.put("lastUseItemPacketTick", nullableTick(lastUseItemPacketTick));
        frame.put("blockDistanceFront", blockDistanceInDirection(0.0F));
        frame.put("blockDistanceBack", blockDistanceInDirection(180.0F));
        frame.put("blockDistanceLeft", blockDistanceInDirection(-90.0F));
        frame.put("blockDistanceRight", blockDistanceInDirection(90.0F));
        frame.put("aForward", actionForward());
        frame.put("aStrafe", actionStrafe());
        frame.put("aJump", key(mc.gameSettings.keyBindJump) ? 1 : 0);
        frame.put("aAttack", attackPressed || lastAttackTick == currentTick ? 1 : 0);
        frame.put("aUseItem", useItemPressed || lastUseItemPacketTick == currentTick ? 1 : 0);

        frame.put("targetExists", targetExists);
        frame.put("targetRelX", targetRelX);
        frame.put("targetRelY", targetRelY);
        frame.put("targetRelZ", targetRelZ);
        frame.put("targetDistance", targetDistance);
        frame.put("targetMotionX", targetExists ? target.motionX : null);
        frame.put("targetMotionY", targetExists ? target.motionY : null);
        frame.put("targetMotionZ", targetExists ? target.motionZ : null);
        frame.put("targetYaw", targetExists ? target.rotationYaw : null);
        frame.put("targetPitch", targetExists ? target.rotationPitch : null);
        frame.put("targetDeltaYaw", targetYawVel);
        frame.put("targetDeltaPitch", targetPitchVel);
        frame.put("targetHp", targetExists ? target.getHealth() : null);
        frame.put("targetHurtTime", targetExists ? target.hurtTime : null);
        frame.put("targetHurtResistantTime", targetExists ? target.hurtResistantTime : null);
        frame.put("targetOnGround", targetExists ? target.onGround : null);
        frame.put("targetSprinting", targetExists ? target.isSprinting() : null);
        frame.put("targetSwingProgress", targetExists ? target.swingProgress : null);
        frame.put("targetUsingItem", targetExists ? target.isUsingItem() : null);
        frame.put("targetHeldItemType", targetExists ? getHeldItemType(target) : null);
        frame.put("targetSpeedPotion", targetExists ? hasPotion(target, Potion.moveSpeed) : null);
        frame.put("targetSpeedDurationTicks", targetExists ? getPotionDuration(target, Potion.moveSpeed): null);
        frame.put("targetFood", targetExists ? target.getFoodStats().getFoodLevel() : null);
        frame.put("lineOfSight", targetExists && getMouseOverEntity() == target);

        frame.put("yawDiffToTarget", yawDiffToTarget);
        frame.put("pitchDiffToTarget", pitchDiffToTarget);
        frame.put("hpDiff", hpDiff);
        writeFrame(frame);

    }



    private void updateCombos(EntityPlayer target) {
        if (target.hurtTime > lastTargetHurtTime) {
            myCombo++;
            targetCombo = 0;
            lastHitLandedTick = recordingTickIndex;
        }
        lastTargetHurtTime = target.hurtTime;

        if (mc.thePlayer.hurtTime > lastMyHurtTime) {
            targetCombo++;
            myCombo = 0;
            lastDamageTakenTick = recordingTickIndex;
        }
        lastMyHurtTime = mc.thePlayer.hurtTime;
    }

    private EntityPlayer findClosestPlayer() {
        if (mc.theWorld == null) return null;
        EntityPlayer closestPlayer = null;
        double closestDistance = Double.MAX_VALUE;

        for (Object playerObject : mc.theWorld.playerEntities) {
            if (playerObject instanceof EntityPlayer) {
                EntityPlayer player = (EntityPlayer) playerObject;
                if (player != mc.thePlayer && !player.isInvisible() && player.getHealth() > 0 && !player.isDead) {
                    double distance = mc.thePlayer.getDistanceToEntity(player);
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestPlayer = player;
                    }
                }
            }
        }
        return closestPlayer;
    }

    private boolean isHoldingItem(String displayName) {
        if (mc.thePlayer.getHeldItem() != null) {
            return mc.thePlayer.getHeldItem().getDisplayName().contains(displayName);
        }
        return false;
    }

    private boolean hasItem(String displayName) {
        for (int i = 0; i < 36; i++) {
            if (mc.thePlayer.inventory.getStackInSlot(i) != null && mc.thePlayer.inventory.getStackInSlot(i).getDisplayName().contains(displayName)) {
                return true;
            }
        }
        return false;
    }

    private boolean key(KeyBinding key) {
        return key.getIsKeyPressed();
    }

    private void installPacketHook() {
        NetHandlerPlayClient netHandler = mc.getNetHandler();
        if (netHandler == null) {
            hookedNetworkManager = null;
            hookedChannel = null;
            return;
        }

        NetworkManager networkManager = netHandler.getNetworkManager();
        if (networkManager == null || networkManager == hookedNetworkManager) return;

        Channel channel = getNetworkChannel(networkManager);
        if (channel == null) return;

        ChannelPipeline pipeline = channel.pipeline();
        if (pipeline.get(PACKET_HOOK_NAME) != null) {
            hookedNetworkManager = networkManager;
            hookedChannel = channel;
            return;
        }

        ChannelDuplexHandler handler = new ChannelDuplexHandler() {
            @Override
            public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise) throws Exception {
                if (msg instanceof Packet) {
                    handleOutgoingPacket((Packet) msg);
                }

                super.write(ctx, msg, promise);
            }
        };

        try {
            if (pipeline.get("packet_handler") != null) {
                pipeline.addBefore("packet_handler", PACKET_HOOK_NAME, handler);
            } else {
                pipeline.addLast(PACKET_HOOK_NAME, handler);
            }
            hookedNetworkManager = networkManager;
            hookedChannel = channel;
        } catch (IllegalArgumentException ignored) {
            hookedNetworkManager = networkManager;
            hookedChannel = channel;
        }
    }

    private Channel getNetworkChannel(NetworkManager networkManager) {
        try {
            Field channelField = NetworkManager.class.getDeclaredField("channel");
            channelField.setAccessible(true);

            return (Channel) channelField.get(networkManager);
        } catch (Exception ex) {
            return null;
        }
    }

    private void handleOutgoingPacket(Packet packet) {
        if (!recording) return;

        int tick = recordingTickIndex;
        if (packet instanceof C02PacketUseEntity) {
            handleUseEntityPacket((C02PacketUseEntity) packet, tick);
        } else if (packet instanceof C08PacketPlayerBlockPlacement) {
            handleUseItemPacket((C08PacketPlayerBlockPlacement) packet, tick);
        }
    }

    private void handleUseEntityPacket(C02PacketUseEntity packet, int tick) {
        if (packet.getAction() != C02PacketUseEntity.Action.ATTACK) return;

        lastAttackTick = tick;
        lastHitLandedTick = tick;
    }

    private void handleUseItemPacket(C08PacketPlayerBlockPlacement packet, int tick) {
        ItemStack stack = packet.getStack();
        if (stack == null && mc.thePlayer != null) {
            stack = mc.thePlayer.getHeldItem();
        }

        lastUseItemPacketTick = tick;

        if (isSplashHealthPotion(stack)) {
            lastPotTick = tick;
        }

        if (isFood(stack)) {
            lastEatTick = tick;
        }

        if (isSpeedPotion(stack)) {
            lastSpeedTick = tick;
        }

        if (isPearl(stack)) {
            lastPearlTick = tick;
            pearlCooldownTicksRemaining = PEARL_COOLDOWN_TICKS;
        }
    }

    private String getHeldItemType(EntityPlayer player) {
        ItemStack stack = player.getHeldItem();
        if (stack == null || stack.getItem() == null) return null;

        return stack.getItem().getUnlocalizedName();
    }

    private Integer getPotionDuration(EntityPlayer player, Potion potion) {
        if (!player.isPotionActive(potion)) return null;

        return player.getActivePotionEffect(potion).getDuration();
    }

    private boolean hasPotion(EntityPlayer player, Potion potion) {
        return player.isPotionActive(potion);
    }

    private boolean isHoldingPearl() {
        ItemStack held = mc.thePlayer.getHeldItem();

        return isPearl(held);
    }

    private List<String> getHotbarSnapshot() {
        List<String> hotbar = new ArrayList<String>();

        for (int i = 0; i < 9; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            hotbar.add(getItemType(stack));
        }

        return hotbar;
    }

    private int countFoodItems() {
        return countInventoryItems(new ItemMatcher() {
            public boolean matches(ItemStack stack) {
                return isFood(stack);
            }
        });
    }

    private int countSpeedPotions() {
        return countInventoryItems(new ItemMatcher() {
            public boolean matches(ItemStack stack) {
                return isSpeedPotion(stack);
            }
        });
    }

    private int countSplashHealthPotions() {
        return countInventoryItems(new ItemMatcher() {
            public boolean matches(ItemStack stack) {
                return isSplashHealthPotion(stack);
            }
        });
    }

    private int countPearls() {
        return countInventoryItems(new ItemMatcher() {
            public boolean matches(ItemStack stack) {
                return isPearl(stack);
            }
        });
    }

    private int countInventoryItems(ItemMatcher matcher) {
        int count = 0;

        for (int i = 0; i < 36; i++) {
            ItemStack stack = mc.thePlayer.inventory.getStackInSlot(i);
            if (stack != null && matcher.matches(stack)) {
                count += stack.stackSize;
            }
        }

        return count;
    }

    private String getItemType(ItemStack stack) {
        if (stack == null || stack.getItem() == null) return null;

        return stack.getItem().getUnlocalizedName();
    }

    private String getDisplayNameLower(ItemStack stack) {
        return stack == null ? "" : stack.getDisplayName().toLowerCase(Locale.US);
    }

    private String getItemTypeLower(ItemStack stack) {
        String type = getItemType(stack);

        return type == null ? "" : type.toLowerCase(Locale.US);
    }

    private boolean isFood(ItemStack stack) {
        String type = getItemTypeLower(stack);

        return type.contains("cooked_beef") || type.contains("golden_carrot");
    }

    private boolean isSpeedPotion(ItemStack stack) {
        String name = getDisplayNameLower(stack);

        return name.contains("swiftness") && name.contains("ii");
    }

    private boolean isSplashHealthPotion(ItemStack stack) {
        String name = getDisplayNameLower(stack);

        return name.contains("splash") && (name.contains("instant health ii") || name.contains("healing ii"));
    }

    private boolean isPearl(ItemStack stack) {
        return getItemTypeLower(stack).contains("ender_pearl");
    }

    private boolean wallInDirection(float yawOffset) {
        Integer distance = blockDistanceInDirection(yawOffset);

        return distance != null && distance <= WALL_CHECK_DISTANCE;
    }

    private Integer blockDistanceInDirection(float yawOffset) {
        if (mc.theWorld == null || mc.thePlayer == null) return null;

        double yawRadians = Math.toRadians(mc.thePlayer.rotationYaw + yawOffset);
        double dirX = -Math.sin(yawRadians);
        double dirZ = Math.cos(yawRadians);

        for (int distance = 1; distance <= WALL_CHECK_DISTANCE; distance++) {
            int x = MathHelper.floor_double(mc.thePlayer.posX + dirX * distance);
            int y = MathHelper.floor_double(mc.thePlayer.posY + 1.0D);
            int z = MathHelper.floor_double(mc.thePlayer.posZ + dirZ * distance);
            Block block = mc.theWorld.getBlock(x, y, z);

            if (block != null && block.getMaterial().isSolid()) return distance;
        }

        return null;
    }

    private Entity getMouseOverEntity() {
        if (mc.objectMouseOver == null) return null;

        return mc.objectMouseOver.entityHit;
    }

    private String getCrosshairTargetType() {
        if (mc.objectMouseOver == null || mc.objectMouseOver.typeOfHit == null) return null;

        return mc.objectMouseOver.typeOfHit.name();
    }

    private Integer nullableTick(int tick) {
        return tick < 0 ? null : tick;
    }

    private Integer ticksSince(int tick, int currentTick) {
        return tick < 0 ? null : currentTick - tick;
    }

    private int actionForward() {
        int forward = key(mc.gameSettings.keyBindForward) ? 1 : 0;
        int back = key(mc.gameSettings.keyBindBack) ? 1 : 0;

        return forward - back;
    }

    private int actionStrafe() {
        int right = key(mc.gameSettings.keyBindRight) ? 1 : 0;
        int left = key(mc.gameSettings.keyBindLeft) ? 1 : 0;

        return right - left;
    }

    private interface ItemMatcher {
        boolean matches(ItemStack stack);
    }

    private void writeFrame(Map<String, Object> frame) {
        if (framesWriter == null) return;

        try {
            JSONL_GSON.toJson(frame, framesWriter);
            framesWriter.write("\n");
            framesWriter.flush();
        } catch (IOException ex) {
            ex.printStackTrace();
            sendMessage("Failed to write frame: " + ex.getMessage());
        }
    }

    private void writeSessionMeta() {
        if (sessionDir == null) return;

        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("sessionId", sessionId);
        meta.put("recorderVersion", RECORDER_VERSION);
        meta.put("startTimeMs", startTimeMs);
        meta.put("endTimeMs", endTimeMs);
        meta.put("startWorldTick", startWorldTick);
        meta.put("serverIp", serverIp);

        try {
            File out = new File(sessionDir, "session_meta.json");
            FileWriter w = new FileWriter(out);
            GSON.toJson(meta, w);
            w.close();
        } catch (IOException ex) {
            ex.printStackTrace();
            sendMessage("Failed to write session_meta.json: " + ex.getMessage());
        }
    }

    private String createSessionId(long timeMs) {
        String base = "session_" + timeMs;
        String candidate = base;
        int suffix = 1;

        while (new File(ROOT_DIR, candidate).exists()) {
            candidate = base + "_" + suffix;
            suffix++;
        }

        return candidate;
    }

    private long getWorldTick() {
        return mc.theWorld == null ? 0 : mc.theWorld.getTotalWorldTime();
    }

    private String getServerIp() {
        if (mc.getCurrentServerData() != null && mc.getCurrentServerData().serverIP != null) {
            return mc.getCurrentServerData().serverIP;
        }

        return "singleplayer";
    }

    private void sendMessage(String message) {
        mc.thePlayer.addChatMessage(new ChatComponentText("\u00A77[\u00A7bLipsy\u00A77] \u00A7f" + message));
    }

    public void startRecording() {
        if (recording) return;

        if (!ROOT_DIR.exists()) ROOT_DIR.mkdirs();

        startTimeMs = System.currentTimeMillis();
        endTimeMs = 0;
        startWorldTick = getWorldTick();
        serverIp = getServerIp();
        sessionId = createSessionId(startTimeMs);
        sessionDir = new File(ROOT_DIR, sessionId);
        recordingTickIndex = 0;
        previousMyYaw = null;
        previousMyPitch = null;
        previousTargetYawById.clear();
        previousTargetPitchById.clear();
        pearlCooldownTicksRemaining = 0;
        previousUseItemPressed = false;
        previousAttackPressed = false;
        lastAttackTick = -1;
        lastEatTick = -1;
        lastSpeedTick = -1;
        lastDamageTakenTick = -1;
        lastHitLandedTick = -1;
        lastPearlTick = -1;
        lastPotTick = -1;
        lastUseItemPacketTick = -1;
        myCombo = 0;
        targetCombo = 0;
        lastMyHurtTime = 0;
        lastTargetHurtTime = 0;

        if (!sessionDir.exists() && !sessionDir.mkdirs()) {
            sendMessage("Failed to create session folder.");
            return;
        }

        try {
            framesWriter = new FileWriter(new File(sessionDir, "frames.jsonl"));
            writeSessionMeta();
            recording = true;
            sendMessage("Recording started: " + sessionId);
        } catch (IOException ex) {
            ex.printStackTrace();
            sendMessage("Failed to start recording: " + ex.getMessage());
            closeFramesWriter();
        }
    }

    public void stopRecording() {
        if (!recording) return;

        recording = false;
        endTimeMs = System.currentTimeMillis();
        writeSessionMeta();
        closeFramesWriter();
        sendMessage("Recording stopped: " + sessionId);
    }

    private void closeFramesWriter() {
        if (framesWriter == null) return;

        try {
            framesWriter.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        framesWriter = null;
    }


    public boolean isRecording() {return recording;}

}
