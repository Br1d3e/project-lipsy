package com.bot.recorder;

import io.netty.channel.Channel;
import io.netty.channel.ChannelDuplexHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.ChannelPromise;
import net.minecraft.client.Minecraft;
import net.minecraft.client.network.NetHandlerPlayClient;
import net.minecraft.network.NetworkManager;
import net.minecraft.network.Packet;

import java.lang.reflect.Field;

public class PacketHandler {
    public interface Listener {
        boolean isRecording();

        int getRecordingTickIndex();

        void onPacketHookStatus(String type, String message);

        void onOutgoingPacket(Packet packet, int tick);

        void onIncomingPacket(Packet packet, int tick);
    }

    private final Minecraft mc;
    private final String hookName;
    private final int hookCheckIntervalTicks;
    private final Listener listener;

    private NetworkManager hookedNetworkManager = null;
    private Channel hookedChannel = null;
    private int lastPacketHookCheckTick = -1;
    private boolean installedLoggedThisSession = false;
    private boolean failedLoggedThisSession = false;

    public PacketHandler(Minecraft mc, String hookName, int hookCheckIntervalTicks, Listener listener) {
        this.mc = mc;
        this.hookName = hookName;
        this.hookCheckIntervalTicks = hookCheckIntervalTicks;
        this.listener = listener;
    }

    public void resetSession() {
        lastPacketHookCheckTick = -1;
        installedLoggedThisSession = false;
        failedLoggedThisSession = false;
    }

    public void install() {
        int hookCheckTick = mc.thePlayer == null ? (int) (System.currentTimeMillis() / 50L) : mc.thePlayer.ticksExisted;
        if (hookedNetworkManager != null && lastPacketHookCheckTick >= 0 && hookCheckTick - lastPacketHookCheckTick < hookCheckIntervalTicks) {
            return;
        }
        lastPacketHookCheckTick = hookCheckTick;

        NetHandlerPlayClient netHandler = mc.getNetHandler();
        if (netHandler == null) {
            hookedNetworkManager = null;
            hookedChannel = null;
            return;
        }

        NetworkManager networkManager = netHandler.getNetworkManager();
        if (networkManager == null) return;

        Channel channel = getNetworkChannel(networkManager);
        if (channel == null) {
            hookedNetworkManager = null;
            hookedChannel = null;
            logStatus("PACKET_HOOK_FAILED", "NetworkManager channel field not found");
            return;
        }

        ChannelPipeline pipeline = channel.pipeline();
        if (pipeline.get(hookName) != null) {
            hookedNetworkManager = networkManager;
            hookedChannel = channel;
            logStatus("PACKET_HOOK_INSTALLED", "existing handler");
            return;
        }

        if (networkManager == hookedNetworkManager) {
            return;
        }

        ChannelDuplexHandler handler = new ChannelDuplexHandler() {
            @Override
            public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
                if (msg instanceof Packet && listener != null && listener.isRecording()) {
                    listener.onIncomingPacket((Packet) msg, listener.getRecordingTickIndex());
                }

                super.channelRead(ctx, msg);
            }

            @Override
            public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise) throws Exception {
                if (msg instanceof Packet && listener != null && listener.isRecording()) {
                    listener.onOutgoingPacket((Packet) msg, listener.getRecordingTickIndex());
                }

                super.write(ctx, msg, promise);
            }
        };

        try {
            if (pipeline.get("packet_handler") != null) {
                pipeline.addBefore("packet_handler", hookName, handler);
            } else {
                pipeline.addLast(hookName, handler);
            }
            hookedNetworkManager = networkManager;
            hookedChannel = channel;
            logStatus("PACKET_HOOK_INSTALLED", "inserted handler");
        } catch (IllegalArgumentException ex) {
            if (pipeline.get(hookName) != null) {
                hookedNetworkManager = networkManager;
                hookedChannel = channel;
                logStatus("PACKET_HOOK_INSTALLED", "handler already present after insert race");
            } else {
                hookedNetworkManager = null;
                hookedChannel = null;
                logStatus("PACKET_HOOK_FAILED", ex.getClass().getSimpleName() + ": " + ex.getMessage());
            }
        }
    }

    private Channel getNetworkChannel(NetworkManager networkManager) {
        String[] channelFieldNames = {"channel", "field_150746_k"};
        for (String fieldName : channelFieldNames) {
            try {
                Field channelField = NetworkManager.class.getDeclaredField(fieldName);
                channelField.setAccessible(true);
                Object value = channelField.get(networkManager);
                if (value instanceof Channel) {
                    return (Channel) value;
                }
            } catch (Exception ignored) {
            }
        }

        for (Field field : NetworkManager.class.getDeclaredFields()) {
            if (!Channel.class.isAssignableFrom(field.getType())) continue;

            try {
                field.setAccessible(true);
                Object value = field.get(networkManager);
                if (value instanceof Channel) {
                    return (Channel) value;
                }
            } catch (Exception ignored) {
            }
        }

        return null;
    }

    private void logStatus(String type, String message) {
        if (listener == null || !listener.isRecording()) return;
        if ("PACKET_HOOK_INSTALLED".equals(type)) {
            if (installedLoggedThisSession) return;
            installedLoggedThisSession = true;
        } else if ("PACKET_HOOK_FAILED".equals(type)) {
            if (failedLoggedThisSession) return;
            failedLoggedThisSession = true;
        }

        listener.onPacketHookStatus(type, message);
    }
}
