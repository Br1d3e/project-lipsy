package com.bot.projectlipsy.features;

import io.netty.channel.*;
import net.minecraft.client.Minecraft;
import net.minecraft.network.NetworkManager;
import net.minecraft.network.Packet;
import net.minecraft.network.play.client.C02PacketUseEntity;
import net.minecraft.network.play.server.S28PacketEffect;
import net.minecraft.network.play.server.S13PacketDestroyEntities;
import net.minecraftforge.fml.common.ObfuscationReflectionHelper;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/** Observe packets without consuming them. Entity lookups happen on the client thread. */
final class FeaturePacketObserver {
    private static final String NAME = "lipsy_hl_features";
    private final Minecraft mc;
    private volatile NetworkManager manager;
    private volatile int tick;
    private final Queue<ObservedPacket> pending = new ConcurrentLinkedQueue<ObservedPacket>();

    FeaturePacketObserver(Minecraft mc) { this.mc = mc; }

    void install(int currentTick) {
        tick = currentTick;
        NetworkManager next = mc.getNetHandler() == null ? null : mc.getNetHandler().getNetworkManager();
        if (next == manager) return;
        pending.clear();
        manager = next;
        if (next == null) return;
        final NetworkManager connection = next;
        Channel channel = ObfuscationReflectionHelper.getPrivateValue(NetworkManager.class, next,
                "channel", "field_150746_k");
        channel.pipeline().addBefore("packet_handler", NAME, new ChannelDuplexHandler() {
            @Override public void write(ChannelHandlerContext ctx, Object message, ChannelPromise promise) throws Exception {
                if (message instanceof C02PacketUseEntity) observe((Packet) message);
                super.write(ctx, message, promise);
            }
            @Override public void channelRead(ChannelHandlerContext ctx, Object message) throws Exception {
                if (message instanceof S28PacketEffect || message instanceof S13PacketDestroyEntities) observe((Packet) message);
                super.channelRead(ctx, message);
            }
            private void observe(Packet packet) {
                if (manager == connection) pending.add(new ObservedPacket(packet, tick));
            }
        });
    }

    ObservedPacket poll() { return pending.poll(); }
    void clear() { pending.clear(); }

    static final class ObservedPacket {
        final Packet packet;
        final int tick;
        ObservedPacket(Packet packet, int tick) { this.packet = packet; this.tick = tick; }
    }
}
