package com.bot.projectlipsy.features;

import net.minecraftforge.common.config.Configuration;

import java.io.File;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class ServerPingConfig {
    private final Map<String, Integer> serverPing = new HashMap<String, Integer>();
    public int defaultPing = 0;

    public ServerPingConfig() {
        serverPing.put("as.minemen.club", 75);
        serverPing.put("wihar.cyou", 25);
        serverPing.put("192.168.2.180", 2);
        serverPing.put("127.0.0.1", 0);
    }

    public int pingFor(String address) {
        if (address == null) return defaultPing;
        String host = address.trim().toLowerCase(Locale.US);
        // Ports are not part of the server identity; keep IPv6 addresses intact.
        int colon = host.indexOf(':');
        if (colon >= 0 && colon == host.lastIndexOf(':')) host = host.substring(0, colon);
        Integer ping = serverPing.get(host);
        return ping == null ? defaultPing : ping;
    }

    public void load(File file) {
        Configuration config = new Configuration(file);
        config.load();
        defaultPing = config.getInt("defaultPing", "features", -1, -1, 10000,
                "Static ping in milliseconds for unconfigured servers; -1 means unknown.");
        for (Map.Entry<String, Integer> entry : serverPing.entrySet()) {
            config.get("ping", entry.getKey(), entry.getValue());
        }
        for (String host : config.getCategory("ping").keySet()) {
            serverPing.put(host.toLowerCase(Locale.US), config.get("ping", host, defaultPing).getInt());
        }
        if (config.hasChanged()) config.save();
    }
}
