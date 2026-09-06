package com.bot.recorder.targetTracker;

/** Updates once per recording tick, using the full inventory rather than the hotbar. */
public class PearlCooldownTracker {
    private int remaining;
    private int previousPearls = -1;

    public void reset(int inventoryPearls) {
        remaining = 0;
        previousPearls = inventoryPearls;
    }

    public boolean update(boolean useAttempt, int inventoryPearls, int cooldownTicks) {
        boolean consumedOne = previousPearls >= 0 && previousPearls - inventoryPearls == 1;
        boolean accepted = consumedOne || (useAttempt && remaining == 0);
        previousPearls = inventoryPearls;
        if (accepted) remaining = cooldownTicks;
        else if (remaining > 0) remaining--;
        return accepted;
    }

    public int getRemaining() {
        return remaining;
    }
}
