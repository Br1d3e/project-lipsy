package com.bot.recorder;

import cpw.mods.fml.common.eventhandler.SubscribeEvent;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiButton;
import net.minecraft.client.gui.GuiScreen;
import net.minecraft.client.gui.ScaledResolution;
import net.minecraftforge.client.event.RenderGameOverlayEvent;
import org.lwjgl.opengl.GL11;

public class RecorderGui extends GuiScreen {

    private final Recorder recorder;

    public RecorderGui(Recorder recorder) {
        this.recorder = recorder;
    }

    @SubscribeEvent
    public void onRenderOverlay(RenderGameOverlayEvent.Post event) {
        if (event.type != RenderGameOverlayEvent.ElementType.ALL) return;
        if (!recorder.isRecording()) return;

        Minecraft minecraft = Minecraft.getMinecraft();
        ScaledResolution scaled = new ScaledResolution(minecraft, minecraft.displayWidth, minecraft.displayHeight);
        int width = scaled.getScaledWidth();
        int panelWidth = 60;
        int x = width - panelWidth - 3;
        int y = 4;
        int lineHeight = 7;
        float scale = 0.5F;

        drawRect(x - 3, y - 3, width - 3, y + lineHeight * 4 + 1, 0x80000000);
        GL11.glPushMatrix();
        GL11.glScalef(scale, scale, 1.0F);
        int sx = (int) (x / scale);
        int sy = (int) (y / scale);
        int scaledLineHeight = (int) (lineHeight / scale);
        minecraft.fontRendererObj.drawStringWithShadow("RECORDING", sx, sy, 0xFFFF5555);
        minecraft.fontRendererObj.drawStringWithShadow("sid: " + shortSessionId(recorder.getSessionId()), sx, sy + scaledLineHeight, 0xFFFFFFFF);
        minecraft.fontRendererObj.drawStringWithShadow("tick: " + recorder.getRecordingTickIndex(), sx, sy + scaledLineHeight * 2, 0xFFFFFFFF);
        minecraft.fontRendererObj.drawStringWithShadow("ms: " + recorder.getCurrentTimeMs(), sx, sy + scaledLineHeight * 3, 0xFFFFFFFF);
        GL11.glPopMatrix();
    }

    private String shortSessionId(String sessionId) {
        if (sessionId == null) return "N/A";
        if (sessionId.length() <= 6) return sessionId;

        return sessionId.substring(sessionId.length() - 6);
    }

    @Override
    public void initGui() {
        this.buttonList.clear();
        int buttonWidth = 200;
        int buttonHeight = 20;
        int spacing = 2;
        int startY = this.height / 2 - (buttonHeight + spacing);

        this.buttonList.add(new GuiButton(0, this.width / 2 - buttonWidth / 2, startY, buttonWidth, buttonHeight, "Recording: " + (recorder.isRecording() ? "ON" : "OFF")));
        this.buttonList.add(new GuiButton(1, this.width / 2 - buttonWidth / 2, startY + (buttonHeight + spacing), buttonWidth, buttonHeight, "Close"));
    }

    @Override
    protected void actionPerformed(GuiButton button) {
        switch (button.id) {
            case 0:
                if (recorder.isRecording()) recorder.stopRecording();
                else recorder.startRecording();
                break;
            case 1:
                this.mc.displayGuiScreen(null);
                break;
        }
        // Update button text
        initGui();
    }

    @Override
    public void drawScreen(int mouseX, int mouseY, float partialTicks) {
        // Draw a semi-transparent background
        drawRect(0, 0, this.width, this.height, 0x80000000);
        GL11.glEnable(GL11.GL_BLEND);
        this.drawCenteredString(this.fontRendererObj, "Recorder Control", this.width / 2, 20, 0xFFFFFF);
        super.drawScreen(mouseX, mouseY, partialTicks);
    }

    @Override
    public boolean doesGuiPauseGame() {
        return false;
    }
}
