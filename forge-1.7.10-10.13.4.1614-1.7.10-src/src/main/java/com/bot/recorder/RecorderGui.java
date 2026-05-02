package com.bot.recorder;

import net.minecraft.client.gui.GuiButton;
import net.minecraft.client.gui.GuiScreen;
import org.lwjgl.opengl.GL11;

public class RecorderGui extends GuiScreen {

    private final Recorder recorder;

    public RecorderGui(Recorder recorder) {
        this.recorder = recorder;
    }

    @Override
    public void initGui() {
        this.buttonList.clear();
        int buttonWidth = 200;
        int buttonHeight = 20;
        int spacing = 2;
        int startY = this.height / 2 - (4 * (buttonHeight + spacing));

        this.buttonList.add(new GuiButton(0, this.width / 2 - buttonWidth / 2, startY, buttonWidth, buttonHeight, "Aim Recording: " + (recorder.isRecordingAim() ? "ON" : "OFF")));
        this.buttonList.add(new GuiButton(1, this.width / 2 - buttonWidth / 2, startY + (buttonHeight + spacing), buttonWidth, buttonHeight, "WTap Recording: " + (recorder.isRecordingWTap() ? "ON" : "OFF")));
        this.buttonList.add(new GuiButton(2, this.width / 2 - buttonWidth / 2, startY + 2 * (buttonHeight + spacing), buttonWidth, buttonHeight, "Strafe Recording: " + (recorder.isRecordingStrafe() ? "ON" : "OFF")));
        this.buttonList.add(new GuiButton(3, this.width / 2 - buttonWidth / 2, startY + 3 * (buttonHeight + spacing), buttonWidth, buttonHeight, "Retreat Recording: " + (recorder.isRecordingRetreat() ? "ON" : "OFF")));
        this.buttonList.add(new GuiButton(4, this.width / 2 - buttonWidth / 2, startY + 4 * (buttonHeight + spacing), buttonWidth, buttonHeight, "ThrowPot Recording: " + (recorder.isRecordingThrowPot() ? "ON" : "OFF")));
        this.buttonList.add(new GuiButton(5, this.width / 2 - buttonWidth / 2, startY + 5 * (buttonHeight + spacing), buttonWidth, buttonHeight, "Tryhard Pearl: " + (recorder.isRecordingTryhardPearl() ? "ON" : "OFF")));
        this.buttonList.add(new GuiButton(6, this.width / 2 - buttonWidth / 2, startY + 6 * (buttonHeight + spacing), buttonWidth, buttonHeight, "Aggro Pearl: " + (recorder.isRecordingAggroPearl() ? "ON" : "OFF")));
        this.buttonList.add(new GuiButton(7, this.width / 2 - buttonWidth / 2, startY + 7 * (buttonHeight + spacing), buttonWidth, buttonHeight, "Escape Pearl: " + (recorder.isRecordingEscapePearl() ? "ON" : "OFF")));
        this.buttonList.add(new GuiButton(8, this.width / 2 - buttonWidth / 2, startY + 8 * (buttonHeight + spacing), buttonWidth, buttonHeight, "Close"));
    }

    @Override
    protected void actionPerformed(GuiButton button) {
        switch (button.id) {
            case 0:
                if (recorder.isRecordingAim()) recorder.stopAimRecording();
                else recorder.startAimRecording();
                break;
            case 1:
                if (recorder.isRecordingWTap()) recorder.stopWTapRecording();
                else recorder.startWTapRecording();
                break;
            case 2:
                if (recorder.isRecordingStrafe()) recorder.stopStrafeRecording();
                else recorder.startStrafeRecording();
                break;
            case 3:
                if (recorder.isRecordingRetreat()) recorder.stopRetreatRecording();
                else recorder.startRetreatRecording();
                break;
            case 4:
                if (recorder.isRecordingThrowPot()) recorder.stopThrowPotRecording();
                else recorder.startThrowPotRecording();
                break;
            case 5:
                if (recorder.isRecordingTryhardPearl()) recorder.stopTryhardPearlRecording();
                else recorder.startTryhardPearlRecording();
                break;
            case 6:
                if (recorder.isRecordingAggroPearl()) recorder.stopAggroPearlRecording();
                else recorder.startAggroPearlRecording();
                break;
            case 7:
                if (recorder.isRecordingEscapePearl()) recorder.stopEscapePearlRecording();
                else recorder.startEscapePearlRecording();
                break;
            case 8:
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
