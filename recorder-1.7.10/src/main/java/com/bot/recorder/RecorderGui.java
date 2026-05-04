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
