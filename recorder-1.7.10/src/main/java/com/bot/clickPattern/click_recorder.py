import time
import json
from pynput import mouse, keyboard

# --- Configuration ---
OUTPUT_FILENAME = "click_intervals.json"
TOGGLE_KEY = keyboard.Key.f8 # Use F8 to start/stop recording

# --- Globals ---
click_intervals = []
last_click_time = None
recording = False

def on_click(x, y, button, pressed):
    """Callback function for mouse clicks."""
    global last_click_time

    if not recording:
        return

    if button == mouse.Button.left and pressed:
        current_time = time.time()

        if last_click_time is not None:
            # Calculate interval in milliseconds
            interval_ms = (current_time - last_click_time) * 1000
            click_intervals.append(round(interval_ms))
            print(f"Interval captured: {round(interval_ms)} ms")

        last_click_time = current_time

def save_intervals():
    """Saves the recorded intervals to a JSON file."""
    if not click_intervals:
        print("No intervals recorded, nothing to save.")
        return

    with open(OUTPUT_FILENAME, 'w') as f:
        json.dump(click_intervals, f, indent=4)
    print(f"\nRecording stopped. Saved {len(click_intervals)} intervals to {OUTPUT_FILENAME}")

def on_press(key):
    """Callback function for key presses to toggle recording."""
    global recording, last_click_time

    if key == TOGGLE_KEY:
        recording = not recording
        if recording:
            click_intervals.clear()
            last_click_time = None
            print("--- Recording started. Press F8 again to stop. ---")
        else:
            save_intervals()

if __name__ == "__main__":
    print(f"--- Click Interval Recorder ---")
    print(f"Press '{TOGGLE_KEY}' to start/stop recording.")
    print("Press 'Ctrl+C' to exit.")

    # Set up listeners
    mouse_listener = mouse.Listener(on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_press)

    mouse_listener.start()
    keyboard_listener.start()
    mouse_listener.join()
    keyboard_listener.join()