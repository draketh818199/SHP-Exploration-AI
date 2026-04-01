import dearpygui.dearpygui as dpg
import random
import math
import threading
import time

# -------------------------
# Shared State
# -------------------------
state = {
    "path": [],
    "reward": [],
    "loss": [],
    "running": False
}

# -------------------------
# Fake AI Loop (background)
# -------------------------
def ai_loop():
    x, y = 100, 100
    while True:
        if state["running"]:
            # simulate movement
            x += random.randint(-5, 5)
            y += random.randint(-5, 5)
            state["path"].append((x, y))

            # simulate metrics
            state["reward"].append(len(state["path"]) * 0.1)
            state["loss"].append(math.exp(-len(state["path"]) * 0.01))

        time.sleep(0.05)

# -------------------------
# UI Update
# -------------------------
def update_ui():
    # Update path drawing
    dpg.delete_item("path_layer", children_only=True)

    if len(state["path"]) > 1:
        for i in range(len(state["path"]) - 1):
            p1 = state["path"][i]
            p2 = state["path"][i + 1]
            dpg.draw_line(p1, p2, parent="path_layer", thickness=2)

    # Update plots
    if state["reward"]:
        dpg.set_value("reward_series", [list(range(len(state["reward"]))), state["reward"]])

    if state["loss"]:
        dpg.set_value("loss_series", [list(range(len(state["loss"]))), state["loss"]])

# -------------------------
# Controls
# -------------------------
def start_callback():
    state["running"] = True

def stop_callback():
    state["running"] = False

def reset_callback():
    state["path"].clear()
    state["reward"].clear()
    state["loss"].clear()

# -------------------------
# UI Layout
# -------------------------
dpg.create_context()

with dpg.window(label="AI Dashboard", width=1200, height=800):

    # Top Controls
    with dpg.group(horizontal=True):
        dpg.add_button(label="Start", callback=start_callback)
        dpg.add_button(label="Stop", callback=stop_callback)
        dpg.add_button(label="Reset", callback=reset_callback)

    # Main Layout
    with dpg.group(horizontal=True):

        # ---------------- LEFT PANEL ----------------
        with dpg.child_window(width=200):
            dpg.add_text("Agents")
            dpg.add_listbox(items=["Agent 1", "Agent 2"], num_items=4)

            dpg.add_separator()
            dpg.add_text("Logs")
            dpg.add_text("Session initialized...")
        
        # ---------------- CENTER PANEL ----------------
        with dpg.child_window():
            dpg.add_text("Path Visualization")

            with dpg.drawlist(width=-1, height=500):
                dpg.draw_rectangle((0, 0), (800, 500), color=(255, 255, 255))
                with dpg.draw_layer(tag="path_layer"):
                    pass

        # ---------------- RIGHT PANEL ----------------
        with dpg.child_window(width=350):

            with dpg.tab_bar():

                # ---- Performance Tab ----
                with dpg.tab(label="Performance"):
                    dpg.add_text("Reward Over Time")

                    with dpg.plot(height=200):
                        dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                        with dpg.plot_axis(dpg.mvYAxis, label="Reward"):
                            dpg.add_line_series([], [], tag="reward_series")

                # ---- Training Tab ----
                with dpg.tab(label="Training"):
                    dpg.add_text("Loss Curve")

                    with dpg.plot(height=200):
                        dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                        with dpg.plot_axis(dpg.mvYAxis, label="Loss"):
                            dpg.add_line_series([], [], tag="loss_series")

                # ---- Diagnostics Tab ----
                with dpg.tab(label="Diagnostics"):
                    dpg.add_text("Stats")
                    dpg.add_text(lambda: f"Steps: {len(state['path'])}")

# -------------------------
# Setup + Run
# -------------------------
dpg.create_viewport(title='AI Dashboard', width=1200, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()

# Start AI thread
threading.Thread(target=ai_loop, daemon=True).start()

# Main loop
while dpg.is_dearpygui_running():
    update_ui()
    dpg.render_dearpygui_frame()

dpg.destroy_context()