import dearpygui.dearpygui as dpg
import random
import math
import threading
import time
from multiprocessing import Queue
from agentManager import AgentManager


#--------------------------
# Improvements needed
#--------------------------
# Add graphs
# Add tabs to display - Session path - Live movement
# Ensure all functionality is done in UI


# -------------------------
# Shared State
# -------------------------
manager = AgentManager()
data_queue = manager.get_data_queue()
control_queue = Queue()
state = {
    "agents": {
        "agent_0": {
            "path": [],
            "rewards": [],
            "grid": None
        }
    }
}



def process_queue():
    while not data_queue.empty():
        msg = data_queue.get()
        agent="agent_0"

        if agent not in state["agents"]:
            state["agents"][agent] = {
                "path": [],
                "rewards": [],
                "grid": None
            }

        if msg["type"] == "step":
            state["agents"][agent]["grid"] = msg["grid"]

        elif msg["type"] == "episode":
            state["agents"][agent]["path"] = msg["path"]
            state["agents"][agent]["rewards"].append(msg["reward"])


#--------------------------
# Rendering
#--------------------------
def draw_grid(agent):
    dpg.delete_item("grid_layer", children_only=True)

    grid = state["agents"][agent]["grid"]
    if grid is None:
        return

    tile_size = 20

    color_map = {
        0: (255, 255, 255),  # empty
        1: (50, 50, 50),     # wall
        2: (0, 0, 255),      # player
        3: (0, 255, 0)       # goal
    }

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            val = grid[r][c]

            dpg.draw_rectangle(
                (c * tile_size, r * tile_size),
                ((c + 1) * tile_size, (r + 1) * tile_size),
                fill=color_map[val],
                parent="grid_layer"
            )

def draw_path(agent):
    dpg.delete_item("path_layer", children_only=True)

    path = state["agents"][agent]["path"]
    if len(path) < 2:
        return

    tile_size = 20

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        dpg.draw_line(
            (y1 * tile_size + tile_size//2, x1 * tile_size + tile_size//2),
            (y2 * tile_size + tile_size//2, x2 * tile_size + tile_size//2),
            parent="path_layer",
            thickness=2
        )


# -------------------------
# UI Update
# -------------------------
def update_ui():
    # Update path drawing
    dpg.delete_item("path_layer", children_only=True)
    process_queue()
    agent = "agent_0"

    draw_grid(agent)
    draw_path(agent)

    # update graphs
    rewards = state["agents"][agent]["rewards"]
    if rewards:
        dpg.set_value("reward_series", [list(range(len(rewards))), rewards])

# -------------------------
# Controls
# -------------------------
def start_callback():
    if not manager.agents:
        manager.create_agent()
    manager.start_all()

def stop_callback():
    manager.stop_all()

def reset_callback():
    manager.reset_all()

# -------------------------
# UI Layout
# -------------------------
dpg.create_context()


# -------------------------
# Setup + Run
# -------------------------

def setup_ui(manager):
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
                dpg.add_listbox(items=["Agent 0"], num_items=4)

                dpg.add_separator()
                dpg.add_text("Logs")
                dpg.add_text("Session initialized...")
            
            # ---------------- CENTER PANEL ----------------
            with dpg.child_window():
                dpg.add_text("Path Visualization")

                with dpg.drawlist(width=-1, height=500):
                    dpg.draw_rectangle((0, 0), (800, 500), color=(255, 255, 255))
                    with dpg.draw_layer(tag="grid_layer"):
                        pass
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
                        dpg.add_text(lambda: f"Steps: {len(state["agents"]["agent_0"]["path"])}")


def start(manager):
    if not manager.agents:
        manager.create_agent()
    manager.start_all()

def stop(manager):
    manager.stop_all()

def run_ui():
    dpg.create_viewport(title='AI Dashboard', width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

# -------------------------
# ENTRY POINT
# -------------------------

def main():
    manager = AgentManager()

    dpg.create_context()
    print("Going to setup")
    setup_ui(manager)
    print("Setup")
    run_ui()

if __name__ == "__main__":
    main()