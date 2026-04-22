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
# Add tabs to display - Session path - Live movement - heatMap?
# add trained model saving and selecting 
# test training values setting
# save and load training values w/ model
# add more data & graphs & logs
# better button press feadback (reset, start, etc)
# add variable run lenght
# reset reward graph on agent reset
# find why dashboard is delayed (seems like dataQueue is being prcessed slower than its being added to)


# -------------------------
# Shared State
# -------------------------
draw_width = 800
draw_height = 500
lr = 1e-3
gamma = .99
entropy_scaler = .01
t_max = 50
control_queue = Queue()
state = {
    "agents": {
        "agent_0": {
            "path": [],
            "rewards": [],
            "grid": None,
            "logs": [],
            "status": "idle",
            "episode": [],
            "loss": []
        }
    }
}



def process_queue(data_queue):
    while not data_queue.empty():
        msg = data_queue.get()
        agent = msg.get("agent", "agent_0")

        if agent not in state["agents"]:
            state["agents"][agent] = {
                "path": [],
                "rewards": [],
                "grid": None,
                "logs": [],
                "status": "idle",
                "episode": []
            }

        if msg["type"] == "step":
            state["agents"][agent]["grid"] = msg["grid"]

        elif msg["type"] == "episode":
            state["agents"][agent]["path"] = msg["path"]
            state["agents"][agent]["rewards"].append(msg["rewards"])
            state["agents"][agent]["episode"] = msg["episode"]
            #state["agents"][agent]["loss"].append(msg["loss"])

        elif msg["type"] == "log":
            logs = state["agents"][agent]["logs"]
            logs.append({
                "level": msg.get("level", "info"),
                "message": msg["message"]})
            if len(logs) > 200:
                logs.pop(0)
        
        elif msg["type"] == "status":
            state["agents"][agent]["status"] = msg["status"]



#--------------------------
# Rendering
#--------------------------
def draw_grid(agent):
    dpg.delete_item("grid_layer", children_only=True)

    grid = state["agents"][agent]["grid"]
    if grid is None:
        return

    color_map = {
        0: (255, 255, 255),  # empty
        1: (50, 50, 50),     # wall
        2: (0, 0, 255),      # player
        3: (0, 255, 0)       # goal
    }

    rows = len(grid)
    cols = len(grid[0])

    tile_size = min(draw_width // cols, draw_height // rows)

    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]

            dpg.draw_rectangle(
                (c * tile_size, r * tile_size),
                ((c + 1) * tile_size, (r + 1) * tile_size),
                fill=color_map.get(int(val), (255, 0, 0)),
                parent="grid_layer"
            )

def draw_path(agent):
    dpg.delete_item("path_layer", children_only=True)

    path = state["agents"][agent]["path"]
    grid = state["agents"][agent]["grid"]
    if len(path) < 2:
        return
    rows = len(grid)
    cols = len(grid[0])


    tile_size = min(draw_width // cols, draw_height // rows)

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        dpg.draw_line(
            (y1 * tile_size + tile_size//2, x1 * tile_size + tile_size//2),
            (y2 * tile_size + tile_size//2, x2 * tile_size + tile_size//2),
            color = (255, 255, 0),
            parent="path_layer",
            thickness=2
        )


# -------------------------
# UI Update
# -------------------------
def update_ui(data_queue):
    # Update path drawing
    process_queue(data_queue)
    agent = "agent_0"

    draw_grid(agent)
    draw_path(agent)
    update_logs(agent)

    status = state["agents"][agent]["status"]
    episode = state["agents"][agent]["episode"]

    dpg.set_value("status_text", f"Status: {status}")
    dpg.set_value("episode_text", f"Episode: {episode}")

    # update graphs
    # reward graph
    rewards = state["agents"][agent]["rewards"]
    if rewards:
        x = list(range(len(rewards)))
        y = rewards
        dpg.set_value("reward_series", [x, y])
        xmin, xmax = 0, len(rewards)
        ymin, ymax = min(rewards), max(rewards)
        padding = 1
        dpg.set_axis_limits("x_axis", xmin, xmax)
        dpg.set_axis_limits("y_axis", ymin - padding, ymax + padding)
    
    # loss graph
    loss = state["agents"][agent]["loss"]
    if loss:
        x = list(range(len(rewards)))
        y = loss
        dpg.set_value("loss_series", [x, y])
        xmin, xmax = 0, len(loss)
        ymin, ymax = min(loss), max(loss)
        padding = 1
        dpg.set_axis_limits("x_axis", xmin, xmax)
        dpg.set_axis_limits("y_axis", ymin - padding, ymax + padding)


def update_logs(agent):
    if not dpg.does_item_exist("log_window"):
        return

    dpg.delete_item("log_window", children_only=True)

    logs = state["agents"][agent]["logs"]

    for log in logs[-50:]:  # show last 50
        msg = log["message"]
        level = log["level"]

        color = {
            "info": (255, 255, 255),
            "warning": (255, 255, 0),
            "error": (255, 0, 0)
        }.get(level, (200, 200, 200))

        dpg.add_text(msg, color=color, parent="log_window")

# -------------------------
# Controls
# -------------------------
def start_callback(sender, app_data, user_data):
    manager = user_data
    if not manager.agents:
        manager.create_agent()
    manager.start_all()

def stop_callback(sender, app_data, user_data):
    manager = user_data
    manager.stop_all()

def reset_callback(sender, app_data, user_data):
    manager = user_data
    manager.reset_all()

def update_speed(sender, app_data, user_data):
    manager = user_data
    max_delay = 0.1      # slowest
    k = 5                # controls curve steepness
    simulation_delay = max_delay * (math.pow(app_data, k))
    if simulation_delay < 0.001:
        simulation_delay = 0.0
    manager.speed_control(simulation_delay)

def update_param(sender, app_data, user_data, manager):
    param_name, manager = user_data
    manager.update_param(param_name, app_data)

def save_callback(sender, app_data, user_data):
    manager = user_data
    name = dpg.get_value("save_name")
    if not name:
        print("Enter a name")
        return
    manager.save_agent(name)
    # refresh dropdown
    dpg.configure_item("load_dropdown", items=manager.list_saved_agents())

def load_callback(sender, app_data, user_data):
    manager = user_data
    selected = dpg.get_value("load_dropdown")
    if not selected:
        print("Select a model")
        return
    # IMPORTANT: stop agents before loading
    manager.stop_all()
    time.sleep(0.2)

    manager.load_agent(selected)

    

# -------------------------
# UI Layout setup
# -------------------------


def setup_ui(manager):
    with dpg.window(label="AI Dashboard", width=1600, height=800):

        # Top Controls
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start", callback=start_callback, user_data=manager)
            dpg.add_button(label="Stop", callback=stop_callback, user_data=manager)
            dpg.add_button(label="Reset", callback=reset_callback, user_data=manager)
            dpg.add_input_text(label="Save Name", tag="save_name", width=150)
            dpg.add_button(label="Save", callback=save_callback, user_data=manager)

            dpg.add_combo(
                label="Saved Models",
                tag="load_dropdown",
                items=manager.list_saved_agents(),
                width=150
            )
            dpg.add_button(label="Load", callback=load_callback, user_data=manager)

        # Main Layout
        with dpg.table(header_row=False, resizable=True):
            dpg.add_table_column(width_fixed=True, init_width_or_weight=200)  # left
            dpg.add_table_column(width_fixed=True, init_width_or_weight=825) 
            dpg.add_table_column(width_fixed=True, init_width_or_weight=500)  # right

            with dpg.table_row():


                # ---------------- LEFT PANEL ----------------
                with dpg.child_window():
                    dpg.add_text("Agents")
                    dpg.add_listbox(items=["Agent 0"], num_items=4)

                    dpg.add_separator
                    dpg.add_text(tag="status_text")
                    dpg.add_text(tag="episode_text")

                    dpg.add_separator()
                    with dpg.child_window(tag="log_window", autosize_x=True, height=200):
                        pass
                
                # ---------------- CENTER PANEL ----------------
                with dpg.child_window():
                    dpg.add_text("Simulation Speed")
                    dpg.add_slider_float(
                        label="Delay (seconds)",
                        default_value=1.0,
                        min_value=0.0,
                        max_value=1.0,
                        callback=update_speed,
                        user_data=manager,
                        tag="speed_slider"
                    )

                    dpg.add_text("Path Visualization")

                    with dpg.drawlist(width=draw_width, height=draw_height, tag="main_drawlist"):
                        dpg.draw_rectangle((0, 0), (800, 500), fill=(100, 100, 100))
                        with dpg.draw_layer(tag="grid_layer"):
                            pass
                        with dpg.draw_layer(tag="path_layer"):
                            pass

                # ---------------- RIGHT PANEL ----------------
                with dpg.child_window():

                    with dpg.tab_bar():

                        # ---- Performance Tab ----
                        with dpg.tab(label="Performance"):
                            dpg.add_text("Reward Over Time")

                            with dpg.plot(height=200):
                                dpg.add_plot_axis(dpg.mvXAxis, label="Step", tag="x_axis")
                                with dpg.plot_axis(dpg.mvYAxis, label="Reward", tag="y_axis"):
                                    dpg.add_line_series([], [], tag="reward_series")

                        # ---- Training Tab ----
                        with dpg.tab(label="Training"):
                            dpg.add_text("Loss Curve")

                            with dpg.plot(height=200):
                                dpg.add_plot_axis(dpg.mvXAxis, label="Step")
                                with dpg.plot_axis(dpg.mvYAxis, label="Loss"):
                                    dpg.add_line_series([], [], tag="loss_series")
                            dpg.add_text("Training Parameters")

                            dpg.add_input_float(
                                label="Learning Rate",
                                default_value=1e-3,
                                callback=update_param,
                                user_data=("lr", manager)
                            )

                            dpg.add_input_float(
                                label="Gamma",
                                default_value=0.99,
                                callback=update_param,
                                user_data=("gamma", manager)
                            )

                            dpg.add_input_float(
                                label="Entropy",
                                default_value=0.01,
                                callback=update_param,
                                user_data=("entropy", manager)
                            )

                            dpg.add_input_int(
                                label="T_MAX",
                                default_value=50,
                                callback=update_param,
                                user_data=("t_max", manager)
                            )

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

def run_ui(data_queue):
    dpg.create_viewport(title='AI Dashboard', width=1200, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        update_ui(data_queue)
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

# -------------------------
# ENTRY POINT
# -------------------------

def main():
    manager = AgentManager()
    data_queue = manager.get_data_queue()

    dpg.create_context()
    print("Going to setup")
    setup_ui(manager)
    print("Setup")
    run_ui(data_queue)

if __name__ == "__main__":
    main()