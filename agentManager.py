import torch.multiprocessing as mp
from multiprocessing import Queue
from A3C import ActorCritic, Agent, SharedAdam
import os
import torch

SAVE_DIR = "saved_agents"

class AgentManager:
    def __init__(self):
        self.agents = []
        self.data_queue = Queue()
        self.control_queue = Queue()

        self.global_actor_critic = None
        self.optimizer = None
        self.global_ep = None

        self.initialized = False
        self.lr = 1e-4

    # -------------------------
    # Initialize shared model
    # -------------------------
    def init_model(self):
        if self.initialized:
            return


        input_dims = [98]
        n_actions = 4

        self.global_actor_critic = ActorCritic(input_dims, n_actions)
        self.global_actor_critic.share_memory()

        self.optimizer = SharedAdam(self.global_actor_critic.parameters(), lr=self.lr)
        self.global_ep = mp.Value('i', 0)

        self.initialized = True

    # -------------------------
    # Create agent
    # -------------------------
    def create_agent(self):
        self.init_model()

        agent_id = f"agent_{len(self.agents)}"

        agent = Agent(
            self.global_actor_critic,
            self.optimizer,
            input_dims=[98],
            n_actions=4,
            gamma=0.99,
            lr=1e-4,
            name=agent_id,
            global_ep_idx=self.global_ep,
            env_id=None,
            data_queue=self.data_queue,
            control_queue=self.control_queue
        )

        self.agents.append(agent)

    # -------------------------
    # Agent controls
    # -------------------------
    def start_all(self):
        for agent in self.agents:
            if not agent.is_alive():
                agent.start()
        self.control_queue.put({"type": "action", "action": "start"})

    def stop_all(self):
        self.control_queue.put({"type": "action", "action": "stop"})

    def reset_all(self):
        self.stop_all()
        ActorCritic.reset_weights(self.global_actor_critic)
        self.optimizer = SharedAdam(self.global_actor_critic.parameters(), lr=self.lr)
        self.control_queue.put({"type": "action", "action": "reset"})
    
    def speed_control(self, speed):
        self.control_queue.put({"type": "speed","value": speed})
    
    def update_param(self, param_name, app_data):
        self.control_queue.put({"type": "param","name": param_name,"value": app_data})
    
    # -------------------------
    # Save Controls
    # -------------------------
    
    def save_agent(self, name):
        if not name:
            print("Invalid name")
            return
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, f"{name}.pt")

        checkpoint = {
            "model_state": self.global_actor_critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "episode": self.global_ep.value,
        }
        torch.save(checkpoint, path)
        self.data_queue.put({
            "type": "log",
            "level": "info",   # info, warning, error
            "message": "Agent initialized"
        })
        self.log(f"Saved: {path}")


    def list_saved_agents(self):
        if not os.path.exists(SAVE_DIR):
            return []
        return [f.replace(".pt", "") for f in os.listdir(SAVE_DIR) if f.endswith(".pt")]
    
    def load_agent(self, name):
        path = os.path.join(SAVE_DIR, f"{name}.pt")

        if not os.path.exists(path):
            self.log("File not found", level="error")
            return

        checkpoint = torch.load(path)
        self.init_model()
        self.global_actor_critic.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.global_ep.value = checkpoint.get("episode", 0)

        self.log(f"Loaded: {path}")


    # -------------------------
    # Access data queue
    # -------------------------
    def get_data_queue(self):
        return self.data_queue

    def log(self, message, level="info", agent="manager"):
        self.data_queue.put({
            "type": "log",
            "agent": agent,
            "level": level,
            "message": message
        })