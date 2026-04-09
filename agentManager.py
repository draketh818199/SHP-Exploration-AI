import torch.multiprocessing as mp
from multiprocessing import Queue
from A3C import ActorCritic, Agent, SharedAdam

class AgentManager:
    def __init__(self):
        self.agents = []
        self.data_queue = Queue()
        self.control_queue = Queue()

        self.global_actor_critic = None
        self.optimizer = None
        self.global_ep = None

        self.initialized = False

    # -------------------------
    # Initialize shared model
    # -------------------------
    def init_model(self):
        if self.initialized:
            return

        lr = 1e-4
        input_dims = [49]
        n_actions = 4

        self.global_actor_critic = ActorCritic(input_dims, n_actions)
        self.global_actor_critic.share_memory()

        self.optimizer = SharedAdam(self.global_actor_critic.parameters(), lr=lr)
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
            input_dims=[49],
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
        self.control_queue.put({"type": "action", "action": "reset"})
    
    def speed_control(self, speed):
        self.control_queue.put({"type": "speed","value": speed})

    # -------------------------
    # Access data queue
    # -------------------------
    def get_data_queue(self):
        return self.data_queue