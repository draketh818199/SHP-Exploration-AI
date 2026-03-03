import copy
import math
import numpy as np
import pygame

from pettingzoo import ParallelEnv
from gymnasium import spaces

# =========================
# Improvements needed
# =========================
# larger map & more map option (probably in another file maybe procedural)
# improved reward function
# maybe add continuous movement & ray vision


# =========================
# MAP DEFINITION
# =========================

MAP0 = [
    ["███","███","███","███","███","███","███","███","███","███"],
    ["███","   ","   ","   ","   ","   ","   ","   ","   ","███"],
    ["███","   ","   ","   ","   ","   ","   ","   ","   ","███"],
    ["███","   ","   ","   ","   ","   ","   ","   ","   ","███"],
    ["███","   ","   ","   ","   ","   ","   ","   ","   ","███"],
    ["███","   ","███","███","   ","   ","   ","   ","   ","███"],
    ["███","   "," G ","███","   ","   ","   ","   ","   ","███"],
    ["███","   ","   ","███","   ","   ","   ","   ","   ","███"],
    ["███","   ","   ","███","   ","   ","   ","   ","   ","███"],
    ["███","███","███","███"," O ","███","███","███","███","███"]
]

# =========================
# Constant Variables
# =========================

RENDER_FPS = 30
MAX_STEPS = 200


# =========================
# PETTINGZOO ENVIRONMENT
# =========================

class env(ParallelEnv):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "GridMapEnv_v0"
    }

    def __init__(self, vision_radius=3, render_mode=None, max_steps = MAX_STEPS):

        # Single agent setup
        self.max_steps = max_steps
        self.possible_agents = ["agent_0"]
        self.agents = []

        self.vision_radius = vision_radius
        self.render_mode = render_mode

        self.original_map = MAP0
        self.size = len(MAP0)

        # Actions: up, down, left, right
        self._action_spaces = {
            "agent_0": spaces.Discrete(4)
        }

        # Observation: visible square
        view_size = 2 * vision_radius + 1
        self._observation_spaces = {
            "agent_0": spaces.Box(
                low=-1,
                high=3,
                shape=(view_size, view_size),
                dtype=np.int8
            )
        }

    # Required properties
    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def action_spaces(self):
        return self._action_spaces

    # =========================
    # CORE LOGIC
    # =========================

    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.grid = copy.deepcopy(self.original_map)

        self.player_pos = self._find_player_start()
        self.grid[self.player_pos[0]][self.player_pos[1]] = " P "

        obs = self._get_observation()
        reward = 0
        terminated = False

        observations = {"agent_0": obs}
        rewards = {"agent_0": reward}
        terminations = {"agent_0": terminated}
        truncations = {"agent_0": False}
        infos = {"agent_0": {}}

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def step(self, actions):    

        self.step_count += 1
        if not self.agents:
            return {}, {}, {}, {}, {}

        action = actions["agent_0"]

        dx_dy = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }  

        dx, dy = dx_dy[action]
        x, y = self.player_pos
        nx, ny = x + dx, y + dy

        #add more complex reward function here
        reward = -0.01
        terminated = False

        # bounds check
        if 0 <= nx < self.size and 0 <= ny < self.size:

            if self.grid[nx][ny] != "███":

                if self.grid[nx][ny] == " G ":
                    reward = 1.0
                    terminated = True

                self.grid[x][y] = "   "
                self.grid[nx][ny] = " P "
                self.player_pos = (nx, ny)

        obs = self._get_observation()

        observations = {"agent_0": obs}
        rewards = {"agent_0": reward}
        terminations = {"agent_0": terminated}
        truncations = {"agent_0": False}
        infos = {"agent_0": {}}

        if self.step_count >= self.max_steps:
            truncations["agent_0"] = True

        if terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    # =========================
    # OBSERVATION LOGIC
    # =========================

    def _get_observation(self):

        px, py = self.player_pos
        radius = self.vision_radius
        view_size = 2 * radius + 1

        obs = np.full((view_size, view_size), -1, dtype=np.int8)

        for i, r in enumerate(range(px - radius, px + radius + 1)):
            for j, c in enumerate(range(py - radius, py + radius + 1)):

                dist = math.sqrt((r - px)**2 + (c - py)**2)
                if dist > radius:
                    continue

                if 0 <= r < self.size and 0 <= c < self.size:
                    obs[i, j] = self._encode_tile(self.grid[r][c])

        return obs

    def _encode_tile(self, tile):
        if tile == "███":
            return 1
        if tile == " G ":
            return 2
        if tile == " P ":
            return 3
        return 0

    def _find_player_start(self):
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if self.grid[r][c] == " O ":
                    self.grid[r][c] = "   "
                    return (r, c)
        raise ValueError("No start position found")

    # =========================
    # RENDERING
    # =========================

    def render(self):

        if self.render_mode == "human":
            tile_size = 40

            if not hasattr(self, "_pygame_initialized"):
                pygame.init()
                self.window_size = self.size * tile_size
                self.screen = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
                pygame.display.set_caption("GridMapEnv")
                self.clock = pygame.time.Clock()
                self._pygame_initialized = True

            # Handle window close events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # Build RGB array (same as before)
            img = np.zeros(
                (self.size * tile_size,
                self.size * tile_size,
                3),
                dtype=np.uint8
            )

            color_map = {
                "███": (50, 50, 50),
                "   ": (255, 255, 255),
                " G ": (0, 255, 0),
                " P ": (0, 0, 255),
            }

            for r in range(self.size):
                for c in range(self.size):
                    color = color_map[self.grid[r][c]]
                    img[
                        r*tile_size:(r+1)*tile_size,
                        c*tile_size:(c+1)*tile_size
                    ] = color

            # Convert numpy array to pygame surface
            surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

            self.screen.blit(surface, (0, 0))
            pygame.display.flip()

            self.clock.tick(RENDER_FPS)  # control FPS

        elif self.render_mode == "rgb_array":
            tile_size = 20
            img = np.zeros(
                (self.size * tile_size,
                 self.size * tile_size,
                 3),
                dtype=np.uint8
            )

            color_map = {
                "███": (50, 50, 50),
                "   ": (255, 255, 255),
                " G ": (0, 255, 0),
                " P ": (0, 0, 255),
            }

            for r in range(self.size):
                for c in range(self.size):
                    color = color_map[self.grid[r][c]]
                    img[
                        r*tile_size:(r+1)*tile_size,
                        c*tile_size:(c+1)*tile_size
                    ] = color

            return img
