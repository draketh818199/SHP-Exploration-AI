import copy
import math
import numpy as np
import pygame
from environement import mapReader

from pettingzoo import ParallelEnv
from gymnasium import spaces

# =========================
# Improvements needed
# =========================
# test improved reward
# maybe add continuous movement & ray vision
# random start location



# =========================
# Constant Variables
# =========================

RENDER_FPS = 30
MAX_STEPS = 250
MAP_ID = 15


get_map = mapReader.load_map(MAP_ID)

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

        self.original_map = get_map
        self.size = len(get_map)

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
                shape=(2, view_size, view_size),
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
        self.grid[self.player_pos[0]][self.player_pos[1]] = 2
        self.goal_pos = self._find_goal()

        self.seen = np.zeros((self.size, self.size), dtype=bool)
        self.visit_count = np.zeros((self.size, self.size))
        self.visited = np.zeros((self.size, self.size), dtype=np.float32)

        obs, _ = self._get_observation()
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
        self.visited *= 0.99
        self.visited[x][y] = 1.0

        old_pos = self.player_pos
        new_pos = (nx, ny)
        terminated = False

        
        obs, new_tiles = self._get_observation()
        reward = -.5

        # bounds check
        if 0 <= nx < self.size and 0 <= ny < self.size:

            if self.grid[nx][ny] != 1:

                new_pos = (nx, ny)

                terminated = self.grid[nx][ny] == 3

                self.grid[x][y] = 0
                self.grid[nx][ny] = 2
                self.player_pos = (nx, ny)

                reward = self._calculate_reward(old_pos, new_pos, new_tiles, terminated)

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


    #reward calculated by distance to goal
    def _calculate_reward(self, old_pos, new_pos, new_tiles, reached_goal):

        #gx, gy = self.goal_pos
        #old_dist = math.sqrt((old_pos[0] - gx)**2 + (old_pos[1] - gy)**2)
        #new_dist = math.sqrt((new_pos[0] - gx)**2 + (new_pos[1] - gy)**2)
        reward = -.2
        

        # exploration (maybe remove)
        x, y = self.player_pos
        self.visit_count[x][y] += 1
        intrinsic_reward = 1.0 / np.sqrt(self.visit_count[x][y])
        reward += 0.05 * intrinsic_reward

        # new tiles 
        reward += 0.02 * new_tiles

        # goal
        if reached_goal:
            reward += 10
        

        return reward



    # =========================
    # OBSERVATION LOGIC
    # =========================

    def _get_observation(self):

        px, py = self.player_pos
        radius = self.vision_radius
        view_size = 2 * radius + 1
        new_tiles = 0

        obs = np.full((2, view_size, view_size), -1, dtype=np.int8)

        for i, r in enumerate(range(px - radius, px + radius + 1)):
            for j, c in enumerate(range(py - radius, py + radius + 1)):

                if 0 <= r < self.size and 0 <= c < self.size:

                    # channel 0: environment
                    obs[0, i, j] = self.grid[r][c]

                    # channel 1: visited
                    obs[1, i, j] = self.visited[r][c]

                    if not self.seen[r][c]:
                        self.seen[r][c] = True
                        new_tiles += 1
                        
                dist = math.sqrt((r - px)**2 + (c - py)**2)
                if dist > radius:
                    continue

        return obs, new_tiles

    def _find_player_start(self):
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if self.grid[r][c] == 2:
                    self.grid[r][c] = 0
                    return (r, c)
        raise ValueError("No start position found")
    
    def _find_goal(self):
        for r in range(len(self.grid)):
            for c in range(len(self.grid[r])):
                if self.grid[r][c] == 3:
                    return (r, c)
        raise ValueError("No goal found")

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
                1: (50, 50, 50),
                0: (255, 255, 255),
                3: (0, 255, 0),
                2: (0, 0, 255),
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
                1: (50, 50, 50),
                0: (255, 255, 255),
                3: (0, 255, 0),
                2: (0, 0, 255),
            }

            for r in range(self.size):
                for c in range(self.size):
                    color = color_map[self.grid[r][c]]
                    img[
                        r*tile_size:(r+1)*tile_size,
                        c*tile_size:(c+1)*tile_size
                    ] = color

            return img
