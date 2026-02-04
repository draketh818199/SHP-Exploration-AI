import math
import sys
import copy

#changes that need to happen
'''
reduse redunant functions
'''

'''
test changes
'''

#future imporvements
'''
make player movment continuous
add more and larger maps
add walls limitive view
move screen printing to playerAccessEnvironment
need a way for oustide access to map
'''


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
maps = [MAP0]

# Global variables
SIZE = 10
VISION_RADIUS = 3
STARTED = False
current_global_map = []
player_pos = (0, 0)



# called to select the used map
def set_map(map):
    global current_global_map
    current_global_map = map


#gets full map
def get_map():
    return current_global_map


#prints current map
def print_map(grid, player_pos, radius, cell_width=3, hidden=" "):
    px, py = player_pos
    size = len(grid)

    # Viewport bounds centered on player
    for r in range(px - radius, px + radius + 1):
        for c in range(py - radius, py + radius + 1):

            # Circular visibility check
            dist = math.sqrt((r - px)**2 + (c - py)**2)
            if dist > radius:
                print(hidden.ljust(cell_width), end="")
                continue

            # Bounds check
            if 0 <= r < size and 0 <= c < size:
                print(str(grid[r][c]).ljust(cell_width), end="")
            else:
                print(hidden.ljust(cell_width), end="")

        print()


#find o in map to get player start location
def find_player_start(grid, start_marker=" O "):
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == start_marker:
                grid[r][c] = "   "   # clear marker
                return (r, c)
    raise ValueError("No player start marker found ('o')")


#creates a copy of the map to use
def create_local_map(selection):
    index = selection

    if not (0 <= index < len(maps)):
        raise ValueError(f"Map {selection} does not exist.")

    # Return a DEEP COPY so the original is untouched
    return copy.deepcopy(maps[index])
    

#gets map and returns array
def get_view_array(grid, player_pos, radius, hidden=" "):
    """
    Returns a 2D array representing what the player can see.

    grid        : NxN map
    player_pos  : (row, col)
    radius      : visibility radius
    hidden      : character for unseen tiles
    """

    px, py = player_pos
    size = len(grid)

    visible = []

    for r in range(px - radius, px + radius + 1):
        row = []
        for c in range(py - radius, py + radius + 1):

            # Circular visibility check
            dist = math.sqrt((r - px)**2 + (c - py)**2)
            if dist > radius:
                row.append(hidden)
                continue

            # Bounds check
            if 0 <= r < size and 0 <= c < size:
                row.append(grid[r][c])
            else:
                row.append(hidden)

        visible.append(row)

    return visible


#prints a given array
def print_array(array, cell_width=3):
    """
    Prints a 2D array.

    array      : list of lists
    cell_width : spacing for each cell
    """

    for row in array:
        for cell in row:
            print(str(cell).ljust(cell_width), end="")
        print()


#moves player using wasd
def move_player(grid, player_pos, direction, wall="███", goal=" G "):
    x, y = player_pos
    moves = {
        "w": (-1, 0),
        "s": (1, 0),
        "a": (0, -1),
        "d": (0, 1)
    }

    if direction not in moves:
        return player_pos, False  # invalid input

    dx, dy = moves[direction]
    nx, ny = x + dx, y + dy

    # Bounds check
    if not (0 <= nx < len(grid) and 0 <= ny < len(grid)):
        return player_pos

    # Collision check
    if grid[nx][ny] == wall:
        return player_pos, False
    
    # Goal check
    reached_goal = (grid[nx][ny] == goal)

    # Move player
    grid[x][y] = "   "
    grid[nx][ny] = " P "
    return (nx, ny), reached_goal



# called first to start a run
def start_map(selection):
    global STARTED

    #gets map from list
    map = create_local_map(selection)

    #sets up initial conidtions
    global player_pos
    player_pos = find_player_start(map)
    map[player_pos[0]][player_pos[1]] = " P "
    set_map(map)
    STARTED = True
    print_map(map, player_pos, VISION_RADIUS)


# function called to run one turn returns current visible array
def action(move):
    # global variables
    global player_pos
    global STARTED

    # confirms a map has been selected
    if not STARTED:
        raise ValueError ("map not started")
    
    # quit game option
    if move == "q":
        return

    # move the player
    player_pos, won = move_player(get_map(), player_pos, move)

    # print current view (for debug purposes)
    current_array = get_view_array(get_map(), player_pos, VISION_RADIUS)
    print_array(current_array)

    # check for win condition
    if won:
        print("🎉 You reached the goal! You win!")
        return

    # return the current visible array
    return current_array