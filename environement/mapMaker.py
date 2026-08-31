import random
import sqlite3
import json


# =======================
# Improvments
# =======================
# Make dense building generation work
# Reuce non-connecting rooms in spread generation
# make both variable size


# =======================
# CONFIG CONSTANTS
# =======================
ROWS = 30
COLS = 30

MIN_LEAF_SIZE = 6
MAX_LEAF_SIZE = 12

ROOM_MIN_SIZE = 3
ROOM_MAX_SIZE = 8

DB_NAME = "maps.db"

EMPTY = 0
WALL = 1
PLAYER = 2
OBJECTIVE = 3
DOOR = 0  # doorway is just empty space

DENSE_ROOM_ROWS = 3
DENSE_ROOM_COLS = 3

# =======================
# DATABASE SETUP
# =======================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS maps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        grid TEXT
    )
    """)

    conn.commit()
    conn.close()

# =======================
# BSP NODE
# =======================
class Leaf:
    def __init__(self, r, c, h, w):
        self.r = r
        self.c = c
        self.h = h
        self.w = w
        self.left = None
        self.right = None
        self.room = None

    def split(self):
        if self.left or self.right:
            return False

        split_h = random.choice([True, False])

        if self.h > self.w and self.h / self.w >= 1.25:
            split_h = True
        elif self.w > self.h and self.w / self.h >= 1.25:
            split_h = False

        max_split = (self.h if split_h else self.w) - MIN_LEAF_SIZE
        if max_split <= MIN_LEAF_SIZE:
            return False

        split = random.randint(MIN_LEAF_SIZE, max_split)

        if split_h:
            self.left = Leaf(self.r, self.c, split, self.w)
            self.right = Leaf(self.r + split, self.c, self.h - split, self.w)
        else:
            self.left = Leaf(self.r, self.c, self.h, split)
            self.right = Leaf(self.r, self.c + split, self.h, self.w - split)

        return True

    def create_room(self):
        room_w = random.randint(ROOM_MIN_SIZE, min(ROOM_MAX_SIZE, self.w - 2))
        room_h = random.randint(ROOM_MIN_SIZE, min(ROOM_MAX_SIZE, self.h - 2))

        room_r = random.randint(self.r + 1, self.r + self.h - room_h - 1)
        room_c = random.randint(self.c + 1, self.c + self.w - room_w - 1)

        self.room = (room_r, room_c, room_h, room_w)
        return self.room

# =======================
# MAP GENERATION
# =======================


def generate_dense_building():
    grid = create_grid()

    inner_h = ROWS - 2
    inner_w = COLS - 2

    cell_h = inner_h // DENSE_ROOM_ROWS
    cell_w = inner_w // DENSE_ROOM_COLS

    rooms = []

    # Create rooms inside each cell (NOT full cell)
    for i in range(DENSE_ROOM_ROWS):
        for j in range(DENSE_ROOM_COLS):
            base_r = 1 + i * cell_h
            base_c = 1 + j * cell_w

            # Irregular room size within the cell
            h = random.randint(cell_h // 2, cell_h - 1)
            w = random.randint(cell_w // 2, cell_w - 1)

            # Random offset inside the cell (keeps walls intact)
            r = base_r + random.randint(0, cell_h - h)
            c = base_c + random.randint(0, cell_w - w)

            room = (r, c, h, w)
            rooms.append(room)

            carve_room(grid, room)

    # Add doorways between adjacent rooms
    for i in range(DENSE_ROOM_ROWS):
        for j in range(DENSE_ROOM_COLS):
            idx = i * DENSE_ROOM_COLS + j
            r, c, h, w = rooms[idx]

            # Connect to right neighbor
            if j < DENSE_ROOM_COLS - 1:
                neighbor = rooms[idx + 1]

                # find overlapping vertical range
                r1 = max(r, neighbor[0])
                r2 = min(r + h - 1, neighbor[0] + neighbor[2] - 1)

                if r1 <= r2:
                    door_r = random.randint(r1, r2)
                    door_c = c + w
                    grid[door_r][door_c] = EMPTY

            # Connect to bottom neighbor
            if i < DENSE_ROOM_ROWS - 1:
                neighbor = rooms[idx + DENSE_ROOM_COLS]

                # find overlapping horizontal range
                c1 = max(c, neighbor[1])
                c2 = min(c + w - 1, neighbor[1] + neighbor[3] - 1)

                if c1 <= c2:
                    door_c = random.randint(c1, c2)
                    door_r = r + h
                    grid[door_r][door_c] = EMPTY

    # Place player and objective
    empty_tiles = [(r, c) for r in range(ROWS) for c in range(COLS) if grid[r][c] == EMPTY]
    random.shuffle(empty_tiles)

    pr, pc = empty_tiles.pop()
    or_, oc = empty_tiles.pop()

    grid[pr][pc] = PLAYER
    grid[or_][oc] = OBJECTIVE

    return grid


def create_grid():
    return [[WALL for _ in range(COLS)] for _ in range(ROWS)]


def create_empty_map():
    grid = create_grid()

    for r in range(1, ROWS - 1):
        for c in range(1, COLS - 1):
            grid[r][c] = EMPTY

    grid[1][1] = PLAYER
    grid[5][3] = OBJECTIVE
    return grid

def carve_room(grid, room):
    r, c, h, w = room
    for i in range(r, r + h):
        for j in range(c, c + w):
            grid[i][j] = EMPTY

def carve_hallway(grid, p1, p2):
    r1, c1 = p1
    r2, c2 = p2

    if random.choice([True, False]):
        for c in range(min(c1, c2), max(c1, c2) + 1):
            grid[r1][c] = EMPTY
        for r in range(min(r1, r2), max(r1, r2) + 1):
            grid[r][c2] = EMPTY
    else:
        for r in range(min(r1, r2), max(r1, r2) + 1):
            grid[r][c1] = EMPTY
        for c in range(min(c1, c2), max(c1, c2) + 1):
            grid[r2][c] = EMPTY

def get_room_center(room):
    r, c, h, w = room
    return (r + h // 2, c + w // 2)

def create_bsp_tree(root):
    leaves = [root]
    did_split = True

    while did_split:
        did_split = False
        for leaf in leaves[:]:
            if not leaf.left and not leaf.right:
                if leaf.h > MAX_LEAF_SIZE or leaf.w > MAX_LEAF_SIZE or random.random() > 0.5:
                    if leaf.split():
                        leaves.append(leaf.left)
                        leaves.append(leaf.right)
                        did_split = True
    return leaves


def generate_building():
    #if random.choice([True, False]):
        return generate_bsp_building()  # your existing BSP function
    #else:
        #return generate_dense_building()


def generate_bsp_building():
    grid = create_grid()

    root = Leaf(1, 1, ROWS - 2, COLS - 2)
    leaves = create_bsp_tree(root)

    rooms = []

    # Create rooms
    for leaf in leaves:
        if not leaf.left and not leaf.right:
            room = leaf.create_room()
            carve_room(grid, room)
            rooms.append(room)

    # Connect rooms
    for i in range(len(rooms) - 1):
        c1 = get_room_center(rooms[i])
        c2 = get_room_center(rooms[i + 1])
        carve_hallway(grid, c1, c2)

    # Ensure doorway per room (connect room edge to hallway)
    for room in rooms:
        r, c, h, w = room

        # pick a wall tile of the room
        side = random.choice(["top", "bottom", "left", "right"])

        if side == "top":
            dr = r
            dc = random.randint(c, c + w - 1)
            grid[dr - 1][dc] = EMPTY
        elif side == "bottom":
            dr = r + h - 1
            dc = random.randint(c, c + w - 1)
            grid[dr + 1][dc] = EMPTY
        elif side == "left":
            dr = random.randint(r, r + h - 1)
            dc = c
            grid[dr][dc - 1] = EMPTY
        else:
            dr = random.randint(r, r + h - 1)
            dc = c + w - 1
            grid[dr][dc + 1] = EMPTY

    # Place player and objective
    empty_tiles = [(r, c) for r in range(ROWS) for c in range(COLS) if grid[r][c] == EMPTY]
    random.shuffle(empty_tiles)

    pr, pc = empty_tiles.pop()
    or_, oc = empty_tiles.pop()

    grid[pr][pc] = PLAYER
    grid[or_][oc] = OBJECTIVE

    return grid

# =======================
# DATABASE FUNCTIONS
# =======================
def save_map(grid):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO maps (grid) VALUES (?)",
        (json.dumps(grid),)
    )

    conn.commit()
    conn.close()

def delete_map(map_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM maps WHERE id = ?", (map_id,))
    conn.commit()

    if cursor.rowcount == 0:
        print("No map found with that ID.")
    else:
        print(f"Map {map_id} deleted.")

    conn.close()

# =======================
# MAIN MENU
# =======================
if __name__ == "__main__":
    init_db()

    print("1: Generate building map")
    print("2: Empty Map")
    print("3: Delete a map")
    choice = input("Select option: ")

    if choice == "1":
        ammount = input("How many: ")
        for i in range (int(ammount)):
            new_map = generate_building()
            save_map(new_map)

        print("Generated map:")
        for row in new_map:
            print(row)

    elif choice == "2":
        empty_map = create_empty_map()
        save_map(empty_map)
        print("Empty map:")
        for row in empty_map:
            print(row)

    elif choice == "3":
        try:
            map_id = int(input("Enter map ID to delete: "))
            delete_map(map_id)
        except ValueError:
            print("Invalid ID.")

    else:
        print("Invalid option.")