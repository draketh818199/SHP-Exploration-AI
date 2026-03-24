import sqlite3
import json

DB_NAME = "maps.db"

def load_map(map_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT grid FROM maps WHERE id = ?", (map_id,))
    result = cursor.fetchone()

    conn.close()

    if result is None:
        raise IndexError("Map not found")

    return json.loads(result[0])

if __name__ == "__main__":
    try:
        i = int(input("Enter map ID: "))
        grid = load_map(i)

        print("Loaded map:")
        for row in grid:
            print(row)

    except Exception as e:
        print("Error:", e)