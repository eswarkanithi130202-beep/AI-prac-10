import numpy as np

# Grid World: 3x3 grid
rows, cols = 3, 3
goal = (2, 2)
gamma = 0.9
theta = 0.001
actions = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}
V = np.zeros((rows, cols))

def next_state(r, c, action):
    dr, dc = actions[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < rows and 0 <= nc < cols:
        return nr, nc
    return r, c

# Value Iteration
while True:
    delta = 0
    for r in range(rows):
        for c in range(cols):
            if (r, c) == goal:
                continue
            old_value = V[r, c]
            values = []
            for action in actions:
                nr, nc = next_state(r, c, action)
                if (nr, nc) == goal:
                    reward = 10
                else:
                    reward = -1
                values.append(reward + gamma * V[nr, nc])
            V[r, c] = max(values)
            delta = max(delta, abs(old_value - V[r, c]))
    if delta < theta:
        break

# Extract optimal policy
policy = [["" for _ in range(cols)] for _ in range(rows)]
for r in range(rows):
    for c in range(cols):
        if (r, c) == goal:
            policy[r][c] = "G"
            continue
        best_action = None
        best_value = float("-inf")
        for action in actions:
            nr, nc = next_state(r, c, action)
            reward = 10 if (nr, nc) == goal else -1
            value = reward + gamma * V[nr, nc]
            if value > best_value:
                best_value = value
                best_action = action
        policy[r][c] = best_action

# Display results
print("State Values:")
print(np.round(V, 2))
print("\nOptimal Policy:")
for row in policy:
    print(" ".join(row))
