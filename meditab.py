import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time  # Import time for measuring episode duration
import tkinter as tk  # Import tkinter for GUI
from tkinter import messagebox

# Define constants for actions
ACTION_FORWARD = 0
ACTION_REVERSE = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_WAIT = 4

# Define the environment
class WarehouseEnv:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start_positions = {}
        self.destination_positions = {}
        self.obstacles = []
        self.find_positions()
        self.reset()

    def find_positions(self):
        """Identify the positions of the autobots and destinations."""
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i, j]
                if cell.startswith('A'):
                    self.start_positions[cell] = (i, j)
                elif cell.startswith('B'):
                    self.destination_positions[cell] = (i, j)
                elif cell == 'X':
                    self.obstacles.append((i, j))

    def reset(self):
        """Reset the environment to the initial state."""
        self.autobots = []
        for bot_name, start_pos in self.start_positions.items():
            dest_name = 'B' + bot_name[1:]  # Find the corresponding destination
            dest_pos = self.destination_positions.get(dest_name)
            if dest_pos is None:
                raise ValueError(f"Destination not found for bot {bot_name}")
            self.autobots.append({
                'name': bot_name,
                'row': start_pos[0],
                'col': start_pos[1],
                'direction': 'up',
                'destination_row': dest_pos[0],
                'destination_col': dest_pos[1],
                'reached': False  # Track if the bot has reached its destination
            })
        return self.get_state()

    def get_state(self):
        """Get the current state representation."""
        state = 0
        for i, autobot in enumerate(self.autobots):
            direction_index = {'up': 0, 'down': 1, 'left': 2, 'right': 3}[autobot['direction']]
            state += (autobot['row'] * self.cols + autobot['col'] + direction_index * self.rows * self.cols) * (5 ** i)
        return state % (self.rows * self.cols * 4 ** len(self.autobots))

    def step(self, actions):
        """Take a step in the environment based on actions taken by autobots."""
        done = all(autobot['reached'] for autobot in self.autobots)  # Check if all reached
        reward = 0

        for i, autobot in enumerate(self.autobots):
            if not autobot['reached']:  # Only move the autobot if it hasn't reached its destination
                self.execute_action(autobot, actions[i])
                if self.is_at_destination(autobot):
                    reward += 100  # Reward for reaching destination
                    autobot['reached'] = True  # Mark as reached

        return self.get_state(), reward, done

    def execute_action(self, autobot, action):
        """Execute the given action for the autobot."""
        if action == ACTION_FORWARD:
            self.move_autobot(autobot, 'forward')
        elif action == ACTION_REVERSE:
            self.move_autobot(autobot, 'reverse')
        elif action == ACTION_LEFT:
            autobot['direction'] = self.turn('left', autobot['direction'])
        elif action == ACTION_RIGHT:
            autobot['direction'] = self.turn('right', autobot['direction'])
        elif action == ACTION_WAIT:
            pass  # No movement, just wait

    def move_autobot(self, autobot, move_type):
        """Move the autobot in the specified direction."""
        current_row, current_col = autobot['row'], autobot['col']
        new_row, new_col = self.get_new_position(current_row, current_col, autobot['direction'], move_type)

        if self.is_valid_move(new_row, new_col):
            autobot['row'], autobot['col'] = new_row, new_col

    def get_new_position(self, row, col, direction, move_type):
        """Calculate new position based on current position, direction, and move type."""
        move_offsets = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
        }
        dy, dx = move_offsets[direction] if move_type == 'forward' else (-move_offsets[direction][0], -move_offsets[direction][1])
        return row + dy, col + dx

    def is_valid_move(self, row, col):
        """Check if the move is valid (within bounds and not into an obstacle)."""
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row, col] != 'X'

    def is_at_destination(self, autobot):
        """Check if the autobot has reached its destination."""
        return autobot['row'] == autobot['destination_row'] and autobot['col'] == autobot['destination_col']

    def turn(self, turn_direction, current_direction):
        """Change the direction of the autobot based on turn direction."""
        directions = ['up', 'right', 'down', 'left']
        idx = directions.index(current_direction)
        return directions[(idx - 1) % 4] if turn_direction == 'left' else directions[(idx + 1) % 4]

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay_rate=0.99):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

    def choose_action(self, state):
        """Choose an action based on the current state using an epsilon-greedy strategy."""
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(np.arange(ACTION_WAIT + 1))  # Choose random action
        return np.argmax(self.q_table[state, :])  # Choose best action

    def learn(self, state, actions, reward, next_state, done):
        """Update the Q-table based on the action taken and the reward received."""
        for i, action in enumerate(actions):
            target = reward + (self.discount_factor * np.max(self.q_table[next_state, :]) if not done else 0)
            self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])

    def decay_exploration_rate(self):
        """Decay the exploration rate after each episode."""
        self.exploration_rate *= self.exploration_decay_rate

# GUI Class
class WarehouseGUI:
    def __init__(self, master, env, agent):
        self.master = master
        self.env = env
        self.agent = agent

        self.master.title("Warehouse Navigation")

        # Create a canvas for the grid
        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()

        # Start button
        self.start_button = tk.Button(master, text="Start Training", command=self.start_training)
        self.start_button.pack()

        # Test button
        self.test_button = tk.Button(master, text="Test Agent", command=self.test_agent)
        self.test_button.pack()

        # Variable to track training status
        self.training = False

    def draw_grid(self):
        """Draw the warehouse grid on the canvas."""
        self.canvas.delete("all")  # Clear previous drawings
        cell_size = 40

        for i in range(self.env.rows):
            for j in range(self.env.cols):
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                cell = self.env.grid[i, j]

                # Draw the cells based on their content
                if cell.startswith('A'):  # Autobots
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue")
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=cell, fill="white")
                elif cell.startswith('B'):  # Destinations
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="green")
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=cell, fill="white")
                elif cell == 'X':  # Obstacles
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="red")
                else:  # Empty space
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")

        # Draw the autobots' positions
        for autobot in self.env.autobots:
            self.canvas.create_text((autobot['col'] * cell_size) + cell_size / 2,
                                    (autobot['row'] * cell_size) + cell_size / 2,
                                    text=autobot['name'], fill="black")

    def start_training(self):
        """Start the training process."""
        self.training = True
        num_episodes = 1000# Number of episodes to train
        best_episode = None
        min_time = float('inf')
        best_time = None

        for episode in range(num_episodes):
            start_time = time.time()
            state = self.env.reset()  # Reset the environment for each episode
            done = False
            total_reward = 0

            while not done:
                actions = [self.agent.choose_action(state) for _ in range(len(self.env.autobots))]
                next_state, reward, done = self.env.step(actions)
                self.agent.learn(state, actions, reward, next_state, done)
                total_reward += reward
                state = next_state

            episode_time = time.time() - start_time
            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}, Time: {episode_time:.2f} seconds")

            # Update minimum time and best episode if necessary
            if episode_time < min_time:
                min_time = episode_time
                best_episode = episode

            self.agent.decay_exploration_rate()  # Decay exploration rate

            # Update the grid after each episode
            self.draw_grid()
            self.master.update()  # Update the GUI

        messagebox.showinfo("Training Complete", f"Best Episode: {best_episode + 1} with Time: {min_time:.2f} seconds")

    def test_agent(self):
        """Test the trained agent using the learned Q-values."""
        self.env.reset()  # Reset environment to the initial state
        state = self.env.get_state()
        done = False
        total_reward = 0

        print("\nTesting Agent")
        while not done:
            actions = [self.agent.choose_action(state) for _ in range(len(self.env.autobots))]
            next_state, reward, done = self.env.step(actions)
            total_reward += reward
            self.draw_grid()  # Update grid display
            self.master.update()  # Update the GUI
            
            # Add a delay to visualize movements clearly
            time.sleep(0.5)

            print(f"Actions Taken: {actions}, State: {state}, Next State: {next_state}, Reward: {reward}")
            state = next_state

        print(f"Total Reward During Testing: {total_reward}")

# Main execution
if __name__ == "__main__":
    # Define a simple grid (modify as needed)
    grid = np.array([
        ['A1', ' ', ' ', ' ', 'B1'],
        ['X', 'X', 'X', ' ', 'X'],
        ['A2', ' ', 'X', ' ', 'B2'],
        [' ', ' ', ' ', ' ', ' '],
        ['X', 'X', 'X', ' ', 'X'],
        ['A3', ' ', ' ', ' ', 'B3']
    ])

    env = WarehouseEnv(grid)
    agent = QLearningAgent(state_space_size=env.rows * env.cols * 4 ** len(env.start_positions), action_space_size=ACTION_WAIT + 1)

    root = tk.Tk()
    app = WarehouseGUI(root, env, agent)
    root.mainloop()
