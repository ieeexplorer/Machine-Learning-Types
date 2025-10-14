import numpy as np
import random

class MazeEnvironment:
    """A simple 4x4 maze where agent learns to reach goal"""
    def __init__(self):
        self.grid = [
            ['S', ' ', ' ', ' '],
            [' ', '#', ' ', '#'],
            [' ', ' ', ' ', ' '],
            ['#', ' ', ' ', 'G']
        ]
        self.actions = ['up', 'down', 'left', 'right']
        self.state = (0, 0)  # Start position
        self.goal = (3, 3)   # Goal position
        self.obstacles = [(1,1), (1,3), (3,0)]
        
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        row, col = self.state
        
        if action == 'up' and row > 0:
            row -= 1
        elif action == 'down' and row < 3:
            row += 1
        elif action == 'left' and col > 0:
            col -= 1
        elif action == 'right' and col < 3:
            col += 1
            
        new_state = (row, col)
        
        # Calculate reward
        if new_state in self.obstacles:
            reward = -10  # Big penalty for hitting obstacle
            new_state = self.state  # Stay in current position
        elif new_state == self.goal:
            reward = 100  # Big reward for reaching goal
        else:
            reward = -1   # Small penalty for each step
            
        self.state = new_state
        done = (new_state == self.goal)
        
        return new_state, reward, done
    
    def render(self):
        for i, row in enumerate(self.grid):
            display_row = []
            for j, cell in enumerate(row):
                if (i, j) == self.state:
                    display_row.append('A')  # Agent
                else:
                    display_row.append(cell)
            print(' '.join(display_row))
        print()

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = 0.01
        
        # Initialize Q-table with zeros
        self.q_table = {}
        for i in range(4):
            for j in range(4):
                self.q_table[(i, j)] = {action: 0 for action in env.actions}
    
    def choose_action(self, state):
        # Exploration vs Exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.env.actions)  # Explore
        else:
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]  # Exploit
    
    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            # Max Q-value for next state
            max_next_q = max(self.q_table[next_state].values())
            target = reward + self.gamma * max_next_q
            
        # Update Q-value
        self.q_table[state][action] += self.lr * (target - current_q)
        
        # Decay exploration rate
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Training the RL agent
print("REINFORCEMENT LEARNING: MAZE SOLVER")
print("Agent learns through trial and error!\n")

env = MazeEnvironment()
agent = QLearningAgent(env)

# Training phase
episodes = 500
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 100:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        steps += 1
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}, Epsilon = {agent.epsilon:.3f}")

# Test the trained agent
print("\n" + "="*50)
print("TESTING TRAINED AGENT:")
print("="*50)

state = env.reset()
done = False
steps = 0

print("Initial maze:")
env.render()

while not done and steps < 20:
    action = max(agent.q_table[state].items(), key=lambda x: x[1])[0]  # Always choose best action
    next_state, reward, done = env.step(action)
    
    print(f"Step {steps + 1}: Action = {action}, Reward = {reward}")
    env.render()
    
    state = next_state
    steps += 1

if done:
    print("🎉 Agent successfully reached the goal!")
else:
    print("❌ Agent failed to reach the goal within step limit")

# Show learned Q-values for start position
print("\nLearned Q-values for start position (0,0):")
for action, q_value in agent.q_table[(0,0)].items():
    print(f"  {action}: {q_value:.2f}")
