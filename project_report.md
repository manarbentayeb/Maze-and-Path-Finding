# Reinforcement Learning for Maze Navigation
**Final Project Report**

---

## 1. Executive Summary

This project implements and evaluates Reinforcement Learning (RL) agents in custom grid-based maze environments. The primary objective was to analyze the performance, sample efficiency, and convergence of classical Tabular Q-Learning versus Deep Q-Networks (DQN) across progressively complex spatial navigation challenges. 

The project was divided into three core phases:
1. **Phase 1**: Static fully-observable mazes (Simple Maze).
2. **Phase 2**: Multi-stage dependencies (Door and Key Maze).
3. **Phase 3**: Custom environmental mechanics (Weighted-Cost and Dynamic Obstacle Mazes).

Across all phases, the project developed a robust pipeline comprising random maze generation via recursive backtracker, standard Gymnasium environment APIs, decoupled reward functions, integrated Pytest suites, and automated visualization pipelines (learning curves, Q-value heatmaps, and policy maps).

---

## 2. Phase 1: The Simple Maze (Baseline)

The foundation of the project focused on a static grid where the agent must navigate from a fixed start to a fixed goal while avoiding walls.

### Environments & Agents
- **Environment**: A static matrix where walls are generated using a phase-1 obstacle block pattern. The state space is a discrete scalar index, and the action space consists of 4 directional movements (Up, Down, Left, Right).
- **Reward Structure**: A baseline `SparseWithPenalty` reward was utilized: `+1.0` for reaching the goal, `-0.5` for hitting a wall, and `-0.01` per step to encourage the shortest path.
- **Algorithms Evaluated**: 
  - **Tabular Q-Learning**: Maintained a precise Q-table.
  - **DQN**: Utilized a Multi-Layer Perceptron (MLP) with a replay buffer and target network.

### Key Results
Tabular Q-Learning rapidly converged on the optimal path (100% success rate in small mazes). DQN also successfully solved the environment but required significantly more episodes to stabilize due to the complexities of neural network function approximation.

---

## 3. Phase 2: Door and Key Maze

To test the agents' ability to handle sequential dependencies, the environment was expanded to include a key that must be collected before a door can be unlocked to reach the goal.

### Challenges Introduced
- **State Expansion**: The state representation was augmented to track the agent's inventory (whether the key was collected). This effectively doubled the tabular state space and forced the DQN to learn condition-dependent representations.
- **Credit Assignment**: The agent faced a harder credit assignment problem, as early actions (collecting the key) were necessary for the final sparse reward.

### Key Results
Tabular Q-Learning maintained its dominance, successfully learning the two-stage policy. It effectively backpropagated the terminal reward back to the key-pickup state.

---

## 4. Phase 3: Custom Mechanics (Weighted & Dynamic)

The final phase pushed the agents to their limits by breaking standard assumptions about the environment.

### 4.1 Weighted-Cost Maze
- **Mechanic**: Replaced random free cells with "Mud" patches. Walking onto a mud patch incurs a harsh penalty (`-0.05`), actively discouraging the agent from using those cells unless strictly necessary.
- **Results**: 
  - **Q-Learning** (2,000 episodes) achieved a 100% success rate. The generated policy maps explicitly showed the agent routing *around* mud patches whenever a cleaner path existed.
  - **DQN** (500 episodes) struggled to balance the negative penalties, achieving a 32% success rate within the allotted training budget. The mud penalties created complex local minima.

### 4.2 Dynamic Obstacle Maze
- **Mechanic**: Walls randomly spawn and disappear during an episode based on a dynamic probability threshold. A Breadth-First Search (BFS) solvability checker ensures the goal is never permanently blocked, but the optimal path constantly shifts.
- **Results**:
  - **Q-Learning** achieved a 97.9% success rate. Because Q-Learning rapidly updates its table based on immediate states, it managed to learn a resilient "average" policy that succeeded despite the shifting walls.
  - **DQN** achieved a 0% success rate within 500 episodes. The probabilistic walls fundamentally break the Markov property for standard DQNs. Because the agent cannot "see" the walls moving until it hits them, the transition dynamics become non-stationary, severely confusing the replay buffer.

---

## 5. Conclusion & Future Work

The project successfully demonstrated that while Tabular Q-Learning is exceptionally efficient for discrete, fully-observable grid worlds, it does not scale to massive state spaces. DQN acts as a powerful function approximator but is highly sensitive to hyperparameter tuning and non-stationary environments.

**Future Extensions:**
1. **Recurrent Neural Networks (LSTM/GRU)**: To solve the Dynamic Obstacle maze with Deep RL, replacing the DQN's MLP with a recurrent layer would allow the agent to retain a memory of disappearing walls and build an internal belief state of the shifting maze.
2. **Prioritized Experience Replay (PER)**: Implementing PER would drastically speed up the DQN's learning on the difficult Door-Key and Weighted mazes by sampling the rare, successful trajectories more frequently.
3. **Reward Shaping**: Utilizing the built-in `PotentialShaping` reward stubs to provide dense, distance-based rewards without altering the optimal policy, which would massively accelerate DQN convergence.
