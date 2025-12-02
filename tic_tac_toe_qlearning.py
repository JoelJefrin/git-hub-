import random
import pickle
from collections import defaultdict, deque

WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def check_winner(board):
    for a,b,c in WIN_LINES:
        if board[a] == board[b] == board[c] and board[a] != 0:
            return board[a]
    if 0 not in board:
        return 0
    return None

def board_to_state(board):
    return tuple(board)

def available_actions(board):
    return [i for i,v in enumerate(board) if v==0]

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.2):
        self.q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q[(state, action)]
    
    def choose_action(self, board):
        state = board_to_state(board)
        actions = available_actions(board)
        if random.random() < self.epsilon:
            return random.choice(actions)
        qvals = [self.get_q(state,a) for a in actions]
        maxq = max(qvals)
        max_actions = [a for a,qv in zip(actions,qvals) if qv==maxq]
        return random.choice(max_actions)
    
    def learn(self, state, action, reward, next_state, done):
        key = (state, action)
        cur_q = self.q[key]
        if done:
            target = reward
        else:
            next_actions = [a for a,v in enumerate(next_state) if v==0]
            if next_actions:
                next_qs = [self.get_q(next_state,a) for a in next_actions]
                target = reward + self.gamma * max(next_qs)
            else:
                target = reward
        self.q[key] = cur_q + self.alpha * (target - cur_q)

def play_episode(agent, opponent='random', learn=True, starting_player=1):
    board = [0]*9
    player = starting_player
    states_actions = []
    while True:
        if player == 1:
            action = agent.choose_action(board)
            state = board_to_state(board)
            board[action] = 1
            states_actions.append((state, action))
        else:
            actions = available_actions(board)
            action = random.choice(actions)
            board[action] = 2
        
        winner = check_winner(board)
        if winner is not None:
            if winner == 1:
                reward = 1.0
            elif winner == 2:
                reward = -1.0
            else:
                reward = 0.0
            if learn:
                for (s,a) in reversed(states_actions):
                    agent.learn(s,a,reward, tuple(board), done=True)
                    reward = 0.0
            return winner
        
        if player == 1 and learn:
            next_state = board_to_state(board)
            agent.learn(state, action, 0.0, next_state, done=False)
        
        player = 1 if player==2 else 2

def evaluate(agent, games=1000):
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    results = {'X_win':0, 'O_win':0, 'draws':0}
    for _ in range(games):
        winner = play_episode(agent, learn=False, starting_player=1)
        if winner == 1:
            results['X_win'] += 1
        elif winner == 2:
            results['O_win'] += 1
        else:
            results['draws'] += 1
    agent.epsilon = old_eps
    return results

def train(agent, episodes=20000, eval_every=4000):
    for ep in range(1, episodes+1):
        start = 1 if ep % 2 == 1 else 2
        play_episode(agent, learn=True, starting_player=start)
        if ep % eval_every == 0:
            eval_res = evaluate(agent, games=1000)
            print(f"Episode {ep}: Eval -> X wins: {eval_res['X_win']}, O wins: {eval_res['O_win']}, Draws: {eval_res['draws']}")

agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0.2)
train(agent, episodes=20000, eval_every=4000)

final_eval = evaluate(agent, games=2000)
print("\nFinal evaluation (2000 games):")
print(final_eval)

print("\nSample game:")
agent.epsilon = 0.0
board = [0]*9
mapping = {0:'.',1:'X',2:'O'}

player = 1
while True:
    if player == 1:
        action = agent.choose_action(board)
        board[action] = 1
    else:
        action = random.choice(available_actions(board))
        board[action] = 2
    
    for r in range(3):
        print(''.join(mapping[v] for v in board[3*r:3*r+3]))
    print("---")
    
    winner = check_winner(board)
    if winner is not None:
        if winner == 1:
            print("X (agent) wins")
        elif winner == 2:
            print("O (random) wins")
        else:
            print("Draw")
        break
    player = 1 if player==2 else 2

with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(agent.q), f)

print("\nSaved Q-table as q_table.pkl")

#hi this is joel
