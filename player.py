import numpy as np
from scipy.special import softmax
from copy import deepcopy
from tqdm import tqdm
import torch
from gym import Env
from mcts import Node, NodeUCT, NodePUCT
from mcts import selection, expansion, backpropagation
from model import ToricNetwork, AlphaZeroLoss
from utils import play_game, rollout


class Player:
    def __init__(self, env: Env):
        self.env = env
        self.observation_shape = env.observation_space.shape
        self.action_n = env.action_space.n

    def next_action(self, observation):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def evaluation(self, times=10):
        win = 0

        for _ in tqdm(range(times)):
            o = self.env.reset()
            while True:
                a = self.next_action(o)
                o, r, done, _ = self.step(a)
                if done:
                    if r == 1:
                        win += 1
                    break

        return win / times


class RandomPlayer(Player):
    def __init__(self, env: Env):
        super().__init__(env)

    def next_action(self, observation):
        return np.random.randint(0, self.action_n)

    def step(self, action):
        return self.env.step(action)


class MCTS(Player):
    def __init__(self, env: Env, simulation_times=100, rollout_times=1):
        super().__init__(env)
        self.env_copied = deepcopy(env)
        self.root = Node()
        self.selected = Node()
        self.node_type = NodeUCT
        self.simulation_times = simulation_times
        self.rollout_times = rollout_times

    def next_action(self, observation) -> int:
        if not np.array_equal(self.root.observation, observation):
            self.root = self.node_type(observation=observation)

        for i in range(self.simulation_times):
            self.env_copied = deepcopy(self.env)
            self.selected, actions = selection(self.root)

            o, r, done = play_game(self.env_copied, actions)
            if not done:
                if o is not None:
                    self.selected.observation = o
                r += self._simulation()

            backpropagation(node=self.selected, result=r)

        scores = [n.q_average() for n in self.root.children]
        index = scores.index(max(scores))
        return index

    def step(self, action: int):
        self.root = self.root.children[action]
        self.root.parent = None
        return self.env.step(action)

    def _simulation(self):
        expansion(node=self.selected, n=self.action_n)
        return rollout(self.env_copied, self.rollout_times)


class AlphaZero(MCTS):
    def __init__(self, env: Env, simulation_times=100):
        super().__init__(env, simulation_times)
        self.node_type = NodePUCT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nn = ToricNetwork(self.observation_shape, self.action_n).to(self.device)

    def _simulation(self):
        o = self.selected.observation[np.newaxis, :]
        o = torch.Tensor(o).to(self.device)
        p, v = self.nn(o)
        p = p.tolist()[0]
        v = v.item()
        expansion(node=self.selected, n=self.action_n, p=p)
        return v

    def root_probs(self, temperature=1.0):
        return softmax([c.n ** (1 / temperature) for c in self.root.children]).tolist()

    def train(self, epochs=1000, lr=0.1, weight_decay=1e-4):
        optimizer = torch.optim.SGD(self.nn.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = AlphaZeroLoss()
        loss_record = []
        p_done = softmax([1] * self.action_n).tolist()

        for _ in tqdm(range(epochs)):
            o_gained = []
            p_gained = []
            v_gained = []

            o = self.env.reset()
            while True:
                a = self.next_action(o)
                o_gained.append(self.root.observation.tolist())
                p_gained.append(self.root_probs())
                v_gained.append([self.root.q_average()])

                o, r, done, _ = self.step(a)
                if done:
                    o_gained.append(o.tolist())
                    p_gained.append(p_done)
                    v_gained.append([r])
                    break

            o_gained = torch.Tensor(o_gained).to(self.device)
            p_gained = torch.Tensor(p_gained).to(self.device)
            v_gained = torch.Tensor(v_gained).to(self.device)
            p, v = self.nn(o_gained)

            optimizer.zero_grad()
            loss = loss_fn(p, p_gained, v, v_gained)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())

        return loss_record
