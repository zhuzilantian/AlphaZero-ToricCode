import os
import csv
import numpy as np
from scipy.special import softmax
from copy import deepcopy
from typing import Union, Iterable
from gym import Env


def play_game(env: Env, actions):
    observation = None
    reward = 0
    done = False

    for a in actions:
        observation, r, done, _ = env.step(a)
        reward += r
        if done:
            break

    return observation, reward, done


def rollout(env: Env, times: int):
    action_n = env.action_space.n

    reward = 0
    for i in range(times):
        e = deepcopy(env)
        while True:
            a = np.random.randint(0, action_n)
            _, r, done, _ = e.step(a)
            reward += r
            if done:
                break

    return reward / times


def get_probs(data: Iterable[Union[int, float]], temperature=1.0):
    return softmax([d ** (1 / temperature) for d in data]).tolist()


def save_record(path, data):
    dirs, _ = os.path.split(path)
    if dirs != '':
        os.makedirs(dirs, exist_ok=True)

    try:
        f = open(path, 'w', newline='')
    except OSError as e:
        print(e.strerror)
    else:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


def load_record(path):
    data = None

    try:
        f = open(path, newline='')
    except OSError as e:
        print(e.strerror)
    else:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        data = next(reader)
        f.close()
    return data
