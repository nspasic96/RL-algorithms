from run import DQNSolver, STATE_BUFFER_SIZE, INPUT_SIZE, ENV_NAME, processState
import gym
from collections import deque
import numpy as np

weights_path = r"C:\SpaceInvadors\env_SpaceInvaders-v0_pg_batch_size=32_apply_steps=11000_state_bufS=4_batch_norm_False_numberAc_6\model_weights_184_max_685.0.h5"
number_of_games = 100


q_network = DQNSolver()
q_network.load_weights(weights_path)

def spaceInvadersPlay(episodes):
    print("PLAyiininininninninining")
    env = gym.make(ENV_NAME)
    
    while True:
        stateBuffer = deque(maxlen = STATE_BUFFER_SIZE)
        for _ in range(STATE_BUFFER_SIZE):
            stateBuffer.append(np.zeros(shape=[*INPUT_SIZE[0:2],1]))

        terminal = False
        state = env.reset()

        state = processState(state, stateBuffer)

        cumulative_reward = 0

        while(not terminal):
            env.render()
            action = q_network.next_move(state, -1)
            state_next, reward, terminal, _ = env.step(action)

            state_next = processState(state_next, stateBuffer)

            cumulative_reward += reward
            state = state_next

        print("Episode {} over, total reward is {}".format(e, cumulative_reward))

if __name__ == "__main__":
    spaceInvadersPlay(number_of_games)

