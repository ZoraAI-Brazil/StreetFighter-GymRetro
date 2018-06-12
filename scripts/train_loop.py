import retro
import random
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 40

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def act(self, state):
        act_rand = np.random.randint(2, size=self.action_size)
        if np.random.rand() <= self.epsilon:
            return act_rand
        act_values = self.model.predict(state)

        for i in range(len(act_values)):
            if act_values[0][i] > 0:
                act_rand[i] = 1
        return act_rand  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            state = np.reshape(state, [1, state_size])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def main():
    env = retro.make(game='StreetFighterII-Genesis', state='round1')
    obs = env.reset()
    state_size = obs.size
    action_size = env.action_space.n
    print(state_size)
    state = np.reshape(obs, [1, state_size])
    e = 0
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/streetfighter-dqn.h5")
    batch_size = 10000
    rew = 0

    while True:
        #action = env.action_space.sample()
        acao = agent.act(state)
        obs, _, done, info = env.step(acao)
        #env.render()

        #print("Ação do keras " + str(acao))
        #print("Ação como tem q ser " + str(action))
        next_state = np.reshape(obs, [1, state_size])
        agent.remember(obs, acao, rew, next_state, done)
        #print(obs)
        if done:
            obs = env.reset()
            e += 1
            print("Iteração " + str(e))
            #print(rew)

            print("episode: {}/{}, score: {}, e: {:.2}"
            .format(e, EPISODES, rew, agent.epsilon))

            if e >= EPISODES: break


        else:
            rew = info["health"]-info["enemyHealth"]
            if info["enemyHealth"] <= 2:
                rew += 1000
            if len(agent.memory) > batch_size:
                 agent.replay(batch_size)
                 #print(len(agent.memory))
    if e % 10 == 0:
        agent.save("./save/streetfighter-dqn.h5")
    obs = env.close()
if __name__ == '__main__':
    gen = 0
    while True:
        gen += 1
        print("GENERATION 1: "+ str(gen))
        main()
