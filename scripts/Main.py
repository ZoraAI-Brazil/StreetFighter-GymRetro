from Model import *
from Settings import *

import gym
import numpy as np
import random
from heapq import nlargest

from keras.layers import Dense
from keras.models import Sequential

# Cria a rede neural do modelo
def makeBrain():
    brain = Sequential()

    brain.add(Dense(units=6, activation='relu', input_dim=INPUTS))
    brain.add(Dense(units = 12, activation='relu'))
    brain.add(Dense(units=OUTPUTS, activation='softmax'))

    return brain

environment = gym.make(GAME)
environment.reset()

# Cria a população inicial
def createInitialPopulation():
    models = []
    scores = []

    for _ in range(POPULATION_SIZE):
        model = Model(makeBrain())
        environment.reset()
        game_memory = []
        prev_obs = []
        for _ in range(STEPS):
            action = random.randrange(0, OUTPUTS)
            obs, rew, done, info = environment.step(action)

            if len(prev_obs) > 0:
                game_memory.append([prev_obs, action])

            prev_obs = obs
            model.score += rew
            if done:
                break

        for data in game_memory:
            if data[1] == 1:
                data[1] = [0,1]
            elif data[1] == 0:
                data[1] = [1,0]

        model.memory = game_memory
        models.append(model)
        scores.append(model.score)
        model.score = 0
    
    return models

initial_pop = createInitialPopulation()

# Cria uma nova população a partir dos melhores modelos da geração anterior
def breedModels(models):
    pass

# Gera uma mutação em alguns modelos dependendo da probabilidade
def mutateModels(models):
    pass

# Roda a população no jogo
def runModels(models):
    pass

# Da um reward para o modelo dependendo do desempenho
def giveReward(models):
    pass

# Loop de treino
def trainingLoop():
    pass