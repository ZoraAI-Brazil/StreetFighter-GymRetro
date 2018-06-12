from Model import *
from Settings import *

import gym
import numpy as np
import random
from heapq import nlargest

from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.models import save_model

# Cria a rede neural do modelo
def makeBrain():
    brain = Sequential()

    brain.add(Dense(units=6, activation='relu', input_dim=INPUTS))
    brain.add(Dense(units = 12, activation='relu'))
    brain.add(Dense(units=OUTPUTS, activation='softmax'))

    return brain

# Cria a população inicial
def createInitialPopulation():
    models_list = []
    scores = []

    for _ in range(POPULATION_SIZE):
        model = Model(makeBrain())
        environment.reset()
        game_memory = []
        prev_obs = []
        for _ in range(STEPS):
            if RENDER:
                environment.render()
            action = random.randrange(0, OUTPUTS)
            obs, rew, done, info = environment.step(action)

            if len(prev_obs) > 0:
                game_memory.append([prev_obs, action])

            prev_obs = obs
            model.score += rew
            if done:
                break

        model.memory = game_memory
        models_list.append(model)
        scores.append(model.score)
        model.score = 0
    
    best_models = getBestModels(models_list, scores)
    saveBestModels(best_models)

# Cria uma nova população a partir dos melhores modelos da geração anterior
def breedModels():
    models = []
    best_models = loadBestModels()
    for model in best_models:
        models.append(model)
    while len(models) != POPULATION_SIZE:
        parent1 = random.randint(0, len(best_models)-1)
        parent1 = best_models[parent1]
        parent1_weights = parent1.brain.get_weights()
        parent2 = random.randint(0, len(best_models)-1)
        parent2 = best_models[parent2]
        parent2_weights = parent2.brain.get_weights()
        new_model = Model(makeBrain())
        new_model_weights = new_model.brain.get_weights()
        for i in range(len(new_model_weights)):
            if i < len(new_model_weights/2):
                new_model_weights[i] = parent1_weights[i]
            else:
                new_model_weights[i] = parent2_weights[i]

        new_model.brain.set_weights(new_model_weights)
        models.append(new_model)
        
    return models

# Gera uma mutação em alguns modelos dependendo da probabilidade
def mutateModels(models):
    mutated_models = []
    for model in models:
        chance = random.randint(1, 100)
        if chance <= 5:
            weights = model.brain.get_weights()
            for i in weights:
                chance2 = random.randint(0,1)
                if chance2 == 0:
                    weights[i] = random.randrange(0,1)

            model.brain.set_weights(weights)

        mutated_models.append(model)

    return mutated_models

# Roda a população no jogo
def runModels(mutated_models):
    scores = []
    models_list = []

    for model in mutated_models:
        environment.reset()
        game_memory = []
        prev_obs = []
        for _ in range(STEPS):
            if RENDER:
                environment.render()
            if len(prev_obs) == 0:
                action = random.randrange(0, OUTPUTS)
            else:
                action = model.think(prev_obs)

            obs, rew, done, info = environment.step(action)
            prev_obs = obs
            game_memory.append([prev_obs, action])
            model.score += rew

            if done:
                break

        model.memory = game_memory
        models_list.append(model)
        scores.append(model.score)
        model.score = 0

    best_models = getBestModels(models_list, scores)
    saveBestModels(best_models)

# Define os  N melhores modelos a partir de uma lista de scores
def getBestModels(models, scores):
    best_models = []
    best_index = nlargest(N_BEST, scores)
    for i in best_index:
        best_models.append(models[i])
    
    return best_models

# Salva os melhores modelos
def saveBestModels(best_models):
    for i in range(len(best_models)):
        best_models[i].save(SAVE_NAME+"{}".format(i))

# Carrega os melhores modelos
def loadBestModels():
    models = []
    for i in range(N_BEST):
        model = load_model(SAVE_NAME+"{}".format(i))
        models.append(model)
    return models

# Da um reward para o modelo dependendo do desempenho
def giveReward(models):
    pass

# Loop de treino
# def trainingLoop():
#     models = breedModels()
#     mutated_models = mutateModels(models)
#     runModels(mutated_models)
#     pass


environment = gym.make(GAME)
environment.reset()
# createInitialPopulation()
# trainingLoop()