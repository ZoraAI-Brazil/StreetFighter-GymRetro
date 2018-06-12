class Model:

    def __init__(self, brain):
        # Neural Network do modelo
        self.brain = brain
        # Pontuacao do modelo
        self.score = 0
        # Memoria do modelo (Dados e acoes que foram executadas a partir deles)
        self.memory = []

    # Tomada de decisao do modelo
    def think(self, inputs):
       return self.brain.predict(inputs)
    
