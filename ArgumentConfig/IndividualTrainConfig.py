class IndividualTrainConfig(object):

    def __init__(self, algorithm, hyperparams, combo):
        self.algorithm = algorithm
        self.hyperparams = hyperparams
        self.combo = combo

    def shouldTrainIndividualCombo(self):
        return self.algorithm is not None and self.hyperparams is not None and \
               self.combo is not None
