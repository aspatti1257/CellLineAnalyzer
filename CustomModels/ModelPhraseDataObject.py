class ModelPhraseDataObject(object):

    def __init__(self, model, phrase, score):
        self.model = model
        self.score = score
        self.phrase = phrase
