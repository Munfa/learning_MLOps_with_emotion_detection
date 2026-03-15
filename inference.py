import torch
from data_module import DataModule
from lightning_module import EmoModel

class EmoPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = EmoModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

    def predict(self, text):
        sample = {'sentence': text}
        processed = self.processor.tokenize_data(sample)
        logits = self.model(
            torch.tensor([processed['input_ids']]),
            torch.tensor([processed['attention_mask']])
        )
        scores = self.softmax(logits[0]).to_list()
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({'label': label, 'score': score})
        return predictions