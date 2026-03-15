from datasets import load_dataset
from data_module import DataModule
from lightning_module import EmoModel
from train import train_model
from inference import EmoPredictor

emotion_ds = load_dataset("dair-ai/emotion", "split")

# print(emotion_ds['train']['features']['label'].unique())

data_model = DataModule(emotion_ds)
emo_model = EmoModel()
# train_model(data_model, emo_model)

sentence = "The news ruined my day"
predictor = EmoPredictor("./models/epoch=4-step=2500.ckpt")
pred = predictor.predict(sentence)
print(pred)

# data_model.prepare_data(emotion_ds)
# data_model.setup()

# print(next(iter(data_model.train_dataloader()))["input_ids"].shape)