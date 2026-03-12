from datasets import load_dataset
from data_module import DataModule
from lightning_module import EmoModel

emotion_ds = load_dataset("dair-ai/emotion", "split")

# print(emotion_ds)

data_model = DataModule()
data_model.prepare_data(emotion_ds)
data_model.setup()

print(next(iter(data_model.train_dataloader()))["input_ids"].shape)