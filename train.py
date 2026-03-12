import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightnint.callbacks.early_stopping import EarlyStopping 

def train_model(data_model, emo_model):
    checkpoint_callback = ModelCheckpoint(
        dir_path="./models", monitor="val_loss", mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience="3", verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        default_root_dir=" logs",
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=5,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger("logs/", name="emotion", version=1),
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    trainer.fit(data_model, emo_model)