import numpy as np
from keras.callbacks import Callback
from utils.CustomModelSaverUtil import CustomModelSaverUtil


class CustomModelSaver(Callback):
    def __init__(self, model_path, loss_path, best_loss=np.Inf):
        super().__init__()
        self.helper = CustomModelSaverUtil()
        self.model_path = model_path
        self.loss_path = loss_path
        self.best_loss = best_loss

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        loss = logs.get('loss')
        if loss < self.best_loss:
            print("Loss improved from %f to %f, saving model." % (self.best_loss, loss))
            self.best_loss = loss
            self.helper.save_model_and_loss(self.model, loss, self.model_path, self.loss_path)
        else:
            print("Loss did not improve from %f." % self.best_loss)
