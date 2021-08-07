import numpy as np
from tensorflow.keras.callbacks import Callback
from custom_callbacks.CustomModelSaverUtil import CustomModelSaverUtil


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
        if loss is not None:
            if loss < self.best_loss:
                print("\nLoss improved from %f to %f, saving model." % (self.best_loss, loss))
                self.best_loss = loss
                self.helper.save_model_and_loss(self.model, loss, self.model_path, self.loss_path)
            else:
                print("\nLoss did not improve from %f." % self.best_loss)
        else:
            print("\nLoss is None Type. Check your network settings.")
