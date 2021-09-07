import gc

import tensorflow.keras.backend
from tensorflow.keras.callbacks import Callback


class MemCleanerCheckpoint(Callback):
    def on_batch_end(self, batch, logs=None):
        gc.collect()
        tensorflow.keras.backend.clear_session()
        return super().on_batch_end(batch, logs)
