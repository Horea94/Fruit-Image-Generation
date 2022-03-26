import gc

import tensorflow.python.keras.backend
from tensorflow.python.keras.callbacks import Callback


class MemCleanerCheckpoint(Callback):
    def on_batch_end(self, batch, logs=None):
        gc.collect()
        tensorflow.python.keras.backend.clear_session()
        return super().on_batch_end(batch, logs)
