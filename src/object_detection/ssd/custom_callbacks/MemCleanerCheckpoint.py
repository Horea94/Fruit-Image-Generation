import gc

import tensorflow


class MemCleanerCheckpoint(tensorflow.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        gc.collect()
        tensorflow.keras.backend.clear_session()
        return super().on_batch_end(batch, logs)
