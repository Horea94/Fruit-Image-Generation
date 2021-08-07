from tensorflow.keras.callbacks import ModelCheckpoint


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, best=None, *args, **kwargs):
        super(MyModelCheckpoint, self).__init__(*args, **kwargs)
        if best is not None:
            self.best = best
