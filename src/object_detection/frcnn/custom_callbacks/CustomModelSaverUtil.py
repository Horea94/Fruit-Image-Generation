import os
import numpy as np


class CustomModelSaverUtil:

    def save_model_and_loss(self, model, loss, model_path, loss_path):
        self.save_model_weights(model, model_path)
        self.save_loss(loss, loss_path)

    def save_model_weights(self, model, model_path):
        model.save_weights(model_path)

    def save_loss(self, loss, loss_path):
        f = open(loss_path, mode='w')
        f.write("%f" % loss)
        f.close()

    def load_model_weights(self, model, model_path):
        if os.path.exists(model_path):
            model.load_weights(model_path, by_name=True)
        else:
            print('%s file not found! Weights are initialized with default values.' % model_path)

    def load_last_loss(self, loss_path):
        loss = np.Inf
        if os.path.exists(loss_path):
            f = open(loss_path, mode='r')
            loss = float(f.readline().strip())
            f.close()
            print('Previous best loss was %f' % loss)
        else:
            print('%s file not found! Previous best loss not defined.' % loss_path)
        return loss
