import os
import numpy as np


class CustomModelSaverUtil:

    def save_model_and_loss(self, model, loss, model_path, loss_path):
        model.save_weights(model_path)
        f = open(loss_path, mode='w')
        f.write("%f" % loss)
        f.close()

    def load_model_weigths(self, model, model_path):
        if os.path.exists(model_path):
            model.load_weights(model_path)
        else:
            print('%s file not found! Weights are initialized with default values.' % model_path)

    def load_last_loss(self, loss_path):
        loss = np.Inf
        if os.path.exists(loss_path):
            f = open(loss_path, mode='r')
            loss = float(f.readline().strip())
            f.close()
            print('Previous best rpn loss was %f' % loss)
        else:
            print('%s file not found! Previous best loss not defined.' % loss_path)
        return loss
