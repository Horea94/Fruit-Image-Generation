import numpy as np


class CustomLearningRateMonitor:
    def __init__(self, model, lr, min_lr, reduction_factor=0.1, patience=3, best_loss=np.Inf):
        self.patience = patience
        self.current_count = 0
        self.best_loss = best_loss
        self.model = model
        self.lr = lr
        self.min_lr = min_lr
        self.reduction_factor = reduction_factor

    def reduce_lr_on_plateau(self, loss):
        if self.lr > self.min_lr:
            if loss < self.best_loss:
                self.current_count = 0
                self.best_loss = loss
            else:
                self.current_count += 1
                if self.current_count >= self.patience:
                    self.lr *= self.reduction_factor
                    print("Loss did not improve in the past %d epochs, updating learning rate for %s to %f" % (self.current_count, self.model.name, self.lr))
                    self.current_count = 0
                    self.model.optimizer.learning_rate = self.lr
