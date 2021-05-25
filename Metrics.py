import numpy as np
import torch
from sklearn.metrics import f1_score


class TrainingMetrics:
    def __init__(self, n_epochs):
        self.losses = np.zeros(n_epochs)
        self.macro_f1s = np.zeros(n_epochs)
        self.best_epoch = None

    def update(self, epoch, loss, macro_f1=None):
        self.losses[epoch] = loss
        self.macro_f1s[epoch] = macro_f1


class Metrics:
    def __init__(self, N):
        """

        :param N: total number of data points
        """
        self._N = N
        self._loss = 0.0

        self.y_preds = []
        self.y_trues = []
        self._f1 = None
    
    @property
    def loss(self):
        return self._loss / self._N

    @property
    def macro_f1(self):
        if self._f1 is None:
            f1 = f1_score(y_true=self.y_trues, y_pred=self.y_preds,
                        average='macro')
            self._f1 = f1
        return self._f1

    def update_outputs(self, y_true, y_out=None):
        y_pred = torch.argmax(y_out, dim=1).cpu().numpy().tolist()
        y_true = y_true.detach().cpu().numpy().tolist()

        self.y_preds += y_pred
        self.y_trues += y_true

    def update_loss(self, loss, batch_size):
        # convert average loss to sum of loss
        self._loss += loss * batch_size
