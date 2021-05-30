import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


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
        self.confusion_matrix = None
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

    @property
    def per_class_stats(self):
        df = pd.DataFrame({'y_true': self.y_trues, 'y_pred': self.y_preds})
        tp = df.groupby('y_true').apply(
            lambda x: (x['y_pred'] == x.name).sum())
        fn = df.groupby('y_true').apply(
            lambda x: (x['y_pred'] != x.name).sum())
        fp = df.groupby('y_pred').apply(
            lambda x: (x['y_true'] != x.name).sum())

        res = pd.DataFrame({'tp': tp, 'fp': fp, 'fn':
            fn})
        res['precision'] = res['tp'] / (res['tp'] + res['fp'])
        res['recall'] = res['tp'] / (res['tp'] + res['fn'])
        res['f1'] = 2 * res['precision'] * res['recall'] / (res['precision']
                                                            + \
                                                            res['recall'])
        res['n'] = res['tp'] + res['fn']

        res = res.drop(['tp', 'fp', 'fn'], axis=1)
        return res

    def update_outputs(self, y_true, y_out=None,
                       update_confusion_matrix=False):
        y_pred = torch.argmax(y_out, dim=1).cpu().numpy().tolist()
        y_true = y_true.detach().cpu().numpy().tolist()

        self.y_preds += y_pred
        self.y_trues += y_true

        if update_confusion_matrix:
            if self.confusion_matrix is None:
                self.confusion_matrix = torch.zeros((y_out.shape[1],
                                                     y_out.shape[1]))

            self.confusion_matrix[y_true, y_pred] += 1

    def update_loss(self, loss, batch_size):
        # convert average loss to sum of loss
        self._loss += loss * batch_size


def main():
    m = Metrics(N=5)

    y_out = torch.tensor([[0, .9, .1], [.1, .5, .4], [0, .8, .2]])
    m.update_outputs(y_true=torch.tensor([1, 1]), y_out=y_out,
                     update_scores=True)

    y_out = torch.tensor([[.5, .5, 0], [.2, .4, .4], [.05, .8, .15]])
    m.update_outputs(y_true=torch.tensor([1, 1]), y_out=y_out,
                     update_scores=True)

    print(m.top_n(n=2))


if __name__ == '__main__':
    main()