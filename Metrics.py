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
        self.y_scores = []
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
        f1 = df.groupby('y_true')\
            .apply(lambda x: f1_score(y_true=[x.name] * x.shape[0],
                                      y_pred=x['y_pred'], pos_label=x.name))
        precision = df.groupby('y_true')\
            .apply(lambda x: precision_score(y_true=[x.name] * x.shape[0],
                                      y_pred=x['y_pred'], pos_label=x.name))
        recall = df.groupby('y_true')\
            .apply(lambda x: recall_score(y_true=[x.name] * x.shape[0],
                                      y_pred=x['y_pred'], pos_label=x.name))
        n = df.groupby('y_true').size()
        res = pd.DataFrame({'f1': f1, 'precision': precision, 'recall':
            recall, 'n': n})
        return res

    def top_n(self, n=5):
        scores = torch.vstack(self.y_scores)

        mean_scores = torch.mean(scores, dim=0)
        top = torch.argsort(mean_scores)[-n:]

        mean_scores = mean_scores[top].cpu().numpy()
        top = top.cpu().numpy()

        res = pd.DataFrame({'mean_score': mean_scores, 'category': top})
        res = res.sort_values('mean_score', ascending=False)
        return res

    def update_outputs(self, y_true, y_out=None, update_scores=False):
        y_pred = torch.argmax(y_out, dim=1).cpu().numpy().tolist()
        y_true = y_true.detach().cpu().numpy().tolist()

        self.y_preds += y_pred
        self.y_trues += y_true

        if update_scores:
            self.y_scores.append(y_out)

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