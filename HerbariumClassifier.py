import datetime
import logging
import os
import sys

import mlflow_util
import mlflow
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from Metrics import Metrics, TrainingMetrics

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)


class Classifier:
    def __init__(self, artifact_path, n_epochs: int,
                 model: torch.nn.Module = None,
                 load_model_from_checkpoint=False,
                 scheduler=None,
                 scheduler_step_after_batch=False,
                 early_stopping=30, experiment_name='test',
                 lr=1e-3):
        self.n_epochs = n_epochs
        self._artifact_path = artifact_path
        self._model = model
        self.scheduler = scheduler
        self.scheduler_step_after_batch = scheduler_step_after_batch
        self.use_cuda = torch.cuda.is_available()
        self.early_stopping = early_stopping
        self.logger = logging.getLogger(__name__)
        self._exeriment = mlflow.get_experiment_by_name(experiment_name)
        self._best_epoch_val_loss = float('inf')
        self._run_id = None
        self._cur_epoch = 0

        if not load_model_from_checkpoint and model is None:
            raise ValueError('Must either load model from checkpoint or '
                             'supply model')

        if load_model_from_checkpoint:
            self._load_model_from_checkpoint()

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._criterion = torch.nn.CrossEntropyLoss()

        if not os.path.exists(f'{self._artifact_path}'):
            os.makedirs(f'{self._artifact_path}')

        if self.use_cuda:
            self._model.cuda()

    def train(self, train_loader: DataLoader,
              valid_loader: DataLoader,
              log_after_each_epoch=True):
        all_train_metrics = TrainingMetrics(n_epochs=self.n_epochs)
        all_val_metrics = TrainingMetrics(n_epochs=self.n_epochs)

        time_since_best_epoch = 0

        with mlflow.start_run(experiment_id=self._exeriment.experiment_id,
                                   run_id=self._run_id):
            for epoch in range(self._cur_epoch, self.n_epochs):
                n_train = len(train_loader.dataset)

                epoch_train_metrics = Metrics(N=n_train)

                self._model.train()

                pb = tqdm(enumerate(train_loader), total=len(train_loader),
                          desc=f'Epoch {epoch} Train', position=0, leave=True)
                for batch_idx, sample in pb:
                    data = sample['image']
                    target = sample['label']

                    # move to GPU
                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    self._optimizer.zero_grad()
                    output = self._model(data)
                    loss = self._criterion(output, target)
                    loss.backward()
                    self._optimizer.step()

                    epoch_train_metrics.update_loss(loss=loss.item(),
                                                    batch_size=data.shape[0])
                all_train_metrics.update(epoch=epoch,
                                         loss=epoch_train_metrics.loss)

                epoch_val_metrics = self.predict(valid_loader=valid_loader,
                                                 epoch=epoch)

                all_val_metrics.update(epoch=epoch,
                                       loss=epoch_val_metrics.loss,
                                       macro_f1=epoch_val_metrics.macro_f1)

                if epoch_val_metrics.loss < self._best_epoch_val_loss:
                    mlflow.pytorch.log_model(self._model,
                                                  artifact_path=self._artifact_path)
                    mlflow.set_tags(tags={'best_epoch': epoch})

                    all_train_metrics.best_epoch = epoch
                    all_val_metrics.best_epoch = epoch
                    self._best_epoch_val_loss = epoch_val_metrics.loss
                    time_since_best_epoch = 0
                else:
                    time_since_best_epoch += 1
                    if time_since_best_epoch > self.early_stopping:
                        self.logger.info('Stopping due to early stopping')
                        return all_train_metrics, all_val_metrics

                if not self.scheduler_step_after_batch:
                    if self.scheduler is not None:
                        if isinstance(self.scheduler,
                                      torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(epoch_val_metrics.macro_f1)
                        else:
                            self.scheduler.step()

                if log_after_each_epoch:
                    self.logger.info(f'Epoch: {epoch + 1} \t'
                                     f'Train loss: '
                                     f'{epoch_train_metrics.loss:.6f} \t'
                                     f'Val Loss: {epoch_val_metrics.loss:.6f}\t'
                                     f'Val F1: {epoch_val_metrics.macro_f1:.6f}\t')

                metrics = {
                    'train loss': epoch_train_metrics.loss,
                    'val loss': epoch_val_metrics.loss,
                    'val F1': epoch_val_metrics.macro_f1
                }
                mlflow.log_metrics(metrics=metrics, step=epoch)

        return all_train_metrics, all_val_metrics

    def predict(self, valid_loader: DataLoader, epoch=None,
                return_confusion_matrix=False,
                return_last_layer_feature_vectors=False,
                return_raw_scores=False):
        pb_desc = f'Epoch {epoch} Val' if epoch is not None else 'Val'
        pb = tqdm(enumerate(valid_loader), total=len(valid_loader),
                  desc=pb_desc, position=0, leave=True)

        n_val = len(valid_loader.dataset)
        epoch_val_metrics = Metrics(N=n_val)

        self._model.eval()
        for batch_idx, sample in pb:
            data = sample['image']
            target = sample['label']

            # move to GPU
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = self._model(data)
                if return_last_layer_feature_vectors:
                    yield data.shape[0]
                if return_raw_scores:
                    yield output, data.shape[0]

                loss = self._criterion(output, target)

                epoch_val_metrics.update_loss(loss=loss.item(),
                                              batch_size=data.shape[0])
                epoch_val_metrics.update_outputs(y_true=target,
                                                 y_out=output,
                                                 update_confusion_matrix=return_confusion_matrix)

        return epoch_val_metrics

    def get_last_layer_features(self, valid_loader: DataLoader):
        global view_output

        def hook_fn(module, input, output):
            global view_output
            view_output = output

        hook = self._model.avgpool.register_forward_hook(hook_fn)

        last_layer_outputs = torch.zeros((len(valid_loader.dataset),
                                          self._model.fc.in_features))
        start_idx = 0
        for batch_size in self.predict(valid_loader=valid_loader,
                                               return_last_layer_feature_vectors=True):
            output = view_output.reshape(view_output.shape[0], -1)
            last_layer_outputs[start_idx:start_idx+batch_size] = output
            start_idx += batch_size
        hook.remove()

        return last_layer_outputs

    def get_classifier_scores(self, valid_loader):
        scores = torch.zeros((len(valid_loader.dataset),
                              self._model.fc.out_features))

        start_idx = 0
        softmax = torch.nn.Softmax(dim=1)
        for output, batch_size in self.predict(valid_loader=valid_loader,
                                               return_raw_scores=True):
            with torch.no_grad():
                scores[start_idx:start_idx+batch_size] = softmax(output)
            start_idx += batch_size
        return scores

    def _load_model_from_checkpoint(self):
        model, run_id, val_loss, epoch = \
            mlflow_util.load_model_from_checkpoint(
            experiment_id=self._exeriment.experiment_id,
            use_cuda=self.use_cuda
        )

        self._model = model
        self._run_id = run_id
        self._best_epoch_val_loss = val_loss
        self._cur_epoch = epoch + 1


def main():
    import torchvision
    from torch import nn
    import torchvision.transforms as transforms
    from HerbariumDataset import HerbariumDataset
    from DataLoader import TrainDataLoader
    import numpy as np
    from util import split_image_metadata

    mlflow.set_tracking_uri(
        'http://mlflo-mlflo-16mqjx084gpy-1597208273.us-west-2.elb.amazonaws.com')

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(448),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(448),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    np.random.seed(1234)

    split_image_metadata(path='data/images.csv', valid_frac=0.4)

    train_data = HerbariumDataset(
        annotations_file='data/annotations.csv',
                                  image_metadata_file='data/train_images.csv',
                                  img_dir='data', transform=train_transform)
    valid_data = HerbariumDataset(
        annotations_file='data/annotations.csv',
                                  image_metadata_file='data/valid_images.csv',
                                  img_dir='data', transform=valid_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=0)

    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features

    num_targets = train_data.annotations['category_id'].nunique()
    model.fc = nn.Linear(num_ftrs, num_targets)

    classifier = Classifier(artifact_path='checkpoints',
                            n_epochs=100, early_stopping=3,
                            load_model_from_checkpoint=True)
    classifier.train(train_loader=train_loader, valid_loader=valid_loader)


if __name__ == '__main__':
    main()
