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

        if not load_model_from_checkpoint and model is None:
            raise ValueError('Must either load model from checkpoint or '
                             'supply model')

        if load_model_from_checkpoint:
            self._load_model_from_checkpoint()

        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            for epoch in range(self.n_epochs):
                n_train = len(train_loader.dataset)
                n_val = len(valid_loader.dataset)

                epoch_train_metrics = Metrics(N=n_train)
                epoch_val_metrics = Metrics(N=n_val)

                self._model.train()

                for batch_idx, sample in enumerate(tqdm(train_loader,
                                                        total=len(train_loader))):
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
                    mlflow.log_text(f'''
                    Epoch {epoch}
                    Train
                    Batch {batch_idx}/{len(train_loader)}
                    ''', 'train_log.txt')

                all_train_metrics.update(epoch=epoch,
                                         loss=epoch_train_metrics.loss)

                self._model.eval()
                for batch_idx, sample in enumerate(tqdm(valid_loader,
                                                        total=len(valid_loader))):
                    data = sample['image']
                    target = sample['label']

                    # move to GPU
                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    # update the average validation loss
                    with torch.no_grad():
                        output = self._model(data)
                        loss = self._criterion(output, target)

                        epoch_val_metrics.update_loss(loss=loss.item(),
                                                      batch_size=data.shape[0])
                        epoch_val_metrics.update_outputs(y_true=target,
                                                         y_out=output)
                    mlflow.log_text(f'''
                    Epoch {epoch}
                    Val
                    Batch {batch_idx}/{len(valid_loader)}
                    ''', 'train_log.txt')

                all_val_metrics.update(epoch=epoch,
                                       loss=epoch_val_metrics.loss,
                                       macro_f1=epoch_val_metrics.macro_f1)

                if epoch_val_metrics.loss < best_epoch_val_loss:
                    mlflow.pytorch.log_model(self._model,
                                                  artifact_path=self._artifact_path)

                    all_train_metrics.best_epoch = epoch
                    all_val_metrics.best_epoch = epoch
                    best_epoch_val_loss = epoch_val_metrics.loss
                    time_since_best_epoch = 0
                else:
                    time_since_best_epoch += 1
                    if time_since_best_epoch > self.early_stopping:
                        self.logger.info('Stopping due to early stopping')
                        mlflow.log_text(f'''
                        Epoch {epoch}
                        Stopping due to early stopping
                        ''', 'train_log.txt')
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

    def _load_model_from_checkpoint(self):
        model, run_id, val_loss = mlflow_util.load_model_from_checkpoint(
            experiment_id=self._exeriment.experiment_id,
            use_cuda=self.use_cuda
        )

        self._model = model
        self._run_id = run_id
        self._best_epoch_val_loss = val_loss


def main():
    import torchvision
    from torch import nn
    import torchvision.transforms as transforms
    from HerbariumDataset import HerbariumDataset
    from DataLoader import TrainDataLoader
    import numpy as np
    from util import split_image_metadata

    mlflow_util.set_tracking_uri('http://mlflo-mlflo-16mqjx084gpy-1597208273.us-west-2.elb.amazonaws.com')

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    classifier = Classifier(model=model, optimizer=optimizer,
                            criterion=criterion, artifact_path='checkpoints',
                            n_epochs=100, early_stopping=3,
                            load_model_from_checkpoint=True)
    classifier.train(train_loader=train_loader, valid_loader=valid_loader)


if __name__ == '__main__':
    main()
