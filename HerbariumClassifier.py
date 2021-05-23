import logging
import os
import sys

import mlflow
import torch
from torch.utils.data import DataLoader

from Metrics import Metrics, TrainingMetrics

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)


class Classifier:
    def __init__(self, model: torch.nn.Module, n_epochs: int, optimizer,
                 criterion, save_path, scheduler=None,
                 scheduler_step_after_batch=False,
                 early_stopping=30):
        self.n_epochs = n_epochs
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_after_batch = scheduler_step_after_batch
        self.criterion = criterion
        self.use_cuda = torch.cuda.is_available()
        self.save_path = save_path
        self.early_stopping = early_stopping
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(f'{self.save_path}'):
            os.makedirs(f'{self.save_path}')

        if self.use_cuda:
            self.model.cuda()

    def train(self, train_loader: DataLoader,
              valid_loader: DataLoader,
              log_after_each_epoch=True, save_model=False):
        all_train_metrics = TrainingMetrics(n_epochs=self.n_epochs)
        all_val_metrics = TrainingMetrics(n_epochs=self.n_epochs)

        best_epoch_val_loss = float('inf')
        time_since_best_epoch = 0

        with mlflow.start_run():
            for epoch in range(self.n_epochs):
                n_train = len(train_loader.sampler)
                n_val = len(valid_loader.sampler)

                epoch_train_metrics = Metrics(N=n_train)
                epoch_val_metrics = Metrics(N=n_val)

                self.model.train()
                for batch_idx, sample in enumerate(train_loader):
                    data = sample['image']
                    target = sample['label']

                    # move to GPU
                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

                    epoch_train_metrics.update_loss(loss=loss.item(),
                                                    batch_size=data.shape[0])

                all_train_metrics.update(epoch=epoch,
                                         loss=epoch_train_metrics.loss)

                self.model.eval()
                for batch_idx, sample in enumerate(valid_loader):
                    data = sample['image']
                    target = sample['label']

                    # move to GPU
                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    # update the average validation loss
                    with torch.no_grad():
                        output = self.model(data)
                        loss = self.criterion(output, target)

                        epoch_val_metrics.update_loss(loss=loss.item(),
                                                      batch_size=data.shape[0])
                        epoch_val_metrics.update_outputs(y_true=target,
                                                         y_out=output)

                all_val_metrics.update(epoch=epoch,
                                       loss=epoch_val_metrics.loss,
                                       macro_f1=epoch_val_metrics.macro_f1)

                if epoch_val_metrics.loss < best_epoch_val_loss:
                    if save_model:
                        torch.save(self.model.state_dict(),
                                   f'{self.save_path}/model.pt')
                    all_train_metrics.best_epoch = epoch
                    all_val_metrics.best_epoch = epoch
                    best_epoch_val_loss = epoch_val_metrics.loss
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


def main():
    import torchvision
    from torch import nn
    import torchvision.transforms as transforms
    from HerbariumDataset import HerbariumDataset
    from DataLoader import TrainDataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(448),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_data = HerbariumDataset(annotations_file='data/annotations.csv',
                                  image_metadata_file='data/images.csv',
                                  img_dir='data', transform=transform)
    train_data_loader = TrainDataLoader(data=train_data, valid_frac=0.5)
    train_loader, valid_loader = train_data_loader.data_loaders

    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features

    num_targets = train_data.annotations['category_id'].nunique()
    model.fc = nn.Linear(num_ftrs, num_targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    classifier = Classifier(model=model, optimizer=optimizer,
                            criterion=criterion, save_path='checkpoints',
                            n_epochs=100, early_stopping=3)
    classifier.train(train_loader=train_loader, valid_loader=valid_loader)


if __name__ == '__main__':
    main()
