import mlflow
import torch


def load_model_from_checkpoint(experiment_id, use_cuda=True):
    run_infos = mlflow.list_run_infos(
        experiment_id=experiment_id, order_by=[
            'metric.val_loss'])
    best_run_info = run_infos[0]
    best_run = mlflow.get_run(run_id=best_run_info.run_id)
    run_id = best_run_info.run_id
    epoch = int(best_run.data.tags['best_epoch'])

    model_uri = f'runs:/{run_id}/model'
    if not use_cuda:
        model = mlflow.pytorch.load_model(model_uri=model_uri,
                                          map_location=torch.device('cpu'))
    else:
        model = mlflow.pytorch.load_model(model_uri=model_uri)

    return model, run_id, best_run.data.metrics['val loss'], epoch
