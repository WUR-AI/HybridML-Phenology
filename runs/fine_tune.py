import argparse

import torch

from datasets.dataset import Dataset
from models.base_torch_accumulation import BaseTorchAccumulationModel
from models.components.param_v2 import AccumulationParameterMapping
from runs.fit_eval_util import configure_argparser_main, configure_argparser_dataset, get_configured_dataset, \
    configure_argparser_evaluation, get_args, _OPTIMIZERS, evaluate_model_using_args

if __name__ == '__main__':

    args = get_args()  # TODO -- create new function

    model_name = args.model_name or args.model_cls.__name__

    # Configure the dataset based on the provided arguments
    dataset, _ = get_configured_dataset(args)
    assert isinstance(dataset, Dataset)

    model = args.model_cls.load(model_name)

    assert isinstance(model, BaseTorchAccumulationModel)

    if isinstance(model.parameter_model, AccumulationParameterMapping):
        model.parameter_model.ungroup()

    model.freeze_operator_weights()

    device = torch.device('cuda') if torch.cuda.is_available() and not args.disable_cuda else torch.device('cpu')

    model_kwargs = dict()
    kwargs = {
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'scheduler_step_size': args.scheduler_step_size,
        'scheduler_decay': args.scheduler_decay,
        'clip_gradient': args.clip_gradient,
        'f_optim': _OPTIMIZERS[args.optimizer],
        'optim_kwargs': {
            'lr': args.lr,
        },
        'device': device,
        'model': model,
        'model_kwargs': model_kwargs,
    }

    model, _ = type(model).fit(
        dataset,
        **kwargs,
    )

    # If the model is a PyTorch model -> load on cpu
    if isinstance(model, torch.nn.Module):
        model.cpu()

    evaluate_model_using_args(
        args,
        model,
        f'{model_name}_finetuned',
        dataset,
    )
