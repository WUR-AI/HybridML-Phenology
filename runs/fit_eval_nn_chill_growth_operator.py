
import argparse

from evaluation.evaluation import evaluate
from models.components.param import LocalParams
from models.nn_chill_operator import NNChillGrowthModel
from runs.util import configure_argparser_main, configure_argparser_dataset, get_configured_dataset, \
    fit_torch_model_using_args, configure_argparser_fit_torch

if __name__ == '__main__':

    run_name = 'Fit & Eval Torch NN Chill Growth Operator'

    parser = argparse.ArgumentParser(run_name)
    configure_argparser_main(parser)
    configure_argparser_dataset(parser)
    configure_argparser_fit_torch(parser)

    args = parser.parse_args()

    dataset, _ = get_configured_dataset(args)

    model_name = args.model_name or NNChillGrowthModel.__name__

    if args.skip_fit:
        model = NNChillGrowthModel.load(model_name)
    else:
        model, _ = fit_torch_model_using_args(NNChillGrowthModel,
                                              dataset,
                                              args,
                                              model_kwargs={
                                                  'param_model': LocalParams(),
                                              }
                                              )
        model.cpu()
        model.save(model_name)

    if not args.skip_eval:
        evaluate(model,
                 dataset,
                 model_name,
                 )  # TODO -- eval args


