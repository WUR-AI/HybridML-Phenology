
import argparse

from evaluation.evaluation import evaluate
from models.mean import MeanModel
from runs.util import configure_argparser_main, configure_argparser_dataset, get_configured_dataset

if __name__ == '__main__':

    run_name = 'Fit & Eval Mean model'

    parser = argparse.ArgumentParser(run_name)
    configure_argparser_main(parser)
    configure_argparser_dataset(parser)

    args = parser.parse_args()

    assert not args.hold_out_locations, 'This model does not support held-out locations'

    dataset, _ = get_configured_dataset(args)

    model_name = args.model_name or MeanModel.__name__

    if args.skip_fit:
        model = MeanModel.load(model_name)
    else:
        model, _ = MeanModel.fit(dataset)
        model.save(model_name)

    if not args.skip_eval:
        evaluate(model,
                 dataset,
                 model_name,
                 )  # TODO -- eval args


