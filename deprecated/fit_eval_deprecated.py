#
# from datasets.dataset import Dataset
#
# from models.base import BaseModel
# from runs.fit_eval_util import get_args, get_configured_dataset, validate_args, fit_model, set_config_using_args, \
#     evaluate_model_using_args
#
# if __name__ == '__main__':
#
#     # Get the provided program arguments
#     args = get_args()
#     # Perform some checks for validity of the run configuration
#     validate_args(args)
#     # Set global config (e.g. seeds)
#     set_config_using_args(args)
#
#     # Obtain the model class that will be trained/evaluated
#     model_cls = args.model_cls
#     assert issubclass(model_cls, BaseModel)
#     # Get the model name (if specified, otherwise use the class name)
#     model_name = args.model_name or model_cls.__name__
#
#     # Configure the dataset based on the provided arguments
#     dataset, _ = get_configured_dataset(args)
#     assert isinstance(dataset, Dataset)
#
#     # If training should be skipped -> load the model from disk
#     if args.skip_fit:
#         model = model_cls.load(model_name)
#     # Otherwise, fit a new model
#     else:
#         model, _ = fit_model(args, dataset)
#
#         # Save the trained model
#         model.save(model_name)
#
#     # Evaluate the model if not disabled
#     if not args.skip_eval:
#         evaluate_model_using_args(
#             args,
#             model,
#             model_name,
#             dataset,
#         )
#
