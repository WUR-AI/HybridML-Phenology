
from datasets.dataset import Dataset


class BaseModel:
    """

        Base class for a Cherry Blossom DOY prediction model

    """

    # Define doy value that will be returned if there is no bloom according to the model
    NO_FIT = Dataset.SEASON_LENGTH - Dataset.DOY_SHIFT  # DOY at end of season
    NO_FIT_IX = Dataset.SEASON_LENGTH

    # Some models behave differently under train/test conditions
    # Models can be set to these modes to specify their desired behaviour
    MODE_TRAIN = 'train'
    MODE_TEST = 'test'

    def __init__(self):
        self._mode = BaseModel.MODE_TRAIN

    def predict_ix(self, x: dict) -> tuple:
        """
        Make a bloom doy prediction
        :param x: a dict containing a single data point
        :return: a three-tuple of:
            - the index (int) of the bloom doy (in the given temperature time series)
            - a bool indicating whether there was a bloom event according to the model
            - a dict containing additional info
        """
        raise NotImplementedError

    def predict(self, x: dict) -> tuple:
        """
        Make a bloom doy prediction
        :param x: a dict containing a single data point
        :return: a three-tuple of:
            - the bloom doy
            - a bool indicating whether there was a bloom event according to the model
            - a dict containing additional info
        """
        ix, bloom, info = self.predict_ix(x)
        doy = Dataset.index_to_doy(ix) if bloom else BaseModel.NO_FIT
        return doy, bloom, info

    def batch_predict_ix(self, xs: list) -> list:
        return [self.predict_ix(x) for x in xs]

    def batch_predict(self, xs: list) -> list:
        return [self.predict(x) for x in xs]

    @classmethod
    def fit(cls,
            dataset: Dataset,
            method: str = None,
            ) -> tuple:
        raise NotImplementedError

    def set_mode_train(self):
        """
        Set the model to train mode
        """
        self._mode = BaseModel.MODE_TRAIN

    def set_mode_test(self):
        """
        Set the model to test mode
        """
        self._mode = BaseModel.MODE_TEST

    @property
    def mode(self) -> str:
        return self._mode

    def is_in_mode_train(self) -> bool:
        return self._mode == BaseModel.MODE_TRAIN

    def is_in_mode_test(self) -> bool:
        return self._mode == BaseModel.MODE_TEST

    def save(self, model_name: str):
        raise NotImplementedError

    @classmethod
    def load(cls, model_name: str) -> 'BaseModel':
        raise NotImplementedError

