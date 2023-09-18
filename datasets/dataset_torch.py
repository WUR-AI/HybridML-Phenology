import numpy as np
import torch
import torch.utils.data

import config
from datasets.dataset import Dataset
from util.normalization import min_max_normalize, normalize_latitude, normalize_longitude, normalize_altitude, \
    mean_std_normalize
from util.torch import batch_tensors


class TorchDataset(torch.utils.data.Dataset):

    """
    Allows lists of data to be wrapped in PyTorch Dataset objects
    """

    def __init__(self, data: list):
        super(TorchDataset, self).__init__()
        self._data = list(data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


class TorchDatasetWrapper:

    """
    Wraps around the TemperatureBlossomDataset so it returns PyTorch Dataset objects
    """

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def get_train_data(self, device=torch.device('cpu'),) -> TorchDataset:
        samples = self._dataset.get_train_data()
        samples = [self.cast_sample_to_tensors(sample, device=device) for sample in samples]
        return TorchDataset(samples)

    def get_test_data(self, device=torch.device('cpu'),) -> TorchDataset:
        samples = self._dataset.get_test_data()
        samples = [self.cast_sample_to_tensors(sample, device=device) for sample in samples]
        return TorchDataset(samples)

    @staticmethod
    def cast_sample_to_tensors(sample: dict,
                               dtype=config.TORCH_DTYPE,
                               device=torch.device('cpu'),
                               ) -> dict:  # TODO -- make including original sample optional?

        new_sample = {
            'lat': torch.tensor(sample['lat'], dtype=dtype).to(device),
            'lon': torch.tensor(sample['lon'], dtype=dtype).to(device),
            'alt': torch.tensor(sample['alt'], dtype=dtype).to(device),
            'location': sample['location'],
            'year': sample['year'],
            'bloom_doy': torch.tensor(sample['bloom_doy']).to(device),
            'bloom_ix': torch.tensor(sample['bloom_ix']).to(device),
            'bloom_date': sample['bloom_date'],
            'original': sample,
        }

        if 'temperature' in sample.keys():
            new_sample['temperature'] = torch.tensor(sample['temperature'], dtype=dtype).to(device)

        return new_sample

    @staticmethod
    def collate_fn(samples: list):
        """
        Function to batch individual samples for use in a DataLoader
        :param samples: a list of data points (as dicts) obtained from this dataset
        :return: a dict of batched data points
        """
        assert len(samples) > 0  # Need the list to be nonemtpy to check for the presence of optional data

        batched_sample = {
            'year': [sample['year'] for sample in samples],
            'location': [sample['location'] for sample in samples],
            'bloom_doy': batch_tensors(*[sample['bloom_doy'] for sample in samples]),
            'bloom_ix': batch_tensors(*[sample['bloom_ix'] for sample in samples]),
            'lat': batch_tensors(*[sample['lat'] for sample in samples]),
            'lon': batch_tensors(*[sample['lon'] for sample in samples]),
            'alt': batch_tensors(*[sample['alt'] for sample in samples]),
            'original': [sample['original'] for sample in samples],
        }

        if 'temperature' in samples[0].keys():
            batched_sample['temperature'] = batch_tensors(*[sample['temperature'] for sample in samples])

        return batched_sample

    @staticmethod
    def normalize(sample: dict, revert: bool = False):
        """

        Normalization parameters:
        - Temperature: {'min': -31.465029999999985, 'max': 56.44580000000002, 'avg': 6.649877167687087, 'std': 9.094601227419858}


        :param sample:
        :param revert:
        :return:
        """

        # sample['temperature_raw'] = sample['temperature']
        # sample['temperature'] = min_max_normalize(sample['temperature'], t_min, t_max)

        if 'temperature' in sample.keys():  # TODO -- photoperiod and others
            sample['temperature'] = TorchDatasetWrapper.normalize_temperature(sample['temperature'], revert=revert)

        sample['lat'] = normalize_latitude(sample['lat'], revert=revert)
        sample['lon'] = normalize_longitude(sample['lon'], revert=revert)
        sample['alt'] = normalize_altitude(sample['alt'], revert=revert)
        return sample

    @staticmethod
    def normalize_temperature(ts: torch.Tensor, revert: bool = False):
        return mean_std_normalize(ts, mean=6.65, std=9.09, revert=revert)

def get_normalization_parameters_temperature(dataset: Dataset) -> dict:
    if dataset.includes_temperature:
        data_train = dataset.get_train_data()

        ts = np.concatenate([d['temperature'] for d in data_train])

        tmin = np.min(ts)
        tmax = np.max(ts)

        tavg = np.mean(ts)
        tstd = np.std(ts)

        return {
            'min': tmin,
            'max': tmax,
            'avg': tavg,
            'std': tstd,
        }
    else:
        return {}


if __name__ == '__main__':
    from datasets.dataset import Dataset
    from sklearn.model_selection import train_test_split

    _dataset = Dataset(
        year_split=train_test_split(Dataset.YEAR_RANGE, random_state=config.SEED, shuffle=True),
        include_temperature=True,
    )

    _dataset_wrapped = TorchDatasetWrapper(_dataset)

    print(get_normalization_parameters_temperature(_dataset))

