from cka_reg import PROJECT_ROOT, get_images
import numpy as np
import torch as ch
import xarray as xr
import glob
import argparse
from torchvision import transforms as transform_lib
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from .datamodules_utils import Partition, CustomTensorDataset


class BehaviorDataModule(LightningDataModule):
    name = "BehaviorData"
    """
    A DataLoader for behavior data. Uses a dataconstructer class to format behavior data.
    """

    def __init__(
        self, hparams, behavior_dataset=None, num_workers=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        behavior_dataset = (
            behavior_dataset
            if behavior_dataset is not None
            else hparams.behavior_dataset
        )
        num_workers = num_workers if num_workers is not None else hparams.num_workers

        if isinstance(hparams, argparse.Namespace):
            self.hparams.update(vars(hparams))
        else:
            self.hparams.update(hparams)

        hparams = self.hparams
        self.image_size = hparams.image_size
        self.dims = (3, self.image_size, self.image_size)
        self.num_workers = num_workers
        self.batch_size = hparams.batch_size
        self.constructor = SOURCES[behavior_dataset](hparams)
        self.n_stimuli = int(1e10) if hparams.stimuli == "All" else int(hparams.stimuli)

        # data augmentation parameters
        self.behavior_train_transform = hparams.behavior_train_transform

        self.translate = eval(hparams.translate)
        self.rotate = eval(hparams.rotate)
        self.scale = eval(hparams.scale)
        self.shear = eval(hparams.shear)
        self.class_type = "category_name"

    def _get_DataLoader(self, *args, **kwargs):
        return DataLoader(*args, **kwargs)

    def train_transform(self):
        if self.behavior_train_transform:
            print("Using transforms on behavioral training data")
            transforms = [
                transform_lib.ToPILImage(),
                transform_lib.Resize(self.image_size),
                transform_lib.RandomAffine(
                    degrees=self.rotate,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear,
                    fillcolor=127,
                ),
                transform_lib.ToTensor(),
            ]
        else:
            print("No transforms on behavioral training data")
            transforms = [
                transform_lib.ToPILImage(),
                transform_lib.Resize(self.image_size),
                transform_lib.ToTensor(),
            ]
        preprocessing = transform_lib.Compose(transforms)
        return preprocessing

    def val_transform(self):
        preprocessing = transform_lib.Compose(
            [
                transform_lib.ToPILImage(),
                transform_lib.Resize(self.image_size),
                transform_lib.ToTensor(),
                # imagenet_normalization(),
            ]
        )
        return preprocessing

    def get_stimuli(self, stimuli_partition):
        """Stimuli are always the same, it's the target that changes"""
        return self.constructor.get_stimuli(stimuli_partition=stimuli_partition).astype(
            "float32"
        )[: self.n_stimuli]

    def get_target(self, stimuli_partition, animals, n_trials, trial_avg=True):
        # behavioral responses
        H = self.constructor.get_behavior_responses(
            animals=animals,
            n_trials=n_trials,
            trial_avg=trial_avg,
            stimuli_partition=stimuli_partition,
        ).astype("float32")[: self.n_stimuli]
        # if stimuli_partition == 'train':
        #     H = self.control_manipulate(H)
        Y = self.constructor.get_labels(
            stimuli_partition=stimuli_partition, class_type=self.class_type
        )[: self.n_stimuli]

        return (H, Y)

    def control_manipulate(self, H):
        raise NotImplementedError("Method not implemented")
        # def shuffle(X):
        #     ntotal = X.shape[0]
        #     shuffled_inds = np.random.choice(ntotal, ntotal, replace=False)
        #     return X[shuffled_inds]

        # def randn_like(X):
        #     return np.random.randn(*X.shape)

        # if 'shuffle' in self.hparams.controls:
        #     H = shuffle(H)

        # if 'random' in self.hparams.controls:
        #     H = randn_like(H)

        # if 'rank' in self.hparams.controls:
        #     raise NameError('Rank control not implemented!')

        # if 'spectrum' in self.hparams.controls:
        #     raise NameError('spectrum control not implemented!')

        # return H

    def train_dataloader(self):
        """Uses the train split from provided behavior data path"""
        hparams = self.hparams

        X = self.get_stimuli(stimuli_partition="train")
        Y = self.get_target(
            stimuli_partition="train",
            animals=hparams.fit_animals,
            n_trials=hparams.trials,
        )
        dataset = CustomTensorDataset((X, *Y), self.train_transform())
        print(self.num_workers)
        loader = self._get_DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        if self.hparams.verbose:
            print(f"behavioral train set shape: {X.shape}, {[y.shape for y in Y]}")
        return loader

    def val_dataloader(
        self,
        stimuli_partition="test",
        animals=None,
        batch_size=None,
        n_trials="All",
        trial_avg=True,
    ):
        """Uses the validation split of behavior data"""
        hparams = self.hparams
        animals = animals if animals is not None else hparams.test_animals
        batch_size = batch_size if batch_size is not None else self.batch_size
        X = self.get_stimuli(stimuli_partition=stimuli_partition)
        Y = self.get_target(
            stimuli_partition=stimuli_partition,
            animals=animals,
            n_trials=n_trials,
            trial_avg=trial_avg,
        )
        dataset = CustomTensorDataset((X, *Y), self.train_transform())
        loader = self._get_DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        if self.hparams.verbose:
            print(f"behavioral validation set shape: {X.shape}, {[y.shape for y in Y]}")
        return loader


############ Behavioral Data construction tools ############
class BehaviorDataConstructor:
    def __init__(self, hparams, partition_scheme, *args, **kwargs):
        self.hparams = hparams
        self.partition = Partition(*partition_scheme, seed=hparams.seed)
        self.verbose = hparams.verbose

    def get_stimuli(self, *args, **kwargs):
        # overwrite method with dataset specific operations
        raise NameError("Method not implemented")

    def get_behavior_responses(self, *args, **kwargs):
        # overwrite method with dataset specific operations
        raise NameError("Method not implemented")

    def get_labels(self, *args, **kwargs):
        # overwrite method with dataset specific operations
        raise NameError("Method not implemented")


def MuriDataConstructer(hparams):
    return _MuriDataConstructer(hparams)


def MuriValDataConstructer(hparams):
    return _MuriDataConstructer(hparams, partition_scheme=(660, 0, 660, 0))


class _MuriDataConstructer(BehaviorDataConstructor):
    ds = xr.open_dataset(f"{PROJECT_ROOT}/data/muri_monkey_behavior.nc")

    def __init__(
        self,
        hparams,
        variations="All",
        #  partition_scheme=(1320, 1220, 100, 0),
        partition_scheme=(1320, 0, 1320, 0),
        *args,
        **kwargs,
    ):
        super().__init__(hparams, partition_scheme, *args, **kwargs)
        if variations == "All":
            # return all stimuli
            self.idxs = self.ds.image_index.values
            assert partition_scheme[0] == 1320
        else:
            raise NotImplementedError("Only all stimuli supported for now")

    def get_stimuli(self, stimuli_partition):
        self.ds.sel(image_index=self.idxs)
        image_paths = self.get_image_paths(self.idxs)
        X = (np.array(get_images(image_paths)) / 255.0).transpose(
            0, 3, 1, 2
        )  # -> (stimuli, channels, height, width)
        # partition the stimuli
        X_Partitioned = self.partition(X)[stimuli_partition]
        return X_Partitioned

    def get_labels(self, stimuli_partition, class_type):
        # Muri 0-9 labels
        X = self.ds.class_labels.sel(image_index=self.idxs).values
        X_Partitioned = self.partition(X)[stimuli_partition]
        return X_Partitioned

    @staticmethod
    def get_image_paths(image_indices):
        return [f"{PROJECT_ROOT}/data/kar2019_images/im{i}.png" for i in image_indices]

    def get_behavior_responses(self, animals, n_trials, trial_avg, stimuli_partition):
        if self.verbose:
            print(
                f"constructing {stimuli_partition} data with\n"
                + f"animals:{animals}\n"
                + f"trials:{n_trials}\n"
            )
        # transform "All" to all dataset's animals
        animals = self.expand(animals)
        n_trials = int(1e10) if n_trials == "All" else int(n_trials)
        X = np.concatenate(
            [
                self._get_behavior_responses(animal, n_trials, trial_avg)
                for animal in animals
            ]
        )
        if self.verbose:
            print(f"Behavioral data shape:\n(stimuli) : {X.shape}")
        X_Partitioned = self.partition(X)[stimuli_partition]
        return X_Partitioned

    def _get_behavior_responses(self, animal, n_trials, trial_avg):
        X = self.ds.sel(
            animal=animal,
            image_index=self.idxs,
            trial=np.arange(min(n_trials, len(self.ds.trial))),
        ).correct.values
        if self.verbose:
            print(f"{animal} shape:\n(stimuli, trials) : {X.shape}")
        if self.verbose:
            print(f"(stimuli, trials) : {X.shape}")

        if trial_avg:  # take mean over trials
            X = np.nanmean(X, axis=-1)
            assert ~np.isnan(np.sum(X))

        if self.verbose:
            print(f"(stimuli) : {X.shape}")

        return X

    @staticmethod
    def expand(animals):
        if animals[0] == "All":
            return ["bento", "magneto"]
        return animals


# class _MuriDataConstructer(NeuralDataConstructor):

#     data_root = "/braintree/data2/active/users/kohitij/for_guy"
#     data = h5.File(f"{data_root}/monkey_behavior.h5", "r")
#     # ds = xr.DataArray(data, dims=['trial', 'x'],
#     #                   coords={'x':
#     #                       np.array(pd.read_csv(f"{data_root}/column_names.csv")).flatten()}).to_dataset('x')
#     # ds = ds.assign_coords({'image_number': ds.image_number})

#     def __init__(self, hparams, partition_scheme=(1360, 1160, 200, 0), *args, **kwargs):
#         super().__init__(hparams, partition_scheme, *args, **kwargs)
#         self.n_heldout_neurons = 0

#     def get_stimuli(self, stimuli_partition):
#         image_paths = glob.glob(f"{self.data_root}/kar2019_images/*")
#         X = get_images(image_paths).transpose(0,3,1,2)
#         # partition the stimuli
#         X_Partitioned = self.partition(X)[stimuli_partition]
#         return X_Partitioned

#     def get_labels(self, stimuli_partition, class_type):
#         # TODO
#         # get label data -- already converted into integers corresponding to HVM category labels
#         X = self.data['category_name_HVM_aligned'][()][self.idxs]

#         X_Partitioned = self.partition(X)[stimuli_partition]

#         return X_Partitioned

#     def get_neural_responses(self, animals, n_neurons, n_trials, neuron_partition, stimuli_partition, hparams):
#         if self.verbose:
#             print(
#                 f'constructing {stimuli_partition} data with\n' +
#                 f'animals:{animals}\n' +
#                 f'neurons:{n_neurons}\n' +
#                 f'trials:{n_trials}\n'
#             )
#         # transform "All" to all dataset's animals
#         animals = self.expand(animals)
#         n_neurons = int(1e10) if n_neurons=='All' else int(n_neurons)
#         n_trials = int(1e10) if n_trials=='All' else int(n_trials)
#         X = np.concatenate([
#             self._get_neural_responses(animal, n_trials, neuron_partition, hparams)
#             for animal in animals
#         ], axis=1)

#         # only return [:n_neurons] if it's not the heldout set of neurons
#         if neuron_partition == 0:
#             # should be taking a random sample not just first n. can we reuse partition neurons?
#             X = X[:, :n_neurons]

#         if self.verbose: print(f'Neural data shape:\n(stimuli, sites) : {X.shape}')

#         X_Partitioned = self.partition(X)[stimuli_partition]
#         return X_Partitioned

#     def _get_neural_responses(self, animal, n_trials, neuron_partition, hparams):
#         X = self.data[animal]['rates'][()][self.idxs]

#         if self.verbose:
#             print(
#                 f'{animal} shape:\n(stimuli, sites, trials) : {X.shape}'
#             )

#         """
#         get subset of neurons to fit/test on.
#         return_heldout==0 => fitting set,
#         return_heldout==1 => heldout set
#         """
#         if self.n_heldout_neurons != 0:
#             X = self.partition_neurons(
#                 X, X.shape[1]-self.n_heldout_neurons, seed=hparams.seed
#             )[neuron_partition]

#         if self.verbose:
#             print(f'(stimuli, sites, trials) : {X.shape}')

#         # take mean over trials
#         X = X[:,:,:n_trials]
#         X = np.nanmean(X, axis=2)

#         if self.verbose:
#             print(f'(stimuli, sites) : {X.shape}')

#         assert ~np.isnan(np.sum(X))
#         return X

#     @staticmethod
#     def expand(animals):
#         if animals[0] == 'All':
#             return ['bento', 'magneto']
#         return animals


SOURCES = {
    "muri": MuriDataConstructer,  # Similar to 'many monkeys'
    "muri_val": MuriValDataConstructer,  # Similar to 'many monkeys'
}
