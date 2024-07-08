import torch as ch
import torch.nn as nn
from copy import deepcopy
from cka_reg import *
import glob
import numpy as np
from tqdm import tqdm


LAYER_MAPS = {
    "efficientnet_b0": {
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "V1": None,
        "V2": None,
        "V4": None,
        "IT": "_blocks.15",
        "decoder": "_blocks.15",
    },
    "mobilenet_v2": {
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "V1": None,
        "V2": None,
        "V4": None,
        "IT": "features.14",
        "decoder": "classifier",
    },
    "mobilenet_v3_large": {
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "V1": None,
        "V2": None,
        "V4": None,
        "IT": "features.13",
        "decoder": "avgpool",
    },
    "resnet18": {
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "V1": None,
        "V2": None,
        "V4": None,
        "IT": "layer4.1",
        "decoder": "avgpool",
        "output": "fc",
    },
    "resnet50": {
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "V1": None,
        "V2": None,
        "V4": None,
        "IT": "layer4.0",
        "decoder": "avgpool",
        "output": "fc",
    },
    "cornet_s": {
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "V1": "V1.output",
        "V2": "V2.output",
        "V4": "V4.output",
        "IT": "IT.output",
        "IT_temp": {"time_steps": 2, "output_step": 1},
        "decoder": "decoder.avgpool",
        "output": "decoder.linear",
    },
    "vonecornet_s": {
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "V1": "model.V1.output",
        "V2": "model.V2.output",
        "V4": "model.V4.output",
        "IT": "model.IT.output",
        "IT_temp": {"time_steps": 2, "output_step": 1},
        "decoder": "model.decoder.avgpool",
        "output": "model.decoder.linear",
    },
}


class Normalize(nn.Module):
    def __init__(self, normalization):
        super(Normalize, self).__init__()
        self.mean = ch.tensor(normalization["mean"]).reshape(3, 1, 1)  # .cuda()
        self.std = ch.tensor(normalization["std"]).reshape(3, 1, 1)  # .cuda()

    def forward(self, x):
        x = x - self.mean.type_as(x)
        x = x / self.std.type_as(x)
        return x


class Mask(nn.Module):
    """
    Mask the outputs of a network to only spit out a specific section of labels.
    """

    def __init__(self, model, start, stop):
        super(Mask, self).__init__()
        self.model = model
        self.start = start
        self.stop = stop

    def forward(self, x):
        x = self.model(x)
        return x[:, self.start : self.stop]


def copy_bns(model):
    bns = {}
    for name, module in model.named_modules():
        if ".norm" in name:
            bns[name] = deepcopy(module)

    return bns


def paste_bns(model, bns):
    for name, module in bns.items():
        name = name[1:]
        exec(f"model[1]{name} = module")
        # not sure why this way doesn't work?
        # setattr(model, name, module)

    return model


def add_normalization(model, **kwargs):
    return nn.Sequential(Normalize(**kwargs), model)


def add_outputs(model, out_name, n_outputs=8, verbose=False):
    # walk model to get to final output module (layer4.3.linear => linear)
    model_ = model[1]
    model_ = model_.module if hasattr(model_, "module") else model_

    layer_tree = [model_]
    for l in out_name.split("."):
        layer_tree.append(getattr(layer_tree[-1], l))

    old_out = layer_tree[-1]

    if verbose:
        print(old_out)

    # assuming output is always linear w a bias..
    new_out = ch.nn.Linear(
        in_features=old_out.in_features, out_features=old_out.out_features + n_outputs
    ).cuda()

    # write params of old output to new output
    for old_p, new_p in zip(old_out.parameters(), new_out.parameters()):
        if verbose:
            print(f"old_p.shape: {old_p.shape}")
        if verbose:
            print(f"new_p.shape: {new_p.shape}")
        new_p.requires_grad = False
        new_p[: old_p.shape[0]] = old_p.data
        new_p.requires_grad = True

    # set the new output to the output layer
    setattr(layer_tree[-2], l, new_out)
    return model


# extract intermediate representations
class Hook:
    """#### DON'T USE THIS WITH MULTIPLE GPUs!! ####"""

    def __init__(self, module, backward=False, time_steps=1, output_step=1):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

        self.time_steps = time_steps
        self.output_step = output_step
        self.counter = 0
        self.output = None

    def hook_fn(self, module, input, output):
        """
        works with multiple timesteps, ie CORnet-S
        if it's one step (standard) the counter will restart every time and the same
        output will be written every time.
        if time_steps is > 1, the counter will count to timesteps before restarting,
        and only the matching output_step will be written to output.
        """
        self.counter += 1
        if self.counter == self.output_step:
            self.output = output

        if self.counter == self.time_steps:
            self.counter = 0

    def close(self):
        self.hook.remove()


class FeatureExtractor:
    def __init__(self, cfg):
        cfg.dims = (3, cfg.image_size, cfg.image_size)
        self.cfg = cfg
        assert cfg.arch in ["cornet_s"], f"Unsupported architecture {cfg.arch}"
        self.layer_map = LAYER_MAPS[cfg.arch]
        self.model, self.weight_path = self.get_model(cfg.logdir)
        self.layer_hooks = self.hook_layers(self.model)

    def __call__(self, loader, verbose=False):
        feats_list = []
        with ch.no_grad():
            for x, *_ in tqdm(loader):
                # for x in tqdm(chunks(X, 100), desc="Extracting features"):
                _ = self.model(x.cuda())
                feats_list.append(self.layer_hooks["IT"].output.cpu().numpy())
        feats = np.vstack(feats_list)
        feats = feats.reshape(len(feats), -1)
        if verbose:
            cprint1(f"feats shape: {feats.shape}")
        return feats

    def get_model(self, logdir):
        # get all the model weight paths in a folder (searches recursively)
        weight_paths = [
            y
            for x in os.walk(f"{logdir}")
            for y in glob.glob(os.path.join(x[0], "epoch*.ckpt"))
        ]
        assert (
            len(weight_paths) == 1
        ), f"Found {len(weight_paths)} weights in {logdir}, expected 1"
        weight_path = weight_paths[0]

        # load model
        model = self.load_model(weight_path)
        return model, weight_path

    def load_model(self, path_to_weights):
        from cka_reg.model_lightning import models_dict

        # load the model architecture with the normalization preprocessor
        arch = self.cfg.arch
        model = models_dict[arch](pretrained=False)
        model = add_normalization(
            model, normalization=LAYER_MAPS[arch]["normalization"]
        )
        model = add_outputs(model, out_name="decoder.linear", n_outputs=8)
        # load weights and strip pesky 'model.' prefix
        state_dict = ch.load(path_to_weights)
        weights = {
            k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
        }
        # load the architecture with the trained model weights
        model.load_state_dict(weights)
        model.cuda()
        # model in eval mode.
        model.eval()

        # # models handle input normalization etc internally.
        # mean = np.array([0, 0, 0]).reshape((3, 1, 1))
        # std = np.array([1, 1, 1]).reshape((3, 1, 1))

        # model = PyTorchClassifier(
        #     model=model,
        #     clip_values=(0, 1),
        #     preprocessing=(mean, std),
        #     loss=nn.CrossEntropyLoss(),
        #     input_shape=self.cfg.dims,
        #     nb_classes=1008,
        # )

        return model

    def hook_layers(self, model):
        layer_hooks = {}
        layer_names = self.cfg.hook_layer.split(",")
        for ln in listify(layer_names):
            # this allows us to specify layer4.downsample0.maxpool for instance to get the maxpool in layer4.downsample0
            # [1] gets model instead of normalization layer [0]
            model = model[1]
            layer = model.module if hasattr(model, "module") else model
            # iteratively find layer to hook
            for id_ in self.layer_map[ln].split("."):
                layer = getattr(layer, id_)

            if f"{ln}_temp" in self.layer_map:
                layer_hooks[ln] = Hook(layer, **self.layer_map[f"{ln}_temp"])
            else:
                layer_hooks[ln] = Hook(layer)

        return layer_hooks

    def __repr__(self) -> str:
        """Return a string representation of this object by listing all its fields."""
        return (
            self.__class__.__name__
            + ":\n"
            + "\n".join(f"\t- {k}={v}" for k, v in self.cfg.__dict__.items())
            + "\n"
        )
