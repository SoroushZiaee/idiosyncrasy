from cka_reg.datamodules.imagenet_datamodule import ImagenetDataModule
from cka_reg.datamodules.neural_datamodule import NeuralDataModule
from cka_reg.datamodules.behavior_datamodule import BehaviorDataModule

DATAMODULES = {
    "ImageNet": ImagenetDataModule,
    "NeuralData": NeuralDataModule,
    #'StimuliClassification' : StimuliDataModule
    "BehaviorData": BehaviorDataModule,
}
