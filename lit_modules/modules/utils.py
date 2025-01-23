from enum import Enum
from typing import List


class ModelLayers(Enum):
    RESNET101 = [
        "maxpool",
        "layer1.1.add",
        "layer2.0.add",
        "layer2.3.add",
        "layer3.1.add",
        "layer3.4.add",
        "layer3.7.add",
        "layer3.10.add",
        "layer3.13.add",
        "layer3.16.add",
        "layer3.2.bn1",
    ]
    VGG16 = [
        "x",
        "features.3",
        "features.7",
        "features.12",
        "features.16",
        "features.21",
        "features.25",
        "features.30",
        "classifier.1",
        "classifier.5",
    ]
    VGG19 = [
        "features.4",
        "features.9",
        "features.18",
        "features.27",
        "features.36",
    ]
    INCEPTION_V3 = [
        "maxpool1",
        "maxpool2",
        "Mixed_5b.avg_pool2d",
        "Mixed_5c.avg_pool2d",
        "Mixed_5d.avg_pool2d",
        "Mixed_6b.avg_pool2d",
        "Mixed_6c.avg_pool2d",
        "Mixed_6d.avg_pool2d",
        "Mixed_6e.avg_pool2d",
        "Mixed_7b.avg_pool2d",
        "Mixed_7c.avg_pool2d",
        "Mixed_7a.branch3x3_1.bn",
    ]
    ALEXNET = [
        "x",
        "features.1",
        "features.3",
        "features.6",
        "features.8",
        "features.11",
        "features.12",
        "avgpool",
        "classifier.1",
        "classifier.3",
        "classifier.6",
    ]
    VIT_B_16 = [
        "encoder.layers.encoder_layer_0.mlp",
        "encoder.layers.encoder_layer_1.mlp",
        "encoder.layers.encoder_layer_2.mlp",
        "encoder.layers.encoder_layer_3.mlp",
        "encoder.layers.encoder_layer_4.mlp",
        "encoder.layers.encoder_layer_6.mlp",
        "encoder.layers.encoder_layer_7.mlp",
        "encoder.layers.encoder_layer_8.mlp",
        "encoder.layers.encoder_layer_9.mlp",
        "encoder.layers.encoder_layer_11.mlp",
    ]
    VIT_B_32 = [
        "encoder.layers.encoder_layer_0.mlp",
        "encoder.layers.encoder_layer_1.mlp",
        "encoder.layers.encoder_layer_2.mlp",
        "encoder.layers.encoder_layer_3.mlp",
        "encoder.layers.encoder_layer_4.mlp",
        "encoder.layers.encoder_layer_6.mlp",
        "encoder.layers.encoder_layer_7.mlp",
        "encoder.layers.encoder_layer_8.mlp",
        "encoder.layers.encoder_layer_9.mlp",
        "encoder.layers.encoder_layer_11.mlp",
    ]
    RESNET50 = [
        "maxpool",
        "layer1.0.add",
        "layer1.2.add",
        "layer2.0.add",
        "layer2.2.add",
        "layer3.0.downsample.0",
        "layer3.1.add",
        "layer3.3.add",
        "layer3.5.add",
        "layer4.0.add",
        "layer3.2.bn1",
    ]
    RESNET18 = [
        "maxpool",
        "layer1.0.add",
        "layer1.1.add",
        "layer2.0.add",
        "layer2.1.add",
        "layer3.0.add",
        "layer3.1.add",
        "layer4.0.add",
        "layer4.1.add",
        "avgpool",
        "layer4.0.relu",
    ]
    EFFICIENTNET_V2_S = [
        "features.1.0.add",
        "features.2.2.add",
        "features.3.2.add",
        "features.4.2.add",
        "features.4.5.add",
        "features.5.3.add",
        "features.5.6.add",
        "features.6.1.add",
        "features.6.4.add",
        "features.6.7.add",
        "features.6.7.stochastic_depth",
    ]


class InferioTemporalLayer(Enum):
    ALEXNET = "features.12"
    RESNET50 = "layer3.2.bn1"
    RESNET101 = "layer3.2.bn1"
    VGG16 = "features.30"
    VGG19 = "features.36"
    INCEPTION_V3 = "Mixed_7a.branch3x3_1.bn"
    VIT_B_16 = "encoder.layers.encoder_layer_8.mlp"
    VIT_B_32 = "encoder.layers.encoder_layer_8.mlp"
    EFFICIENTNET_B0 = "features.6.2.stochastic_depth"
    RESNET18 = "layer4.0.relu"
