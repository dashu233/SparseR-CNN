  
detectron2's structure
# Configure
detectron2 has a universial variable ```cfg```, define any config you need for building a model/trainer

using ```cfg = get_cfg()``` to get a default configer. look at ```detectron2/config/deafults.py``` for more details.

to add new config into ```cfg``` do as 
```
cfg.MODEL.YOUR_CONFIG_NAME = CN()
cfg.MODEL.YOUR_CONFIG_NAME.YOUR_CONFIG_ATTRIBUTE = "this is a config"
``` 

if your class need cfg, using ```@configurable``` before your ```__init__(...)```, and def ```from_config(class,cfg,OTHER_INPUT)``` as a class method. the output of ```from_config``` is the dict of all para in ```__init__()```

I have to say this way is a little bit complicate, so if you ensure your code is only use in dectectron2 framework, just add cfg in ```__init__(...,cfg,...)``` is better

# Trainer
extends from TrainBase. If I want to write my own model, TrainDefault is sometimes useful

## model
using Trainer.build_model(cfg) to build a model
```
def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
```

it will use ```cfg.MODEL.META_ARCHITECTURE``` to find a registered model and create it.

To register a model using
```
@META_ARCH_REGISTRY.register()
```
before your class

usually, a model has 4 parts to do inference

* self.preprocess_image
* self.backbone
* self.proposal_generator
* self.head

if we want to train a model, ```head``` and ```proposal_generator``` also need a loss as second output

### preprocess_image

I guess this part is not useful for me. usually we can just using some existed function. such as DefaultTrainer's ```preprocess_image```

### backbone

using cfg.MODEL.BACKBONE.NAME to assign a specific NN to the backbone, and then call
```
self.backbone = detectron2.modeling.build_backbone(cfg)
```
if you create a backbone by yourself, don't forget to add
```
@BACKBONE_REGISTRY.register()
```
before your building_backbone function

### proposal_generator

usually it was a learnable NN, the input of this NN is the feature map & some image info & target boxes(ground truth)
target boxes are used in training process

the output of NN has 2 parts, proposal region & losses term, the losses term usually used in training process

### head

same like proposal_generator, it's just a NN, so you can do whatever you want to do.

## optimizer
## ...
