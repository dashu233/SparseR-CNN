  
detectron2's structure
# Configure
detectron2 has a universial variable ```cfg```, define any config you need for building a model/trainer

using ```cfg = get_cfg()``` to get a default configer. look at ```detectron2/config/deafults.py``` for more details.

to add new config into ```cfg``` do as 
```
cfg.MODEL.YOUR_CONFIG_NAME = CN()
cfg.MODEL.YOUR_CONFIG_NAME.YOUR_CONFIG_ATTRIBUTE = "this is a config"
``` 

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

I guess this part is not useful for me. usually we can just using some exists. such as DefaultTrainer's ```preprocess_image```

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

## optimizer
## ...
