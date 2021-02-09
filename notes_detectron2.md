  
detectron2's structure

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

I guess this part is not useful for me. usually we can just using some exists 


## optimizer
## ...
