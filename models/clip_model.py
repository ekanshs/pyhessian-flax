from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp
import jax

from transformers import (AutoConfig, 
                          FlaxCLIPModel,
                          FlaxCLIPVisionModel
                          )

import tqdm

ModuleDef=Any

vision_module = FlaxCLIPVisionModel.module_class

class CLIPModelwithClassifier(nn.Module):
  config: Any
  vision_module : ModuleDef
  classifier_module : ModuleDef
  dtype: jnp.dtype = jnp.float32  

  @nn.compact
  def __call__(self, pixel_values):
    encoder_outputs = self.vision_module(self.config.vision_config, dtype=self.dtype, name='encoder')(
            pixel_values=pixel_values
        )
    pooled_outputs = encoder_outputs[1]
    image_features = nn.Dense(
        self.config.projection_dim,
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.normal(0.02),
        use_bias=False,
        name='visual_projection'
    )(pooled_outputs)
    
    # normalize features
    image_features /= jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    
    logits = self.classifier_module(logit_scale = self.param("logit_scale", lambda _, shape: jnp.ones(shape) * self.config.logit_scale_init_value, []),
                                    dtype=self.dtype, 
                                    name='classifier'
                                    )(image_features) 
    return logits
  
  def permutation_spec(self, skip_classifier=False):
    ## TBD:
    pass

class MultiheadClassifier(nn.Module):
  """Classifier Layer."""
  num_classes : Sequence[int]
  logit_scale: Any
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, x):
    logits = []
    for classes in self.num_classes:
      logits += [jnp.exp(self.logit_scale) * jnp.asarray(nn.Dense(classes, dtype=self.dtype, use_bias=False)(x), self.dtype)]
    return logits

class Classifier(nn.Module):
  """Classifier Layer."""
  num_classes : int
  logit_scale: Any
  dtype: Any = jnp.float32
  
  @nn.compact
  def __call__(self, x):
    logits = jnp.exp(self.logit_scale) * jnp.asarray(nn.Dense(self.num_classes, dtype=self.dtype, use_bias=False)(x), self.dtype)
    return logits

def get_clip_model_with_classifier(*, num_classes, **kwargs):
  try:
    nheads = len(num_classes)
    classifier = partial(MultiheadClassifier, num_classes=num_classes)
  except:
    classifier = partial(Classifier, num_classes=num_classes)
  return CLIPModelwithClassifier(classifier_module=classifier, **kwargs)


def ViTB16(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('openai/clip-vit-base-patch16')
  return get_clip_model_with_classifier(num_classes=num_classes, 
                                        vision_module=vision_module, 
                                        config=model_config,
                                        dtype=dtype)

def ViTB32(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('openai/clip-vit-base-patch32')
  return get_clip_model_with_classifier(num_classes=num_classes, 
                                        vision_module=vision_module, 
                                        config=model_config,
                                        dtype=dtype)

def ViTL14(*, num_classes, dtype, **kwargs):
  model_config = AutoConfig.from_pretrained('openai/clip-vit-large-patch14')
  return get_clip_model_with_classifier(num_classes=num_classes, 
                                        vision_module=vision_module, 
                                        config=model_config,
                                        dtype=dtype)

