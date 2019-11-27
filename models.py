from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP."""

    def __init__(self, model_config):
        """Initializes a MLP.

        Arguments:
          model_config: A OrderedDict of lists. The keys of the dict indicate
            the names of different parts of the model. Each value of the dict
            is a list indicating the configs of layers in the corresponding
            part. Each element of the list is a list [layer_type, arguments],
            where layer_type is a string and arguments is a dict.
        """
        super(MLP, self).__init__()
        self.model_config = model_config

    def forward(self, inputs):
        outputs = None
        # TODO: Implement forward.
        raise NotImplementedError("`forward` not implemented.")
        return outputs
