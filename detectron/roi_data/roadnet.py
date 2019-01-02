
"""Minibatch construction for Roadline Networks (RoadNet)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_roadnet_blob_names(is_training=True):
    if is_training:
        blob_names = ['roadline_int32']
    else:
        blob_names = []
    return blob_names

def get_roadnet_blobs(blobs, im_scale, roidb):
    for im_i, entry in enumerate(roidb):
        _label = np.array([entry['roadline']],dtype=np.int32)
        # _label = _label.reshape(1,1,1,1)
        blobs['roadline_int32'].append(_label)

    v = blobs['roadline_int32']
    if isinstance(v, list) and len(v) > 0:
        blobs['roadline_int32'] = np.concatenate(v)
    logger.debug('Road line:{}'.format(blobs['roadline_int32']))
    return True