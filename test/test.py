from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils
from detectron.core.config import save_to_json
c2_utils.import_detectron_ops()
from detectron.utils.collections import AttrDict

args = AttrDict()
args.cfg_file = '/home/xuzhewei/code/Detectron/configs/scut/e2e-frcnn-VGG16-C5.yaml'
args.opts = ['TEST.WEIGHTS','/home/xuzhewei/code/Detectron/output/scut/e2e-frcnn-VGG16-C5/detectron-output/train/scut_train_10x/generalized_rcnn/model_final.pkl']


workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
if args.cfg_file is not None:
    merge_cfg_from_file(args.cfg_file)
if args.opts is not None:
    merge_cfg_from_list(args.opts)
assert_and_infer_cfg()
logger = setup_logging(__name__)
logger.info('Called with args:')
logger.info(args)
logger.info('Testing with config:')
logger.info(pprint.pformat(cfg))
save_to_json(cfg, training=False)

while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
    logger.info('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
    time.sleep(10)

run_inference(
    cfg.TEST.WEIGHTS,
    ind_range=None,
    multi_gpu_testing=False,
    check_expected_results=True,
)