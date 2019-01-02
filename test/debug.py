
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import numpy as np
import pprint

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
from detectron.utils.logging import setup_logging_file
import detectron.utils.c2 as c2_utils
import detectron.utils.train
from detectron.core.config import save_to_json
c2_utils.import_contrib_ops()
c2_utils.import_detectron_ops()
from detectron.utils.collections import AttrDict

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
def main():

    OUTPUT_DIR = 'output/scut/S0-G0-e2e_frcnn_VGG16-C5_roadline/detectron-output'
    NUM_GPUS = 2
    EXP_ID = '20181217D01'

    args = AttrDict()
    args.skip_test = False
    args.multi_gpu_testing = False
    args.cfg_file = 'configs/scut/e2e_frcnn_VGG16-C5_roadline.yaml'
    args.opts = ['OUTPUT_DIR', OUTPUT_DIR, 'NUM_GPUS', NUM_GPUS, 'EXP_ID', EXP_ID]

    # Initialize C2
    workspace.GlobalInit(
        ['caffe2', '--caffe2_log_level=0', '--caffe2_gpu_memory_tracking=1']
    )
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    # Set up logging and load config options
    logger = setup_logging_file(__name__, cfg.OUTPUT_DIR+'/log-'+cfg.EXP_ID)
    logging.getLogger('detectron.roi_data.loader').setLevel(logging.INFO)
    logger.info('Called with args:')
    logger.info(args)
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))
    save_to_json(cfg,training=True)
    # Note that while we set the numpy random seed network training will not be
    # deterministic in general. There are sources of non-determinism that cannot
    # be removed with a reasonble execution-speed tradeoff (such as certain
    # non-deterministic cudnn functions).
    np.random.seed(cfg.RNG_SEED)
    # Execute the training run
    checkpoints = detectron.utils.train.train_model()
    # Test the trained model
    if not args.skip_test:
        test_model(checkpoints['final'], args.multi_gpu_testing, args.opts)


def test_model(model_file, multi_gpu_testing, opts=None):
    """Test a model."""
    # Clear memory before inference
    workspace.ResetWorkspace()
    # Run inference
    run_inference(
        model_file, multi_gpu_testing=multi_gpu_testing,
        check_expected_results=True,
    )

main()

