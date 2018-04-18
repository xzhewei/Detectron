# Copyright (c) 2018, Zhewei Xu
# [xzhewei-at-gmail.com]
# Licensed under The MIT License [see LICENSE for details]

"""Caltech Pedestrian dataset evaluation interface."""

import logging
import os
import uuid
import shutil

import pydatatool as pdt

from core.config import cfg
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import DEVKIT_DIR

logger = logging.getLogger(__name__)

def evaluate_boxes(
    json_dataset,
    all_boxes,
    output_dir,
    use_salt=False,
    cleanup=False,
    use_matlab=True
):
    res_dir = os.path.join(
        output_dir, 'caltech_eval', cfg.METHOD_NAME
    )
    if use_salt:
        res_dir += '_{}'.format(str(uuid.uuid4()))
    
    _write_caltech_results_files(json_dataset, all_boxes, res_dir)
    #_do_python_eval(json_dataset, salt, output_dir)
    if use_matlab:
        do_matlab_eval(
            json_dataset.name, 
            res_dir, 
            os.path.join(output_dir, 'caltech_eval'),
            )
    if cleanup:
        pass
        # for filename in filenames:
        #     shutil.copy(filename, output_dir)
        #     os.remove(filename)
    return None

def _write_caltech_results_files(json_dataset, all_boxes, res_dir):
    logger.info(
        'Writing bbox results to: {}'.format(os.path.abspath(res_dir)))
    pass
    classes = json_dataset.classes
    imgs = json_dataset.COCO.imgs.values()
    img_names = [os.path.splitext(img['file_name'])[0] for img in imgs]
    img_names.sort()
    pdt.caltech.write_voc_results_file(
        all_boxes, img_names, res_dir, classes)
    
def do_matlab_eval(dname, res_dir, output_dir):
    import subprocess
    logger.info('-----------------------------------------------------')
    logger.info('Computing results with the official MATLAB eval code.')
    logger.info('-----------------------------------------------------')
    path = os.path.join(DATASETS[dname][DEVKIT_DIR],'datatool')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format('matlab')
    cmd += '-r "dbstop if error;'
    cmd += 'startup;'
    cmd += 'caltech_eval(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
        .format(
            os.path.abspath(res_dir), 
            os.path.abspath(output_dir), 
            cfg.METHOD_NAME)
    logger.info('Running:\n{}'.format(cmd))
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    while p.poll() is None:  
        line = p.stdout.readline()  
        line = line.strip()  
        if line:  
            logger.info('Subprogram output: [{}]'.format(line))  
    if p.returncode == 0:  
        logger.info('Subprogram success')  
    else:  
        logger.info('Subprogram failed') 


