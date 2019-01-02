# Copyright (c) 2018, Zhewei Xu
# [xzhewei-at-gmail.com]
# Licensed under The MIT License [see LICENSE for details]

"""SCUT Pedestrian dataset evaluation interface."""

import logging
import os
import uuid
import shutil

import pydatatool as pdt

from detectron.core.config import cfg
from detectron.datasets.dataset_catalog import get_devkit_dir

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
        output_dir, 'scut_eval', cfg.METHOD_NAME
    )
    if use_salt:
        res_dir += '_{}'.format(str(uuid.uuid4()))
    
    _write_scut_results_files(json_dataset, all_boxes, res_dir)
    #_do_python_eval(json_dataset, salt, output_dir)
    if use_matlab:
        do_matlab_eval(
            json_dataset.name, 
            res_dir, 
            os.path.join(output_dir, 'scut_eval'),
            )
    if cleanup:
        pass
        # for filename in filenames:
        #     shutil.copy(filename, output_dir)
        #     os.remove(filename)
    return None

def evaluate_roadline(
    json_dataset,
    all_lines,
    output_dir
):
    import numpy as np
    import scipy.io as sio
    lines = np.argmax(all_lines, 1)
    mat_file = os.path.join(output_dir,'roadline.mat')
    sio.savemat(mat_file, {'roadline': lines})

def _write_scut_results_files(json_dataset, all_boxes, res_dir):
    print(
        'Writing bbox results to: {}'.format(os.path.abspath(res_dir)))
    classes = json_dataset.classes
    imgs = json_dataset.COCO.imgs.values()
    img_names = [os.path.splitext(img['file_name'])[0] for img in imgs]
    img_names.sort()
    pdt.scut.write_voc_results_file(
        all_boxes, img_names, res_dir, classes)
    
def do_matlab_eval(dname, res_dir, output_dir):
    import subprocess
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(get_devkit_dir(dname),'datatool')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format('matlab')
    cmd += '-r "dbstop if error;'
    cmd += 'startup;'
    cmd += 'scut_eval(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
        .format(
            os.path.abspath(res_dir), 
            os.path.abspath(output_dir), 
            cfg.METHOD_NAME)
    print('Running:\n{}'.format(cmd))
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    while p.poll() is None:  
        line = p.stdout.readline()  
        line = line.strip()  
        if line:  
            print('Subprogram output: [{}]'.format(line))  
    if p.returncode == 0:  
        print('Subprogram success')  
    else:  
        print('Subprogram failed') 


