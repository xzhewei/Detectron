path = '/home/xzw/code/Detectron/output/scut/S1_G0-3-S1_e2e_frcnn_VGG16-C5_roadline/detectron-output/test/scut_test_1x_roadline/generalized_rcnn/'
pkl_path=path+'detections.pkl'

import pydatatool as pdt
detections = pdt.load_pkl(pkl_path)
roadline_cls_prob = detections['all_lines']
import numpy as np
lines = np.argmax(roadline_cls_prob,1)
import scipy.io as sio
sio.savemat(path+'roadline.mat',{'roadline':lines})