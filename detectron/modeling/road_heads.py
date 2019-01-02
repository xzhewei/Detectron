
from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
import detectron.utils.blob as blob_utils
# ---------------------------------------------------------------------------- #
# Road line outputs and losses
# ---------------------------------------------------------------------------- #
def add_road_outputs(model, blob_in, dim_in, spatial_scale):
    dim_out = 1
    model.Conv(
        blob_in,
        'conv_road',
        dim_in,
        dim_out,
        kernel=1,
        pad=0,
        stride=1,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    model.Relu('conv_road', 'conv_road')
    # Road line regress
    # blob_out = model.Conv(
    #     'conv_road',
    #     'conv_road_reg',
    #     dim_in,
    #     1,
    #     kernel=1,
    #     pad=0,
    #     stride=1,
    #     weight_init=gauss_fill(0.01),
    #     bias_init=const_fill(0.0)
    # )
    # model.Relu('conv_road', 'conv_road')
    # TODO: upsampling the feature to image size
    blob_out = model.FC(
        'conv_road',
        'roadline_cls_score',
        dim_out*107*134,
        576, # TODO: change to fit the feature size
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('roadline_cls_score', 'roadline_cls_prob', engine='CUDNN')
    return blob_out

def add_road_losses(model):
    roadline_cls_prob, loss_roadline_cls = model.net.SoftmaxWithLoss(
        ['roadline_cls_score', 'roadline_int32'],
        ['roadline_cls_prob', 'loss_roadline_cls'],
        scale=model.GetLossScale()*0.01
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_roadline_cls])
    model.AddLosses('loss_roadline_cls')
    model.Accuracy(['roadline_cls_prob', 'roadline_int32'], 'accuracy_roadline')
    model.AddMetrics('accuracy_roadline')
    return loss_gradients

