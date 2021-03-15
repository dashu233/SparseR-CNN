import torch
import numpy as np


def _meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def grid_points(featmap_size, stride=16, device='cuda'):
    feat_h, feat_w = featmap_size
    shift_x = torch.arange(0., feat_w, device=device) * stride
    shift_y = torch.arange(0., feat_h, device=device) * stride

    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)

    stride = shift_x.new_full((shift_xx.shape[0],), stride)
    shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
    all_points = shifts.to(device)

    return all_points


def valid_flags(featmap_size, valid_size, device='cuda'):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
    valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = _meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    return valid

def generate_all_points(featmap_sizes,img_metas,point_strides=[8, 16, 32, 64, 128]):
    num_imgs = len(img_metas)
    num_levels = len(featmap_sizes)

    # since feature map sizes of all images are the same, we only compute
    # points center for one time
    multi_level_points = []
    for i in range(num_levels):
        points = grid_points(
            featmap_sizes[i], point_strides[i])
        multi_level_points.append(points)
    points_list = [[point.clone() for point in multi_level_points] for _ in range(num_imgs)]
    # for each image, we compute valid flags of multi level grids
    valid_flag_list = []
    for img_id, img_meta in enumerate(img_metas):
        multi_level_flags = []
        for i in range(num_levels):
            point_stride = point_strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = img_meta['pad_shape'][:2]
            valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
            flags = valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w))
            multi_level_flags.append(flags)
        valid_flag_list.append(multi_level_flags)
    # print(valid_flag_list[0][0].size())
    return points_list, valid_flag_list