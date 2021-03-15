import torch
import torch.nn as nn

def points2bbox(pts, y_first=True,transform_method='moment'):
    """Converting the points set into bounding box.
    :param pts: the input points sets (fields), each points
        set (fields) is represented as 2n scalar.
    :param y_first: if y_fisrt=True, the point set is represented as
        [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
        represented as [x1, y1, x2, y2 ... xn, yn].
    :return: each points set is converting to a bbox [x1, y1, x2, y2].
    """
    pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
    pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]
    if transform_method == 'minmax':
        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                         dim=1)
    elif transform_method == 'partial_minmax':
        pts_y = pts_y[:, :4, ...]
        pts_x = pts_x[:, :4, ...]
        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                         dim=1)
    elif transform_method == 'moment':
        pts_y_mean = pts_y.mean(dim=1, keepdim=True)
        pts_x_mean = pts_x.mean(dim=1, keepdim=True)
        pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
        pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
        moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)
        moment_mul = 0.01
        moment_transfer = (moment_transfer * moment_mul) + (
                moment_transfer.detach() * (1 - moment_mul))
        moment_width_transfer = moment_transfer[0]
        moment_height_transfer = moment_transfer[1]
        half_width = pts_x_std * torch.exp(moment_width_transfer)
        half_height = pts_y_std * torch.exp(moment_height_transfer)
        bbox = torch.cat([
            pts_x_mean - half_width, pts_y_mean - half_height,
            pts_x_mean + half_width, pts_y_mean + half_height
        ],
            dim=1)
    elif transform_method == "exact_minmax":
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_reshape = pts_reshape[:, :2, ...]
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]
        bbox_left = pts_x[:, 0:1, ...]
        bbox_right = pts_x[:, 1:2, ...]
        bbox_up = pts_y[:, 0:1, ...]
        bbox_bottom = pts_y[:, 1:2, ...]
        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
    else:
        raise NotImplementedError
    return bbox

from functools import partial

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret

def offset_to_pts(center_list, pred_list,point_strides=[8, 16, 32, 64, 128],num_points=9):
    """Change from point offset to point coordinate."""
    pts_list = []
    for i_lvl in range(len(point_strides)):
        pts_lvl = []
        for i_img in range(len(center_list)):
            pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                1, num_points)
            pts_shift = pred_list[i_lvl][i_img]
            yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                -1, 2 * num_points)
            y_pts_shift = yx_pts_shift[..., 0::2]
            x_pts_shift = yx_pts_shift[..., 1::2]
            xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
            xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
            pts = xy_pts_shift * point_strides[i_lvl] + pts_center
            pts_lvl.append(pts)
        pts_lvl = torch.stack(pts_lvl, 0)
        pts_list.append(pts_lvl)
    return pts_list