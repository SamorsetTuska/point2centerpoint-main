from copy import deepcopy
from os import path as osp
import os

import numpy as np
import trimesh


def _write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def _write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox(list[ndarray] or ndarray): xyz pos of center and
            3 lengths (dx,dy,dz) and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename(str): Filename.
    """

    def heading2rotmat(heading_angle):
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        try:
            box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        except Exception:
            # import ipdb;
            # ipdb.set_trace()
            return None
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to obj file
    # trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')
    mesh_list.export(out_filename, file_type='obj')

    return


def show_result(points,
                gt_bboxes,
                pred_bboxes,
                out_dir,
                filename):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool): Visualize the results online. Defaults to False.
        snapshot (bool): Whether to save the online results. Defaults to False.
    """
    # result_path = osp.join(out_dir, filename)
    result_path = out_dir

    if not osp.exists(result_path):
        os.makedirs(result_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_bboxes is not None:
        # bottom center to gravity center
        # gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2
        # the positive direction for yaw in meshlab is clockwise
        # gt_bboxes[:, 6] *= -1
        _write_oriented_bbox(gt_bboxes,
                             osp.join(result_path, f'{filename}_gt.obj'))

    if pred_bboxes is not None:
        # bottom center to gravity center
        # pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
        # the positive direction for yaw in meshlab is clockwise
        # pred_bboxes[:, 6] *= -1
        _write_oriented_bbox(pred_bboxes,
                             osp.join(result_path, f'{filename}_pred.obj'))


def show_det_result_meshlab(data,
                            result,
                            out_dir,
                            score_thr=0.0):
    """Show 3D detection result by meshlab."""
    points = data['points'][..., 1:5].cpu().numpy()
    file_name = str(data['frame_id'][0]).zfill(6)

    pred_bboxes = result[0]['pred_boxes'].cpu().numpy()
    # pred_bboxes[:, 3:] = 0.1
    pred_scores = result[0]['pred_scores'].cpu().numpy()

    gt = data['gt_boxes'][0].cpu().numpy() if data.keys().__contains__('gt_boxes') else None

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]

    show_bboxes = deepcopy(pred_bboxes)

    show_result(
        points,
        gt,
        show_bboxes,
        out_dir,
        file_name)

    return file_name
