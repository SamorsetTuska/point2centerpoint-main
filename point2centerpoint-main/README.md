<img src="docs/open_mmlab.png" align="right" width="30%">

# Point2Center

## 网络结构

<img src="docs\point2center_pipeline.png" align="center" width="100%">

> 网络结构文件：/pcdet/models/dense_heads/point_center_head.py
>
> loss和匹配文件: /pcdet/models/dense_heads/center_loss.py

- 3D Backbone: VoxelResBackBone8x
- 2D Backbone: 一层卷积修改channel，一层3*3卷积
- neck：SharedConvResV1

## 使用方法

训练：

```python
python -m torch.distributed.launch --nproc_per_node=4 train.py --launcher pytorch --cfg_file cfgs/kitti_models/point2centerpoint.yaml --workers=4
```

测试：

```python
python -m torch.distributed.launch --nproc_per_node=4 test.py --launcher pytorch --cfg_file cfgs/kitti_models/point2centerpoint.yaml --ckpt /path/to/checkpoint --workers=4
```

