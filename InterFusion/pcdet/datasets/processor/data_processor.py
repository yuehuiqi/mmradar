from functools import partial

import numpy as np

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except ImportError:
    pass


class VoxelGeneratorWrapper:
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except ImportError:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except ImportError:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels,
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels,
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                return (
                    voxel_output["voxels"],
                    voxel_output["coordinates"],
                    voxel_output["num_points_per_voxel"],
                )
            return voxel_output

        assert tv is not None, "cumm.tensorview is required for spconv 2.x voxelization"
        tv_voxels, tv_coordinates, tv_num_points = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
        return tv_voxels.numpy(), tv_coordinates.numpy(), tv_num_points.numpy()


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.voxel_generator = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        if 'points' in data_dict:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]
        else:
            lidar_mask = common_utils.mask_points_by_range(data_dict['lidar_points'], self.point_cloud_range)
            radar_mask = common_utils.mask_points_by_range(data_dict['radar_points'], self.point_cloud_range)
            data_dict['lidar_points'] = data_dict['lidar_points'][lidar_mask]
            data_dict['radar_points'] = data_dict['radar_points'][radar_mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            if 'points' in data_dict:
                points = data_dict['points']
                shuffle_idx = np.random.permutation(points.shape[0])
                points = points[shuffle_idx]
                data_dict['points'] = points
            else:
                lidar_points = data_dict['lidar_points']
                radar_points = data_dict['radar_points']
                lidar_shuffle_idx = np.random.permutation(lidar_points.shape[0])
                radar_shuffle_idx = np.random.permutation(radar_points.shape[0])
                lidar_points = lidar_points[lidar_shuffle_idx]
                radar_points = radar_points[radar_shuffle_idx]
                data_dict['lidar_points'] = lidar_points
                data_dict['radar_points'] = radar_points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        # Initialize the parameters needed to convert the point cloud to a pillar.
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, config=config)

        if 'points' in data_dict:
            points = data_dict['points']
            if self.voxel_generator is None:
                self.voxel_generator = VoxelGeneratorWrapper(
                    vsize_xyz=config.VOXEL_SIZE,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features=points.shape[1],
                    max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                )
            # Generate the output of the pillar.
            voxel_output = self.voxel_generator.generate(points)
            # voxels: data of each generated pillar, represented by the dimension tensor as [M,32,4].
            # coordinates: the right-angle coordinates of each generated pillar,
            # represented by the dimension tensor as [M,3], where the third dimension is always 0, i.e. z=0.
            # num_points: the number of valid points in each generated pillar, expressed as [M,] in the dimension tensor,
            # which is filled to 0 when the value of the second dimension is less than 32.
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output

            if not data_dict['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
        else:
            # Generate output voxel_output for different modalities in sequence.
            lidar_points = data_dict['lidar_points']
            radar_points = data_dict['radar_points']
            if self.voxel_generator is None:
                self.voxel_generator = VoxelGeneratorWrapper(
                    vsize_xyz=config.VOXEL_SIZE,
                    coors_range_xyz=self.point_cloud_range,
                    num_point_features=lidar_points.shape[1],
                    max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                    max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                )
            lidar_voxel_output = self.voxel_generator.generate(lidar_points)
            radar_voxel_output = self.voxel_generator.generate(radar_points)
            if isinstance(lidar_voxel_output, dict):
                lidar_voxels, lidar_coordinates, lidar_num_points = \
                    lidar_voxel_output['voxels'], lidar_voxel_output['coordinates'], lidar_voxel_output['num_points_per_voxel']
            else:
                lidar_voxels, lidar_coordinates, lidar_num_points = lidar_voxel_output
            if isinstance(radar_voxel_output, dict):
                radar_voxels, radar_coordinates, radar_num_points = \
                    radar_voxel_output['voxels'], radar_voxel_output['coordinates'], radar_voxel_output['num_points_per_voxel']
            else:
                radar_voxels, radar_coordinates, radar_num_points = radar_voxel_output

            if not data_dict['use_lead_xyz']:
                lidar_voxels = lidar_voxels[..., 3:]  # remove xyz in voxels(N, 3)
                radar_voxels = radar_voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if 'points' in data_dict:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        else:
            data_dict['lidar_voxels'] = lidar_voxels
            data_dict['lidar_voxel_coords'] = lidar_coordinates
            data_dict['lidar_voxel_num_points'] = lidar_num_points
            data_dict['radar_voxels'] = radar_voxels
            data_dict['radar_voxel_coords'] = radar_coordinates
            data_dict['radar_voxel_num_points'] = radar_num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            # near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                if num_points >= len(points)*2:
                    extra_choice = np.random.choice(choice, num_points - len(points), replace=True)
                else:
                    extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
