import torch
from easydict import EasyDict

from pcdet.models.backbones_3d.vfe.pfa_vfe import (
    DynamicGraphRadarPillarFeatureAttention,
    RadarPillarFeatureAttention,
)


def _model_config(name):
    return EasyDict(
        {
            "NAME": name,
            "USE_NORM": True,
            "WITH_DISTANCE": False,
            "USE_ABSLOTE_XYZ": True,
            "NUM_FILTERS": [32],
            "NUM_HEADS": 2,
            "MEAN_POOL_SCALE": 0.5,
            "FINAL_FUSION_GATE_INIT": 0.25,
            "INFERENCE_FUSION_SCALE": 1.0,
            "GRAPH": {
                "NUM_LAYERS": 2,
                "K": 3,
                "NEIGHBOR_MODE": "hybrid",
                "FEATURE_WEIGHT": 1.0,
                "POSITION_WEIGHT": 0.25,
                "POSITION_SCALE": 4.0,
                "QUERY_CHUNK_SIZE": 2,
                "USE_POSITION_DELTA": True,
                "USE_GLOBAL_CONTEXT": True,
                "RESIDUAL_SCALE": 0.5,
            },
            "AUX_ATTENTION": {
                "ENABLED": True,
                "MAX_POINTS": 5,
                "MOTION_FEATURE_INDEX": 4,
                "INTENSITY_FEATURE_INDEX": 3,
                "USE_STRUCTURE": True,
                "USE_RANGE": True,
                "GATE_SCALE": 0.5,
            },
        }
    )


def _batch():
    torch.manual_seed(7)
    num_voxels, max_points, num_features = 7, 5, 5
    voxels = torch.zeros(num_voxels, max_points, num_features)
    point_counts = torch.tensor([5, 3, 2, 1, 4, 2, 1], dtype=torch.int32)
    coordinates = torch.tensor(
        [
            [0, 0, 30, 40],
            [0, 0, 31, 41],
            [0, 0, 29, 43],
            [0, 0, 33, 45],
            [1, 0, 42, 70],
            [1, 0, 43, 72],
            [1, 0, 40, 74],
        ],
        dtype=torch.int32,
    )
    for index, count in enumerate(point_counts.tolist()):
        xyz = torch.randn(count, 3) * 0.15
        xyz[:, 0] += coordinates[index, 3].float() * 0.5 - 16.0
        xyz[:, 1] += coordinates[index, 2].float() * 0.5 - 20.0
        xyz[:, 2] += -2.0 + index * 0.2
        voxels[index, :count, :3] = xyz
        voxels[index, :count, 3] = torch.rand(count) * 4.0
        voxels[index, :count, 4] = torch.randn(count) * 0.8
    return {
        "voxels": voxels,
        "voxel_num_points": point_counts,
        "voxel_coords": coordinates,
        "batch_size": 2,
    }


def _build(model_class, name):
    return model_class(
        model_cfg=_model_config(name),
        num_point_features=5,
        voxel_size=[0.5, 0.5, 18.0],
        point_cloud_range=[-16.0, -20.0, -8.0, 64.0, 20.0, 8.0],
    )


def test_original_and_improved_models_keep_the_same_output_contract():
    batch = _batch()
    baseline = _build(RadarPillarFeatureAttention, "RadarPillarFeatureAttention")
    improved = _build(
        DynamicGraphRadarPillarFeatureAttention,
        "DynamicGraphRadarPillarFeatureAttention",
    )
    baseline.eval()
    improved.eval()

    baseline_output = baseline({key: value.clone() if torch.is_tensor(value) else value
                                for key, value in batch.items()})
    improved_output = improved({key: value.clone() if torch.is_tensor(value) else value
                                for key, value in batch.items()})

    assert baseline_output["pillar_features"].shape == (7, 32)
    assert improved_output["pillar_features"].shape == (7, 32)
    assert torch.isfinite(baseline_output["pillar_features"]).all()
    assert torch.isfinite(improved_output["pillar_features"]).all()


def test_extra_modules_do_not_change_baseline_initialization_stream():
    torch.manual_seed(19)
    baseline = _build(RadarPillarFeatureAttention, "RadarPillarFeatureAttention")
    baseline_rng_state = torch.random.get_rng_state()

    torch.manual_seed(19)
    improved = _build(
        DynamicGraphRadarPillarFeatureAttention,
        "DynamicGraphRadarPillarFeatureAttention",
    )
    improved_rng_state = torch.random.get_rng_state()

    assert torch.equal(baseline_rng_state, improved_rng_state)
    baseline.eval()
    improved.eval()
    baseline_output = baseline(_batch())["pillar_features"]
    improved_output = improved(_batch())["pillar_features"]
    torch.testing.assert_close(baseline_output, improved_output)


def test_graph_never_connects_different_batch_items():
    model = _build(
        DynamicGraphRadarPillarFeatureAttention,
        "DynamicGraphRadarPillarFeatureAttention",
    )
    model.eval()
    first_batch = _batch()
    second_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in first_batch.items()
    }
    second_batch["voxels"][4:, :, :3] += 25.0

    first_output = model(first_batch)["pillar_features"][:4]
    second_output = model(second_batch)["pillar_features"][:4]
    torch.testing.assert_close(first_output, second_output)


def test_improved_model_backpropagates_through_all_enabled_branches():
    model = _build(
        DynamicGraphRadarPillarFeatureAttention,
        "DynamicGraphRadarPillarFeatureAttention",
    )
    model.train()
    batch = _batch()
    batch["voxels"].requires_grad_(True)
    output = model(batch)["pillar_features"]
    output.square().mean().backward()

    assert batch["voxels"].grad is not None
    assert torch.isfinite(batch["voxels"].grad).all()
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)
