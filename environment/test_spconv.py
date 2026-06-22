#!/usr/bin/env python3
import torch
import spconv.pytorch as spconv


def main():
    torch.manual_seed(0)
    indices = torch.cartesian_prod(
        torch.arange(1, device="cuda", dtype=torch.int32),
        torch.arange(4, device="cuda", dtype=torch.int32),
        torch.arange(4, device="cuda", dtype=torch.int32),
        torch.arange(4, device="cuda", dtype=torch.int32),
    )
    features = torch.randn((len(indices), 4), device="cuda", requires_grad=True)
    sparse = spconv.SparseConvTensor(features, indices, spatial_shape=[4, 4, 4], batch_size=1)
    layer = spconv.SubMConv3d(4, 8, kernel_size=3, padding=1, bias=False).cuda()
    output = layer(sparse)
    loss = output.features.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None and torch.isfinite(features.grad).all()
    print(
        f"spconv={spconv.__version__ if hasattr(spconv, '__version__') else 'ok'} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)} loss={loss.item():.6f} backward=ok"
    )


if __name__ == "__main__":
    main()
