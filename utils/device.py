"""设备与混合精度工具。

统一全项目的设备选择，并把对 CUDA 的硬编码改为按可用性回退，
以便在 macOS (MPS) / Linux+CUDA / 纯 CPU 上都能运行。

用法::

    from utils.device import get_device, autocast_context, make_grad_scaler

    DEVICE = get_device()
    with autocast_context(DEVICE):
        ...
"""

import contextlib
import torch

# 优先级: CUDA -> MPS(Apple Silicon GPU) -> CPU
_DEVICE_ORDER = ("cuda", "mps", "cpu")


def available_devices():
    """按优先级返回当前环境可用的设备名列表。"""
    out = []
    if torch.cuda.is_available():
        out.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        out.append("mps")
    out.append("cpu")
    return out


def resolve_device(spec="auto"):
    """把设备名解析为实际可用设备。

    Parameters
    ----------
    spec : str
        "auto" 自动选择（CUDA -> MPS -> CPU）；也可显式指定
        "cuda" / "cuda:0" / "mps" / "cpu"。

    Returns
    -------
    str : 可用的设备名；指定的设备不可用时回退到 "cpu" 并给出提示。
    """
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        return available_devices()[0]

    base = spec.split(":")[0]
    if base == "cuda" and not torch.cuda.is_available():
        print(f"[device] 本机无可用 CUDA，回退 CPU（请求: {spec}）")
        return "cpu"
    mps = getattr(torch.backends, "mps", None)
    if base == "mps" and not (mps is not None and mps.is_available()):
        print(f"[device] 本机无可用 MPS，回退 CPU（请求: {spec}）")
        return "cpu"
    return spec


def get_device(spec="auto"):
    """返回 torch.device，并打印实际选择结果。"""
    device = torch.device(resolve_device(spec))
    print(f"Device: {device}")
    return device


def is_cuda(device):
    """判断是否为 CUDA 设备（供混合精度开关使用）。"""
    try:
        return torch.device(device).type == "cuda"
    except (RuntimeError, TypeError):
        return False


def autocast_context(device, enabled=None):
    """返回适用于该设备的 autocast 上下文。

    仅 CUDA 启用混合精度：MPS 上 autocast 支持有限，CPU 上无收益，
    二者均直接用 ``nullcontext``，从而彻底避开 `torch.cuda.amp` 的警告。
    """
    if not is_cuda(device) or enabled is False:
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)


def make_grad_scaler(device):
    """返回与该设备匹配的 GradScaler（非 CUDA 时禁用缩放）。"""
    return torch.amp.GradScaler("cuda", enabled=is_cuda(device))
