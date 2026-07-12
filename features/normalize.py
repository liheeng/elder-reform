import numpy as np

def normalize_keypoints(kpts, confs=None, target_bh=100.0):
    """以 hip_center 为原点，按 body_height 缩放关键点。
    
    Parameters
    ----------
    kpts : ndarray (17, 2)  COCO 17 关键点
    confs : ndarray (17,) or None
    target_bh : float  目标 body_height（默认 100.0）
    
    Returns
    -------
    kpts_norm : ndarray (17, 2)  归一化后的关键点
    confs : ndarray (17,)  原样返回（置信度不受归一化影响）
    or (None, None) if body_height < 5（人不可见）
    """
    kpts = np.asarray(kpts, dtype=np.float32)
    if kpts.shape != (17, 2):
        return None, None
    
    nose = kpts[0]
    la, ra = kpts[15], kpts[16]
    bh = float(np.linalg.norm(nose - (la + ra) / 2))
    
    if bh < 5.0:
        return None, None
    
    hip_center = (kpts[11] + kpts[12]) / 2.0
    scale = target_bh / bh
    kpts_norm = (kpts - hip_center) * scale
    
    if confs is not None:
        confs = np.asarray(confs, dtype=np.float32)
    
    return kpts_norm.astype(np.float32), confs
