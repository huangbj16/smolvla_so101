"""F2: give ACT a learned per-camera identity embedding. Idempotent, asserts both edits land."""
import sys
from pathlib import Path
import lerobot.policies.act.modeling_act as m

f = Path(m.__file__)
src = f.read_text()
if "camera_id_embed" in src:
    print("already patched"); sys.exit(0)

A_OLD = """        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
"""
A_NEW = """        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
            # F2: a learned identity vector per camera. Upstream ACT gives every camera's token block
            # the SAME 2D sinusoidal positional embedding, so the model can only tell the streams apart
            # by appearance. Zero-init makes this exactly equivalent to upstream at step 0.
            self.camera_id_embed = nn.Embedding(len(self.config.image_features), config.dim_model)
            nn.init.zeros_(self.camera_id_embed.weight)
"""
B_OLD = """            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
"""
B_NEW = """            for cam_idx, img in enumerate(batch[OBS_IMAGES]):
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                # F2: tag the block with which camera it came from (broadcast over h, w)
                cam_features = cam_features + self.camera_id_embed.weight[cam_idx].view(1, -1, 1, 1)
"""
for old, new, name in ((A_OLD, A_NEW, "__init__"), (B_OLD, B_NEW, "forward")):
    assert src.count(old) == 1, f"anchor not found exactly once in {name} (lerobot version mismatch?)"
    src = src.replace(old, new)
f.write_text(src)
print(f"patched {f}")
