"""Print all 19 joints of Panoptic GT in frame 150 to identify the correct indices."""

import json
import numpy as np

fpath = r"e:\Chaos\Projects\aimocap_re\data\panoptic\171204_pose3\hdPose3d_stage1_coco19\body3DScene_00000150.json"
with open(fpath) as f:
    d = json.load(f)

k = np.array(d["bodies"][0]["joints19"]).reshape(19, 4)

print("Index: [X, Y, Z, Score]")
print("-" * 30)
for i in range(19):
    print(f"Joint {i:2d}: [{k[i,0]:8.2f}, {k[i,1]:8.2f}, {k[i,2]:8.2f}] (score: {k[i,3]:.2f})")
