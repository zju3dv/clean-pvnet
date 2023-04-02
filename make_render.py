# # built-in lin
# import os
# # third-party lib
# import cv2
# import numpy as np
# from tqdm import tqdm
# from pycocotools.coco import COCO
# # personal lib
# from lib.utils.linemod.opengl_renderer import OpenGLRenderer

# coco = COCO("data/linemod/cat/train.json")
# len = range(len(coco.getAnnIds()))

# ply = OpenGLRenderer("data/linemod/cat/cat.ply")

# render_dir = "render"

# for i in tqdm(len):
#     anno =  coco.loadAnns(i+1)[0]
#     img_info = coco.loadImgs(i+1)[0]

#     K = np.array(anno["K"])
#     pose = np.array(anno["pose"])
#     img_size = np.array([img_info["width"],img_info["height"]])

#     render_img =  ply.render(pose,K,img_size,render_type="rgb")

#     cv2.imwrite(os.path.join(render_dir,"{:0>6d}.jpg".format(i)),render_img)

from lib.utils.linemod.opengl_renderer import OpenGLRenderer
from tools.handle_custom_dataset import get_model_corners
import numpy as np

model = OpenGLRenderer("data/linemod/cat/cat.ply")

print(get_model_corners(model.model["pts"]))