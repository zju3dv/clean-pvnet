from pycocotools.coco import COCO

coco = COCO("data/LINEMOD/cat/train.train.json")

print(coco.loadAnns(0))