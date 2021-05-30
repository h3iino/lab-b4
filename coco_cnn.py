from pycocotools.coco import COCO

anno_path = "coco/annotations/instances_train2014.json"
coco = COCO(anno_path)

# try
print(coco.getCatIds(catNms=["dog", "cat"]))
print(coco.getCatIds(supNms=["vehicle"]))
