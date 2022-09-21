import os

# COCO
with open(os.path.join(os.path.dirname(__file__), "data/coco_category.txt"), "r") as f:
    lines = f.readlines()
DETECTION_CATEGORIES = [line.rstrip() for line in lines]
COCO_CATEGORIES = DETECTION_CATEGORIES
# 40 category of interests
with open(os.path.join(os.path.dirname(__file__), "data/matterport_category.txt"), "r") as f:
    lines = f.readlines()
CATEGORIES = {}
CATEGORIES['mp3d'] = [line.rstrip() for line in lines]
CATEGORIES['gibson'] = DETECTION_CATEGORIES
with open(os.path.join(os.path.dirname(__file__), "data/cv_colors.txt"), "r") as f:
    lines = f.readlines()
STANDARD_COLORS = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/mp3d_trainset.txt"), "r") as f:
    lines = f.readlines()
MP3D_TRAIN_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/mp3d_valset.txt"), "r") as f:
    lines = f.readlines()
MP3D_VAL_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/gibson_trainset.txt"), "r") as f:
    lines = f.readlines()
GIBSON_TINY_TRAIN_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/gibson_testset.txt"), "r") as f:
    lines = f.readlines()
GIBSON_TINY_TEST_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/mp3d_catset.txt"), "r") as f:
    lines = f.readlines()
MP3D_CAT_TRAIN_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/hm3d_trainset.txt"), "r") as f:
    lines = f.readlines()
HM3D_TRAIN_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/hm3d_valset.txt"), "r") as f:
    lines = f.readlines()
HM3D_VAL_SCENE = [line.rstrip() for line in lines]

with open(os.path.join(os.path.dirname(__file__), "data/hm3d_minivalset.txt"), "r") as f:
    lines = f.readlines()
HM3D_MINIVAL_SCENE = [line.rstrip() for line in lines]
