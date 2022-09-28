import numpy as np;import os

object_categories = ['mug', 'bottle', 'pot', 'bowl', 'chair', 'table', 'clock', 'bag', 'sofa', 'laptop', 'bed', 'microwave', 'cabinet', 'bookshelf', 'stove', 'printer', 'pillow']

CATEGORIES = [
    'void',
     'wall', 'floor', 'chair',  'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa',
     'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs', 'ceiling', 'toilet', 'stool', 'towel',
     'mirror', 'tv_monitor', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting', 'beam', 'railing',
     'shelving', 'blinds', 'gym_equipment', 'seating', 'board', 'furniture', 'appliances', 'clothes', 'objects', 'misc'
]
glb_path_dir = "./data/objects"
files = np.sort([file for file in os.listdir(glb_path_dir) if file.split('.')[-1] == "glb"])
object_infos = {}
for category_name in object_categories:
    dic = {}
    category_files = [f for f in files if category_name in f]
    for cf in category_files:
        name = cf.split(".")[0]
        dic[name] = {'height': 0.0}
    object_infos[category_name] = dic

DETECT_CATEGORY = ['mug', 'bottle', 'pot', 'bowl', 'chair', 'table', 'clock', 'bag', 'sofa', 'laptop', 'bed', 'microwave', 'cabinet', 'bookshelf', 'stove', 'printer', 'pillow']