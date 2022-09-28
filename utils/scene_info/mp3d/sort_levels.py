import json

dataset_name = "gibson"
data = json.load(open("./scene_info/{}/{}_scan_levels.json"))

for scan, floors in sorted(data.items()):
    new_floors = []
    prev = None
    for floor in floors["z_axes"]:
        floor = [str("%0.2f" % float(floor[0])), str("%0.2f" % float(floor[1]))]
        if prev is not None and prev != floor[0]:
            floor[0] = prev
        prev = floor[1]
        new_floors.append(floor)
    data[scan]["z_axes"] = new_floors

json.dump(
    data,
    open("./scene_info/scan_levels.json", "w"),
    indent=3,
    sort_keys=True,
)
