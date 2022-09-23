# CollectHabitatMap
It provides RGB/Semantic(Instance, Region, Place) Top Down Map for Habitat-Sim.
The position on the map can be converted to the position in the habitat-sim scene (vice versa).

## Example Output
* RGB Top Down Map <br>
<img src="sample/hm3d.png" alt="HM3D" width="200"/>
* Instance Top Down Map <br>
<img src="sample/hm3d_semantic_inst.png" alt="HM3DSemanticInstance" width="200"/>
* Region Top Down Map <br>
<img src="sample/hm3d_semantic_region.png" alt="HM3DSemanticRegion" width="200"/>
* Place Top Down Map <br>
<img src="sample/hm3d_semantic_place.png" alt="HM3DSemanticPlace" width="200"/>

## Download
You can download the generated top down maps [Here]().

## Play with Map
After collecting the map, play with it using play_with_2dmap.py

### Usage
* You can use 'w/a/s/d' buttons to move an agent in the simulator.
* Double click the map and press 'm' to move to the clicked position.
* Press 'n' to move random point in the map.
* Press 'v' to see next house.