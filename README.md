# CollectHabitatMap
It provides RGB/Semantic(Instance, Category, Region, Place) Top Down Maps for Habitat-Sim.<br>
The position on the map can be converted to the position in the habitat-sim scene (vice versa).

## Note
gibson, mp3d, and gibson_tiny datasets are collected using habitat 0.2.1.
hm3d dataset is collected using habitat 0.2.2.

## Example Output
* RGB Top Down Map <br>
<img src="sample/rgb.png" alt="HM3D" width="400"/>
* Instance Top Down Map <br>
<img src="sample/inst.png" alt="HM3DSemanticInstance" width="400"/>
* Category Top Down Map <br>
<img src="sample/cat.png" alt="HM3DSemanticInstance" width="400"/>
* Region Top Down Map <br>
<img src="sample/region.png" alt="HM3DSemanticRegion" width="400"/>
* Place Top Down Map <br>
<img src="sample/place.png" alt="HM3DSemanticPlace" width="400"/>

## Download
You can download the generated top down maps [Here](https://mysnu-my.sharepoint.com/:f:/g/personal/blackfoot_seoul_ac_kr/EvtGmk7nR2xIl6ddsOQXP4oBguEAJm5yt3WrMl8Cv4ZUaw?e=WNxYIs).

## Play with Map
After collecting the map, play with it using play_with_2dmap.py

### Usage
* You can use 'w/a/s/d' buttons to move an agent in the simulator.
* Double click the map and press 'm' to move to the clicked position.
* Press 'n' to move random point in the map.
* Press 'v' to see next house.

## Add Objects on Environments
After collecting the map, add objects using a map add_object_with_2dmap.py

### Usage
* Use "edit" command to manually add objects on the map.
* Otherwise, the code will automatically add objects on the environment.
* You can use 'w/a/s/d' buttons to move an agent in the simulator.
* Double click the map and press 'm' to move to the clicked position.
* Press 'b' to move random point in the map.
* Press 'n' to see next house.
* Press 'i' to add an object to the clicked position.
* Press 'z' to remove the recently added object.
* Press 'c' to clear all the objects.
* Press 's' to save the added objects.
* Press 'l' to load the added objects.
* Press 'q' to quit.
* Press 'h' to see the help.
* Press 'p' to see the added objects.
* Press 'p' to see the added objects.
* You can change the object rotation by pressing 'j' and 'k' buttons.
* You can change the object scale by pressing 'u' and 'i' buttons.
* You can change the object position by pressing 'o' and 'p' buttons.
