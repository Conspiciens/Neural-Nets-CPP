# Road Segmentation For Cal Poly Pomona 
The goal of this project is for the automation project at Cal Poly Pomona that creates masks for the sidewalk currently. The scope of the project will eventually increase and more issues will be created, but for now just a mask for sidewalks/roads. 

# Currently working on 
Switching from pre-trained Mask RCNN to U-Net Neural Network, as the Mask RCNN is for instance segmentation (Used for detecting objects) and not semantic segmentation, which is what we are looking for


# Issues 
- [ ] https://github.com/Conspiciens/auto-car/issues/1

# Goals 
 - [ ] Develop a system for the Camera to detect sidewalks/roads
 - [ ] Create a model for U-Net using the sidewalk dataset 
 - [ ] Eventually develop a RCNN from scratch for LIDAR/Camera
 - [ ] Finetune the current RCNN using pytorch RCNN and truly understand the mathematics behind the current training
