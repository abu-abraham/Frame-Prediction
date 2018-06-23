### Detecting Objects in a Frame

This repository contains the code for detecting object in a frame. This is an extention built on top of [Single Shot MultiBox Object Detector](https://github.com/amdegroot/ssd.pytorch). 

As a sub-module in the frame prediction project, the functionality of this module is to detect objects, its position in the frame, and also features such as the color of the object. The frames of humans recognized is to be passed to the action recognition module, to recognize actions. Together with the action recognition, the objective is to get results such as => [{object: Person, Action: Running, color: red, xPos: 34, yPos: 43},{object: Ball, Action: none, color: white, xPos: 34, yPos: 43}]


> To Run : demo > object_detector.py
> TODO: https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
