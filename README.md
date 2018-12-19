# Real Self-Driving Car :: System Integration

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a 
Real Self-Driving Car. A system integration project the involves taking multiple ROS nodes and creating
a complete system and working as a team. This project utilizes waypoint following and traffic light detection to navigate
environments. One of these environments being the Udacity Self-Driving Car in it's own test lot.

## Frameworks Used

There are a few frameworks/libraries used:


- ROS - provides scaffolding and messaging middleware for integration
- Tensorflow - Deep learning framework for running machine learning model
- Keras - High level machine learning library using `Tensorflow` as the backend
- Autoware Waypoint Follower 
- Dataspeed DBW - controls actuation on the car

# Light Detection

## Detection Approach

Our approach was to treat this as a classification problem. Are there red lights in the RGB capture
or not. However, we decided not to use binary classification, we used 4 classes to allow other 
sub-systems to make more intelligent decisions.

## Classes

- Red Light
- Yellow Light
- Green Light
- Unknown

## Model

We fine-tuned `MobileNet` to our dataset pulled from the `Keras Applications` providing our own top. 

To prevent over-fitting we used image flipping and L2 bias regularization. 

# Bridge

The provided web-socket bridge did not work very well when the camera was on. When converting the 
base64 image into a cv2 image it caused some blocking in the event handler. This processing was moved
and processed in a `ROS` node. However, on powerful systems this may not cause a problem.

# Waypoint Publishing

The future waypoints were published with desired speed which included when we had to stop when a red light
is detected. The `Autoware` follower had to be updated to always update to keep the car from diverging
from the line.

# Team

- Thomas Milas
- Yunying-Chen
- Tianzhi Yang
