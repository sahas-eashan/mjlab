# Unitree D1

This D1 arm description was sourced from the public
`elijah-waichong-chan/hq-pcot` ROS 2 workspace, package `d1_description`.
The package metadata declares a BSD license.

The URDF is a SolidWorks export with mesh collision geometry and zero
effort/velocity limits. mjlab supplies conservative placeholder actuator
limits in `d1_constants.py`. The combined Go2+D1 asset uses simplified box
colliders on the gripper fingers for experimental manipulation while keeping
the detailed finger, wrist, and arm-link meshes visual-only.
