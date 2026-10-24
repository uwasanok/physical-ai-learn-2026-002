#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
set -u
mkdir -p /tmp/toilet_ur/share/ament_index/resource_index/packages /work/output/ur5e
touch /tmp/toilet_ur/share/ament_index/resource_index/packages/ur_description
ln -s /work/vendor/ur_description /tmp/toilet_ur/share/ur_description
export AMENT_PREFIX_PATH="/tmp/toilet_ur:${AMENT_PREFIX_PATH}"
xacro /work/vendor/ur_description/urdf/ur.urdf.xacro ur_type:=ur5e name:=toilet_ur5e > /work/output/ur5e/ur5e.urdf
