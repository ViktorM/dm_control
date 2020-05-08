# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Humanoid Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

_DEFAULT_TIME_LIMIT = 16.6
_CONTROL_TIMESTEP = 0.0166

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10


SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('humanoid2.xml'), common.ASSETS


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = HumanoidSimple(move_speed=_RUN_SPEED, pure_state=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add()
def run_pure_state(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = HumanoidSimple(move_speed=_RUN_SPEED, pure_state=True, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Walker domain."""

  def unscale(self, x, limits):
    delta = limits[1] - limits[0]
  #  print(limits)
  #  print(delta)
    delta[abs(delta) < 1e-6] = 1.0
    return (2.0 * x - limits[1] - limits[0]) / delta

  def torso_upright(self):
    """Returns projection from z-axes of torso to the z-axes of world."""
    return self.named.data.xmat['torso', 'zz']

  def torso_forward(self):
    """Returns projection from x-axes of torso to the x-axes of world."""
    return self.named.data.xmat['torso', 'xx']

  def head_height(self):
    """Returns the height of the torso."""
    return self.named.data.xpos['head', 'z']

  def center_of_mass_position(self):
    """Returns position of the center-of-mass."""
    return self.named.data.subtree_com['torso'].copy()

  def center_of_mass_velocity(self):
    """Returns the velocity of the center-of-mass."""
    return self.named.data.sensordata['torso_subtreelinvel'].copy()

  def torso_vertical_orientation(self):
    """Returns the z-projection of the torso orientation matrix."""
    return self.named.data.xmat['torso', ['zx', 'zy', 'zz']]

  def torso_forward_orientation(self):
    """Returns the x-projection of the torso orientation matrix."""
    return self.named.data.xmat['torso', ['xx', 'xy', 'xz']]

  def joint_angles(self, limits=None):
    """Returns the state without global orientation or position."""
    if limits is None:
      angles = self.data.qpos[7:].copy()
    else:
      angles = self.unscale(self.data.qpos[7:].copy(), limits)

    return angles  # Skip the 7 DoFs of the free root joint.

  def joint_velocities(self):
    """Returns the state without global orientation or position."""
    return self.data.qvel[6:].copy()  # Skip linear and angular velocities of the free root joint.

  def extremities(self):
    """Returns end effector positions in egocentric frame."""
    torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
    torso_pos = self.named.data.xpos['torso']
    positions = []
    for side in ('left_', 'right_'):
      for limb in ('hand', 'foot'):
        torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
        positions.append(torso_to_limb.dot(torso_frame))
    return np.hstack(positions)

  def contact_forces(self):
    """Returns feet contact forces."""

  #  for side in ('left_', 'right_'):
  #    for limb in ('foot'):
    
    pass


class HumanoidSimple(base.Task):
  """A humanoid task."""

  def __init__(self, move_speed, pure_state, random=None):
    """Initializes an instance of `Humanoid`.

    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      pure_state: A bool. Whether the observations consist of the pure MuJoCo
        state or includes some useful features thereof.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._move_speed = move_speed
    self._pure_state = pure_state
    self._terminate_at_height = 1.05
    self._joint_velocity_scale = 0.1
    self._joint_limits = np.zeros((2, 21))
    print("Termination heigth = ", self._terminate_at_height)
    super(HumanoidSimple, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `Physics`.

    """
    # Find a collision-free random initial configuration.
    '''penetrating = True
    while penetrating:
      randomizers.randomize_limited_and_rotational_joints(physics, self.random)
      # Check for collisions.
      physics.after_reset()
      penetrating = physics.data.ncon > 0'''

    random = np.random

    penetrating = True
    while penetrating:
      hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
      slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
      ball = mjbindings.enums.mjtJoint.mjJNT_BALL
      free = mjbindings.enums.mjtJoint.mjJNT_FREE

      qpos = physics.named.data.qpos

      limited_joint_id = 0
      for joint_id in range(physics.model.njnt):
        joint_name = physics.model.id2name(joint_id, 'joint')
        joint_type = physics.model.jnt_type[joint_id]
        is_limited = physics.model.jnt_limited[joint_id]
        range_min, range_max = physics.model.jnt_range[joint_id]

        if is_limited:
          if joint_type == hinge or joint_type == slide:
            self._joint_limits[0][limited_joint_id] = range_min
            self._joint_limits[1][limited_joint_id] = range_max
          #  print("Joint ", limited_joint_id)
          #  print("Range ", range_min, range_max)
            limited_joint_id += 1

            delta_max = range_max - qpos[joint_name]
            delta_min = range_min - qpos[joint_name]

            qpos[joint_name] = qpos[joint_name] + 0.3 * random.uniform(delta_min, delta_max)

      # Check for collisions.
      physics.after_reset()
      penetrating = physics.data.ncon > 0

    self._failure_termination = False
    super(HumanoidSimple, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns either the pure state or a set of egocentric features."""
    obs = collections.OrderedDict()
    if self._pure_state:
      obs['position'] = physics.position()
      obs['velocity'] = physics.velocity()
    else:
      obs['joint_angles'] = physics.joint_angles(self._joint_limits)
      obs['head_height'] = physics.head_height()
      obs['extremities'] = physics.extremities()
      obs['torso_vertical'] = physics.torso_vertical_orientation()
      obs['com_velocity'] = physics.center_of_mass_velocity()
      obs['velocity'] = self._joint_velocity_scale * physics.velocity()
      obs['prev_actions'] = physics.control()
    return obs

  def after_step(self, physics, random_state=None):
    self._failure_termination = False

  #  print("Failure termination = ", self._failure_termination)
  #  print("Head height = ", physics.head_height())

    if physics.head_height() < self._terminate_at_height:
      self._failure_termination = True

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    standing = rewards.tolerance(physics.head_height(),
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/4)
    upright = rewards.tolerance(physics.torso_upright(),
                                bounds=(0.9, float('inf')), sigmoid='linear',
                                margin=1.9, value_at_margin=0)
    stand_reward = standing * upright
    small_control = rewards.tolerance(physics.control(), margin=1,
                                      value_at_margin=0,
                                      sigmoid='quadratic').mean()
    small_control = (4 + small_control) / 5
    if self._move_speed == 0:
      horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
      dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
      return small_control * stand_reward * dont_move
    else:
      com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
      '''move = rewards.tolerance(com_velocity,
                               bounds=(self._move_speed, float('inf')),
                               margin=self._move_speed, value_at_margin=0,
                               sigmoid='linear')'''
      
      move = physics.center_of_mass_velocity()[0] * physics.torso_forward()

      # get number joint at limits
      joint_angles_norm = np.abs(physics.joint_angles(self._joint_limits)) - 0.98
      joint_angles_norm[joint_angles_norm < 0.0] = 0.0

      joint_angles_norm = joint_angles_norm / (1.0 - 0.98)
      joint_angles_norm[joint_angles_norm > 1.0] = 1.0
      #print("Joint angles norm2", joint_angles_norm)

      joints_at_limit_cost = 0.15 * np.sum(joint_angles_norm)
      #print ("Joints at limits cost", joints_at_limit_cost)

      electricity_cost = 0.005 * np.sum(np.abs(physics.control() * physics.joint_velocities()))
    #  print ("Electricity cost", electricity_cost)
    #  print ("Joint velocities", physics.joint_velocities())

      #move = com_velocity * physics.torso_forward()
      return move + 2.0 + 0.1 * upright - electricity_cost - joints_at_limit_cost

  def should_terminate_episode(self, physics):
  #  print("Failure termination2 = ", self._failure_termination)
    return self._failure_termination

  def get_discount(self, physics):
    if self._failure_termination:
      return 0.
    else:
      return 1.
