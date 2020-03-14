import unittest
import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from human_action_node import *
from robot_action_node import *
from chef_world_environment import *


class RobotActionNodeUnitTest(unittest.TestCase):
    def setUp(self):
        robot_actions = [[1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]]
        human_actions = [[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1]]
        reward_space = [[[4, 2, 2], 2],
                        [[0, 6, 2], 1],
                        [[0, 0, 8], 1]]
        initial_world_state = [0, 0, 0]
        true_theta = 0
        true_chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]
        true_chi_r = [[1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]]
        capability_calibration_distance = "intersection"
        human_behaviour = "rational"
        beta = 1
        gamma = 0.9
        c = 6
        e = 0.1
        self.env = ChefWorldEnvironment(robot_actions, human_actions, reward_space,
                                        initial_world_state, true_theta, true_chi_h, true_chi_r, capability_calibration_distance, human_behaviour, beta, gamma, c, e)
        self.robot_action_node = RobotActionNode(self.env)

    def test_initChildren(self):
        self.assertTrue(self.robot_action_node.human_node_children == [
                        "empty"] * len(self.env.observation_space))

    def test_augmentedValue(self):
        self.robot_action_node.value = 2.5
        self.robot_action_node.visited = 2

        self.assertEqual(
            round(self.robot_action_node.augmented_value(4), 2), 4.50)

    def test_updateValue(self):
        self.robot_action_node.value = 2.5
        self.robot_action_node.visited = 2

        self.robot_action_node.update_value(3)

        self.assertEqual(round(self.robot_action_node.value, 2), 2.75)

    def test_updateVisited(self):
        self.robot_action_node.visited = 2

        self.robot_action_node.update_visited()

        self.assertEqual(self.robot_action_node.visited, 3)


if __name__ == '__main__':
    unittest.main()
