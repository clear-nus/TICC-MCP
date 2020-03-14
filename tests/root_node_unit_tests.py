import unittest
import numpy as np
import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from robot_action_node import *
from root_node import *
from chef_world_environment import *


class RootNodeUnitTest(unittest.TestCase):
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

        world_state_0 = [0, 0, 0]
        world_state_1 = [0, 0, 1]
        theta = 0
        chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]]
        chi_r = [[1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]]

        augmented_state_0 = [world_state_0, theta, chi_h, chi_r]
        augmented_state_1 = [world_state_1, theta, chi_h, chi_r]

        self.env = ChefWorldEnvironment(robot_actions, human_actions, reward_space,
                                        initial_world_state, true_theta, true_chi_h, true_chi_r, capability_calibration_distance, human_behaviour, beta, gamma, c, e)
        self.belief = [augmented_state_0, augmented_state_1]
        self.root_node = RootNode(self.env, self.belief)

    def test_initChildren(self):
        self.assertTrue(self.root_node.robot_node_children == [
                        "empty"] * len(self.env.action_space))

    def test_optimalRobotAction(self):
        robot_child_0 = RobotActionNode(self.env)
        robot_child_1 = RobotActionNode(self.env)
        robot_child_2 = RobotActionNode(self.env)
        robot_child_0.value = 2
        robot_child_1.value = 3
        robot_child_2.value = 4
        robot_child_0.visited = 1
        robot_child_1.visited = 1
        robot_child_2.visited = 1
        self.root_node.robot_node_children = [
            robot_child_0, robot_child_1, robot_child_2, "empty", "empty", "empty", "empty"]

        self.assertTrue(
            self.root_node.optimal_robot_action(0) == [0, 0, 1, 0, 0, 0, 0])

    def test_sampleState(self):
        sampled_state = self.root_node.sample_state()

        self.assertTrue(sampled_state ==
                        self.belief[0] or sampled_state == self.belief[1])

    def test_updateVisited(self):
        pass

    def test_updateValue(self):
        pass

    def test_updateBelief(self):
        pass

    def test_getChildrenValues(self):
        robot_child_0 = RobotActionNode(self.env)
        robot_child_1 = RobotActionNode(self.env)
        robot_child_2 = RobotActionNode(self.env)
        robot_child_0.value = 2
        robot_child_1.value = 3
        robot_child_2.value = 4

        self.root_node.robot_node_children = [
            robot_child_0, robot_child_1, robot_child_2, "empty", "empty", "empty", "empty"]

        expected_children_values = [2, 3, 4, 0, 0, 0, 0]

        self.assertEqual(self.root_node.get_children_values(),
                         expected_children_values)


if __name__ == '__main__':
    unittest.main()
