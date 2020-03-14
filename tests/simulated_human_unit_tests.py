import unittest
import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from chef_world_environment import *
from simulated_human import *


class SimulatedHumanUnitTests:
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
        true_chi_h = [[0, 1, 0, 0, 0, 0, 0, 0],
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
        human_behaviour = "rational"
        beta = 1
        gamma = 0.9
        c = 6
        e = 0.1
        self.env = ChefWorldEnvironment(robot_actions, human_actions, reward_space,
                                        initial_world_state, true_theta, true_chi_h, true_chi_r, human_behaviour, beta, gamma, c, e)
        self.simulated_human = SimulatedHuman(env)

    def test_simulateHumanAction(self):
        curr_world_state = [2, 1, 2]
        actual_robot_action = [1, 0, 0, 0, 0, 0, 0]

        self.simulated_human.pedagogy_constant = 0

        expected_intended_human_action = [0, 0, 1, 0, 0, 0, 0, 0]
        expected_actual_human_action = [0, 0, 1, 0, 0, 0, 0, 0]

        self.assertEqual(self.simulated_human.simulateHumanAction(
            curr_world_state, actual_robot_action)[0], expected_intended_human_action)
        self.assertEqual(self.simulated_human.simulateHumanAction(
            curr_world_state, actual_robot_action)[1], expected_actual_human_action)

        self.simulated_human.pedagogy_constant = 1

        expected_intended_human_action = [0, 1, 0, 0, 0, 0, 0, 0]
        expected_actual_human_action = [0, 1, 0, 0, 0, 0, 0, 0]

        self.assertEqual(self.simulated_human.simulateHumanAction(
            curr_world_state, actual_robot_action)[0], expected_intended_human_action)
        self.assertEqual(self.simulated_human.simulateHumanAction(
            curr_world_state, actual_robot_action)[1], expected_actual_human_action)


if __name__ == '__main__':
    unittest.main()
