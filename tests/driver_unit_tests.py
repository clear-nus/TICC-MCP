import unittest
import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from driver import *
from chef_world_environment import *
from human_action_node import *
from simulated_human import *


class DriverUnitTest(unittest.TestCase):
    def setUp(self):
        robot_action_space = [[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]]
        human_action_space = [[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]]
        reward_space = [[[4, 2, 2], 1],
                        [[0, 4, 4], 1],
                        [[8, 0, 0], 1]]
        initial_world_state = [0, 0, 0]
        true_theta = 1
        true_chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
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
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r = [[1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1]]
        capability_calibration_distance = "intersection"
        human_behaviour = "rational"
        beta = 1
        gamma = 0.9
        c = 6
        e = 0.1
        epsilon = math.pow(0.9, 5)
        num_iter = 400000
        num_steps = 5
        initial_belief = []
        for i, _ in enumerate(reward_space):
            belief_theta = i
            initial_belief.append(
                [initial_world_state, belief_theta, belief_chi_h, belief_chi_r])

        self.env = ChefWorldEnvironment(robot_action_space, human_action_space, reward_space,
                                        initial_world_state, true_theta, true_chi_h, true_chi_r,
                                        capability_calibration_distance, human_behaviour, beta, gamma, c, e)
        root_node = RootNode(self.env, initial_belief)
        TICC_POMCP_solver = TICCPOMCPSolver(
            epsilon, self.env, root_node, num_iter, c)
        simulated_human = SimulatedHuman(self.env)

        self.driver = Driver(self.env, TICC_POMCP_solver,
                             num_steps, simulated_human)

    def test_invigorateBelief(self):
        parent_human_action_node = HumanActionNode(self.env)
        prev_world_state_0 = [0, 0, 0]
        prev_world_state_1 = [1, 1, 1]
        curr_world_state = [0, 0, 0]
        intended_robot_action = [0, 1, 0, 0, 0, 0, 0]
        actual_robot_action = [0, 1, 0, 0, 0, 0, 0]
        intended_human_action = [1, 0, 0, 0, 0, 0, 0, 0]
        actual_human_action = [0, 1, 0, 0, 0, 0, 0, 0]
        prev_chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]
        curr_chi_h = [[1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]
        prev_chi_r = [[1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]]
        curr_chi_r = [[1, 1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]]

        for i in range(len(self.env.reward_space)):
            parent_human_action_node.update_belief(
                [prev_world_state_0, i, prev_chi_h, prev_chi_r])
            parent_human_action_node.update_belief(
                [prev_world_state_1, i, prev_chi_h, prev_chi_r])

        expected_curr_human_action_node = HumanActionNode(self.env)
        for i in range(len(self.env.reward_space)):
            expected_curr_human_action_node.update_belief(
                [curr_world_state, i, curr_chi_h, curr_chi_r])

        actual_curr_human_action_node = HumanActionNode(self.env)
        self.driver.invigorate_belief(actual_curr_human_action_node, parent_human_action_node,
                                      intended_robot_action, actual_robot_action, intended_human_action, actual_human_action, self.env)

        self.assertTrue(actual_curr_human_action_node.belief ==
                        expected_curr_human_action_node.belief)

    def test_updateRootCapabilitiesBelief(self):
        belief_state_0 = [0, 0, 0]
        belief_state_1 = [1, 2, 3]
        belief_theta_0 = 0
        belief_theta_1 = 1
        belief_theta_2 = 2
        belief_chi_h_0 = [[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_h_1 = [[2, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_h_2 = [[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 2, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r_0 = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r_1 = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 2, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r_2 = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 2, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]]

        root_node = RootNode(
            self.env, [[belief_state_0, belief_theta_0, belief_chi_h_0, belief_chi_r_0]])
        curr_node = RootNode(
            self.env, [[belief_state_1, belief_theta_1, belief_chi_h_1, belief_chi_r_1],
                       [belief_state_1, belief_theta_1, belief_chi_h_2, belief_chi_r_2]])

        expected_updated_root_node = RootNode(
            self.env, [[belief_state_0, belief_theta_0, belief_chi_h_1, belief_chi_r_1],
                       [belief_state_0, belief_theta_0,
                           belief_chi_h_2, belief_chi_r_2],
                       [belief_state_0, belief_theta_1,
                           belief_chi_h_1, belief_chi_r_1],
                       [belief_state_0, belief_theta_1,
                           belief_chi_h_2, belief_chi_r_2],
                       [belief_state_0, belief_theta_2,
                           belief_chi_h_1, belief_chi_r_1],
                       [belief_state_0, belief_theta_2, belief_chi_h_2, belief_chi_r_2]])

        self.driver.updateRootCapabilitiesBelief(root_node, curr_node)

        self.assertEqual(root_node.belief, expected_updated_root_node.belief)

    def test_finalCapabilityCalibrationScores(self):
        sampled_beliefs = []
        belief_state_0 = [0, 0, 0]
        belief_state_1 = [1, 2, 3]
        belief_theta_0 = 0
        belief_theta_1 = 1
        belief_chi_h_0 = [[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_h_1 = [[1, 2, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 2, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 2, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r_0 = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r_1 = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 2, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]]
        sampled_beliefs = [[belief_state_0, belief_theta_0, belief_chi_h_0, belief_chi_r_0],
                           [belief_state_1, belief_theta_1, belief_chi_h_1, belief_chi_r_1]]
        human_action_node = HumanActionNode(self.env)
        human_action_node.belief = sampled_beliefs

        expected_final_robot_capability_score = 0
        expected_final_human_capability_score = 0
        for belief in sampled_beliefs:
            expected_final_robot_capability_score += self.env.robotCapabilityCalibrationScore(
                belief)
            expected_final_human_capability_score += self.env.humanCapabilityCalibrationScore(
                belief)
        expected_final_robot_capability_score /= len(sampled_beliefs)
        expected_final_human_capability_score /= len(sampled_beliefs)

        actual_final_robot_capability_score, actual_final_human_capability_score = self.driver.finalCapabilityCalibrationScores(
            human_action_node)
        self.assertEqual(actual_final_robot_capability_score,
                         expected_final_robot_capability_score)
        self.assertEqual(actual_final_human_capability_score,
                         expected_final_human_capability_score)

    def test_updateBeliefWorldState(self):
        human_action_node = HumanActionNode(self.env)
        self.env.world_state = [0, 0, 1]
        belief_world_state = [0, 0, 0]
        belief_theta_0 = 0
        belief_theta_1 = 1
        belief_chi_h_0 = [[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_h_1 = [[2, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r_0 = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r_1 = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 2, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]]

        human_action_node.update_belief(
            [belief_world_state, belief_theta_0, belief_chi_h_0, belief_chi_r_0])
        human_action_node.update_belief(
            [belief_world_state, belief_theta_1, belief_chi_h_1, belief_chi_r_1])

        expected_human_action_node = HumanActionNode(self.env)
        expected_human_action_node.update_belief(
            [self.env.world_state, belief_theta_0, belief_chi_h_0, belief_chi_r_0])
        expected_human_action_node.update_belief(
            [self.env.world_state, belief_theta_1, belief_chi_h_1, belief_chi_r_1])

        self.driver.updateBeliefWorldState(human_action_node, self.env)

        self.assertTrue(human_action_node.belief ==
                        expected_human_action_node.belief)

    def test_beliefRewardScore(self):
        human_action_node = HumanActionNode(self.env)
        belief_world_state = [0, 0, 0]
        belief_theta_0 = 0
        belief_theta_1 = 1
        belief_chi_h_0 = [[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_h_1 = [[2, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r_0 = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]]
        belief_chi_r_1 = [[1, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 2, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 1]]

        human_action_node.update_belief(
            [belief_world_state, belief_theta_0, belief_chi_h_0, belief_chi_r_0])
        human_action_node.update_belief(
            [belief_world_state, belief_theta_1, belief_chi_h_1, belief_chi_r_1])

        self.assertEqual(self.driver.beliefRewardScore(
            human_action_node.belief), 0.5)


if __name__ == '__main__':
    unittest.main()
