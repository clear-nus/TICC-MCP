import unittest
import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from chef_world_environment import *
from robot_action_node import *
from human_action_node import *


class ChefWorldEnvironmentUnitTest(unittest.TestCase):
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

    def test_augmentedTransition(self):
        curr_world_state = [0, 0, 0]
        next_world_state = [1, 0, 0]
        curr_theta = 0
        next_theta = 0
        curr_chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]
        next_chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]]
        curr_chi_r = [[1, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]]
        next_chi_r = [[2, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1]]

        intended_human_action = [0, 0, 0, 1, 0, 0, 0, 0]
        actual_human_action = [0, 0, 0, 1, 0, 0, 0, 0]
        intended_robot_action = [1, 0, 0, 0, 0, 0, 0]
        actual_robot_action = [1, 0, 0, 0, 0, 0, 0]

        curr_augmented_state = [curr_world_state,
                                curr_theta, curr_chi_h, curr_chi_r]

        actual_next_augmented_state = self.env.augmentedTransition(
            curr_augmented_state, intended_robot_action, actual_robot_action, intended_human_action, actual_human_action)
        expected_next_augmented_state = [
            next_world_state, next_theta, next_chi_h, next_chi_r]

        self.assertTrue(actual_next_augmented_state ==
                        expected_next_augmented_state)

    def test_worldStateTransition(self):
        curr_world_state = [0, 0, 0]
        actual_human_action = [0, 0, 1, 0, 0, 0, 0, 0]
        actual_robot_action = [1, 0, 0, 0, 0, 0, 0]

        actual_next_world_state = self.env.worldStateTransition(
            curr_world_state, actual_robot_action, actual_human_action)
        expected_next_world_state = [1, 1, 0]

        self.assertTrue(actual_next_world_state == expected_next_world_state)

    def test_robotCapabilityCalibrationScore(self):
        world_state = [0, 0, 0]
        theta = 1
        chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]]
        chi_r = [[1, 1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 2, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]]

        augmented_state = [world_state, theta, chi_h, chi_r]

        # Test KL divergence
        self.env.capability_calibration_distance = "kl-divergence"
        self.assertEqual(
            round(self.env.robotCapabilityCalibrationScore(augmented_state), 3), 0.306)

        # Test intersection distance
        self.env.capability_calibration_distance = "intersection"
        self.assertEqual(
            round(self.env.robotCapabilityCalibrationScore(augmented_state), 3), 0.444)

        # Test chi-square distance
        self.env.capability_calibration_distance = "chi-square"
        self.assertEqual(
            round(self.env.robotCapabilityCalibrationScore(augmented_state), 3), 0.370)

    def test_huamnCapabilityCalibrationScore(self):
        world_state = [0, 0, 0]
        theta = 1
        chi_h = [[1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 2, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 0, 0],
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

        augmented_state = [world_state, theta, chi_h, chi_r]

        # Test KL divergence
        self.env.capability_calibration_distance = "kl-divergence"
        self.assertEqual(
            round(self.env.humanCapabilityCalibrationScore(augmented_state), 3), 0.306)

        # Test intersection distance
        self.env.capability_calibration_distance = "intersection"
        self.assertEqual(
            round(self.env.humanCapabilityCalibrationScore(augmented_state), 3), 0.444)

        # Test chi-square distance
        self.env.capability_calibration_distance = "chi-square"
        self.assertEqual(
            round(self.env.humanCapabilityCalibrationScore(augmented_state), 3), 0.370)

    def test_reward(self):
        theta = 0
        chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]]
        chi_r = [[1, 1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 2, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]]

        world_state_none_fulfilled = [0, 0, 0]
        world_state_all_fulfilled = [4, 2, 2]
        world_state_partially_fulfilled = [3, 1, 1]
        world_state_overfulfilled = [5, 2, 2]

        augmented_state_none_fulfilled = [
            world_state_none_fulfilled, theta, chi_h, chi_r]
        augmented_state_all_fulfilled = [
            world_state_all_fulfilled, theta, chi_h, chi_r]
        augmented_state_partially_fulfilled = [
            world_state_partially_fulfilled, theta, chi_h, chi_r]
        augmented_state_overfulfilled = [
            world_state_overfulfilled, theta, chi_h, chi_r]
            
        self.assertEqual(round(self.env.reward(augmented_state_none_fulfilled), 3), 0.033)
        self.assertEqual(round(self.env.reward(augmented_state_all_fulfilled), 3), 2)
        self.assertEqual(round(self.env.reward(
            augmented_state_partially_fulfilled), 3), 0.115)
        self.assertEqual(round(self.env.reward(
            augmented_state_overfulfilled), 3), 0.089)

    def test_finalReward(self):
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

        world_state_none_fulfilled = [0, 0, 0]
        world_state_all_fulfilled_goal = [4, 2, 2]
        world_state_all_fulfilled_non_goal = [0, 0, 8]
        world_state_partially_fulfilled = [3, 1, 1]
        world_state_overfulfilled = [5, 2, 2]

        augmented_state_none_fulfilled = [
            world_state_none_fulfilled, theta, chi_h, chi_r]
        augmented_state_all_fulfilled_goal = [
            world_state_all_fulfilled_goal, theta, chi_h, chi_r]
        augmented_state_all_fulfilled_non_goal = [
            world_state_all_fulfilled_non_goal, theta, chi_h, chi_r]
        augmented_state_partially_fulfilled = [
            world_state_partially_fulfilled, theta, chi_h, chi_r]
        augmented_state_overfulfilled = [
            world_state_overfulfilled, theta, chi_h, chi_r]

        self.assertEqual(self.env.finalReward(
            augmented_state_none_fulfilled), 0)
        self.assertEqual(self.env.finalReward(
            augmented_state_all_fulfilled_goal), 2)
        self.assertEqual(round(self.env.finalReward(
            augmented_state_all_fulfilled_non_goal), 3), -0.320)
        self.assertEqual(round(self.env.finalReward(
            augmented_state_partially_fulfilled), 3), 1.167)
        self.assertEqual(round(self.env.finalReward(
            augmented_state_overfulfilled), 3), 1.280)

    def test_rolloutObservation(self):
        world_state = [3, 2, 1]
        theta = 0
        chi_h = [[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]]
        chi_r = [[0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]]
        augmented_state = [world_state, theta, chi_h, chi_r]

        intended_robot_action_towards_goal = [1, 0, 0, 0, 0, 0, 0]
        intended_robot_action_exceeding_goal = [0, 0, 1, 0, 0, 0, 0]
        intended_robot_action_failed = [0, 0, 0, 1, 0, 0, 0]
        intended_robot_action_noop = [0, 0, 0, 0, 0, 0, 1]

        expected_rollout_human_intended_action_towards_goal = [
            0, 0, 0, 0, 1, 0, 0, 0]
        expected_rollout_human_actual_action_towards_goal = [
            0, 0, 0, 0, 0, 1, 0, 0]

        expected_rollout_human_intended_action_exceeding_goal = [
            0, 0, 0, 0, 0, 0, 1, 0]
        expected_rollout_human_actual_action_exceeding_goal = [
            0, 0, 0, 0, 0, 0, 1, 0]

        expected_rollout_human_intended_action_failed_0 = [
            0, 0, 0, 0, 1, 0, 0, 0]
        expected_rollout_human_actual_action_failed_0 = [
            0, 0, 0, 0, 0, 1, 0, 0]
        expected_rollout_human_intended_action_failed_1 = [
            1, 0, 0, 0, 0, 0, 0, 0]
        expected_rollout_human_actual_action_failed_1 = [
            1, 0, 0, 0, 0, 0, 0, 0]

        expected_rollout_human_intended_action_noop_0 = [
            0, 0, 0, 0, 1, 0, 0, 0]
        expected_rollout_human_actual_action_noop_0 = [
            0, 0, 0, 0, 0, 1, 0, 0]
        expected_rollout_human_intended_action_noop_1 = [
            1, 0, 0, 0, 0, 0, 0, 0]
        expected_rollout_human_actual_action_noop_1 = [
            1, 0, 0, 0, 0, 0, 0, 0]

        self.assertTrue(self.env.rolloutObservation(
            augmented_state, intended_robot_action_towards_goal)[0] == expected_rollout_human_intended_action_towards_goal)
        self.assertTrue(self.env.rolloutObservation(
            augmented_state, intended_robot_action_towards_goal)[1] == expected_rollout_human_actual_action_towards_goal)

        self.assertTrue(self.env.rolloutObservation(
            augmented_state, intended_robot_action_exceeding_goal)[0] == expected_rollout_human_intended_action_exceeding_goal)
        self.assertTrue(self.env.rolloutObservation(
            augmented_state, intended_robot_action_exceeding_goal)[1] == expected_rollout_human_actual_action_exceeding_goal)

        actual_rollout_human_intended_action_failed, actual_rollout_human_actual_action_failed = self.env.rolloutObservation(
            augmented_state, intended_robot_action_failed)
        self.assertTrue(actual_rollout_human_intended_action_failed == expected_rollout_human_intended_action_failed_0 or
                        actual_rollout_human_intended_action_failed == expected_rollout_human_intended_action_failed_1)
        self.assertTrue(actual_rollout_human_actual_action_failed == expected_rollout_human_actual_action_failed_0 or
                        actual_rollout_human_actual_action_failed == expected_rollout_human_actual_action_failed_1)

        actual_rollout_human_intended_action_noop, actual_rollout_human_actual_action_noop = self.env.rolloutObservation(
            augmented_state, intended_robot_action_noop)
        self.assertTrue(actual_rollout_human_intended_action_noop == expected_rollout_human_intended_action_noop_0 or
                        actual_rollout_human_intended_action_noop == expected_rollout_human_intended_action_noop_1)
        self.assertTrue(actual_rollout_human_actual_action_noop == expected_rollout_human_actual_action_noop_0 or
                        actual_rollout_human_actual_action_noop == expected_rollout_human_actual_action_noop_1)

    def test_observation(self):
        world_state = [0, 0, 0]
        theta = 0
        chi_h = [[0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]]
        chi_r = [[0, 1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1]]
        augmented_state = [world_state, theta, chi_h, chi_r]

        robot_action_node = RobotActionNode(self.env)
        human_action_node_1 = HumanActionNode(self.env)
        human_action_node_2 = HumanActionNode(self.env)
        human_action_node_3 = HumanActionNode(self.env)
        robot_action_node.human_node_children = [
            "empty", human_action_node_1, human_action_node_2, "empty", human_action_node_3, "empty", "empty", "empty"]
        human_action_node_1.value_list = [2, 0, 0]
        human_action_node_2.value_list = [4, 0, 0]
        human_action_node_3.value_list = [5, 0, 0]

        expected_intended_human_action_not_search = [0, 0, 0, 0, 1, 0, 0, 0]
        expected_actual_human_action_not_search = [0, 0, 0, 0, 0, 1, 0, 0]
        expected_intended_human_action_search = [1, 0, 0, 0, 0, 0, 0, 0]
        expected_actual_human_action_search = [0, 1, 0, 0, 0, 0, 0, 0]

        self.assertTrue(self.env.observation(
            augmented_state, robot_action_node, False)[0] == expected_intended_human_action_not_search)
        self.assertTrue(self.env.observation(
            augmented_state, robot_action_node, False)[1] == expected_actual_human_action_not_search)
        self.assertTrue(self.env.observation(
            augmented_state, robot_action_node, True)[0] == expected_intended_human_action_not_search)
        self.assertTrue(self.env.observation(
            augmented_state, robot_action_node, True)[1] == expected_actual_human_action_not_search)

    def test_isTerminal(self):
        world_state_terminal_0 = [4, 2, 2]
        world_state_terminal_1 = [0, 6, 2]
        world_state_terminal_2 = [0, 0, 8]

        world_state_non_terminal_0 = [4, 2, 3]
        world_state_non_terminal_1 = [0, 0, 0]

        self.assertTrue(self.env.isTerminal(world_state_terminal_0))
        self.assertTrue(self.env.isTerminal(world_state_terminal_1))
        self.assertTrue(self.env.isTerminal(world_state_terminal_2))
        self.assertFalse(self.env.isTerminal(world_state_non_terminal_0))
        self.assertFalse(self.env.isTerminal(world_state_non_terminal_1))

    def test_robotActionDeviation(self):
        world_state = [0, 0, 0]
        theta = 0
        chi_h = [[0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]]
        self.env.true_chi_r = [[0, 1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1]]
        augmented_state = [world_state, theta, chi_h, self.env.true_chi_r]

        intended_robot_action = [1, 0, 0, 0, 0, 0, 0]
        expected_actual_robot_action = [0, 1, 0, 0, 0, 0, 0]

        self.assertEqual(self.env.robotActionDeviation(
            augmented_state, intended_robot_action), expected_actual_robot_action)


if __name__ == '__main__':
    unittest.main()
