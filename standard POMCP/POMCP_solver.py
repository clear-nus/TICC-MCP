import math
import random as rand

from environment import *
from root_node import *
from human_action_node import *
from robot_action_node import *


class POMCPSolver:
    def __init__(self, epsilon, env, root_action_node, num_iter, c):
        """
        Initializes an instance of POMCP solver.

        :param epsilon: tolerance factor to terminate rollout
        :type epsilon: float
        :param env: environment in which the robot and human operate
        :type env: Environment (for now)
        :param root_action_node: action node/history from which search function is called
        :type root_action_node: RootNode or HumanActionNode
        :param num_iter: the number of trajectories/simulations in each search
        :type num_iter: integer
        :param c: exploration constant
        :type c: float
        """
        self.epsilon = epsilon
        self.env = env
        self.root_action_node = root_action_node
        self.num_iter = num_iter
        self.c = c

    def search(self):
        """
        Samples num_iter trajectories and carries out the search process.

        :param belief_chi_h: the current robot's 0 order ToM belief of human capability
        :type belief_chi_h: list representing chi experience counts of human actions
        :param belief_chi_r: the current robot's 1st order ToM belief of human's belief of it's own capability
        :type belief_chi_r: list representing chi experience counts of robot actions

        :return: optimal robot action
        :rtype: list representing one hot vector of robot action
        """
        for _ in range(self.num_iter):
            sampled_augmented_state = self.root_action_node.sample_state()
            self.simulate(sampled_augmented_state, self.root_action_node, 0)

        return self.root_action_node.optimal_robot_action(0)

    def rollout(self, augmented_state, intended_robot_action, actual_robot_action, action_node, depth):
        """
        Calls the rollout function and adds new robot and human nodes created by the rollout to search tree.

        :param augmented_state: the starting rollout state, before robot and human action
        :type augmented_state: list of world_state, theta and chi
        :param intended_robot_action: the starting intended robot action  (original optimal action)
        :type intended_robot_action: list representing one hot vector of intended robot action
        :param actual_robot_action: the starting actual robot action
        :type actual_robot_action: list representing one hot vector of actual robot action
        :param action_node: the action node/history from which the rollout starts
        :param depth: the current depth in the search tree
        :type depth: integer

        :return: rollout value
        :rtype: float
        """
        intended_human_action, actual_human_action = self.env.rolloutObservation(
            augmented_state, intended_robot_action)

        value = self.rollout_helper(
            augmented_state, intended_robot_action, actual_robot_action, intended_human_action, actual_human_action, depth)
        next_augmented_state = self.env.augmentedTransition(
            augmented_state, intended_robot_action, actual_robot_action, intended_human_action, actual_human_action)

        # Creates new robot and human nodes
        new_robot_action_child = RobotActionNode(self.env)
        new_robot_action_child.update_visited()
        new_robot_action_child.update_value(value)

        new_human_action_child = HumanActionNode(self.env)
        new_human_action_child.update_belief(next_augmented_state)
        new_human_action_child.update_visited(augmented_state[1])
        new_human_action_child.update_value(value, augmented_state[1])

        # Adds created new nodes to the tree
        new_robot_action_child.human_node_children[self.env.observation_space.index(
            intended_human_action)] = new_human_action_child
        action_node.robot_node_children[self.env.action_space.index(
            intended_robot_action)] = new_robot_action_child

        return value

    def rollout_helper(self, augmented_state, intended_robot_action, actual_robot_action, intended_human_action, actual_human_action, depth):
        """
        Carries out recursive rollout process.

        :param augmented_state: the starting rollout state, before robot and human action
        :type augmented_state: list of world_state, theta, chi_h and chi_r
        :param intended_robot_action: the current intended robot action
        :type intended_robot_action: list representing one hot vector of intended robot action
        :param actual_robot_action: the current actual robot action
        :type actual_robot_action: list representing one hot vector of actual robot action
        :param intended_human_action: the current intended human action
        :type intended_human_action: list representing one hot vector of intended human action
        :param actual_human_action: the current actual human action 
        :type actual_human_action: list representing one hot vector of actual human action
        :param depth: the current depth in the search tree
        :type depth: integer

        :return: rollout value
        :rtype: float
        """
        # Returns 0 upon reaching maximum depth
        if math.pow(self.env.gamma, depth) < self.epsilon:
            return 0

        # Returns actual reward upon reaching terminal state
        if self.env.isTerminal(augmented_state[0]):
            return self.env.reward(augmented_state)

        # Generates next state and actions
        next_augmented_state = self.env.augmentedTransition(
            augmented_state, intended_robot_action, actual_robot_action, intended_human_action, actual_human_action)
        next_intended_robot_action = rand.choice(self.env.action_space)
        next_actual_robot_action = next_intended_robot_action
        next_intended_human_action, next_actual_human_action = self.env.rolloutObservation(
            next_augmented_state, next_intended_robot_action)

        return self.env.reward(augmented_state, actual_human_action) + self.env.gamma * \
            self.rollout_helper(next_augmented_state, next_intended_robot_action,
                                next_actual_robot_action, next_intended_human_action,
                                next_actual_human_action, depth + 1)

    def simulate(self, augmented_state, action_node, depth):
        """
        Simulates a trajectory from the start state down the search tree, by picking the 
        optimal action at each point in the tree and simulating observations/ human actions. 
        Incrementally builds the search tree and updates the statistics of visited nodes. 
        Returns the value achieved from the simulation.

        :param augmented_state: the starting state
        :param action_node: the current history/action node in the search tree
        :param depth: the current depth in the search tree

        :return: value from the current simulation
        """
        # Returns 0 upon reaching maximum depth
        if math.pow(self.env.gamma, depth) < self.epsilon:
            return 0

        # Update belief
        action_node.update_belief(augmented_state)

        # Returns actual reward upon reaching terminal state
        if self.env.isTerminal(augmented_state[0]):
            terminal_value = self.env.reward(augmented_state)
            return terminal_value

        # Finds optimal robot action
        intended_robot_action = action_node.optimal_robot_action(self.c)
        actual_robot_action = intended_robot_action
        robot_action_node = action_node.robot_node_children[self.env.action_space.index(
            intended_robot_action)]

        # Returns rollout value if next robot action node is not in tree
        if robot_action_node == "empty":
            rollout_value = self.rollout(
                augmented_state, intended_robot_action, actual_robot_action, action_node, depth)
            return rollout_value

        # Simulates intended and actual human action
        intended_human_action, actual_human_action = self.env.observation(
            augmented_state, robot_action_node, True)
        next_augmented_state = self.env.augmentedTransition(
            augmented_state, intended_robot_action, actual_robot_action, intended_human_action, actual_human_action)

        next_action_node = robot_action_node.human_node_children[self.env.observation_space.index(
            intended_human_action)]

        # Creates new node if next human action node is not in tree
        if next_action_node == "empty":
            new_human_action_node = HumanActionNode(self.env)
            next_action_node = robot_action_node.human_node_children[self.env.observation_space.index(
                intended_human_action)] = new_human_action_node

        # Recursion
        value = self.env.reward(augmented_state, actual_human_action) + self.env.gamma * \
            self.simulate(next_augmented_state, next_action_node, depth + 1)

        # Backups/updates statistics
        robot_action_node.update_visited()
        robot_action_node.update_value(value)

        next_action_node.update_visited(augmented_state[1])
        next_action_node.update_value(value, augmented_state[1])

        return value
