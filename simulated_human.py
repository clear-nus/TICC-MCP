import random as rand
import operator

from environment import *


class SimulatedHuman:
    """
    The simulated human.
    """

    def __init__(self, env, pedagogy_constant=0.75, decay=0.9):
        """
        Initializes an instance of simulated human.

        :param env: the environment
        :type: Environment
        :param pedagogy_constant: the chance of human demonstrating incapable action
        :type pedagogy_constant: float
        :param decay: the decay rate of chance of human demonstrating incapable action
        :type decay: float
        """
        self.env = env
        self.pedagogy_constant = pedagogy_constant
        self.decay = decay

    def simulateHumanAction(self, world_state, actual_robot_action):
        """
        Simulates actual human action given the actual robot action.

        :param world_state: the current world state
        :type world_state: list
        :param actual_robot_action: the current actual robot action
        :type actual_robot_action: list representing one hot vector of actual robot action

        :return: rollout intended human action, rollout actual human action
        :rtype: lists representing one hot vector of intended and actual human actions
        """
        goal_world_state = self.env.reward_space[self.env.true_theta][0]
        goal_difference_before_robot_action = list(
            map(operator.sub, goal_world_state, world_state))
        goal_difference = list(
            map(operator.sub, goal_difference_before_robot_action, actual_robot_action[:-1:2]))

        actual_robot_action_index = actual_robot_action.index(1)

        # Determines if negative signal is to be given based on robot action
        if actual_robot_action_index % 2 == 0 and \
                actual_robot_action_index // 2 in range(len(goal_world_state)) and \
                goal_difference_before_robot_action[actual_robot_action_index // 2] <= 0:
            # Negative signal if exceeded goal
            intended_human_action_index = len(self.env.observation_space) - 2

        else:  # Goal oriented
            # If everything is satisfied, human does nothing (no-op)
            intended_human_action_index = len(self.env.observation_space) - 1

            possible_intended_human_action_indices = []
            possible_intended_human_action_weights = []

            if rand.random() < self.pedagogy_constant:  # Demonstrate pedagogy behaviour (incapable action)
                for i, action_chi_h in enumerate(self.env.true_chi_h[:-2:2]):
                    if action_chi_h[i * 2 + 1] > 0:  # Non perfect capable action
                        possible_intended_human_action_indices.append(
                            i * 2 + 1)  # Appends possible action
                        possible_intended_human_action_weights.append(
                            action_chi_h[i * 2 + 1] / sum(action_chi_h[i * 2:i * 2 + 2]))  # Appends possible action weight
            else:  # Human will try to work towards goal no matter what
                for i, diff in enumerate(goal_difference):
                    if diff > 0:
                        possible_intended_human_action_indices.append(
                            i * 2)  # Appends possible action
                        possible_intended_human_action_weights.append(
                            self.env.true_chi_h[i * 2][i * 2] / sum(self.env.true_chi_h[i * 2][i * 2:i * 2 + 2]))  # Appends possible action weight

            if len(possible_intended_human_action_indices) > 0:
                # Chooses intended human action based on weights (success probability)
                action_weights_normalizer = sum(
                    possible_intended_human_action_weights)
                if action_weights_normalizer > 0:
                    intended_action_probabilities = list(map(
                        lambda weight: weight / action_weights_normalizer, possible_intended_human_action_weights))
                    intended_human_action_index = rand.choices(
                        possible_intended_human_action_indices, weights=intended_action_probabilities)[0]

        intended_human_action = self.env.observation_space[intended_human_action_index]

        intended_action_chi_h = self.env.true_chi_h[intended_human_action_index]
        chi_h_normalizer = sum(intended_action_chi_h)
        probabilities = list(map(
            lambda chi_h: chi_h / chi_h_normalizer, intended_action_chi_h))
        actual_human_action_index = rand.choices(
            range(len(self.env.observation_space)), weights=probabilities)[0]
        actual_human_action = self.env.observation_space[actual_human_action_index]

        self.pedagogy_constant *= self.decay

        return intended_human_action, actual_human_action
