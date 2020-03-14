import random as rand
import math
import operator
import copy


class Environment:
    """
    The environment.
    """

    def __init__(self, robot_action_space, human_action_space, reward_space, initial_world_state, true_theta, true_chi_h, true_chi_r, capability_calibration_distance, human_behaviour, beta, gamma, c, e, capability_calibration_constant=0.3, capability_calibration_decay=1):
        """
        Initializes an instance of the environment.

        :param robot_action_space: the list of actions available to the robot
        :type robot_action_space: list of lists
        :param human_action_space: the list of actions available to the human/ possible obervations
        :type human_action_space: list of lists
        :param reward_space: the list of possible goal world states (reward parameters) and associated rewards
        :type reward_space: list of (goal_world_state, reward)
        :param initial_world_state: the initial world state of the environment
        :type initial_world_state: list of number ingredients collected so far
        :param true_theta: the true (hidden) reward parameter of the human
        :type true_theta: index of actual reward function
        :param true_chi_h: the true (hidden) capability parameters of the human
        :type true_chi_h: 2D list representing intended-actual action counts, list[intended][actual]
        :param true_chi_r: the true (hidden) capability parameters of the robot
        :type true_chi_r: 2D list representing intended-actual action counts, list[intended][actual]
        :param capability_calibration_distance: capability calibration distance measure
        :type capability_calibration_distance: string, "kl-divergence"/"intersection"/"chi-square"
        :param human_behaviour: simulated human behaviour
        :type human_behaviour: string, "rational"/"e-greedy"/"boltzmann"
        :param beta: Boltzmann rationality parameter
        :type beta: float, (0, 1)
        :param gamma: discount factor
        :type gamma: float, (0, 1)
        :param c: exploration constant
        :type c: float
        :param e: epsilon greedy factor
        :type e: float, (0, 1)
        :param capability_calibration_constant: weight for capabiilty calibration score
        :type capability_calibration_constant: float
        :param capability_calibration_decay: decay for capabiilty calibration constant
        :type capability_calibration_decay: float
        """
        self.action_space = robot_action_space
        self.observation_space = human_action_space
        self.reward_space = reward_space
        self.world_state = initial_world_state  # True world state of the game
        self.true_theta = true_theta
        self.true_chi_h = true_chi_h
        self.true_chi_r = true_chi_r
        self.capability_calibration_distance = capability_calibration_distance
        self.human_behaviour = human_behaviour
        self.beta = beta
        self.gamma = gamma
        self.c = c
        self.e = e
        self.capability_calibration_constant = capability_calibration_constant
        self.capability_calibration_decay = capability_calibration_decay

    def augmentedTransition(self, current_augmented_state, intended_robot_action, actual_robot_action, intended_human_action, actual_human_action):
        """
        Returns the next augmented state of the game after the robot and human acts.
        Theta remains constant. Used for tree search.

        :param current_augmented_state: the current augmented state
        :type current_augmented_state: list of world_state, theta, chi_h and chi_r
        :param intended_robot_action: the current intended robot action
        :type intended_robot_action: list representing one hot vector of intended robot action
        :param actual_robot_action: the current actual robot action
        :type actual_robot_action: list representing one hot vector of actual robot action
        :param intended_human_action: the current intended human action
        :type intended_human_action: list representing one hot vector of intended human action
        :param actual_human_action: the current actual human action 
        :type actual_human_action: list representing one hot vector of actual human action

        :return: the next augmented state
        :rtype: list of world_state, theta, chi_h and chi_r
        """
        aggregated_action = list(
            map(operator.add, actual_human_action[:-2:2], actual_robot_action[:-1:2]))
        intended_human_action_index = intended_human_action.index(1)
        actual_human_action_index = actual_human_action.index(1)
        intended_robot_action_index = intended_robot_action.index(1)
        actual_robot_action_index = actual_robot_action.index(1)

        next_world_state = list(
            map(operator.add, current_augmented_state[0], aggregated_action))

        # Updates chi_h
        # Intended human failure action
        if intended_human_action_index < len(self.observation_space) - 2 and intended_human_action_index % 2 != 0:
            next_chi_h = copy.deepcopy(current_augmented_state[2])
            next_chi_h[intended_human_action_index - 1][actual_human_action_index] += 1
        else:
            next_chi_h = copy.deepcopy(current_augmented_state[2])
            next_chi_h[intended_human_action_index][actual_human_action_index] += 1

        # Updates chi_r
        # Intended robot failure action
        if intended_robot_action_index < len(self.action_space) - 1 and intended_robot_action_index % 2 != 0:
            next_chi_r = copy.deepcopy(current_augmented_state[3])
            next_chi_r[intended_robot_action_index - 1][actual_robot_action_index] += 1
        else:
            next_chi_r = copy.deepcopy(current_augmented_state[3])
            next_chi_r[intended_robot_action_index][actual_robot_action_index] += 1

        return [next_world_state, current_augmented_state[1], next_chi_h, next_chi_r]

    def worldStateTransition(self, current_world_state, actual_robot_action, actual_human_action):
        """
        Returns the next world state of the game after the robot and human acts.
        Hidden parameters are not updated. Used for taking actual robot action.

        :param current_world_state: the current world state
        :type current_world_state: list
        :param actual_robot_action: the current actual robot action
        :type actual_robot_action: list representing one hot vector of actual robot action
        :param actual_human_action: the current actual human action 
        :type actual_human_action: list representing one hot vector of actual human action

        :return: the next world state
        :rtype: list
        """
        aggregated_action = list(
            map(operator.add, actual_human_action[:-2:2], actual_robot_action[:-1:2]))
        next_world_state = list(
            map(operator.add, current_world_state, aggregated_action))

        return next_world_state

    def robotCapabilityCalibrationScore(self, augmented_state):
        """
        Returns the robot's capability calibration score based on how close is the human's belief of robot's capability
        to the robot's true capability. Done by computing distance between belief and true chi distributions.

        :param augmented_state: the augmented state considered
        :type augmented state: list of world_state, theta, chi_h and chi_r

        :return: current robot capability calibration score
        :rtype float
        """
        belief_chi_r = augmented_state[3]

        capability_calibration_score = 0
        eps = 0.00001

        # Iterates through every robot action
        for row in range(0, len(self.action_space) - 1, 2):
            belief_chi_r_normalizer = sum(belief_chi_r[row][row:row+2])
            true_chi_r_normalizer = sum(self.true_chi_r[row][row:row+2])
            belief_probabilities = list(map(
                lambda chi_r: chi_r / float(belief_chi_r_normalizer), belief_chi_r[row][row:row+2]))
            true_probabilities = list(map(
                lambda chi_r: chi_r / float(true_chi_r_normalizer), self.true_chi_r[row][row:row + 2]))

            # KL-divergence method (D_kl(true || belief))
            # KL-divergence is still buggy (do not use as of now)
            if self.capability_calibration_distance == "kl-divergence":
                distance = 0
                for i in range(len(belief_probabilities)):
                    distance += (true_probabilities[i] + eps) * math.log(
                        (true_probabilities[i] + eps) / (belief_probabilities[i] + eps))
                # Normalizes through non-linear sigmoid function transformation
                capability_calibration_score += 1.0 - \
                    (1.0 / (1.0 + math.exp(-distance)))

            # Intersection distance method
            if self.capability_calibration_distance == "intersection":
                capability_calibration_score += min(belief_probabilities[0], true_probabilities[0]) + \
                    min(belief_probabilities[1], true_probabilities[1])

            # Chi-square method
            if self.capability_calibration_distance == "chi-square":
                distance = 0
                for i in range(len(belief_probabilities)):
                    distance += (belief_probabilities[i] - true_probabilities[i]) ** 2 if true_probabilities[i] == 0 \
                        else (belief_probabilities[i] - true_probabilities[i]) ** 2 / true_probabilities[i]
                capability_calibration_score += 1.0 - distance
        
        # Averages capability calibration score over all robot actions
        capability_calibration_score /= ((len(self.action_space) - 1) / 2)

        return capability_calibration_score

    def humanCapabilityCalibrationScore(self, augmented_state):
        """
        Returns the human's capability calibration score based on how close is the robot's belief of human's capability
        to the human's true capability. Done by computing distance between belief and true chi distributions.

        :param augmented_state: the augmented state considered
        :type augmented state: list of world_state, theta, chi_h and chi_r

        :return: current human capability calibration score
        :rtype float
        """
        belief_chi_h = augmented_state[2]

        capability_calibration_score = 0
        eps = 0.00001

        # Iterates through every human action
        for row in range(0, len(self.observation_space) - 2, 2):
            belief_chi_h_normalizer = sum(belief_chi_h[row][row:row + 2])
            true_chi_h_normalizer = sum(self.true_chi_h[row][row:row + 2])
            belief_probabilities = list(map(
                lambda chi_h: chi_h / float(belief_chi_h_normalizer), belief_chi_h[row][row:row + 2]))
            true_probabilities = list(map(
                lambda chi_h: chi_h / float(true_chi_h_normalizer), self.true_chi_h[row][row:row + 2]))

            # KL-divergence method (D_kl(true || belief))
            # KL-divergence is still buggy (do not use as of now)
            if self.capability_calibration_distance == "kl-divergence":
                distance = 0
                for i in range(len(belief_probabilities)):
                    distance += (true_probabilities[i] + eps) * math.log(
                        (true_probabilities[i] + eps) / (belief_probabilities[i] + eps))
                # Normalizes through non-linear sigmoid function transformation
                capability_calibration_score += 1.0 - \
                    (1.0 / (1.0 + math.exp(-distance)))

            # Intersection distance method
            if self.capability_calibration_distance == "intersection":
                capability_calibration_score += min(belief_probabilities[0], true_probabilities[0]) + \
                    min(belief_probabilities[1], true_probabilities[1])

            # Chi-square method
            if self.capability_calibration_distance == "chi-square":
                distance = 0
                for i in range(len(belief_probabilities)):
                    distance += (belief_probabilities[i] - true_probabilities[i]) ** 2 if true_probabilities[i] == 0 \
                        else (belief_probabilities[i] - true_probabilities[i]) ** 2 / true_probabilities[i]
                capability_calibration_score += 1.0 - distance

        # Averages capability calibration score over all human actions
        capability_calibration_score /= ((len(self.observation_space) - 2) / 2)

        return capability_calibration_score

    def reward(self, augmented_state, actual_human_action=None):
        """
        Returns the reward based on the input augmented state for search.

        :param augmented_state: the augmented state considered
        :type augmented_state: list of world_state, theta, chi_h and chi_r

        :return: current immediate reward consisting of environmental reward and capability calibration score
        :rtype: float
        """
        # If human gives negative signal, returns negative reward
        if actual_human_action != None and actual_human_action.index(1) == len(self.observation_space) - 2:
            return -0.1

        world_state = augmented_state[0]
        theta = augmented_state[1]
        true_reward = self.reward_space[theta][1]

        goal_world_state = self.reward_space[theta][0]
        goal_difference = list(
            map(operator.sub, goal_world_state, world_state))

        # If world state satisfies one of the goal states, return associated full reward
        if world_state == self.reward_space[theta][0]:
            return true_reward

        # Else assigns partial rewards
        # Computes robot capability calibration score
        robot_capability_calibration_score = self.robotCapabilityCalibrationScore(augmented_state)

        # Computes environmental partial reward
        env_reward = 0
        individual_max = true_reward / len(world_state)
        for i in range(len(world_state)):
            if goal_difference[i] == 0:  # Exactly meets what is needed
                env_reward += individual_max * 0.1
            elif goal_difference[i] < 0:  # Exceeds what is needed
                env_reward += individual_max * 0.08 * goal_difference[i]
            else:  # Partially meets what is needed
                env_reward += individual_max * 0.1 * \
                    float(world_state[i]) / goal_world_state[i]

        # Averages between environmental reward and capability calibration reward
        partial_reward = (1 - self.capability_calibration_constant) * env_reward + \
            self.capability_calibration_constant * 0.25 * robot_capability_calibration_score

        return partial_reward

    def finalReward(self, augmented_state):
        """
        Returns the reward based on the input augmented state for final evaluation.

        :param augmented_state: the augmented state considered
        :type augmented_state: list of world_state, theta, chi_h and chi_r

        :return: final environmental reward
        :rtype: float
        """
        world_state = augmented_state[0]
        theta = augmented_state[1]
        true_reward = self.reward_space[theta][1]

        goal_world_state = self.reward_space[theta][0]
        goal_difference = list(
            map(operator.sub, goal_world_state, world_state))

        # If world state satisfies true goal state, return full reward
        if world_state == self.reward_space[theta][0]:
            return true_reward

        # Else assign partial rewards
        partial_reward = 0
        individual_max = true_reward / len(world_state)
        for i in range(len(world_state)):
            if goal_difference[i] == 0:  # Exactly meets what is needed
                partial_reward += individual_max
            elif goal_difference[i] < 0:  # Exceeds what is needed
                partial_reward += individual_max * 0.08 * goal_difference[i]
            else:  # Partially meets what is needed
                partial_reward += individual_max * \
                    float(world_state[i]) / goal_world_state[i]

        return partial_reward

    def rolloutObservation(self, augmented_state, intended_robot_action):
        """
        Returns the intended and actual human actions (observation) for rollout in search tree

        :param augmented_state: the augmented state before robot action
        :type augmented_state: list of world_state, theta, chi_h and chi_r
        :param intended_robot_action: the current intended robot action
        :type intended_robot_action: list representing one hot vector of intended robot action

        :return: rollout intended human action, rollout actual human action
        :rtype: lists representing one hot vector of intended and actual human action
        """
        world_state = augmented_state[0]
        theta = augmented_state[1]
        chi_h = augmented_state[2]

        goal_world_state = self.reward_space[theta][0]
        goal_difference_before_robot_action = list(
            map(operator.sub, goal_world_state, world_state))
        goal_difference = list(
            map(operator.sub, goal_difference_before_robot_action, intended_robot_action[:-1:2]))

        intended_robot_action_index = intended_robot_action.index(1)

        # Determines if negative signal is to be given based on robot action
        if intended_robot_action_index % 2 == 0 and \
            intended_robot_action_index // 2 in range(len(goal_world_state)) and \
            goal_difference_before_robot_action[intended_robot_action_index // 2] <= 0:
            # Negative signal if exceeded goal
            intended_human_action_index = len(self.observation_space) - 2

        else:  # Else no signal
            # If everything is satisfied, human does nothing (no-op)
            intended_human_action_index = len(self.observation_space) - 1

            # Human will try to work towards goal no matter what
            possible_intended_human_action_indices = []
            for i, diff in enumerate(goal_difference):
                if diff > 0:
                    possible_intended_human_action_indices.append(
                        i * 2)  # Always carries out successful action

            if len(possible_intended_human_action_indices) > 0:
                intended_human_action_index = rand.choices(
                    possible_intended_human_action_indices)[0]

        intended_human_action = self.observation_space[intended_human_action_index]

        intended_action_chi_h = chi_h[intended_human_action_index]
        chi_h_normalizer = sum(intended_action_chi_h)
        probabilities = list(map(
            lambda chi_h: chi_h / chi_h_normalizer, intended_action_chi_h))
        actual_human_action_index = rand.choices(
            range(len(self.observation_space)), weights=probabilities)[0]
        actual_human_action = self.observation_space[actual_human_action_index]

        return intended_human_action, actual_human_action

    def observation(self, augmented_state, robot_action_node, is_search):
        """
        Returns the intended and actual human actions (observation) as actual simulation

        :param augmented_state: the augmented state before robot action
        :type augmented_state: list of world_state, theta, chi_h and chi_r
        :param robot_action_node: the current robot action node in the search tree
        :type robot_action_node: RobotActionNode
        :param is_search: indicates whether the observation is generated for the search tree
        :param is_search: Boolean

        :return: simulated intended human action, simulated actual human action
        :rtype: lists representing one hot vector of intended and actual human action
        """
        theta = augmented_state[1]
        chi_h = augmented_state[2]
        Q_values = []

        for human_action_node_child in robot_action_node.human_node_children:
            # Setup exploration bonus for tree search
            # No exploration bonus when simulating actual human action outside tree
            exploration_bonus = 0
            if is_search:
                if human_action_node_child == "empty":
                    exploration_bonus = self.c
                else:
                    exploration_bonus = self.c / \
                        (human_action_node_child.visited_list[theta] + 1)

            # Compute Q values
            if human_action_node_child == "empty":
                Q_values.append(exploration_bonus)
            else:
                Q_values.append(human_action_node_child.value_list[theta] +
                                exploration_bonus)

        # Human is rational
        if self.human_behaviour == "rational":
            intended_human_action_index = Q_values.index(max(Q_values))

        # Human is epsilon greedy
        if self.human_behaviour == "e-greedy":
            if rand.random() > self.e:
                intended_human_action_index = Q_values.index(max(Q_values))
            else:
                intended_human_action_index = rand.choice(
                    range(len(self.observation_space)))

        # Human is boltzmann rational
        if self.human_behaviour == "boltzmann":
            exp_Q_values = list(map(lambda Q_value: math.pow(
                1000000, self.beta * Q_value), Q_values))
            exp_normalizer = sum(exp_Q_values)
            probabilities = list(map(
                lambda exp_Q_value: exp_Q_value / exp_normalizer, exp_Q_values))

            # print(probabilities)
            intended_human_action_index = rand.choices(
                range(len(self.observation_space)), weights=probabilities)[0]

        intended_human_action = self.observation_space[intended_human_action_index]

        intended_action_chi_h = chi_h[intended_human_action_index]
        chi_h_normalizer = sum(intended_action_chi_h)
        probabilities = list(map(
            lambda chi_h: chi_h / chi_h_normalizer, intended_action_chi_h))
        actual_human_action_index = rand.choices(
            range(len(self.observation_space)), weights=probabilities)[0]
        actual_human_action = self.observation_space[actual_human_action_index]

        return intended_human_action, actual_human_action

    def realObservation(self, augmented_state, intended_human_action_index):
        """
        Returns the actual human action given the intended human action input by real human.

        :param augmented_state: the augmented state before robot action
        :type augmented_state: list of world_state, theta, chi_h and chi_r
        :param intended_human_action_index: the index that encodes the intended human action
        :param intended_human_action_index: Int

        :return: real intended human action, real actual human action
        :rtype: lists representing one hot vector of intended and actual human action
        """
        theta = augmented_state[1]
        chi_h = augmented_state[2]

        intended_human_action = self.observation_space[intended_human_action_index]

        intended_action_chi_h = chi_h[intended_human_action_index]
        chi_h_normalizer = sum(intended_action_chi_h)
        probabilities = list(map(
            lambda chi_h: chi_h / chi_h_normalizer, intended_action_chi_h))
        actual_human_action_index = rand.choices(
            range(len(self.observation_space)), weights=probabilities)[0]
        actual_human_action = self.observation_space[actual_human_action_index]

        return intended_human_action, actual_human_action

    def robotActionDeviation(self, augmented_state, intended_robot_action):
        """
        Returns actual robot action from the input intended robot action.

        :param augmented_state: the augmented state considered
        :type augmented_state: list of world_state, theta, chi_h and chi_r
        :param intended_human_action: the intended human action
        :type intended_human_action: list representing one hot vector of intended human action

        :return: the actual robot action
        :rtype: list representing one hot vector of actual robot action
        """
        theta = augmented_state[1]
        chi_r = self.true_chi_r

        intended_robot_action_index = intended_robot_action.index(1)
        intended_action_chi_r = chi_r[intended_robot_action_index]
        chi_r_normalizer = sum(intended_action_chi_r)
        probabilities = list(map(
            lambda chi_r: chi_r / chi_r_normalizer, intended_action_chi_r))
        actual_robot_action_index = rand.choices(
            range(len(self.action_space)), weights=probabilities)[0]
        actual_robot_action = self.action_space[actual_robot_action_index]

        return actual_robot_action

    def isTerminal(self, world_state):
        """
        Checks if input world state is a terminal state.

        :param world_state: input world state
        :type world_state: list

        :return: returns true if state is terminal, otherwise false
        """
        return world_state in [reward[0] for reward in self.reward_space]
