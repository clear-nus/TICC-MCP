import math
import time
import copy
import random as rand
import sys

from environment import *
from TICC_POMCP_solver import *
from human_action_node import *
from robot_action_node import *
from root_node import *
from simulated_human import *


class Driver:
    def __init__(self, env, solver, num_steps, simulated_human):
        """
        Initializes a driver.

        :param env: the environment
        :type env: Environment
        :param solver: the TICC-POMCP solver
        :type solver: TICCPOMCPSolver
        :param num_steps: the number of action steps allowed
        :type num_steps: integer
        :param simulated_human: the simulated human
        :type simulated_human: SimulatedHuman
        """
        self.env = env
        self.solver = solver
        self.num_steps = num_steps
        self.simulated_human = simulated_human

    def invigorate_belief(self, curr_human_action_node, parent_human_action_node, intended_robot_action, actual_robot_action, intended_human_action, actual_human_action, env):
        """
        Invigorates the belief space when a new human action node is created.

        :param curr_human_action_node: new human action node being created
        :type curr_human_action_node: HumanActionNode
        :param parent_human_action_node: previous human action node
        :type parent_human_action_node: HumanActionNode
        :param intended_robot_action: the current intended robot action
        :type intended_robot_action: list representing one hot vector of intended robot action
        :param actual_robot_action: the current actual robot action
        :type actual_robot_action: list representing one hot vector of actual robot action
        :param intended_human_action: the current intended human action
        :type intended_human_action: list representing one hot vector of intended human action
        :param actual_human_action: the current actual human action 
        :type actual_human_action: list representing one hot vector of actual human action
        :param env: the environment
        :type env: Environment
        """
        for belief_state in parent_human_action_node.belief:
            # Checks if belief world state is the same as actual world state
            if belief_state[0] == env.world_state:
                next_augmented_state = env.augmentedTransition(
                    belief_state, intended_robot_action, actual_robot_action, intended_human_action, actual_human_action)
                curr_human_action_node.update_belief(next_augmented_state)

    def updateBeliefWorldState(self, human_action_node, env):
        """
        Updates the world state in belief if there is any discrepancy.

        :param human_action_node: human action node whose belief is to be updated
        :type human_action_node: HumanActionNode
        :param env: the environment
        :type env: Environment
        """
        if human_action_node.belief[0][0] != env.world_state:
            human_action_node.belief = [[env.world_state, belief[1], belief[2], belief[3]]
                                        for belief in human_action_node.belief]

    def updateBeliefChiH(self, human_action_node, actual_human_action):
        """
        Updates the chi h in belief if human demonstrates incapable action.

        :param human_action_node: human action node whose belief is to be updated
        :type human_action_node: HumanActionNode
        :param env: the actual human action
        :type env: list representing one hot vector of actual human action
        """
        actual_human_action_index = actual_human_action.index(1)

        # If human carries out failed actual action
        if actual_human_action_index % 2 != 0 and actual_human_action_index // 2 in range(len(self.env.world_state)):
            for belief in human_action_node.belief:
                belief[2][actual_human_action_index -
                          1][actual_human_action_index] += 1

    def updateRootCapabilitiesBelief(self, root_node, current_node):
        """
        Update the root belief about capabilities to the capabilities belief of the current node.

        :param root_node: the root node to be updated
        :type root_node: RootNode
        :param current_node: the node whose belief will be used for update
        :type current_node: RootNode or HumanActionNode
        """
        initial_world_state = [0] * len(self.env.world_state)
        root_node.belief = []
        num_samples = 1000
        for belief_theta in range(len(self.env.reward_space)):
            sampled_beliefs = rand.sample(current_node.belief, num_samples) if len(
                current_node.belief) > num_samples else current_node.belief
            root_node.belief.extend([[initial_world_state, belief_theta, current_node_belief[2], current_node_belief[3]]
                                     for current_node_belief in sampled_beliefs])

    def finalCapabilityCalibrationScores(self, human_action_node):
        """
        Returns the average capability calibration scorse from particles sampled from input human action node.

        :param human_action_node: the human action node from which particles are sampled to be evaluated
        :type human_action_node: HumanActionNode

        :return: expected robot capability calibration score, expected human capability calibration score
        :rtype: float
        """
        num_samples = 10000
        sampled_beliefs = rand.sample(human_action_node.belief, num_samples) if len(
            human_action_node.belief) > num_samples else human_action_node.belief

        total_robot_capability_score = 0
        total_human_capability_score = 0
        for belief in sampled_beliefs:
            total_robot_capability_score += self.env.robotCapabilityCalibrationScore(
                belief)
            total_human_capability_score += self.env.humanCapabilityCalibrationScore(
                belief)

        return total_robot_capability_score / float(len(sampled_beliefs)), total_human_capability_score / float(len(sampled_beliefs))

    def beliefRewardScore(self, belief):
        """
        Returns the reward belief score for the current belief.

        :param belief: the set of belief particles
        :type belief: list

        :return: belief reward score
        :rtype: float
        """
        belief_reward_counts = [0] * len(self.env.reward_space)
        for particle in belief:
            belief_reward_counts[particle[1]] += 1
        belief_reward_normalizer = sum(belief_reward_counts)
        belief_reward_counts = list(map(
            lambda count: count / float(belief_reward_normalizer), belief_reward_counts))

        true_reward_counts = [0] * len(self.env.reward_space)
        true_reward_counts[self.env.true_theta] = 1

        belief_reward_score = sum([
            min(belief_reward_count, true_reward_count) for (belief_reward_count, true_reward_count) in zip(belief_reward_counts, true_reward_counts)])

        return belief_reward_score

    def execute(self, round_num):
        """
        Executes one round of search.

        :param round_num: the round number of the current execution
        :type round_num: integer

        :return: final environmental reward and capability calibration score
        :rtype: float
        """
        intended_robot_actions = []
        actual_robot_actions = []
        intended_human_actions = []
        actual_human_actions = []

        # Creates a deep copy of the env and solver
        env = copy.deepcopy(self.env)
        solver = copy.deepcopy(self.solver)

        print("Executing round {} of search...".format(round_num))
        start_time = time.time()

        for _ in range(self.num_steps):
            # Gets the next robot action node
            intended_robot_action = solver.search()
            intended_robot_action_node = solver.root_action_node.robot_node_children[env.action_space.index(
                intended_robot_action)]
            if intended_robot_action_node == "empty":
                intended_robot_action_node = RobotActionNode(env)
            actual_robot_action = env.robotActionDeviation(
                [env.world_state, env.true_theta, env.true_chi_h, env.true_chi_r], intended_robot_action)

            print("Robot action values:",
                  solver.root_action_node.get_children_values())
            print("Intended robot action:", intended_robot_action)
            print("Actual robot action:  ", actual_robot_action)
            print("World state after robot action:", env.worldStateTransition(
                env.world_state, actual_robot_action, env.observation_space[-1]))

            intended_human_action, actual_human_action = self.simulated_human.simulateHumanAction(
                env.world_state, actual_robot_action)
            intended_human_action_node = intended_robot_action_node.human_node_children[env.observation_space.index(
                intended_human_action)]

            if intended_human_action_node == "empty":
                intended_human_action_node = intended_robot_action_node.human_node_children[env.observation_space.index(
                    intended_human_action)] = HumanActionNode(env)
                self.invigorate_belief(intended_human_action_node, solver.root_action_node,
                                       intended_robot_action, actual_robot_action, intended_human_action, actual_human_action, env)

            # Updates solver and environment
            solver.root_action_node = intended_human_action_node
            env.world_state = env.worldStateTransition(
                env.world_state, actual_robot_action, actual_human_action)

            # Updates world state in belief to ensure alignment with actual world state
            self.updateBeliefWorldState(intended_human_action_node, env)

            # Updates chi h in belief to ensure alignment with actual human action
            self.updateBeliefChiH(
                intended_human_action_node, actual_human_action)

            print("Intended human action:", intended_human_action)
            print("Actual human action:  ", actual_human_action)
            print("World state after human action:", env.world_state)

            # Prints belief theta at the intended human action node after every turn
            temp_belief = [0] * len(self.env.reward_space)
            for particle in solver.root_action_node.belief:
                temp_belief[particle[1]] += 1
            print("Beliefs at selected human action node:", temp_belief)
            print("Num of particles at selected human action node:",
                  len(solver.root_action_node.belief))
            print("Belief reward score for true theta:")
            print(self.beliefRewardScore(solver.root_action_node.belief))

            intended_robot_actions.append(intended_robot_action)
            actual_robot_actions.append(actual_robot_action)
            intended_human_actions.append(intended_human_action)
            actual_human_actions.append(actual_human_action)
            print("Values for thetas:")
            print(solver.root_action_node.value_list)
            print("========================================================")

            # Terminates if goal is reached
            is_all_zero = True
            for i, subgoal in enumerate(env.reward_space[env.true_theta][0]):
                if subgoal - env.world_state[i] > 0:
                    is_all_zero = False
                    break
            if is_all_zero == True:
                break

        # Transfer current capabilities beliefs to the next round
        self.updateRootCapabilitiesBelief(
            self.solver.root_action_node, solver.root_action_node)

        print("========================================================")
        print("Round {} completed!".format(round_num))
        print("Time taken:")
        print("{} seconds".format(time.time() - start_time))
        print("Intended Robot actions:")
        print(intended_robot_actions)
        print("Actual Robot actions:")
        print(actual_robot_actions)
        print("Intended human actions:")
        print(intended_human_actions)
        print("Actual human actions:")
        print(actual_human_actions)
        print("Final world state for round {}:".format(round_num))
        print(env.world_state)
        print("Values for thetas for round {}:".format(round_num))
        print(solver.root_action_node.value_list)
        final_env_reward = env.finalReward(
            [env.world_state, env.true_theta, env.true_chi_h, env.true_chi_r])
        final_capability_calibration_scores = self.finalCapabilityCalibrationScores(
            solver.root_action_node)
        print("Final environmental reward for round {}:".format(round_num))
        print(final_env_reward)
        print("Final robot capability calibration score for round {}:".format(round_num))
        print(final_capability_calibration_scores[0])
        print("Final human capability calibration score for round {}:".format(round_num))
        print(final_capability_calibration_scores[1])

        return final_env_reward, final_capability_calibration_scores


if __name__ == "__main__":
    # Setup constants
    robot_action_space = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    human_action_space = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    reward_space = [[[4, 3, 0, 2, 3], 1],  # 13
                    [[1, 4, 0, 7, 1], 1],  # 13
                    [[2, 3, 2, 3, 3], 1],  # 13
                    [[5, 4, 2, 0, 2], 1],  # 13
                    [[0, 3, 3, 4, 3], 1],  # 13
                    [[3, 3, 0, 3, 3], 1],  # 12
                    [[6, 3, 0, 1, 2], 1],  # 12
                    [[2, 3, 4, 1, 2], 1],  # 12
                    [[1, 1, 2, 4, 4], 1],  # 12
                    [[0, 3, 2, 5, 2], 1]]  # 12
    initial_world_state = [0, 0, 0, 0, 0]
    true_theta = 0
    true_chi_h = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 9, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    belief_chi_h = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    true_chi_r = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 9, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    belief_chi_r = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    capability_calibration_distance = "intersection"
    human_behaviour = "rational"
    beta = 0.9
    gamma = 0.9
    c = 15
    e = 0.1
    capability_calibration_constant = 0.3
    epsilon = math.pow(0.9, 7)
    num_iter = 50000
    num_steps = 7
    initial_belief = []
    for i, _ in enumerate(reward_space):
        belief_theta = i
        initial_belief.append(
            [initial_world_state, belief_theta, belief_chi_h, belief_chi_r])

    # Executes num_tests of experiments
    rand.seed(sys.argv[1])
    # sys.stdout = open("numSamples_50000_10_{}_TICC.txt".format(sys.argv[1]), "a")
    num_test = 10  # Change this to set number of experiments in one run
    for n in range(num_test):
        print("*********************************************************************")
        print("Executing test number {}......".format(n))
        print("*********************************************************************")

        # Setup driver
        env = Environment(robot_action_space, human_action_space, reward_space,
                          initial_world_state, true_theta, true_chi_h, true_chi_r, capability_calibration_distance, human_behaviour, beta, gamma, c, e, capability_calibration_constant)
        root_node = RootNode(env, initial_belief)
        TICC_POMCP_solver = TICCPOMCPSolver(
            epsilon, env, root_node, num_iter, c)
        simulated_human = SimulatedHuman(env)

        driver = Driver(env, TICC_POMCP_solver, num_steps, simulated_human)

        # Executes num_rounds of search (calibration)
        num_rounds = 5
        total_env_reward = 0
        total_robot_capability_calibration_score = 0
        total_human_capability_calibration_score = 0
        # Resets calibration constant
        driver.env.capability_calibration_constant = capability_calibration_constant
        for i in range(num_rounds):
            # Generates arbitrary theta for different rounds
            driver.env.true_theta = rand.randrange(len(reward_space))
            print("True theta: {}".format(driver.env.true_theta))
            env_reward, capability_calibration_scores = driver.execute(i)
            total_env_reward += env_reward
            total_robot_capability_calibration_score += capability_calibration_scores[0]
            total_human_capability_calibration_score += capability_calibration_scores[1]

        print("========================================================")
        print("========================================================")
        print("Successfully completed {} rounds of search (calibration rounds)".format(
            num_rounds))
        print("Average environmental reward (calibration rounds):")
        print(total_env_reward / float(num_rounds))
        print("Average robot capability calibration score (calibration rounds):")
        print(total_robot_capability_calibration_score / float(num_rounds))
        print("Average human capability calibration score (calibration rounds):")
        print(total_human_capability_calibration_score / float(num_rounds))

        # Executes num_rounds of search (upon stabilization)
        num_rounds = 5
        total_env_reward = 0
        total_robot_capability_calibration_score = 0
        total_human_capability_calibration_score = 0
        # Turns off calibration during evaluation
        driver.env.capability_calibration_constant = 0
        for i in range(num_rounds):
            # Generates arbitrary theta for different rounds
            driver.env.true_theta = rand.randrange(len(reward_space))
            print("True theta: {}".format(driver.env.true_theta))
            env_reward, capability_calibration_score = driver.execute(i)
            total_env_reward += env_reward
            total_robot_capability_calibration_score += capability_calibration_scores[0]
            total_human_capability_calibration_score += capability_calibration_scores[1]

        print("========================================================")
        print("========================================================")
        print("Successfully completed {} rounds of search (stabilized rounds)".format(
            num_rounds))
        print("Average environmental reward (stabilized rounds):")
        print(total_env_reward / float(num_rounds))
        print("Average robot capability calibration score (stabilized rounds):")
        print(total_robot_capability_calibration_score / float(num_rounds))
        print("Average human capability calibration score (stabilized rounds):")
        print(total_human_capability_calibration_score / float(num_rounds))
