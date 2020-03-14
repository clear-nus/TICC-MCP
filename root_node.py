import random as rand
import math


class RootNode:
    def __init__(self, env, belief):
        """ 
        Initializes the special root node of the search tree. 

        :param env: environment in which the robot and human operate
        :type env: Environment (for now)
        :param belief: the belief set of augmented state particles
        :type belief: list of world_state, theta and chi
        """
        self.type = "root"
        self.env = env
        self.belief = belief
        # List of robot action nodes children
        self.robot_node_children = self.init_children()

    def init_children(self):
        """
        Initializes all the robot node children of the root node to "empty".

        :return: initialized robot node children
        :rtype: list
        """
        children = ["empty"] * len(self.env.action_space)
        return children

    def optimal_robot_action(self, c):
        """
        Returns the optimal robot action to take from this node.

        :param c: exploration constant
        :type c: float

        :return: optimal robot action
        :rtype: list
        """
        values = []
        for child in self.robot_node_children:
            if child == "empty":
                values.append(c)
            else:
                values.append(child.augmented_value(c))

        return self.env.action_space[values.index(max(values))]

    def sample_state(self):
        """
        Randomly sample an initial augmented state from the current belief.

        :return: sampled augmented state
        :rtype: list of world_state, theta and chi
        """
        return rand.choice(self.belief)

    def update_visited(self, theta):
        """
        Does not keep/update the number of times of visiting this node.

        :param theta: the theta visiting the node
        :type theta: list
        """
        pass

    def update_value(self, reward, theta):
        """
        Does not keep/update value of the root.

        :param reward: the immediate reward just received
        :type reward: float
        :param theta: the theta visiting the node
        :type theta: list
        """
        pass

    def update_belief(self, augmented_state):
        """
        Does not update belief of the root.

        :param augmented_state: the augmented state visiting this node
        :type augmented_state: list of world_state, theta and chi
        """
        pass

    def get_children_values(self):
        """
        Returns the values of the robot children nodes of this node.

        :return: values of robot children nodes
        :rtype: list of float
        """
        values = [0] * len(self.robot_node_children)
        for i, child in enumerate(self.robot_node_children):
            if child != "empty":
                values[i] = child.value

        return values
