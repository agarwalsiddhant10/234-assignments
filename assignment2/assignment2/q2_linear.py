import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from core.deep_q_learning import DQN
from q1_schedule import LinearExploration, LinearSchedule

from configs.q2_linear import config


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)
        # print(state_shape)

        ##############################################################
        """
        TODO: 
            Add placeholders:
            Remember that we stack 4 consecutive frames together.
                - self.s: batch of states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.a: batch of actions, type = int32
                    shape = (batch_size)
                - self.r: batch of rewards, type = float32
                    shape = (batch_size)
                - self.sp: batch of next states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.done_mask: batch of done, type = bool
                    shape = (batch_size)
                - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: 
            Variables from config are accessible with self.config.variable_name.
            Check the use of None in the dimension for tensorflow placeholders.
            You can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################

        # print(self.config)

        self.s = tf.placeholder(dtype = tf.uint8, shape = (None, state_shape[0], state_shape[1], state_shape[2]*self.config.state_history), name = "s")
        self.a = tf.placeholder(dtype = tf.int32, shape = (None), name = "a")
        self.r = tf.placeholder(dtype = tf.float32, shape = (None), name = "r")
        self.sp = tf.placeholder(dtype = tf.uint8, shape = (None, state_shape[0], state_shape[1], state_shape[2]*self.config.state_history), name = "sp")
        self.done_mask = tf.placeholder(dtype = tf.bool, shape = (None), name = "done_mask")
        self.lr = tf.placeholder(dtype = tf.float32, shape = (), name = "lr")

        ##############################################################
        ######################## END YOUR CODE #######################


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            Implement a fully connected with no hidden layer (linear
            approximation with bias) using tensorflow.

        HINT: 
            - You may find the following functions useful:
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ################## 
        
        out = tf.contrib.layers.flatten(state)
        out = tf.contrib.layers.fully_connected(out, num_actions, activation_fn = None, reuse = reuse, scope = scope)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. 
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: 
            Add an operator self.update_target_op that for each variable in
            tf.GraphKeys.GLOBAL_VARIABLES that is in q_scope, assigns its
            value to the corresponding variable in target_q_scope

        HINT: 
            You may find the following functions useful:
                - tf.get_collection
                - tf.assign
                - tf.group (the * operator can be used to unpack a list)

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        
        col_q = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = q_scope)
        col_target = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = target_q_scope)

        assigns = []

        for i in range(len(col_q)):
            q_name = col_q[i].name.split('/')[1:]
            # print('out', col_q[i])
            for j in range(len(col_target)):
                target_name = col_target[j].name.split('/')[1:]
                # print('in', col_target[j])
                if(q_name == target_name):
                    assign = tf.assign(col_target[j], col_q[i])
                    assigns.append(assign)
        self.update_target_op = tf.group(*assigns)


        ##############################################################
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 
        HINT: 
            - Config variables are accessible through self.config
            - You can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
            - You may find the following functions useful
                - tf.cast
                - tf.reduce_max
                - tf.reduce_sum
                - tf.one_hot
                - tf.squared_difference
                - tf.reduce_mean
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############

        # print(tf.one_hot(self.done_mask, depth = 2))
        done = tf.cast(self.done_mask, tf.float32)
        not_done = tf.cast(tf.math.logical_not(self.done_mask), tf.float32)
        q_sample = self.r*done + (self.r + self.config.gamma*tf.reduce_max(target_q, axis = 1))*not_done
        action_space = tf.one_hot(self.a, num_actions)
        q_ac = tf.reduce_sum(action_space*q, axis = 1)
        self.loss = tf.reduce_sum(tf.squared_difference(q_sample, q_ac))

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
        """

        ##############################################################
        """
        TODO: 
            1. get Adam Optimizer
            2. compute grads with respect to variables in scope for self.loss
            3. if self.config.grad_clip is True, then clip the grads
                by norm using self.config.clip_val 
            4. apply the gradients and store the train op in self.train_op
                (sess.run(train_op) must update the variables)
            5. compute the global norm of the gradients (which are not None) and store 
                this scalar in self.grad_norm

        HINT: you may find the following functions useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variables by writing self.config.variable_name
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############

        optimizer = tf.train.AdamOptimizer(learning_rate = self.lr)
        col = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope)
        print('col')
        print(col)
        var_in_scope = [var for var in col]
        print('var_in_scope')
        print( var_in_scope)
        var_grad = optimizer.compute_gradients(self.loss, var_in_scope)
        print(var_grad)
        grad_list = [grad[1] for grad in var_grad]
        if self.config.grad_clip is True:
            pass
        self.train_op = optimizer.apply_gradients(var_grad)
        self.grad_norm = tf.global_norm(grad_list)
        
        ##############################################################
        ######################## END YOUR CODE #######################
    


if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
