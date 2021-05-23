import numpy as np
import torch

from .network import DeepQNetwork
from .memory import Memory


class DQNAgent:
    def __init__(
            self,
            input_shape,
            action_shape,
            gamma,
            epsilon,
            learning_rate,
            batch_size=32,
            memory_size=10000,
            epsilon_minimum=0.01,
            epsilon_decrement=5e-7,
            target_replace_frequency=1000,
            checkpoint_dir='temp/'

    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_minimum = epsilon_minimum
        self.epsilon_decrement = epsilon_decrement
        self.target_replace_frequency = target_replace_frequency
        self.checkpoint_dir = checkpoint_dir

        self.action_space = [i for i in range(action_shape)]
        self.current_step = 0

        self.replay_memory = Memory(memory_size, input_shape)
        self.eval_network, self.target_network = self.create_networks(
            input_shape, action_shape, learning_rate
        )

    def create_networks(self, *args, **kwargs):
        return (
            DeepQNetwork(*args, **kwargs, checkpoint_file=self.checkpoint_dir + 'dqn_eval'),
            DeepQNetwork(*args, **kwargs, checkpoint_file=self.checkpoint_dir + 'dqn_target')
        )

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)

        state = torch.tensor([observation], dtype=torch.float)
        state = state.to(self.eval_network.device)
        actions = self.eval_network.forward(state)
        return torch.argmax(actions).item()

    def replace_target_network(self):
        if self.current_step % self.target_replace_frequency == 0:
            self.target_network.load_state_dict(self.eval_network.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon -= self.epsilon_decrement
        else:
            self.epsilon = self.epsilon_minimum

    def save_networks(self):
        self.target_network.save_checkpoint()
        self.eval_network.save_checkpoint()

    def load_networks(self):
        self.target_network.load_checkpoint()
        self.eval_network.load_checkpoint()

    def save_memory(self, state, action, reward, new_state, done):
        self.replay_memory.save(state, action, reward, new_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.replay_memory.sample(self.batch_size)
        return (
            self.eval_network.to_tensor(state),
            self.eval_network.to_tensor(action),
            self.eval_network.to_tensor(reward),
            self.eval_network.to_tensor(new_state),
            self.eval_network.to_tensor(done)
        )

    def learn(self):

        # Fill all the replay memory before starting
        if self.replay_memory.memory_counter < self.replay_memory.memory_size:
            return

        self.eval_network.optimizer.zero_grad()
        self.replace_target_network()
        states, actions, rewards, next_states, done_flags = self.sample_memory()

        # For each item in batch we need the action_value of the specific action
        action_values = self.eval_network.forward(states)
        indices = np.arange(self.batch_size)
        action_values = action_values[indices, actions]

        # We need the next state max action values. We forward next_states,
        # through the target network, take a max over the action dimension
        # and return the first value of the tuple (value, indices)
        # Doc here: https://pytorch.org/docs/master/generated/torch.max.html#torch.max
        action_values_next = self.target_network.forward(next_states)
        action_values_next = torch.max(action_values_next, dim=1)[0]
        # Mask everything that is done to zero
        action_values_next[done_flags] = 0.0

        # Calculate target action value using the equation:
        action_value_target = rewards + self.gamma * action_values_next

        # Propagate errors and step
        self.eval_network.backward(action_value_target, action_values)
        self.current_step += 1
        self.decrement_epsilon()
