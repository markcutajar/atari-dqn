import numpy as np


class Memory:
    def __init__(self, memory_size, input_shape):

        # state, action, reward, next_state, done
        memory_shape = [
            ('state', np.float32, input_shape), ('action', np.int64),
            ('reward', np.float32), ('next_state', np.float32, input_shape),
            ('done', np.bool)
        ]

        # Numpy record structure array allows, different data types
        # but with also batching ability
        self.memory = np.rec.array(np.zeros(10, dtype=memory_shape))
        self.memory_size = memory_size
        self.memory_counter = 0

    def save(self, state, action, reward, next_state, done):
        """Save the transition of the into the buffer
        """
        index = self.memory_counter % self.memory_size
        self.memory[index] = (state, action, reward, next_state, done)
        self.memory_counter += 1

    def sample(self, batch_size):
        """Return a sample of batch_size given from memory. We do not use replace
        so the samples are unique.
        """
        maximum_current_memory = min(self.memory_counter, self.memory_size)
        indices = np.random.choice(maximum_current_memory, batch_size, replace=False)
        batch = self.memory[indices]

        return (
            batch.state,
            batch.action,
            batch.reward,
            batch.next_state,
            batch.done
        )
