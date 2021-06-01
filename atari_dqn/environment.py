import collections
import cv2
import gym
import numpy as np

from gym.wrappers import GrayScaleObservation, FrameStack


class RepeatActionInFramesTakeMaxOfTwo(gym.Wrapper):
    def __init__(self, env, repeat=4):
        super().__init__(env)

        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frames = collections.deque(maxlen=2)

        if repeat <= 0:
            raise ValueError('Repeat value needs to be 1 or higher')

    def step(self, action):

        total_reward = 0
        done = False
        info = {}

        for i in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.frames.append(observation)

            if done:
                break

        # Open queue into arguments for np.maximum
        maximum_of_frames = np.maximum(*self.frames)
        return maximum_of_frames, total_reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.frames.clear()
        self.frames.append(observation)
        return observation


class NormResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)

        # Create the new observation space for the env
        # Since we are converting to grayscale we set low of 0 and high of 1
        self.shape = shape

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.shape, dtype=np.float32
        )

    def observation(self, observation):
        """Change from 255 grayscale to 0-1 scale
        """
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return (observation / 255.0).reshape(self.shape)


def prep_environment(env, shape, repeat):
    env = RepeatActionInFramesTakeMaxOfTwo(env, repeat)
    env = GrayScaleObservation(env)
    env = NormResizeObservation(env, shape)
    return FrameStack(env, num_stack=repeat)
