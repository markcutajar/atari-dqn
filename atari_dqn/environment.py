import collections
import cv2
import gym
import numpy as np


class RepeatActionInFrames(gym.Wrapper):
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


class ChannelInvertGrayFrame(gym.ObservationWrapper):
    def __init__(self, env, frame_shape):
        super().__init__(env)

        # We 2nd dimension to be channels as required by pytorch
        # While OpenAI gives channel as last dimension. In the
        # observation fn we need to reshape
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.frame_shape = frame_shape

        # Create the new observation space for the env
        # Since we are converting to grayscale we set low of 0 and high of 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.frame_shape, dtype=np.float32
        )

    def observation(self, observation):
        """Convert from color to gray scale, resize images
         to be smaller to be easier to train, change from 255
         grayscale to 0-1 scale and reshape to channels 2nd dimension
        """
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.shape[1:], interpolation=cv2.INTER_AREA)
        frame = np.array(frame) / 255.0
        return frame.reshape(self.frame_shape)


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super().__init__(env)

        self.observation_stack = collections.deque(maxlen=repeat)

        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis=0),
            env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.float32
        )

    def reshape_stack(self):
        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def reset(self):
        observation = self.env.reset()
        self.stack.clear()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        return self.reshape_stack()

    def observation(self, observation):
        self.stack.append(observation)
        return self.reshape_stack()
