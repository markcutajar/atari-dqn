import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from atari_dqn.agent import DQNAgent
from atari_dqn.environment import prep_environment


if __name__ == '__main__':
    repeat = 4
    frame_shape = (84, 84)
    gamma = 0.99
    epsilon = 1
    learning_rate = 0.0001
    games = 200
    rolling_average_n = 200
    directory = 'temp/'

    if not os.path.exists(directory):
        raise NotADirectoryError('Folder specified to save plots and models does not exist')

    env = gym.make('EnduroNoFrameskip-v4')
    env = prep_environment(env, frame_shape, repeat)

    agent = DQNAgent(
        input_shape=env.observation_space.shape,
        action_shape=env.action_space.n,
        gamma=gamma,
        epsilon=epsilon,
        learning_rate=learning_rate,
        checkpoint_dir=directory
    )

    best_score = 0

    plot_name = f'{directory}dqn_agent_enduro_plot.png'
    scores, steps, rolling_means, epsilons = [], [], [], []
    current_step = 0

    for episode in range(games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward

            agent.save_to_memory(observation, action, reward, new_observation, done)
            agent.learn()
            observation = new_observation
            current_step += 1

        scores.append(score)
        steps.append(current_step)
        epsilons.append(agent.epsilon)

        rolling_mean = np.mean(scores[-rolling_average_n:])
        rolling_means.append(rolling_mean)

        print(f"Ep: {episode} | Score: {score} | Avg: {rolling_mean:.1f} | Best: {best_score:.1f}")

        if score > best_score:
            best_score = score
            agent.save_networks()

    fig, ax = plt.subplots()
    ax.plot(steps, rolling_means, color="red")
    ax.set_xlabel("steps", fontsize=12)
    ax.set_ylabel("Mean Score", color="red", fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(steps, epsilons, color="blue")
    ax2.set_ylabel("Epsilon", color="blue", fontsize=12)
    fig.savefig(plot_name)

