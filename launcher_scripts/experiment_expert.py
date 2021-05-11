import gym
ENV_NAME = "HalfCheetah-v2"
from stable_baselines3 import PPO


def main():
    env = gym.make(ENV_NAME)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()
