from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

ENV_KEY="CartPole-v1"
RENDER_TYPE="ansi"

env = make_vec_env(ENV_KEY, n_envs=4)

def train(timeout:int) -> A2C:
    model = A2C("MlpPolicy", env, verbose=0)
    print("Trainning model")
    model.learn(total_timesteps=timeout, progress_bar=True)
    return model

def test(model:A2C, sample:int):
    obs = env.reset()
    result = []
    for _ in range(sample):
        obs = env.reset()

        action, _ = model.predict(obs)
        obs, _, terminated, _ = env.step(action)
        steps = 1

        while not any(terminated):
            action, _ = model.predict(obs)
            obs, _, terminated, _ = env.step(action)
            steps+=1
        
        result.append(steps)
    
    print(result)
    result.sort()
    print(f"Mean: {sum(result)/len(result)}")
    print(f"Min: {result[0]}\nMax: {result[-1]}")

model = train(25000)
test(model, 10)