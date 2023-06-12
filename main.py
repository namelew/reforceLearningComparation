from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from Algoritmos import Algoritmo

ENV_KEY="CartPole-v1"
LEARNING_RATE=1
ENTROPY=0.01
GAMA=0.01
GAEL=0.01

# n_envs é o número de instâncias em paralelo
env = make_vec_env(ENV_KEY, n_envs=4)

algoritmos = (
    Algoritmo("A2C", A2C("MlpPolicy", env, verbose=0)),
    Algoritmo("PPO", PPO("MlpPolicy", env, verbose=0))
)

for algorimo in algoritmos:
    algorimo.train(1000)
    algorimo.test(10)