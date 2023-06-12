from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from Algoritmos import Algoritmo

ENV_KEY="CartPole-v1"
LEARNING_RATE=0.8
TIMEOUT=1000
TEST_SAMPLE=100
ENTROPY=0.01
GAMA=0.01
GAEL=0.01

env = make_vec_env(ENV_KEY, n_envs=4)

algoritmos = (
    Algoritmo("A2C", A2C("MlpPolicy",env,learning_rate=LEARNING_RATE,gamma=GAMA,gae_lambda=GAEL,ent_coef=ENTROPY,verbose=0)),
    Algoritmo("PPO", PPO("MlpPolicy",env,learning_rate=LEARNING_RATE, gamma=GAMA,gae_lambda=GAEL,ent_coef=ENTROPY,verbose=0))
)

for algorimo in algoritmos:
    algorimo.train(TIMEOUT)
    algorimo.test(TEST_SAMPLE)