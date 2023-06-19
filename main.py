from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
from Algoritmos import Algoritmo

ENV_KEY="CartPole-v1"
RUNS = 1
TIMEOUT=25000
TEST_SAMPLE=100

LEARNING_RATE=0
ENTROPY=0
GAMA=0
GAEL=0


LEARNING_RATE_PASS=0.0001
ENTROPY_PASS=0.1
GAMA_PASS=0.01
GAEL_PASS=0.1

env = make_vec_env(ENV_KEY, n_envs=4)

results = []

for _ in range(RUNS):
    algoritmos = (
        Algoritmo("A2C", A2C("MlpPolicy",env,device="cuda",learning_rate=LEARNING_RATE,gamma=GAMA,gae_lambda=GAEL,ent_coef=ENTROPY,verbose=0)),
        Algoritmo("PPO", PPO("MlpPolicy",env,device="cuda",learning_rate=LEARNING_RATE, gamma=GAMA,gae_lambda=GAEL,ent_coef=ENTROPY,verbose=0))
    )

    for algorimo in algoritmos:
        algorimo.train(TIMEOUT)
        min,max,mean = algorimo.test(TEST_SAMPLE)
        results.append((algorimo.name, ENV_KEY,TIMEOUT, LEARNING_RATE,ENTROPY,GAMA,GAEL,min,max,mean))
    
    LEARNING_RATE += LEARNING_RATE_PASS
    ENTROPY += ENTROPY_PASS
    GAMA += GAMA_PASS if (GAMA + GAMA_PASS) < 1 else 0
    GAEL += GAEL_PASS

dataset = pd.DataFrame(results)
dataset.to_csv("resultados.csv",sep=',', header=["algoritmo", "ambiente", "tempo", "taxa-aprendizado", "entropia", "gama", "gae-lambda", "min-passos", "max-passos", "media-passos"], index=False)