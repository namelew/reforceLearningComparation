from stable_baselines3.common.base_class import BaseAlgorithm

class Algoritmo:
    def __init__(self, name:str, model:BaseAlgorithm) -> None:
        self.name:str = name
        self.model:BaseAlgorithm = model
    def train(self, timeout:int):
        print(f"\nTraining {self.name} model")
        self.model.learn(total_timesteps=timeout, progress_bar=True)
    def test(self, sample:int):
        env = self.model.get_env()
        result = []
        print(f"\nTesting {self.name} model")
        for _ in range(sample):
            obs = env.reset()

            action, _ = self.model.predict(obs)
            obs, _, terminated, _ = env.step(action)
            steps = 1

            while not any(terminated):
                action, _ = self.model.predict(obs)
                obs, _, terminated, _ = env.step(action)
                steps+=1
            
            result.append(steps)
        result.sort()
        print(f"Results - {self.name}")
        print(f"Mean: {sum(result)/len(result)}")
        print(f"Min: {result[0]}\nMax: {result[-1]}")
