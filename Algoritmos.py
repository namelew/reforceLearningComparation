from stable_baselines3.common.base_class import BaseAlgorithm

class Algoritmo:
    def __init__(self, name:str, model:BaseAlgorithm) -> tuple[int,int,int]:
        self.name:str = name
        self.model:BaseAlgorithm = model
    def train(self, timeout:int):
        print(f"\nTraining {self.name} model")
        self.model.learn(total_timesteps=timeout, progress_bar=True)
    def test(self, sample:int):
        env = self.model.get_env()
        result:int = []
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
        mean = round(sum(result)/len(result))
        print(f"Mean: {mean}")
        print(f"Min: {result[0]}\nMax: {result[-1]}")
        return result[0], result[-1], mean
