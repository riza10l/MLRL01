import numpy as np

class EnsembleAgent:
    def __init__(self, agents, weights=None):
        self.agents = agents
        self.weights = weights if weights is not None else [1.0/len(agents)] * len(agents)

    def predict(self, obs, deterministic=True):
        votes = []
        for agent in self.agents:
            action, _ = agent.predict(obs, deterministic=deterministic)
            votes.append(action)
        
        # Simple majority voting for now
        # In a real scenario, you'd want to handle position sizes (P8)
        counts = np.bincount(votes, weights=self.weights, minlength=6)
        return np.argmax(counts), None

    def evaluate(self, env):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = self.predict(obs)
            obs, _, done, _, _ = env.step(action)
        return env.get_equity_curve(), env.get_trade_stats(), env.get_trade_log()