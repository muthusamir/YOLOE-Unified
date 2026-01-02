class PromptCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size

    def get(self, prompt):
        key = hash(prompt)
        return self.cache.get(key)

    def set(self, prompt, embedding):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))  # LRU approximate
        self.cache[hash(prompt)] = embedding
