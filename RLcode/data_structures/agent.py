from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Basic Agent structure.
    """
    @abstractmethod
    def play(self):
        pass
    
    def register_play(self, action):
        pass
    
    def reset(self):
        pass