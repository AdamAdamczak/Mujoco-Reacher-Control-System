import abc


class TrajectoryGenerator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate(self, t):
        pass
