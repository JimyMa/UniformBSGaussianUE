from abc import abstractmethod
from . import BaseBS


class BandConstrainBS(BaseBS):
    def __init__(self,
                 bs_number,
                 layer=1,
                 power=1.0,
                 bs_distribution="uniform"):
        super(BandConstrainBS, self).__init__(bs_number, layer, power, bs_distribution)

    @abstractmethod
    def set_bs_to_region(self):
        # Get initialized BS Position
        pass

    @abstractmethod
    def set_uniform_bs_to_region(self):
        pass

    @abstractmethod
    def select_ue(self):
        pass
