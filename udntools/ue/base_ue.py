from abc import abstractmethod


class BaseUE(object):

    def __init__(self,
                 ue_number,
                 ue_distribution='uniform',
                 ue_sigma=0):
        self.ue_number_ = int(ue_number)
        self.ue_distribution_ = ue_distribution
        self.ue_position_ = None
        self.ue_sigma = ue_sigma

    @abstractmethod
    def set_ue_to_region(self):
        pass
