from abc import abstractmethod


class BaseUE(object):

    def __init__(self,
                 ue_number,
                 ue_distribution='uniform'):
        self.ue_number_ = ue_number
        self.ue_distribution_ = ue_distribution
        self.ue_position_ = None

    @abstractmethod
    def set_ue_to_region(self):
        pass

    @abstractmethod
    def set_uniform_ue_to_region(self):
        pass

