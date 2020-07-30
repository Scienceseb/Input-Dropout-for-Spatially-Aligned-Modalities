import random
import torch.utils.data as data


class InputDropout(object):
    def __init__(self, mode="addit",test=False):
        self.mode = mode
        self.test=test

    def __call__(self, sample):
        if self.test == True:
            data[3, :, :] = 0
        if self.test==False:
            if self.mode == "both":
                if random.uniform(0, 1) <= 0.66:
                    if random.uniform(0, 1) <= 0.5:
                        data[3, :, :] = 0
                    else:
                        data[0, :, :] = 0
                        data[1, :, :] = 0
                        data[2, :, :] = 0

            elif random.uniform(0, 1) <= 0.5:
                if self.mode == "addit":
                    data[3, :, :] = 0

        return data