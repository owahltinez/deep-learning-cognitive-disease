import collections
from torch import nn
from bananas.core.mixins import HighDimensionalMixin
from bananas.statistics.loss import LossFunction
from coconuts.learners.base import BaseNNLearner, BaseNNRegressor, BaseNNClassifier, Flatten
    
class BaseQD(BaseNNLearner, HighDimensionalMixin):
    ''' Network inspired by the architecture used in the "Quick, Draw!" paper '''

    def __init__(self, kernel_size: int = 5,
                 learning_rate: float = .001, loss_function: LossFunction = None,
                 random_seed: int = 0, verbose: bool = False, **kwargs):
        super().__init__(learning_rate=learning_rate, loss_function=loss_function,
                         random_seed=random_seed, verbose=verbose, **kwargs)
        self.kernel_size = kernel_size

    def init_model(self, input_shape: tuple, output_shape: tuple):
        conv_channels = [16, 32]
        dense_sizes = [256, 128]
        
        model = collections.OrderedDict()
        model['conv1'] = nn.Sequential(
            nn.Conv2d(input_shape[0], conv_channels[0], self.kernel_size, bias=False),
            nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        model['conv2'] = nn.Sequential(
            nn.Conv2d(*conv_channels, self.kernel_size, bias=False),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        model['flatten'] = Flatten()
        
        conv1_outdim = (input_shape[-1] - self.kernel_size + 1) // 2
        conv2_outdim = (conv1_outdim - self.kernel_size + 1) // 2
        dimension = conv_channels[-1] * conv2_outdim * conv2_outdim
        model['fc1'] = nn.Sequential(nn.Linear(dimension, dense_sizes[0]), nn.Dropout(0.5))
        model['fc2'] = nn.Sequential(nn.Linear(*dense_sizes), nn.Dropout(0.5))
        model['fc3'] = nn.Sequential(nn.Linear(dense_sizes[-1], output_shape[0]))

        return nn.Sequential(model)


class QDRegressor(BaseQD, BaseNNRegressor):
    pass

class QDClassifier(BaseQD, BaseNNClassifier):
    pass