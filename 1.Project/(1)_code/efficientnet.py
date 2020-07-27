import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

import efficientnet.keras as efn

model = efn.EfficientNetB0(weights='imagenet')