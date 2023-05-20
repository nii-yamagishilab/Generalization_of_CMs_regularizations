import torch
import random
import numpy as np

def set_seed(seed = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
	#  cudnn.deterministic = deterministic
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
