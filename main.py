import random
import torch
import numpy as np
import Trans_mod

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("\nSelected device:", device, end="\n\n")

tmod = Trans_mod.Train_test(dataset='sy30A', device=device, skip_train=True, save=True)  #dataset=sy30A,或者 DC3_20
tmod.run(smry=False)
