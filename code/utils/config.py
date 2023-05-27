import os
import random

import numpy as np
import torch
import hanlp


# tok = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_every_where():
    print("seed every where")
    seed = 7
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
