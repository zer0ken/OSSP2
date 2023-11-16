import os
import torch as T
import torch.nn as nn


class SavableModule(nn.Module):
    def __init__(self, name, chkpt_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.chkpt_file = os.path.join(chkpt_dir, name+'_maddpg')

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))