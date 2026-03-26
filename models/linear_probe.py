import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        return self.linear(x)

    def get_optimizer(self, weight_decay=1e-2, lr=9e-4, betas=(0.9, 0.99)):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}

        assert len(decay & no_decay) == 0, "parameters %s made it into both decay/no_decay sets!" % (
            str(decay & no_decay),)
        assert len(param_dict.keys() - (
                    decay | no_decay)) == 0, "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - (decay | no_decay)),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

    def get_loss_fn(self):
        return nn.CrossEntropyLoss()