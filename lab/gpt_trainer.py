import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dataset.dataset import OthelloDataset
from models.gpt import GPT

torch.set_float32_matmul_precision('high')


class GPTTrainer:
    def __init__(self, rank, world_size, batch_size=32, save_every=100, lr_decay=False,
                 warmup_tokens=100_000_000, final_tokens=1_500_000_000):
        self.rank = rank
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        self.tokens = 0

        self.__setup(world_size)

        dataset = OthelloDataset(train=True)
        self.sampler = DistributedSampler(dataset, num_replicas=world_size, rank=self.rank)
        self.data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=self.sampler, pin_memory=True,
                                      num_workers=2, prefetch_factor=2)
        self.device = torch.device("cuda", self.local_rank)
        self.model = GPT().to(self.device)
        self.model = DDP(self.model)

    def __setup(self, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        torch.cuda.set_device(self.rank)

        dist.init_process_group(backend, rank=self.rank, world_size=world_size)

    def train(self, num_epochs=1000):
        try:
            for epoch in range(num_epochs):
                self.sampler.set_epoch(epoch)

                for batch_idx, (x, y) in enumerate(self.data_loader):
                    pass

        except Exception as e:
            print(e)
            self.__save_checkpoint()

    def __save_checkpoint(self):
        pass

    def __save_model(self):
        pass

    def __cleanup(self):
        dist.destroy_process_group()

    def __del__(self):
        self.__cleanup()