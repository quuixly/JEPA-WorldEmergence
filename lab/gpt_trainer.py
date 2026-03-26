import os
import random
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import logging

from dataset.dataset import OthelloDataset
from game.othello import GameBoard, Piece
from models.gpt import GPT


torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO)


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, top_k=None, block_size=60):
    model.eval()

    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

        logits = model(x_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, idx_next), dim=1)

    return x

def convert_token_to_position(token):
    flat_index = token - 1

    if flat_index >= 33:
        flat_index += 4

    elif flat_index >= 27:
        flat_index += 2

    row = flat_index // 8
    col = flat_index % 8

    return GameBoard.index_to_position((row, col))


class GPTTrainer:
    def __init__(self,batch_size=32, save_every_batch=100, lr_decay=True,
                 warmup_tokens=1_200_000_000, final_tokens=12_000_000_000):
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        self.save_every_batch = save_every_batch
        self.tokens = 0

        self.__setup()

        logging.info("Loading dataset...")
        dataset = OthelloDataset(train=True)
        if self.local_rank == 0:
            self.val_dataset = OthelloDataset(train=False)

        self.sampler = DistributedSampler(dataset)
        self.data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, sampler=self.sampler, pin_memory=True,
                                      num_workers=2, prefetch_factor=2)
        logging.info("Creating model...")
        self.device = torch.device(self.local_rank)
        self.model = GPT().to(self.device)
        self.optimizer = self.model.get_optimizer()
        self.loss_fn = self.model.get_loss_fn()
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def __setup(self):
        self.local_rank = int(os.environ["LOCAL_RANK"])

        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        torch.cuda.set_device(self.local_rank)

        dist.init_process_group(backend)

    def train(self, num_epochs=10):
        logging.info("Training...")

        try:
            for epoch in range(num_epochs):
                self.sampler.set_epoch(epoch)

                pbar = None
                if self.local_rank == 0:
                    pbar = tqdm(total=len(self.data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

                for batch_idx, (x, y) in enumerate(self.data_loader):
                    self.model.train()
                    # Training
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), y.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)
                    self.optimizer.step()

                    # Learning rate decay
                    if self.lr_decay:
                        batch_tokens = (y >= 0).sum()
                        dist.all_reduce(batch_tokens, op=dist.ReduceOp.SUM)
                        self.tokens += batch_tokens.item()
                        if self.tokens < self.warmup_tokens:
                            lr_mult = self.tokens / max(1, self.warmup_tokens)
                        else:
                            progress = (self.tokens - self.warmup_tokens) / \
                                       max(1, self.final_tokens - self.warmup_tokens)
                            progress = min(1.0, progress)
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = self.optimizer.defaults['lr'] * lr_mult
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr

                    # Display statistics
                    if self.local_rank == 0 and pbar is not None:
                        pbar.set_postfix({
                            "loss": f"{loss.item():.4f}",
                            "lr": f"{lr:.2e}" if self.lr_decay else self.optimizer.param_groups[0]['lr']
                        })
                        pbar.update(1)

                    # Save model
                    if self.local_rank == 0 and batch_idx % self.save_every_batch == 0:
                        pass_rate = self.evaluate_legal_moves()
                        self.__save_model(f"batch_{batch_idx}_loss_{loss.item():.4f}_pass_rate_{pass_rate:.4f}")


        except (Exception, KeyboardInterrupt) as e:
            print(e)
            self.__save_checkpoint()

    def evaluate_legal_moves(self, num_games=1000):
        self.model.eval()
        total_nodes = 0
        success_nodes = 0

        games_to_eval = [self.val_dataset[i][0] for i in range(num_games)]
        bar = tqdm(games_to_eval, desc="Evaluating Legal Moves", leave=False)

        with torch.no_grad():
            for game_tensor in bar:
                pad_val = 0

                valid_moves = [move for move in game_tensor.view(-1).tolist() if move != pad_val]
                length_of_whole_game = len(valid_moves)

                for length_of_partial_game in range(1, length_of_whole_game):
                    total_nodes += 1

                    context_tokens = valid_moves[:length_of_partial_game]

                    x = torch.tensor(context_tokens, dtype=torch.long)[None, ...].to(self.device)

                    y = sample(self.model.module, x, steps=1, temperature=0.0)[0]
                    predicted_token = int(y[-1])

                    predicted_move = convert_token_to_position(predicted_token)

                    try:
                        test_board = GameBoard()
                        current_player = Piece.BLACK

                        for token in context_tokens:
                            move_str = convert_token_to_position(token)
                            test_board.add_piece(current_player, move_str)

                            current_player = Piece.WHITE if current_player == Piece.BLACK else Piece.BLACK
                            if not test_board.get_legal_moves(current_player):
                                opponent = Piece.WHITE if current_player == Piece.BLACK else Piece.BLACK
                                if test_board.get_legal_moves(opponent):
                                    current_player = opponent

                        legal_moves_for_current_state = test_board.get_legal_moves(current_player)
                        if predicted_move in legal_moves_for_current_state:
                            success_nodes += 1

                    except Exception:
                        pass

                bar.set_postfix({"pass_rate": f"{success_nodes / max(1, total_nodes) * 100:.2f}%"})

        pass_rate = success_nodes / max(1, total_nodes) * 100
        logging.info(f"\nLegal Move Pass Rate: {pass_rate:.2f}% ({success_nodes}/{total_nodes} nodes)\n")

        self.model.train()
        return pass_rate

    def __save_checkpoint(self):
        if self.local_rank == 0:
            checkpoint = {
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            torch.save(checkpoint, "checkpoints/gpt_checkpoint.pt")

    def __save_model(self, postfix):
        if self.local_rank == 0:
            torch.save(self.model.module.state_dict(), f"checkpoints/gpt_{postfix}.pt")

    def save_baseline(self):
        if self.local_rank == 0:
            torch.save(self.model.module.state_dict(), f"checkpoints/gpt_baseline.pt")

    def __cleanup(self):
        dist.destroy_process_group()

    def __del__(self):
        self.__cleanup()


if __name__ == "__main__":
    trainer = GPTTrainer()
    trainer.train()