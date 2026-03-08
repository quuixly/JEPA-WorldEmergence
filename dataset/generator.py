from multiprocessing import Pool
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import os
import logging

from game.othello import GameBoard


logging.basicConfig(level=logging.INFO)


def generate_single_game(_):
    game_board = GameBoard()
    game_board.simulate_game()

    return game_board


class DatasetGenerator:
    """
    This class generates a dataset (both training and test sets). It guarantees that each sequence in the test set is
    unique across both sets, and that each game is unique.
    Games are saved as uint8 tensors to reduce memory usage, and are generated using multiprocessing for efficiency.
    """
    def __init__(self, train_size = 20_000_000, test_size = 2_000_000, padding_token=0):
        self.GAME_MAX_LENGTH = 60
        self.NUM_OF_WORKERS = os.cpu_count()
        self.BATCH_SIZE = 10_000
        self.train_size = train_size
        self.test_size = test_size
        self.padding_token = padding_token
        self.unique_games = set()

        self.train_dataset = torch.empty((self.train_size,  self.GAME_MAX_LENGTH), dtype=torch.uint8)
        self.test_dataset = torch.empty((self.test_size,  self.GAME_MAX_LENGTH), dtype=torch.uint8)
        self.current_train_index = 0
        self.current_test_index = 0

    def generate(self, load_checkpoint=False):
        if load_checkpoint:
            self.__load_checkpoint()

        try:
            self.__generate_train_dataset()
            self.__generate_test_dataset()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.__save_checkpoint()

    def __load_checkpoint(self):
        if not Path("train_dataset.pt").exists():
            raise FileNotFoundError("train_dataset.pt file does not exists!")
        if not Path("test_dataset.pt").exists():
            raise FileNotFoundError("test_dataset.pt file does not exists!")
        if not Path("checkpoint_data.pt").exists():
            raise FileNotFoundError("checkpoint_data.pt file does not exists!")

        self.train_dataset = torch.load("train_dataset.pt")
        self.test_dataset = torch.load("test_dataset.pt")
        temp_dict = torch.load("checkpoint_data.pt")
        self.unique_games = temp_dict["unique_games"]
        self.current_train_index = temp_dict["current_train_index"]
        self.current_test_index = temp_dict["current_test_index"]

    def __generate_train_dataset(self):
        while self.current_train_index < self.train_size:
            batch = self.generate_batch_of_games()
            deduplicated_batch = self.deduplicate_batch(batch)

            for game in deduplicated_batch:
                if self.current_train_index >= self.train_size:
                    break
                game_tensor = self.convert_game_to_tensor(game)
                self.train_dataset[self.current_train_index] = game_tensor.view(-1)
                self.current_train_index += 1
                if self.current_train_index % 100_000 == 0:
                    logging.info(f"Current train index {self.current_train_index}")

    def __generate_test_dataset(self):
        while self.current_test_index < self.test_size:
            batch = self.generate_batch_of_games()
            deduplicated_batch = self.deduplicate_batch(batch)

            for game in deduplicated_batch:
                if self.current_test_index >= self.test_size:
                    break
                game_tensor = self.convert_game_to_tensor(game)
                self.test_dataset[self.current_test_index] = game_tensor.view(-1)
                self.current_test_index += 1
                if self.current_test_index % 100_000 == 0:
                    logging.info(f"Current test index {self.current_test_index}")

    def generate_batch_of_games(self):
        logging.info(f"Generating batch of {self.BATCH_SIZE} games...")
        with Pool(self.NUM_OF_WORKERS) as p:
            batch = list(tqdm(p.imap(generate_single_game, range(self.BATCH_SIZE)),
                            total=self.BATCH_SIZE))
        logging.info(f"Finished!")

        return batch

    def deduplicate_batch(self, batch):
        logging.info(f"Deduplicating {len(batch)} games...")
        deduplicated_batch = []

        for game in batch:
            game_hash = game.get_hash()
            if game_hash not in self.unique_games:
                self.unique_games.add(game_hash)
                deduplicated_batch.append(game)

        logging.info(f"Finished deduplication, {len(deduplicated_batch)} unique games.")
        return deduplicated_batch

    def convert_game_to_tensor(self, game):
        game_history = game.get_game_history()
        output = torch.full((60, 1), self.padding_token)

        for index, i in enumerate(game_history):
            row, col = GameBoard.position_to_index(i[1])
            flat_index = row * 8 + col

            if flat_index >= 37:
                flat_index -= 4
            elif flat_index >= 29:
                flat_index -= 2

            flat_index += 1
            output[index] = flat_index

        return output

    def __save_checkpoint(self):
        torch.save(self.train_dataset, "train_dataset.pt")
        torch.save(self.test_dataset, "test_dataset.pt")
        torch.save({"current_train_index": self.current_train_index,
                    "current_test_index": self.current_test_index,
                    "unique_games": self.unique_games}, "checkpoint_data.pt")


if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate(False)