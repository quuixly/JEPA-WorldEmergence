from multiprocessing import Pool
import torch
from tqdm import tqdm
from pathlib import Path

from game.othello import GameBoard


class DatasetGenerator:
    """
    This class generates a dataset (both training and test sets). It guarantees that each sequence in the test set is
    unique across both sets, and that each game is unique.
    Games are saved as uint8 tensors to reduce memory usage, and are generated using multiprocessing for efficiency.
    """
    def __init__(self, train_size = 20_000_000, test_size = 1_000_000, padding_token=0):
        self.GAME_MAX_LENGTH = 60
        self.train_size = train_size
        self.test_size = test_size
        self.padding_token = padding_token
        self.unique_games = set()
        self.unique_sequences = set()

        self.train_dataset = torch.full((self.train_size,  self.GAME_MAX_LENGTH), padding_token, dtype=torch.uint8)
        self.test_dataset = torch.full((self.test_size,  self.GAME_MAX_LENGTH), padding_token, dtype=torch.uint8)
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
        self.unique_sequences = temp_dict["unique_sequences"]
        self.current_train_index = temp_dict["current_train_index"]
        self.current_test_index = temp_dict["current_test_index"]

    def __generate_train_dataset(self):
        pass

    def __generate_test_dataset(self):
        pass

    def __generate_single_game(self):
        gameBoard = GameBoard()

    def __save_checkpoint(self):
        torch.save(self.train_dataset, "train_dataset.pt")
        torch.save(self.test_dataset, "test_dataset.pt")
        torch.save({"current_train_index": self.current_train_index,
                    "current_test_index": self.current_test_index,
                    "unique_games": self.unique_games,
                    "unique_sequences": self.unique_sequences}, "checkpoint_data.pt")


if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate(True)