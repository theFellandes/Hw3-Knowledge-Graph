import os

import yaml
from dataclasses import dataclass, field

CONFIG_PATH = "config.yaml"


@dataclass
class YamlReader:
    """Class to read configuration from a YAML file."""
    file_path: str = field(default=CONFIG_PATH)
    neo4j_config: dict = field(init=False)
    wikipedia_config: dict = field(init=False)
    huggingface_config: dict = field(init=False)

    def __enter__(self):
        # Resolve the absolute path of the config file
        absolute_path = os.path.abspath(self.file_path)
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"Configuration file not found at: {absolute_path}")

        with open(absolute_path, 'r') as file:
            self.config = yaml.safe_load(file)
        return self.config

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def get_config(self):
        try:
            with YamlReader(self.file_path) as config_reader:
                self.neo4j_config = config_reader['neo4j']
                self.wikipedia_config = config_reader['wikipedia']
                self.huggingface_config = config_reader['huggingface']

        except FileNotFoundError:
            print("File Not Found")
