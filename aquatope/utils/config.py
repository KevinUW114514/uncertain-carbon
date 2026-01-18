# import configparser
# from pathlib import Path

# config = configparser.ConfigParser()
# config_path = Path(__file__).parent.absolute() / "config.ini"
# config.read_file(open(config_path))


# Constants
NUM_RESOURCES = 2
CPU_MIN = 1000
CPU_MAX = 3000
MEMORY_MIN = 256
MEMORY_MAX = 512
CPU_UNIT_COST = 0.173  # Based on Azure Function
MEMORY_UNIT_COST = 0.0123  # Based on Azure Function

CPU_UNIT_POWER = 0.690
MEMORY_UNIT_POWER = 0.314
CPU_BASE_POWER = 247.3
MEMORY_BASE_POWER = 18.38

FISSION_HOST = "http://localhost:31314"