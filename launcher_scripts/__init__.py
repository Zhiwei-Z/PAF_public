import sys
from os.path import dirname, abspath, join

PROJECT_PATH = dirname(dirname(abspath(__file__)))
print("appending project path to front: ", PROJECT_PATH)
sys.path.insert(0, PROJECT_PATH)

DATA_PATH = join(dirname(PROJECT_PATH), "paf-data", "")
