import sys
sys.path.insert(0, ".")
from src.data_utils import parse_file_data
data = parse_file_data("data/case14_uctest.m")
for g, d in data.gens.items():
    print(f"Gen {g}: c2={d['cost'][0]}, c1={d['cost'][1]}, c0={d['cost'][2]}")
