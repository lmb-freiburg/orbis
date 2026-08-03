# Backward compatibility wrapper pointing to stats.py
from LinearProbe.stats import generate_stats, main

if __name__ == "__main__":
    generate_stats(is_mc=False)
