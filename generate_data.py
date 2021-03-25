import numpy as np
import random

def main():
    with open('./data/data.txt', mode='w') as f:
        for i in range(100):
            s = []
            for _ in range(3):
                s.append(str(random.randint(0, 10)/10))
            f.write('\t'.join(s)+'\n')

    return

if __name__ == "__main__":
    main()