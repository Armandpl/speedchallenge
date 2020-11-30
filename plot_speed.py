import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    with open("data/train.txt", "r") as f:
        lines = f.readlines()

    labels = []
    for l in lines:
        labels.append(float(l))

    print(labels)
    t = np.zeros(len(labels))
    labels =  np.array(labels , dtype=float)

    fig, ax = plt.subplots()

    ax.plot(labels)
    ax.set(xlabel='time', ylabel='speed (mph)',
       title='train.mp4')

    fig.savefig("speed.png")
    # plt.show()