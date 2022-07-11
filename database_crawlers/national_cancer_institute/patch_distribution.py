import matplotlib.pyplot as plt

if __name__ == '__main__':
    res_patch_counts = []
    plt.hist([i[0] for i in res_patch_counts], bins=800)
    plt.xlabel("Patch per slide")
    plt.ylabel("Frequency")
    plt.savefig("patch_distribution.jpeg")
    plt.clf()

    plt.hist([i[0] / (i[1] + 0.00001) for i in res_patch_counts], bins=800)
    plt.xlabel("Patch per slide percent")
    plt.ylabel("Frequency")
    plt.savefig("patch_percent_distribution.jpeg")
    plt.clf()
