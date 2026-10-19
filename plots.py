
from matplotlib import pyplot as plt
import os

def plot_abundance(ground_truth, estimated, em, save_dir):

    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        plt.subplot(2, em, i + 1)
        plt.imshow(ground_truth[:, :, i], cmap='jet',vmin=0, vmax=1)

    for i in range(em):
        plt.subplot(2, em, em + i + 1)
        plt.imshow(estimated[:, :, i], cmap='jet',vmin=0, vmax=1)
    plt.tight_layout()

    plt.savefig(save_dir + "abundance.png")






def plot_endmembers(target, pred, em, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    x_length = len(pred[:, 0])


    for i in range(em):

        plt.figure(figsize=(8, 6), dpi=150)


        plt.plot(pred[:, i], label='_nolegend_')
        plt.plot(target[:, i], label='_nolegend_')
        plt.yticks([0, 0.5, 1])

        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)

        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)


        plt.xlim(left=0)
        plt.ylim(bottom=0)


        plt.xlabel('Bands', fontsize=22)
        plt.ylabel('Normalized Reflectance', fontsize=22)


        plt.gca().set_aspect(x_length)


        save_path = os.path.join(save_dir, f"endmember_{i + 1}")
        plt.savefig(save_path + ".png", bbox_inches='tight', dpi=150, facecolor='white')
        print(f"已保存: {save_path}.png")

        plt.close()


