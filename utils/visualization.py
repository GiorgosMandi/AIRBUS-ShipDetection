import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from skimage.util import montage
from skimage.segmentation import mark_boundaries


def plot_masks(img, all_masks):
    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks)
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()


def multiplot_images(rows, columns, *images):
    total = rows*columns
    if len(images) != total:
        print("ERROR: Wrong number of rows and columns")
        return
    fig, axarr = plt.subplots(rows, columns, figsize=(15*rows, 15*columns))
    for i, img in enumerate(images):
        axarr[i].imshow(img)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()


def plot_ships_frequencies(df):
    count = df['ships'].tolist()
    count = Counter(count)

    plt.bar(count.keys(), count.values(), color='blue')
    plt.xlabel("#Ships")
    plt.ylabel("total")
    plt.title("Ship count distribution")
    plt.show()


montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)


def plot_as_montage(images, masks):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    batch_rgb = montage_rgb(images)
    batch_seg = montage(masks[:, :, :, 0])
    ax1.imshow(batch_rgb)
    ax1.set_title('Images')
    ax2.imshow(batch_seg)
    ax2.set_title('Segmentations')
    ax3.imshow(mark_boundaries(batch_rgb, batch_seg.astype(int)))
    ax3.set_title('Outlined Ships')

