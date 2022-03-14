import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def xy_to_wh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1 
    h = y2 - y1
  
    return w, h

def plot_img_bbox(ax, image, target, channel_after = True, show_labels = True):
    img = image.detach()
    if not channel_after:
        img = img.permute(1, 2, 0)
    ax.imshow(img)

    label_enc = {
        1 : ['with_mask', 'red'],
        2 : ['without_mask', 'blue'],
        3 : ['mask_weared_incorrect', 'green'],
    }

    bboxes = target['boxes']
    labels = target['labels']
    for ind, bbox in enumerate(bboxes):
        w, h = xy_to_wh(bbox)
        rect = Rectangle((bbox[0], bbox[1]), w, h, linewidth = 1, edgecolor = label_enc[labels[ind].item()][1], facecolor='none')
        if show_labels:
            ax.text(bbox[0] + 0.05, bbox[1] + 0.05, label_enc[labels[ind].item()][0], fontsize = 'medium', backgroundcolor = label_enc[labels[ind].item()][1])
        ax.add_patch(rect)
    ax.axis(False);

def get_row(num, col):
    if num % col == 0:
        return num // col
    return (num // col) + 1

def show_img(loader, num, col = 5, channel_after = True, show_labels = True):
    image, target = next(iter(loader))
    num = min(len(image), num)
    row = get_row(num, col)

    fig, ax = plt.subplots(row, col, figsize = (col * 4, row * 4))

    ind = 0
    for i in range(row):
        for j in range(col):
            plot_img_bbox(ax[i, j], image[ind], target[ind], channel_after, show_labels)
            ind += 1


def compute_mean_std(loader, channel_after = True):
    channel_sum, channel_sq_sum, num_batches = 0, 0, 0
    if channel_after:
        dim = [0, 1, 2]
    else:
        dim = [0, 2, 3]

    for x, _ in loader:
        x = torch.stack(x)
        channel_sum    = torch.mean(x, dim = dim)
        channel_sq_sum = torch.mean(x**2, dim = dim)
        num_batches   += 1

    mean = channel_sum / num_batches
    std  = torch.sqrt( channel_sq_sum / num_batches - mean**2 )

    return mean, std


def plot_mean_area(loader):
    areas = []
    for _, y in loader:
        for a in y:
            for i in a['area'].flatten():
                areas.append(i)

    areas = np.array(areas)
    try:
        plt.figure(figsize = (12, 6))
        sns.distplot(areas)
        plt.axvline(areas.mean(), linewidth = 3.5, color = 'red', linestyle = '--')
        plt.title(areas.mean())
        plt.grid(False)
    except:
        print(areas)