import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from pathlib import Path


def gif_from_pngs(png_dir, prefix, max_frames=10, interval=400, repeat_delay=100, rm_pngs=True):
    '''
    Function takes a pathlib.Path dirctory which has .png files
    creates a gif and saves it to the same directory.
    rm_pngs will rm all the unused png to reduce clutter.
    '''
    pngs = sorted(png_dir.glob(f'{prefix}*.png'))
    N = len(pngs)
    if N < 2:
        return
    hop = max(1, len(pngs)//max_frames)
    png_paths_for_gif = pngs[:-1][::hop][:max_frames] + pngs[-1:]  # make sure we catch last
    image_array = [Image.open(f) for f in png_paths_for_gif]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    im = ax.imshow(image_array[0], animated=True)  # Set init image

    def gif_update(i):
        im.set_array(image_array[i])
        return im,
    # Create the animation object
    animation_fig = animation.FuncAnimation(
        fig, gif_update, frames=len(image_array),
        interval=interval, blit=True, repeat_delay=repeat_delay)
    gif_path = png_dir / f'{prefix}.gif'
    animation_fig.save(gif_path, writer='pillow')
    plt.close()
    if rm_pngs:
        for png in pngs:
            if png not in png_paths_for_gif:
                png.unlink()
    return gif_path


def plot_test_pred(test, pred, n_samples, names=None, epoch=None, 
                   show=False, out_path=None):
    n_samples = min(len(test), len(pred), n_samples)

    fig, axs = plt.subplots(nrows=n_samples, ncols=2,
                        sharex=True, sharey=True, figsize=(12, n_samples))
    if epoch is not None:
        fig.suptitle(f'epoch {epoch}')
    axs[0, 0].set_ylim([-1, 1])
    axs[0, 0].set_title('Test')
    axs[0, 1].set_title('Pred')
    
    for i in range(n_samples):
        axs[i, 0].plot(test[i])
        axs[i, 1].plot(pred[i])

        if names:
            axs[i, 0].set_title(names[i])

    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
