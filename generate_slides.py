"""Generate media for slides (requires Python >= 3.10)."""

from pathlib import Path
import glob
import re

from data_morph.data.loader import DataLoader
from data_morph.morpher import DataMorpher
from data_morph.plotting.style import style_context
from data_morph.shapes.factory import ShapeFactory
from matplotlib.animation import FuncAnimation
from matplotlib import ticker
from nbconvert import exporters
from scipy.stats import moment
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytweening


SLIDES_DIR = Path(__file__).parent
DATA_DIR = SLIDES_DIR / 'data'
MEDIA_DIR = SLIDES_DIR / 'media'

def get_data():
    """Get the three datasets for the talk."""
    logo = DataLoader.load_dataset('Python')
    slant_up = DataLoader.load_dataset(DATA_DIR / 'slant_up.csv')
    heart = DataLoader.load_dataset(DATA_DIR / 'heart.csv')
    return logo, slant_up, heart

def truncate(stat):
    """Truncate statistic to 2 decimal points."""
    return np.trunc(stat * 100) / 100


def tweening(frame, iterations, min_value, max_value):
    """Determine the next value with tweening."""
    return (max_value - min_value) * pytweening.easeInOutQuad(
        (iterations - frame) / iterations
    ) + min_value

def generate_example_datasets_plot(datasets):
    """Generate plot of three datasets."""

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), layout='constrained')
    for ax, dataset in zip(axs, datasets):
        dataset.plot(ax=ax, show_bounds=False, title=None)
    plt.savefig(
        MEDIA_DIR / 'example_datasets.png', facecolor='white', bbox_inches='tight'
    )
    plt.close(fig)

def generate_stats_static(datasets):
    """Generate the static version of the summary statistics comparison."""
    fig, axs = plt.subplots(1, 3, figsize=(9, 4))
    for ax, dataset in zip(axs, datasets):
        dataset.plot(ax=ax, show_bounds=False, title=None)
        stats = (dataset.df.describe() * 100).astype(int) / 100
        ax.text(
            0.5, -0.2, f'X mean : {stats.loc["mean", "x"]:>+6}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.5, -0.3, f'Y mean : {stats.loc["mean", "y"]:>+6}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.5, -0.4, f'X stdev: {stats.loc["std", "x"]:>+6}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.5, -0.5, f'Y stdev: {stats.loc["std", "y"]:>+6}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.5, -0.6,
            f'Corr.  : {np.trunc(dataset.df.corr().loc["x", "y"] * 100) / 100:>+6.2f}',
            transform=ax.transAxes, ha='center'
        )
    fig.tight_layout(rect=(0, -0.14, 1, 1))
    fig.savefig(MEDIA_DIR / 'stats.png', facecolor='white', bbox_inches='tight')
    plt.close(fig)

def generate_stats_animation(datasets):
    """Generate the summary statistics animation."""

    def draw(frame):
        for ax, dataset, stat in zip(axs, datasets, stats):
            match frame:
                case 0:
                    dataset.plot(ax=ax, show_bounds=False, title=None)
                case 1:
                    stat[frame - 1].set_text(
                        f'X mean : {truncate(dataset.df.x.mean()):>+6}',
                    )
                case 2:
                    stat[frame - 1].set_text(
                        f'Y mean : {truncate(dataset.df.y.mean()):>+6}',
                    )
                case 3:
                    stat[frame - 1].set_text(
                        f'X stdev: {truncate(dataset.df.x.std()):>+6}',
                    )
                case 4:
                    stat[frame - 1].set_text(
                        f'Y stdev: {truncate(dataset.df.y.std()):>+6}',
                    )
                case 5:
                    stat[frame - 1].set_text(
                        f'Corr.  : {truncate(dataset.df.corr().loc["x", "y"]):>+6.2f}',
                    )

    fig, axs = plt.subplots(1, 3, figsize=(9, 4))
    stats = [
        [
            ax.text(0.5, y, ' ', transform=ax.transAxes, ha='center')
            for y in np.linspace(-0.2, -0.6, num=5)
        ] for ax in axs
    ]
    fig.tight_layout(rect=(0, -0.14, 1, 1))
    ani = FuncAnimation(
        fig,
        draw,
        frames=range(1, 6),
        init_func=lambda: draw(0),
    )
    ani.save(MEDIA_DIR / 'stats.gif', writer='pillow', fps=0.5)
    plt.close(fig)

def generate_moments_animation(datasets):
    """Generate animation of the moments."""

    def draw(frame):
        for ax, dataset, stat in zip(axs, datasets, stats):
            match frame:
                case 0:
                    dataset.plot(ax=ax, show_bounds=False, title=None)
                case 1:
                    stat[frame - 1].set_text(
                        f'2nd moment in X: {np.trunc(moment(dataset.df.x, moment=2)):>+6.0f}',
                    )
                case 2:
                    stat[frame - 1].set_text(
                        f'2nd moment in Y: {np.trunc(moment(dataset.df.y, moment=2)):>+6.0f}',
                    )
                case 3:
                    stat[frame - 1].set_text(
                        f'3rd moment in X: {np.trunc(moment(dataset.df.x, moment=3)):>+6.0f}',
                    )
                case 4:
                    stat[frame - 1].set_text(
                        f'3rd moment in Y: {np.trunc(moment(dataset.df.y, moment=3)):>+6.0f}',
                    )

    fig, axs = plt.subplots(1, 3, figsize=(9, 4))
    stats = [
        [
            ax.text(0.5, y, ' ', transform=ax.transAxes, ha='center')
            for y in np.linspace(-0.25, -0.55, num=4)
        ] for ax in axs
    ]
    fig.tight_layout(rect=(0, -0.1, 1, 1))
    ani = FuncAnimation(
        fig,
        draw,
        frames=range(1, 5),
        init_func=lambda: draw(0),
    )
    ani.save(MEDIA_DIR / 'moments.gif', writer='pillow', fps=0.5)
    plt.close(fig)

def generate_marginals_plot(datasets):
    """Generate the scatter plots with marginal distributions."""
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.5), sharex=True, sharey=True)
    for ax, dataset in zip(axs, datasets):
        ax.set_aspect(1)
        ax.scatter(dataset.df.x, dataset.df.y, s=1, color='black')
        x_hist = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
        x_hist.hist(dataset.df.x, ec='black', bins=15, color='slategray')
        y_hist = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
        y_hist.hist(dataset.df.y, orientation='horizontal', ec='black', bins=15, color='slategray')
        for marginal_ax in [x_hist, y_hist]:
            marginal_ax.xaxis.set_visible(False)
            marginal_ax.yaxis.set_visible(False)
        ax.text(
            0.66, -0.3,
            f'X skewness: {moment(dataset.df.x, moment=3):>+9,.2f}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.66, -0.4,
            f'Y skewness: {moment(dataset.df.y, moment=3):>+9,.2f}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.66, -0.5,
            f'X kurtosis: {moment(dataset.df.x, moment=4):.4g}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.66, -0.6,
            f'Y kurtosis: {moment(dataset.df.y, moment=4):.4g}',
            transform=ax.transAxes, ha='center'
        )
    fig.tight_layout(rect=(0, -0.2, 1, 1))
    fig.savefig(
        MEDIA_DIR / 'with_marginals.png', facecolor='white', bbox_inches='tight'
    )
    plt.close(fig)

def anscombes_quartet():
    """Plot Anscombe's Quartet along with summary statistics."""

    # get data
    anscombe = pd.read_csv(
        'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/anscombe.csv'
    ).groupby('dataset')

    # define subplots and titles
    fig, axes = plt.subplots(1, 4, figsize=(9, 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (group_name, group_data) in zip(axes, anscombe):
        ax.set_aspect(1)

        x, y = group_data.x, group_data.y
        ax.scatter(x, y, s=5, color='k')
        ax.set_title(group_name)

        # plot the regression line
        m, b = np.polyfit(x, y, 1)
        reg_x = np.append([0, 20], x)
        reg_y = [m*num + b for num in reg_x]
        ax.plot(reg_x, reg_y, 'r--', linewidth=0.5)

        # annotate the summary statistics
        ax.text(
            0.5, -0.35, f'X mean : {x.mean():>+6.2f}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.5, -0.5, f'Y mean : {y.mean():>+6.2f}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.5, -0.65, f'X stdev: {x.std():>+6.2f}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.5, -0.8, f'Y stdev: {y.std():>+6.2f}',
            transform=ax.transAxes, ha='center'
        )
        ax.text(
            0.5, -0.95,
            f'Corr.  : {np.corrcoef(x, y)[0, 1]:>+6.2f}',
            transform=ax.transAxes, ha='center'
        )

    # give the plots a title
    fig.tight_layout(rect=(0, 0, 1, 1))
    fig.savefig(MEDIA_DIR / 'anscombe.png', facecolor='white', bbox_inches='tight')
    plt.close(fig)

def generate_bounds_example(dataset):
    """Generate plot showing bounds."""
    _ = dataset.plot().set_title('')
    plt.savefig(MEDIA_DIR / 'bounds.png', facecolor='white', bbox_inches='tight')

def generate_shape_fitting_example():
    """Generate example of how shapes are calculated based on the dataset."""
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    for dataset_name, plot_row in zip(['Python', 'music'], axs):
        dataset = DataLoader.load_dataset(dataset_name)
        factory = ShapeFactory(dataset)
        for ax, shape in zip(plot_row, ['heart', 'slant_up', 'star']):
            ax.scatter(dataset.df.x, dataset.df.y, s=1, color='black', alpha=0.1)
            factory.generate_shape(shape).plot(ax=ax)
    fig.tight_layout()
    fig.savefig(
        MEDIA_DIR / 'fitting_shapes.png', facecolor='white', bbox_inches='tight'
    )
    plt.close(fig)

def generate_builtin_listings(dataset):
    """Generate plots with all built-in shapes and datasets."""
    _ = DataLoader.plot_available_datasets()
    plt.savefig(
        MEDIA_DIR / 'available_datasets.png', facecolor='white', bbox_inches='tight'
    )
    plt.close()

    _ = ShapeFactory(dataset).plot_available_shapes()
    plt.savefig(
        MEDIA_DIR / 'available_shapes.png', facecolor='white', bbox_inches='tight'
    )
    plt.close()

def generate_bald_spot_example():
    """Generate example of the "bald spot" limitation."""
    dataset = DataLoader.load_dataset('dino')
    factory = ShapeFactory(dataset)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    shape = factory.generate_shape('heart')

    for ax, shape_alpha, data_alpha in zip(axs, [1, 0.1], [0.2, 0]):
        ax.scatter(dataset.df.x, dataset.df.y, s=1, alpha=data_alpha)
        shape._alpha = shape_alpha
        shape.plot(ax=ax)
    axs[0].set_title('target')

    actual = pd.read_csv(DATA_DIR / 'dino-to-heart.csv')
    axs[1].scatter(actual.x, actual.y, s=5, color='black', alpha=0.9)
    axs[1].set_title('actual')
    fig.tight_layout()
    fig.savefig(
        MEDIA_DIR / 'bald_spots.png', facecolor='white', bbox_inches='tight'
    )
    plt.close(fig)

def generate_scale_example():
    """Generate example for scale effect."""
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    for ax, scale_reduction in zip(axs, [2, 1, 0.5]):
        dataset = DataLoader.load_dataset('dino', scale=scale_reduction)
        ax.scatter(dataset.df.x, dataset.df.y, s=1, color='black')
        scale = 1/scale_reduction
        ax.set_title(f'{scale:{".1" if scale < 1 else ".0"}f}:1 size')
    fig.tight_layout()
    fig.savefig(
        MEDIA_DIR / 'scale.png', facecolor='white', bbox_inches='tight'
    )
    plt.close(fig)

def generate_easing_example(name, min_value, max_value):
    """Show how value is eased over iterations."""
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

    for iterations, ax in zip([10_000, 50_000, 100_000], axs):
        x = np.arange(iterations)
        ax.plot(x, [tweening(i, iterations, min_value, max_value) for i in x], '-k')
        ax.xaxis.set_major_formatter(ticker.EngFormatter())
        ax.set(xlabel='iteration', ylim=(0, None))
        ax.axhline(min_value, linestyle='dashed', color='black', alpha=0.2)

    axs[0].set_ylabel(name)
    fig.tight_layout()
    fig.savefig(
        MEDIA_DIR / f'{name.replace(" ", "_")}_over_time.png',
        facecolor='white', bbox_inches='tight'
    )
    plt.close(fig)

def generate_simulated_annealing_animation():
    """Generate an animation showing the exact point that moves."""
    dataset = DataLoader.load_dataset('Python')
    target_shape = ShapeFactory(dataset).generate_shape('heart')

    morpher = DataMorpher(
        decimals=2,
        in_notebook=False,
        write_images=False,
        write_data=True,
        output_dir=DATA_DIR,
        seed=1,
    )
    _ = morpher.morph(dataset, target_shape, iterations=100, max_temp=1)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    def reframe(axs):
        for ax in axs:
            ax.set(
                xlim=dataset.plot_bounds.x_bounds, ylim=dataset.plot_bounds.y_bounds
            )
        axs[1].scatter(*target_shape.points.T, s=5, color='k', alpha=0.5)

    alphas = [0.2, 0.1]
    for ax, alpha in zip(axs, alphas):
        ax.set_aspect(1)
        ax.scatter(dataset.df.x, dataset.df.y, s=5, color='black', alpha=alpha)
    reframe(axs)

    iteration_text = fig.text(0.5, 0.97, '', va='center', ha='center')

    fig.tight_layout()

    def update(frame):
        i = frame // 2

        iteration_text.set_text(f'Iteration {i + 1:>2}')

        current = pd.read_csv(DATA_DIR / f'Python-to-heart-data-{i+1:>03}.csv')
        previous = (
            pd.read_csv(DATA_DIR / f'Python-to-heart-data-{i:>03}.csv')
            if i else dataset.df
        )

        idx = current.compare(previous).dropna().index
        unchanged = previous.drop(idx)
        for ax, alpha in zip(axs, alphas):
            ax.clear()
            ax.scatter(unchanged.x, unchanged.y, s=5, color='black', alpha=alpha)

            if frame % 2 == 0:  # show where change will happen
                ax.scatter(
                    previous.loc[idx].x, previous.loc[idx].y, s=20, color='blue',
                )
            else:  # move the point
                ax.scatter(
                    current.loc[idx].x, current.loc[idx].y, s=20, color='red'
                )

        reframe(axs)

    ani = FuncAnimation(fig, update, frames=38)
    ani.save(MEDIA_DIR / 'simulated_annealing.gif', writer='pillow', fps=2)

    plt.close(fig)

    for filepath in glob.glob(str(DATA_DIR / 'Python-to-heart-*.csv')):
        Path(filepath).unlink()

def compile_slides(title):
    """Compile the slides."""
    slides_notebook = nbformat.read(
        SLIDES_DIR / 'slides.ipynb', as_version=nbformat.NO_CONVERT
    )
    output, _ = exporters.export(exporters.SlidesExporter, slides_notebook)

    badges = """
    <div style="float: right; margin-top: -18px;">
        <a href="https://github.com/stefmolin/data-morph"
            style="z-index: 1;" target="_blank" rel="noopener noreferrer">
            <img src="https://img.shields.io/badge/view-repo-black?logo=github"
                alt="repo" style="max-width: 100%; margin: 20px 0 0 0;">
        </a>
        <a href="https://stefmolin.github.io/data-morph/stable/index.html"
            style="z-index: 1;" target="_blank" rel="noopener noreferrer">
            <img src="https://img.shields.io/badge/view-docs-orange?logo=github"
                alt="docs" style="max-width: 100%; margin: 20px 0 0 0;">
        </a>
    </div>
    """

    # footer = """
    # <aside style="display: block; position: absolute; left: 10px; bottom: 0px; font-size: 0.5em; width: 100%;">
    #     <h4 style="text-align: left;"><a href="https://tinyurl.com/data-morph-docs"
    #         style="z-index: 1;" target="_blank" rel="noopener noreferrer">
    #         tinyurl.com/data-morph-docs
    #     </a></h4>
    # </aside>
    # """
    footer = ''

    with open(SLIDES_DIR / 'index.html', 'w') as file:
        file.write(
            re.sub(
                '(<div class="reveal">)',
                rf'\1\n<div class="footer" style="padding: 4px; font-size: 22px;">{title}{badges}</div>\n{footer}',
                re.sub('(<title>).*(</title>)', rf'\1{title}\2', output)
            )
        )

def main():
    """Generate all media and compile the slides."""
    datasets = get_data()

    with style_context():
        generate_example_datasets_plot(datasets)
        generate_stats_static(datasets)
        generate_stats_animation(datasets)
        generate_moments_animation(datasets)
        generate_marginals_plot(datasets)

        generate_bounds_example(datasets[0])
        generate_shape_fitting_example()
        generate_bald_spot_example()
        generate_scale_example()
        generate_easing_example('maximum movement', 0.3, 1)
        generate_easing_example('temperature', 0, 0.4)
        generate_builtin_listings(datasets[0])
        generate_simulated_annealing_animation()

        anscombes_quartet()

    compile_slides(title='Data Morph: A Cautionary Tale of Summary Statistics &ndash; Stefanie Molin')

if __name__ == '__main__':
    main()
