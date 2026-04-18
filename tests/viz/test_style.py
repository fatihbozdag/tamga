"""Tests for publication-style defaults."""

import matplotlib.pyplot as plt

from tamga.viz.style import apply_publication_style, figure_size


def test_apply_publication_style_sets_dpi():
    apply_publication_style(dpi=300)
    fig = plt.figure()
    assert fig.get_dpi() == 300
    plt.close(fig)


def test_figure_size_single_column_is_3_5_inches():
    w, _ = figure_size("single")
    assert w == 3.5


def test_figure_size_double_column_is_7_inches():
    w, _ = figure_size("double")
    assert w == 7.0


def test_apply_publication_style_uses_colorblind_palette():
    apply_publication_style()
    import seaborn as sns

    palette = sns.color_palette()
    assert len(palette) >= 6
