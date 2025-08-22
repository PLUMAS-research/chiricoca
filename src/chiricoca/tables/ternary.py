import ternary


def ternary_scatter(df, scale=1, s=None, ax=None, fig_args=None):
    fig, tax = ternary.figure(scale=scale)

    tax.scatter(df.values, s=s, marker=".", color="magenta")
    tax.right_corner_label(df.columns[0])
    tax.top_corner_label(df.columns[1])
    tax.left_corner_label(df.columns[2])

    texts = []
    for idx, row in df.iterrows():
        tax.annotate(idx, row.values, fontsize="xx-small", va="center")

    tax.gridlines(multiple=0.1, color="blue")
    tax.ticks(
        axis="lbr", linewidth=1, multiple=0.1, tick_formats="%.2f", fontsize="x-small"
    )
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")

    return tax
