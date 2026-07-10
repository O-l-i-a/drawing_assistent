try:
    from ..match_shapes import template_list, template_names
except ImportError:
    from match_shapes import template_list, template_names


def get_figure_templates() -> list[dict]:
    """Build the sidebar's figure list from match_shapes.py's templates.

    Adding a new shape to `template_list`/`template_names` in match_shapes.py
    makes it show up here automatically.
    """
    return [
        {"name": name, "points": points}
        for name, points in zip(template_names, template_list)
    ]
