```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv add --dev pre-commit && uv run pre-commit install
uv sync
uv run python reproduce/reproduce_parts_of_fig_2_and_add_more_models.py
uv run python reproduce/make_heatmaps.py
```
