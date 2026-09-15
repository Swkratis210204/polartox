# Polarized Tree — `polartox.polarized_tree`

`polartox.polarized_tree` builds and inspects **one text's** polarized tree:
the recursive split structure that partitions a text's annotators by
demographic dimension to explain where its rating disagreement comes from.

This is the single-tree building block. Running the method over a whole
corpus (filtering which texts qualify, building a tree per text, aggregating
F/C/P summaries across many trees) is `polartox.pipeline`
(`PolarizedTreesPipeline`) — see [`polarized_trees/README.md`](../polarized_trees/README.md).
`PolarizedTreesPipeline` uses `PolarizedTree` internally, but `PolarizedTree`
can also be used directly on a single text's ratings without a pipeline at
all, which is what this folder demonstrates.

## Quickstart

```python
from polartox.polarized_tree import PolarizedTree

# text_data: one text's rows -- columns 'rating' plus whichever
# demographic dimensions you want to split on.
tree = PolarizedTree.build(
    text_data,
    dims=["gender", "politics", "age"],
    min_size=10,
    h=0.05,
    max_depth=4,
    scale=5,
    theta_stop=0.15,
    text_id=text_id,
)

tree.render()                              # ASCII tree, one line per node
tree.inspect(dataset, show_distributions=True)  # full walk + histograms
```

## API

- `PolarizedTree.build(data, dims, min_size, h, max_depth, scale, theta_pole=None, theta_stop=0.15, variant="beta", beta=1.0, relative_h=False, text_id=None)` — runs the splitting algorithm and returns a built tree.
- `get_root()` / `get_leaves()` — the raw node-dict structures.
- `n_leaves`, `depth` — quick size/shape properties.
- `internal_nodes()` — generator over every non-leaf node, `(depth, split_dim, prg, path, node)`.
- `find_node(path)` — look up the node reached by a sequence of `(dim, value)` splits.
- `node_ratings(dataset, path=())` — the raw ratings belonging to one node (root by default).
- `node_distribution(dataset, path=(), indent=0)` — print that node's rating histogram.
- `leaf_distributions(dataset)` — print the histogram for every leaf.
- `render()` — print the whole tree as ASCII, with each node's pole/p_tox.
- `inspect(dataset, show_distributions=False)` — full walk, optionally with a histogram at every node.

See [`polarized_tree_demo.ipynb`](polarized_tree_demo.ipynb) for a runnable walkthrough.
