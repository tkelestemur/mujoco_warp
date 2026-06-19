# Contrib

Contrib is a home for experiments, helper scripts and other tools that are not officially part of MJWarp.

The contents of this directory are subject to change with no notice.

## Per-World Texture Randomization

`per_world_texture_randomization_viser.py` shows four parallel environments in
Viser with a Panda arm, a table, and live Warp renders. Every 50 physics steps
the table material texture is randomized independently for each world.

Run from the repository root:

```bash
uv run python contrib/per_world_texture_randomization_viser.py
```

The example uses the per-field model batch-size API to allocate one
`mat_texid` row per world:

```python
m = mjw.put_model(mjm, batch_sizes={"mat_texid": num_envs})
```

The texture images are compiled into the MuJoCo model once from
`contrib/assets/textures/table/manifest.json`. At reset/randomization time the
demo only updates `m.mat_texid[:, table_mat_id, rgb_role]`, so texture selection
changes without recompiling the model.
