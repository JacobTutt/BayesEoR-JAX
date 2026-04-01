# Image-Domain FG Migration Log

This note records the implementation steps, commits, and validation results for
the optional image-domain non-monopole foreground parameterization.

## Scope

Goal:

- add an option to build the non-monopole foreground `[eta0 | LSSM]` block in
  image space,
- keep `T_eor` unchanged,
- keep the foreground monopole block `T_mp` unchanged in meaning,
- preserve the old UV-domain behavior as the default.

## Implementation Record

### Step 1: parser / setup / layout plumbing

Added:

- parser flag
  - `--fg-non-monopole-parameterization {uv,image}`
- setup argument propagation
- `args.json` recording of the selected parameterization
- `BuildMatrices` state for the new option
- parameterization-dependent non-monopole FG column counts

### Step 2: direct and stepwise `T` construction

Added:

- direct-build branch for UV vs image-domain non-monopole FG geometry
- stepwise writer branch for UV chunks vs image-pixel chunks
- cheap monopole construction from the constant-image template in image mode

## Test Record

1. `python3 -m py_compile`
   - files:
     - `bayeseor_jax/params.py`
     - `bayeseor_jax/setup.py`
     - `bayeseor_jax/build_matricies/build_utils.py`
     - `bayeseor_jax/build_matricies/overall.py`
     - `scripts/test_stepwise_t.py`
     - `scripts/benchmark_direct_t_stack.py`
   - result: passed

Planned next checks:

2. UV-mode regression:
   - direct vs stepwise `T`
3. image-mode consistency:
   - direct vs stepwise `T`
4. block invariance:
   - EoR block unchanged between UV and image modes
   - FG monopole block unchanged between UV and image modes

## Results

- compile-only validation passed for all modified files listed above
