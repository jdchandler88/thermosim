# WP04: Geometry (liquid grid, shell mesh, lid, probes, slice)

**Wave:** 1 · **Size:** L · **Depends on:** WP00. It reads `Scenario` fields
by name per CONTRACTS §5; until WP01 lands, use a local test-only factory that
builds a `Scenario`-shaped object (for example a `SimpleNamespace`).

## Goal

Turn a scenario into a `Domain`: the liquid voxel grid, the thin-shell wall and
base mesh with its coupling to the liquid, the lumped lid spec, the probes and
the slice specification. Correct areas and volumes here are what make energy
accounting and heating times right.

## Owns

- `src/thermosim/geometry/**` (`mesh.py`, `shell.py`, `lid.py`, `probes.py`,
  `slice.py`, `build.py`)
- `tests/geometry/**`

## Interfaces

Implement CONTRACTS §8 and produce the CONTRACTS §2 types. Get solid
properties from `physics.properties.solid_props`; until WP02 lands, tests may
monkeypatch it.

## Algorithm specification

Notation: R = inner radius, H_v = inner height, h = fill height, t_w = wall
thickness, t_b = base thickness, dx = cell size.

### Liquid grid (`mesh.py`)

- `nx = ny = ceil(2R / dx)`, `nz = ceil(h / dx)`. The origin is
  `(-nx·dx/2, -ny·dx/2, 0)`, so the grid is centered on the axis.
- A cell `[i,j,k]` is liquid if its center (x, y) satisfies x² + y² ≤ R², and
  `k < nz`.
- **Top layer:** f = (h − (nz−1)·dx)/dx, in (0, 1]. Liquid cells in layer
  nz−1 have volume f·dx³. Other liquid cells have dx³.
- **Footprint correction:** the cell-count footprint A_cells = N_col·dx² is
  not exactly πR². Scale *every* cell volume and every horizontal face area
  (`area_z`, `surface_area`) by s = πR²/A_cells, so the total liquid volume
  is exactly πR²h. Vertical face areas (`area_x`, `area_y`) are dx², or f·dx²
  in the top layer, *not* scaled by s (in-plane conduction between columns is
  unaffected). Document this.
- Face area arrays are zero wherever either neighbor isn't liquid.
- `surface_cells` are all liquid cells in layer nz−1. `surface_area` is
  dx²·s for each.

### Shell (`shell.py`)

**Base nodes:** one per liquid column (i, j), kind BASE.
- position: (x_c, y_c, −t_b/2); area: dx²·s; thickness: t_b.
- Coupling to liquid cell (i, j, 0) with `couple_area` = the node area.
- outer DOWN; `outer_area` = area × (π(R+t_w)²/πR²), so the base outer area
  includes the rim; `char_length` = 2(R+t_w)/4.
- In-plane edges join 4-neighbor columns, with conductance
  k·t_b·dx/dx = k·t_b.

**Side nodes:** a cylinder unrolled on an (nθ × nzs) grid, kind SIDE.
- nθ = max(12, round(2πR/dx)); nzs = ceil(H_v/dx), with the top row's height
  possibly partial.
- Node (a, b) sits at θ_a = (a+0.5)·2π/nθ, z_b at the row center, radius
  R + t_w/2.
- area = R·Δθ·Δz_b; thickness = t_w; outer SIDE;
  `outer_area` = (R+t_w)·Δθ·Δz_b; `char_length` = H_v + t_b.
- Edges:
  - around the circumference (periodic): conductance k·t_w·Δz/(R·Δθ);
  - vertically: k·t_w·R·Δθ/Δz.
  - Bottom ring to base: join each bottom-ring side node to the base node(s)
    whose column touches the outer staircase edge nearest to θ_a, with
    conductance k_avg·t_avg·(R·Δθ)/(distance between node centers). Each base
    perimeter column must connect to at least one side node.
- **Side coupling to liquid:**
  - For every liquid cell with a non-liquid x or y neighbor (a staircase
    face), compute the face center's θ and z, and map it to side node
    (a, b), where b is the row containing the face's z range. If a face
    straddles two rows, split it by overlap.
  - Raw face area: dx·(cell height in that layer).
  - Then, **per layer k**, rescale the couple areas so their sum equals
    2πR·(height of layer k). The true wetted wall area is then exact.
  - Then cap each node's total coupled area at its area. Spread any excess to
    θ-neighbors, or document why it can't happen.
- `wetted_area` = sum of couple areas per node; `dry_area` = area −
  wetted_area (≥ 0, with a tolerance of 1e-12).

**Materials and insulation:**
- `rho_c`, `k` and `emissivity` come from the vessel material.
- If insulation covers "side" or "base", those nodes get `insulation_r` =
  thickness/k_ins and `emissivity` = the insulation's.
- `outer_diameter` = 2(R + t_w + insulation thickness on the side, if any).

### Lid (`lid.py`)

- `present` = mode ≠ "none".
- `covered_fraction`: 1.0 for full, the configured value for partial.
- area = f·πR².
- heat capacity = ρ·cp·t·area of the lid material.
- `insulation_r` if covers includes "lid".
- headspace volume = πR²(H_v − h)·f.

### Probes (`probes.py`)

- Trilinear interpolation over the 8 cell centers surrounding the point.
- Weights on non-liquid cells are set to 0 and the rest renormalized.
- Raise `GeometryError` if no liquid cell is among the 8, or the point is
  outside the liquid.
- Near walls, interpolation degenerates to nearest-liquid, which is fine.
  Document it.

### Slice (`slice.py`)

- `j` = the index of the cell containing y = 0, using the cell centered at
  +dx/2 when nx is even.
- `wall_left`: side nodes with θ closest to π, every row, sorted by z.
- `wall_right`: θ closest to 0.
- `base_line`: base nodes with column j, sorted by x.
- `extract_slice` returns `T_liquid[:, j, :]` as float32, with NaN where the
  cell isn't liquid.

### `build_domain`

Calls the above, validates that the element and probes are inside the liquid
(element checks belong to WP05; probes here), and returns a `Domain`.

## Acceptance criteria

All with tests, at dx ∈ {0.004, 0.005, 0.01, 0.02} and several radius and fill
combinations:

1. Σ volume = πR²h, relative error under 1e-12.
2. Σ surface_area = πR²; Σ base wetted area = πR²; Σ side wetted area = 2πRh
   (all under 1e-9 relative).
3. Σ side area = 2πR·H_v; Σ side outer area = 2π(R+t_w)·H_v.
4. Shell graph: connected (one component); no self-loops; no duplicate edges;
   every conductance > 0.
5. Every liquid cell on the outer staircase (with a non-liquid neighbor) has at
   least one coupling row. Every base column has exactly one base coupling.
6. Probe interpolation reproduces a linear field a·x + b·y + c·z + d exactly
   (to 1e-12) at interior points.
7. `extract_slice` on a field T = z gives each column equal to `z_centers`.
8. Building the 50 L example at dx = 0.005 takes under 3 s and peaks below
   500 MB.
9. A `SolverError`-free smoke test is **not** required here (that's WP09).

## Notes

- Performance: use NumPy vectorization. Numba is allowed, but the build runs
  once per run.
- Add a debug helper, `geometry.debug.plot_domain(domain)` (matplotlib, an
  optional import), that draws the slice, the shell nodes and the couplings.
  It is invaluable for review. matplotlib is a dev-only dependency; ask the
  coordinator to add it.
