# Real-world specifications

Numbers the kit is built and tested against. `tests/run_tests.py` checks the ones marked ✓.

## Field of play (IFAB Laws of the Game, Law 1)

| item | value |
|---|---|
| Length (touchline) | 90–120 m; international 100–110 m. Default **105 m** ✓ |
| Width (goal line) | 45–90 m; international 64–75 m. Default **68 m** ✓ |
| Line width | ≤ 12 cm, all lines equal ✓ |
| Lines belong to the areas they bound | measurements are to the outer edge of the boundary lines |
| Centre circle | radius 9.15 m ✓ |
| Goal area | 5.5 m from the inside of each post, 5.5 m into the field |
| Penalty area | 16.5 m from the inside of each post, 16.5 m into the field ✓ |
| Penalty mark | 11 m from the midpoint between the posts ✓ |
| Penalty arc | radius 9.15 m from the penalty mark, outside the penalty area |
| Corner arc | radius 1 m |
| Optional marks | 9.15 m from the corner arc, off the field, at right angles to the lines |
| Corner flagpost | ≥ 1.5 m high, non-pointed top ✓ |

Run-off used by the kit: 6 m beyond the touchlines and 7.5 m beyond the goal lines (UEFA
recommends at least 5 m and 6 m including the technical areas).

## Goals (Law 1)

| item | value |
|---|---|
| Inner width | 7.32 m ✓ |
| Inner height (lower edge of crossbar) | 2.44 m ✓ |
| Post and crossbar width and depth | ≤ 12 cm, equal; same as the goal-line width ✓ |
| Shapes allowed | square, rectangular, round, elliptical |
| Placement | centred on the goal line; posts occupy the line |

Kit goal styles (`catalog/goals.json`):

- **box**: elite-level box net, 2.0 m deep at top and bottom, hung from two free-standing rear
  poles. The roof and back sag between supports.
- **classic**: bent stanchions; the net is 0.8 m deep at the crossbar and 2.0 m at the ground,
  with a ground frame.
- **portable**: grassroots aluminium goal with a welded rear frame (1.0 m top depth, 2.0 m ground
  depth), round 100 mm posts, sandbag weights.

Nets: 3–5 mm braided cord, 100–120 mm mesh, square or hexagonal.

## Grass

The pitch shader uses three effects, written here so any engine can reproduce them:

1. **Base colour** = per-type healthy colour × large/medium/fine noise variation, mixed towards
   straw then soil by the **wear mask** (Gaussians at both goalmouths, the centre spot, and the
   assistant referees' running lanes, broken up by noise, scaled by `wear`).
2. **Mowing stripes (natural grass) are view-dependent.** Each mowing pass leaves blades leaning
   in its direction `L` (a unit vector on the ground plane, given by the pattern). With `V` the
   unit vector from the camera to the surface point:
   `colour *= 1 + k · dot(V.xy, L)`, where `k = stripe_strength(type) · clamp(height / 25 mm, 0.4, 1.6)`.
   Grass leaning away from the viewer looks lighter. The stripes fade when seen from directly
   above (|V.xy| → 0) and reverse when you walk to the other side of the pitch, as on a real pitch.
3. **Artificial turf stripes are colour-based.** The two fibre shades are woven in, so
   `colour *= 1 + k · s` with `s = ±1` per strip, independent of the viewing direction.

Patterns give `L` per point: `stripes` (bands across the pitch, ±Y lean), `stripes_long`,
`checkerboard` (two passes: lean = normalised sum), `diagonal`, `diamond`, `circles`
(tangential lean, alternating per ring), `none`.

Wetness darkens the albedo by up to 30% and drops the roughness from about 0.85 to about 0.6
(mud drops to about 0.35).

### Why grass must not be shaded as a plane

A turf canopy is millions of blades that shadow and mask each other. Shading it as a smooth
dielectric sheet is what makes CG grass look like plastic: a sheet mirrors the bright horizon at
grazing angles (a whitish Fresnel haze across the far pitch) and shows no structure up close. The
kit's grass (Blender `kit/assets/pitch.py`, viewer `viewer/src/shader.wgsl`) therefore uses:

4. **Colour at four scales.** Field tone (≈30 m, ±8%), clumps (≈1.5 m, ±5%, leaning yellow- or
   blue-green), dry-stress patches, and blade scale: noise streaked along the mowing direction
   (≈4 mm across, ≈12 mm along, ±20%), bright tips, ~1% dead blades. Engines fade each octave to its
   mean once the pixel footprint exceeds about half its wavelength, so distant grass averages to
   the same colour instead of shimmering.
5. **Thatch between blades.** Where the blade noise is low, the colour falls towards dark thatch
   (sRGB ≈ #3a3420), weighted by how directly the surface is seen from above: looking straight
   down you see between the blades; across the pitch the blades hide it.
6. **Canopy lighting.** Specular is faint (IOR level ≈0.04 in Blender; the viewer scales GGX by
   0.25), a soft sheen carries the grazing look, ambient is reduced 12–27% for blade
   self-occlusion, an opposition hotspot brightens the canopy up to 45% when the sun is behind the
   viewer, light comes through the blades (a warm green) when looking towards the sun, and within
   ~15 m a parallax ray march through the blade layer shades blade bases darker than tips.
7. **Blades for close-ups.** Render-only blades are pointed, twisted, 0.7–1.3× the cut height,
   each a slightly different green, 40% translucent, with a deep-shaded root and a waxy
   (roughness 0.42) surface; their density fades with distance and towards the patch edges so they
   blend into the shaded surface without a seam.

### Other surfaces

Outdoor powder coat is satin and never lacquered (no clear coat), so painted steel uses roughness
0.45–0.55 plus weathering: dust on upward faces, UV chalking in patches, rain streaks down vertical
faces. Moulded polypropylene seats are spark-eroded satin (roughness ≈0.5) with chalking and dust
in the seat pan. Static perimeter boards have a back panel (printed sheet on a frame).

### Grass types (`catalog/grass.json`)

Gameplay values are **starting estimates** for the match engine to calibrate later. `ball_roll`
is relative ground-roll deceleration (1.0 = perennial ryegrass at 25 mm; soccer-v2's fitted
ground decay k = 0.43/s corresponds to roughly this). `bounce` is the relative coefficient of
restitution; `traction` is relative stud grip; `wear_rate` is the relative wear per match.

| type | climate | mow height | look |
|---|---|---|---|
| perennial_ryegrass | cool temperate (UK, N. Europe) | 22–30 mm | deep fresh green |
| kentucky_bluegrass | cool continental | 25–35 mm | blue-green, dense |
| hybrid_rye | cool temperate, elite (3–5% synthetic fibres) | 22–28 mm | rye look, wears far less |
| bermuda | hot (US South, Middle East, Brazil) | 12–20 mm | finer, yellow-green |
| seashore_paspalum | tropical, coastal, salt-tolerant | 15–25 mm | dark emerald |
| zoysia | warm humid (Japan, Korea) | 15–25 mm | dense, slightly olive |
| artificial_3g | any | 40–60 mm fibre | uniform, rubber infill, woven-in stripes |

## Stands (Green Guide, 6th ed., and FIFA stadium guidance)

| item | value used |
|---|---|
| Seated row depth (tread) | 0.80 m standard, 0.95 m hospitality; seat style sets a minimum ✓ |
| Seatway (clear, seat front to the row behind) | ≥ 305 mm ✓ |
| Seat pitch | ≥ 0.50 m standard, 0.56 m hospitality ✓ |
| Seats between aisles | ≤ 28 ✓ |
| Seated eye height | 1.15 m above tread, 0.30 m forward of the riser behind |
| Standing eye height | 1.60 m; terrace steps 0.38 m deep |
| C-value | target 90 mm (good), minimum 60 mm (acceptable); per profile ✓ |
| Maximum rake | 34 deg ✓ |
| Aisle step riser | ≤ 190 mm (intermediate steps inside each row) |
| Focal point | nearest touchline (goal line for end stands), at pitch level |
| Vomitory headroom | 2.2 m |
| Crush barriers | 1.1 m high, every fifth step, staggered bay to bay |
| Standing density | 1.8 people per metre of step (4.7/m² on 0.38 m steps) |
| Upper-tier headroom over the tier below | ≥ 2.4 m ✓ |

Capacities from the bundled profiles as full bowls on a 105 x 68 pitch: `single_tier_20`
~15k, `three_tier_bowl` ~61k, `mega_bowl` ~153k.

## Roofs

| item | value used |
|---|---|
| Leading edge | clears the back-row sightline to a ball 15 m above the far touchline (+0.3–0.5 m) ✓ |
| Headroom under the roof | ≥ 2.5 m over every covered tread ✓ |
| Cantilever length | ≤ 60 m (longer spans need arches or tension rings, not modelled) ✓ |
| Cantilever rise toward the pitch | 5–9 deg by preset |
| Truss depth | span/7 at the column, span/40 at the tip |

## Floodlighting (UEFA Stadium Infrastructure Regulations; EN 12193)

| item | value used |
|---|---|
| Elite broadcast (UEFA level A) | ~2000 lux horizontal avg, U1 = Emin/Emax ≥ 0.7, U2 = Emin/Eavg ≥ 0.8 ✓ |
| Classic televised ground | 1000 lux (preset), same uniformity ✓ |
| Grassroots (EN 12193 class III) | 75–200 lux, U2 ≥ 0.5 ✓ |
| Corner masts | lamp line ≥ 25 deg above the centre spot ✓; in the gap between stands at open corners, behind the corner stand (≥ 8 m past its back) in a bowl ✓ |
| Roof ribbon | only on roof edges ≥ 10 m up (else corner pylons, with a warning) ✓; fixtures hang from a rail under the edge |
| Aiming | no fixture aims back over its own shoulder (its aim point is in front of it, seen from the pitch's long axis); aims are dealt across the pitch, then re-aimed greedily for uniformity ✓ |
| Poles | on the main-stand side, none within 15 m of halfway (dugouts, tunnel, TV camera) ✓ |
| Colour temperature | LED 5700 K, metal halide 4300 K |
| Blender light units | 1 W/m² ≈ 683 lux; spot power P emits P/4π W/sr inside its cone ✓ (unit test) |

## Pitch-side furniture and advertising

| item | value used |
|---|---|
| Perimeter boards | 0.9–1.0 m high, 4.0 m behind the touchline, 5.5 m behind the goal line; with 3D carpets the end boards move back (typically 7.75–8.75 m, never closer than 0.8 m to the stand front) so each carpet gets a 1 m virtual board ✓; a manual end offset is kept between 3 m (behind the nets) and 0.8 m short of the stand front ✓ |
| LED board face | tilted ~10° back; one LED per texel (128 px per metre of height) |
| 3D carpet | projection of a ~1 m virtual board from the main camera, beside the goal; the whole footprint stays off the field of play (≥ 0.3 m behind the goal line, inside the touchline extended), clear of the goal (no corner within 2.5 m behind the line and 4.26 m of the centre line) and ≥ 0.4 m clear of the end boards; a carpet squeezed under 0.35 m is left out with a warning ✓ |
| Dugouts | front 4.3 m behind the touchline (clear of the run-off), closer (≥ 1.5 m) when the main stand is near, always in front of it ✓; 2 rows |
| Technical area | 1 m beyond the seating either side, to within 1 m of the touchline (dashed) |
| Tunnel | 3.6 m wide, 2.7 m high, at halfway between the dugouts |
| Centre-hung screen | bottom ≥ 18 m (~20–30 m) above the pitch; roofs too low (or none) → screens at the ends, with a warning ✓ |
| End screens | hung below the middle of the roof edge with ≥ 6 m clear below; otherwise on a frame on top of the end stand (above its roof), or freestanding with no stand; frames get a leg every ~8 m ✓ |
| Main TV camera | front of the main-stand tier nearest 16 m, at least 12 m up: a lower tier front gets a gantry slung under the roof edge, or a platform on scaffold over the back rows ✓ |

## Robustness

| item | rule |
|---|---|
| Corner radius | clamped so the corner stands stay ≥ 2 m outside the field corners; the run-off turf and apron stop at the bowl front ✓ |
| Curved corners | need four sides of one profile; otherwise open corners, with a warning ✓ |
| Goalpost roofs | front girder ≤ 25 m; over taller stands a cantilever, with a warning ✓ |
| Seat rows | treads widened to the seat's `min_tread` (305 mm seatway) ✓ |
| Mosaic text | wrapped onto more lines when the tier has the rows, else shortened, with a warning; accents drawn as plain letters, missing glyphs left blank with a warning ✓ |
| Unknown ids and colours | goal style, flag preset, grass, mowing, net pattern/mesh, profile, seat, roof/facade/floodlight preset, brand, screen placement and non-`#rrggbb` colours fall back to a default with a warning (`kit/sanitize.py`, `viewer/gen/src/sanitize.rs`) ✓ |
| Fuzzing | `viewer/gen/tests/fuzz.rs` builds seeded random layouts across the editor's ranges (and some hand-edited mistakes) and checks: nothing buried or on the field of play, screens clear, furniture in front of the stands, carpets clear of goals and boards, floodlights high and aimed down, masts clear of stands, camera on its gantry (`FUZZ_N=1000 cargo test --release -p stadium-gen --test fuzz`) |
