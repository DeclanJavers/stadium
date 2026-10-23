# Stadium Kit viewer and editor

A real-time viewer and live stadium editor (Rust, wgpu on Metal, egui).

- **Editor**: open a layout (`layouts/*.json`) and change anything — pitch size, grass type, cut
  height, mowing, wear; goals and flags; stands per side (profile, seat, length), open or curved
  corners; seat colour patterns and text mosaics; roofs and facades per side; floodlights; ads,
  carpets and LED ribbons; screens, dugouts and tunnel. Every change rebuilds the stadium in the
  background (a few hundred ms even for the 150k bowl) and the readouts update: capacity by seat
  style, sightlines (rake, C-values), roof clearances, floodlight design (fixtures, beam, lux,
  uniformity U1/U2), ad inventory with TV weighting, warnings. Save and open layouts as JSON (the
  same `stadium-layout/1` file the Blender pipeline reads).
- **Viewer**: open a game export (`out/stadiums/*.game.glb`) to see exactly what an engine
  receives from `tools/export_stadium.py --game`.

```bash
./build.sh --offline                                       # release build
target/release/stadium-viewer ../layouts/premier.json      # editor
target/release/stadium-viewer                              # editor with layouts/premier.json
target/release/stadium-viewer ../out/stadiums/premier.game.glb --night   # viewer
target/release/stadium-viewer ../layouts/mega.json --camera aerial --screenshot mega.png   # headless
target/release/stadium-viewer ../layouts/classic.json --ui --screenshot ui.png             # with the editor UI
```

Options: `--night`, `--camera broadcast|aerial|pitch|goal|grass|goalmouth`, `--screenshot out.png`
(headless), `--ui` (draw the editor panels into the screenshot), `--select <part>` (draw the
selection highlight, e.g. `Stand_north`), `--size 1600x900`,
`--exact-lights` (per-pixel floodlight loop instead of the irradiance volume), `--max-lights N`.
`STADIUM_SNAPSHOT=out.png` saves the live window once the stadium has been on screen a few seconds.

Interactive: drag to orbit, right-drag to pan, scroll to zoom, `1`–`6` camera presets, `N`
day/night. In the editor the left panel edits the layout, the right panel shows the readouts, the
top bar has File (new from template, open, save, save as), undo/redo, day/night, cameras and
exposure.

- **Undo/redo**: Cmd+Z / Cmd+Shift+Z (or the buttons). A slider drag is one step.
- **Click to select**: click a stand, roof, facade, the pitch, a goal, the boards, a screen, a
  dugout or a floodlight; its editor section opens and scrolls into view, and the object glows
  through whatever covers it. Esc clears the selection.
- **Image mosaics**: in Seat colours & mosaics switch a mosaic to Image and pick a PNG/JPG. The
  editor keeps a master copy and a version a chosen number of pixels tall in
  `layouts/mosaics/` (one pixel = a block of seats; transparent pixels keep the stand's colours) and
  shows how many rows the tier has. The Blender pipeline reads the same files.
- **Nets**: pattern (solid, bands, checker), two colours, band width, hex or square mesh, mesh
  size and cord thickness. The game's net card texture is regenerated from these.

## How the editor builds a stadium

`gen/` (crate `stadium-gen`) is a Rust port of the Python generators — sightlines, runs and bays,
stand sections, vomitories, seat anchors, rails, barriers, roofs (cantilever and goalpost with the
sightline/headroom solver), facades (six systems, ETFE cushions, gables), floodlight design (masts,
roof ribbons, poles, greedy aim optimisation, lamp power), seat patterns and mosaics, ad placement
and anamorphic carpets, pitch markings — plus the layout assembler (`stadium.rs`). Its output is
renderer-neutral: meshes with material names, instanced props, spots, sockets and a report.

`gen/tests/parity.rs` checks the port against the Python reference outputs
(`tools/dump_reference.py` writes `gen/tests/reference/*.json`): counts and indices exactly,
coordinates to ~1e-9, and whole-layout capacities equal to the Blender assembler's.

Materials and props come from the asset library, `out/library/library.game.glb`, exported by
`blender -b --factory-startup -P tools/export_library.py`: every material the generators reference
(game-ready: baked tileable PBR, `TINT_` detail textures with `tint_gain`) and every prop (seats at
LOD1 with LOD2, goals with net cards, corner flags, floodlight fixtures). Ad creatives and the
scoreboard come from `out/ads/`. Recolours use `tint_gain` (TINT_ materials) or the colour each
material was baked with (`base_hex`).

## Rendering

- Sun with an 8192² shadow map (alpha-tested, so nets cast their cord pattern), GGX + Lambert in
  the kit's units (irradiance in W/m², 683 lux per W/m²), ACES tone map.
- **Sky visibility**: at load, the scene is rendered into 24 depth maps from directions over the
  sky; a compute pass stores, for a ~1 m grid, the fraction of sky each point sees. Ambient and
  sky reflections use it, so the space under roofs and in the lower tiers is shaded as in reality
  instead of being lit by an open sky.
- **Grass** is shaded as a canopy, not a painted plane: field, clump and blade-scale colour
  variation (anti-aliased by pixel footprint), thatch that shows looking down and hides at grazing
  angles, wear and dry stress from the layout, view-dependent mowing stripes (docs/SPECS.md), an
  opposition hotspot, light through the blades, a faint waxy sheen instead of a specular sheet, and
  within ~15 m a parallax ray march through the blade layer. Line paint lets the grass show through.
- Depth pre-pass, then shading with depth EQUAL and no `discard`, so every pixel is shaded once.
- Alpha-tested textures get coverage-preserving mipmaps; net cards are blended, so cords fade to
  faint lines with distance instead of breaking up or vanishing.
- Seats: every frame each instance is assigned the fine mesh (< 45 m) or its coarse mesh; shadows
  always use the coarse mesh. Only the seat shell takes the per-seat colour.
- Floodlights at night: a compute shader bakes all spots into an irradiance volume (an "ambient
  cube" per ~1.5–2 m cell) once at load.

Measured on an M3 Pro at 1600×900 (4× MSAA): elite bowl (61k seats) about 30 ms by day and
20 ms at night.

## Build note

Some macOS Command Line Tools default to an SDK newer than their linker can read
(`tapi error: malformed file ... unknown architecture`). `build.sh` detects that and builds against
the SDK matching the running macOS instead (or set `SDKROOT` yourself for `cargo test`).
