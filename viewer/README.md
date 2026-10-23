# Stadium Kit viewer

A real-time viewer (Rust, wgpu on Metal) for the kit's **game exports**. It is the engine-side proof
that what `tools/export_stadium.py --game` produces renders correctly outside Blender: baked PBR
materials, GPU-instanced seats with per-seat colours, MSFT_lod seat levels, net cards, floodlights,
and the mowing-stripe shader re-implemented from `docs/SPECS.md`.

```bash
./build.sh --offline                       # release build
target/release/stadium-viewer ../out/stadiums/premier.game.glb
target/release/stadium-viewer ../out/stadiums/premier.game.glb --night --screenshot night.png
```

Options: `--night`, `--camera broadcast|aerial|pitch|goal`, `--screenshot out.png` (headless),
`--size 1600x900`, `--exact-lights` (per-pixel floodlight loop instead of the irradiance volume),
`--max-lights N`.

Interactive: drag to orbit, right-drag to pan, scroll to zoom, `1`–`4` camera presets, `N` day/night.

## Rendering

- Sun with an 8192² shadow map (alpha-tested, so nets cast their cord pattern), sky ambient,
  GGX + Lambert in the kit's units (irradiance in W/m², 683 lux per W/m²), ACES tone map.
- Depth pre-pass, then shading with depth EQUAL and no `discard`, so every pixel is shaded once.
- Seats: every frame each instance is assigned the fine mesh (< 45 m) or its MSFT_lod coarse mesh;
  shadows always use the coarse mesh.
- Floodlights at night: a compute shader bakes all spots into an irradiance volume (an "ambient
  cube" per ~1.5–2 m cell) once at load; shading samples it instead of looping over every
  fixture. Spots are unshadowed either way, and the result matches the exact loop closely.
- The broadcast camera is the stadium's own `SOCKET:camera:main` gantry.

Measured on an M3 Pro at 1600×900 (4× MSAA): elite bowl (61k seats) about 45 ms by day and 31 ms at
night; 153k bowl about 81 ms by day.

## Build note

Some macOS Command Line Tools default to an SDK newer than their linker can read
(`tapi error: malformed file ... unknown architecture`). `build.sh` detects that and builds against
the SDK matching the running macOS instead.
