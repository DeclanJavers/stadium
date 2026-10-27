# Asset fixer brief (round 3, the last judged round)

You are improving procedurally generated 3D assets in /Users/declanjavers/Desktop/Code/stadium, a
game asset kit for UK football stadiums and training grounds. The owner's bar is "life-like".
Independent judges scored every asset 1-100 (60 = realistic model, 80 = one of the best game
assets ever = PASS, 100 = a photograph). Round 1: mean 39. Round 2 (after your first pass): mean 51,
best 63, nothing passed. Round 3 is the last judged round: act on the round-2 critiques for YOUR
assets, then the owner's rule is to fix the remaining suggestions and move on.

## Inputs
- Round-2 judge verdicts (fresh, independent judges): /Users/declanjavers/Desktop/Code/stadium/out/qa/judge_r2_b*.json
  (grep for your assets' names; read every suggestion and renderer note). Scores per asset:
  out/qa/scores_r2.md. Round-1 verdicts are out/qa/judge_r1_b*.json if you want history.
- Round-2 render of each asset (what the judges saw): .../scratchpad/review/<name>_r2.png.
- The viewer changed since round 2: real grass blades now grow round props (about 25-35 mm on the
  review lawn), the review ground is back at turf level (z = 0), the close-up of props taller than
  2.5 m frames their head, and the near clip plane scales down for small props. A part that should
  be visible must sit above the turf (an 8 mm agility ladder lost in grass scored 15).
- Real-world dimensions (manufacturer specs where found): .../scratchpad/dimensions.md. Use them.

## How the props are made (read before editing)
- kit/assets/props.py: the mesh toolkit `PM` (box/obox/cyl/lathe(with flutes, per-face mats)/loft/
  sweep/tube (bent tubes)/extrude/sphere/decal/card), `rrect`, `round_path`, `label()` (text
  signs as image materials), `tyre()`, `mesh_image()` (alpha mesh/net cards; give their quads UVs
  in metres), `clear_glass()`, shared materials `mats()`. Each face carries an edge radius
  (`p.r`, metres, set before adding parts): on `p.obj()` vertices are welded and hard edges
  bevelled with that radius (hardened normals). Use p.r = 0 for thin/alpha/tube parts.
- `p.decal(c, n, v, w, h, mat)`: n = outward face normal, v = up.
- Ambient occlusion is baked into vertex colours for every prop automatically (props.build).
- Local frame of every prop: origin on the ground at the footprint centre, +Z up, local +Y is the
  FRONT (where a bench faces, where a car points). Some round-1 assets violate this; fix them.
- Materials: kit/core/materials.py (painted_metal, plastic, rubber, aluminium ...),
  kit/assets/site_materials.py. IMPORTANT game-bake rule (kit/core/game.py `prepare`): a
  procedural node material on a small prop is reduced to ONE flat colour + roughness ("factor")
  unless its name is listed in a TILEABLE dict (game.TILEABLE / site_materials.TILEABLE, baked to
  1024 tileable textures using the mesh's metric box UVs) or it uses an image texture (kept as is).
  So for visible surface detail (wood grain, rust, dirt, decals, stencils, LED grids) either add
  the material to site_materials.TILEABLE with a sensible repeat size, or paint an image in
  numpy (see kit/assets/foliage_paint.py, props.label) and map it with explicit UVs.
  Metallic materials now bake their base colour correctly.
- Text: props.label uses a 5x7 pixel font; judges disliked it. If you need crisp text, write a
  better glyph rasteriser (e.g. a stroke/vector font defined in code, drawn anti-aliased at 2-4x)
  in a NEW module kit/assets/lettering.py and use it; do not break label()'s signature.
- Performance: these are instanced by the hundred. Rough budgets: small street/training props
  <= 6k tris, furniture/cabins <= 15k, vehicles <= 25k (plus a .lod2 if you change topology a
  lot), plant/machines <= 40k; textures <= 1024x1024. Prefer normal-free tricks (bevels, decals,
  AO, image textures) over dense geometry.

## Your loop
1. Edit only YOUR files (listed in your task). If you need a new shared helper, put it in your
   own module or a new module; do not edit props.py except to fix a clear bug, and then keep the
   change minimal (other fixers are editing in parallel).
2. Export just your props to your own partial library (about 20-60 s; it waits for a free Blender
   slot, max 3 run at once across all fixers):
   .../scratchpad/export_props.sh .../scratchpad/lib/<your_id>.glb <comma-separated key words>
   (key words match prop keys by substring, e.g. "bin_,bench,bollard").
3. Render a review sheet exactly as the judges see it:
   STADIUM_LIBRARY=.../scratchpad/lib/<your_id>.glb .../scratchpad/review.sh <full key> <tag>
   -> .../scratchpad/review/<name>_<tag>.png  (use tags w1, w2, ... while working)
   Look at it (Read the png) critically, as the judge would, against the critique. Iterate.
   review.sh allows two viewers at once machine-wide and waits for a slot; never start the viewer
   any other way in parallel loops (several at once preceded a machine restart that wiped /tmp).
4. When you believe an asset would score >= 80 (or you have done all you reasonably can), render
   the final sheet with tag r3.
5. Never run the full library export, never edit other fixers' modules, never commit. The viewer
   source (viewer/src/*) is the coordinator's this round: if a renderer change would help, say so
   in your report instead of editing it.
6. Keep your work recoverable: /private/tmp can be wiped by a restart. Your code lives in the repo
   (safe); keep throwaway scripts in the scratchpad.

## Report
Reply with one line per asset: what you changed and your honest self-estimate. Also list any
critique items you could not address and why.
