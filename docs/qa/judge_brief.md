# Asset judging brief

You are an independent art director judging 3D assets for a game asset kit (a UK football
stadium and training-ground kit). The assets are procedurally generated and shown rendered in the
kit's real-time viewer (forward PBR, sun + sky ambient, shadow maps, no global illumination, no
screen-space reflections). Judge the ASSET (shape, proportions, scale, construction detail,
materials, surface wear, how believable it is as the real object), as it appears in these renders.
If something looks wrong only because of the renderer (lighting, sky, the lawn it stands on), say
so separately and do not mark the asset down for it.

## Scale (1-100)
- 100: indistinguishable from a photograph of the real thing
- 80: one of the best game assets ever made (AAA hero quality)  <- PASS mark
- 60: a realistic model: right shape and proportions, believable, but clearly CG
- 40: recognisable but simplified, toy-like or blocky
- 20: placeholder / greybox

Be strict and calibrated. Most procedural assets will land well below 80; say so honestly.

## For every asset below 80
Give a detailed critique and concrete, prioritised suggestions to raise it, that keep it cheap
enough to instance by the hundred in a game (state rough triangle or texture cost if relevant):
geometry (silhouette, proportions vs the real object, missing parts, bevels), materials (albedo
values, roughness, wear, decals, text), and anything that breaks realism. Be specific (which part,
what dimension, what to add) so a programmer generating the mesh in code can act on it.

## Real-world reference
Real dimensions are in
docs/qa/dimensions.md
(read it once). Use it to check proportions where it covers the asset.

## Images
Each asset has one sheet: four views in a 2x2 grid (three-quarter, opposite three-quarter, side,
close detail from above). Stadium sheets ("stadium_<layout>_overall" / "_detail") show a whole
generated stadium; judge these as separate assets: seating & stands, roof, exterior/facade,
floodlights, pitch & goals & nets, dugouts & tunnel, advertising boards & screens (one score each,
"stadium_<layout>:<part>").

## Output
Write a JSON array to the output file you are given, one object per asset:
{"asset": "<name>", "score": <int>, "pass": <bool>, "summary": "<one line>",
 "critique": "<detailed, for < 80>", "suggestions": ["<prioritised, concrete>", ...],
 "renderer_notes": "<optional>"}
Then reply with one line per asset: name, score, the single most important fix.
