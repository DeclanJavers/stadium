# Asset library audit (2026-10-26)

Every prop in `out/library/library.game.glb` (254 props, not counting their coarse `.lod2` levels)
rendered in the editor from the same three-quarter view, 12 to a sheet, and graded by eye.
Grades: **good** (reads as the real thing), **ok** (fine at a distance or in a room, plain up
close), **weak** (looks wrong or unfinished), **bad** (disfigured; should not ship).

## Summary

| | coded (213) | generated (41) |
|---|---|---|
| good | 186 | 20 |
| ok | 22 | 13 |
| weak | 5 | 0 |
| bad | 0 | 8 |

Code is reliable: no coded prop is disfigured, and the weak ones are plain rather than broken.
Generation is reliable for organic and sculpted things (birds, the fox, busts, the legend and
manager statues, the pillar box, memorial stone, trophies) and unreliable for manufactured objects
with crisp edges, round parts or thin parts.

## Bad (fix or remove first)

All generated; eight more were already removed (water carrier, foam rollers, resistance bands,
bicycle, flags, matchday litter, kit bag, tactics easel; kept in `out/text_to_3d/rejected/`).

| prop | problem | fix |
|---|---|---|
| `genprop.gk_gloves` | open tubes, no fingers | code |
| `genprop.swiss_ball` | lumpy sphere, stray ring at the base | code (a sphere) |
| `genprop.plyo_boxes` | wrinkled where it should be crisp | code |
| `genprop.kit_hamper` | smeared shirts, wicker reads as slats | code basket + generated cloth, or drop |
| `genprop.framed_shirt` | smeared shirt | drop: the coded `prop.framed_shirt` is better |
| `genprop.medals_stand` | smeared ribbons | code |
| `genprop.trophy_cabinet` | blurry contents | drop: the coded `prop.trophy_cabinet` + generated trophies |
| `genprop.ball_bag` | yellow blotches on the balls | code (mesh bag + coded balls) |

## Weak (coded; give them detail)

- `treadmill`: a box with an arch; needs the belt, side rails, console and handrails to read.
- `dexa`: reads as a bench; needs the scanning arm over the table and the operator console.
- `extraction_hood`: a plain box from this angle; needs the visible baffle filters and the sloped front.
- `dumbbell_rack`: dumbbells are grey hexagon blobs; needs separate heads and handles.
- `washroom_mirror`: a beige slab; needs a reflective mirror material and a frame.
- `alter_g` (ok to weak): a white blob; needs the treadmill base and the pressure bag clearly shaped.

## Ok (fine in context, plain up close)

Coded: acoustic panels, agility ladder, bramble, display plinth, gondola (products read as
blocks), kitchen line, kitchen wall units, isokinetic, lectern, makeup station and mirror, grass
tuft, med curtain, phone booth, reception desk, shop counter, shirt rail (shirts read as one red
wall), sauna, sleep pod, studio desk, studio camera, TV camera, vanity basins, WC cubicles.

Generated: statue_mascot_lion (lumpy mane and back), cylinder mower, hose reel, wheelbarrow,
divot bucket, kettlebells, medicine balls, trophy cup (ribbon smear), match ball plinth, crest
sculpture, youth trophies set, bronze football (odd blue base), muddy ball, food van (smeared
windscreen).

## Other findings

- **Car variety is only number plates.** `car_hatch`, `car_saloon` and `car_suv` 1 to 5 share
  one body each; the colour comes per placement. Real variety needs different body shapes, trims
  and wheels.
- **Minibus 1 to 3** are likewise one body.

## Done (2026-10-26)

Steps 1 and 2: the sixteen disfigured or removed generated props are out of the kit (their files
in `out/text_to_3d/rejected/`), and fifteen are rebuilt in code in `kit/assets/props_kit.py`
(gloves, Swiss ball, plyo boxes, kit hamper, medals stand, ball bag, water carrier, foam rollers,
resistance bands, bicycle, kit bag, tactics easel, flag rack, matchday litter, plus a bibs stack
the generator never managed). Each is its generated prop's `fallback` in
`catalog/generated_props.json`, so every existing placement draws the coded version; the framed
shirt and trophy cabinet fall back to the coded ones that already existed.

Step 3: the weak coded props are rebuilt with detail. The treadmill has silver side covers,
aluminium foot rails, a black belt, side handrails, a console with keys and the safety key. The
DEXA scanner has a full C-arm with its detector window. The anti-gravity treadmill's bag is
see-through clear PVC with seam ribs. The dumbbells now lie across the rack as proper pairs with
their weights on the ends. The extraction hood is open underneath with baffle filters, a sloped
front and a duct. The mirrors are silvered glass (the editor has no reflections, so it is a pale
gloss rather than pure metal). `Med_BagVinyl` and `Hos_MirrorGlass` are kept as authored by the
game conversion (`kit/core/game.py` KEEP).

Step 4: every car and minibus variant has its own body (`kit/assets/vehicles.py` VARIANTS, applied
by `_vary`: a length stretch, a cabin-height scale, a new roof line, glasshouse and wheels). Hatch:
the base, a supermini, a hot hatch, a hatch estate with rails, a family hatch. Saloon: the base,
a touring estate, a fastback, an executive saloon, a compact. SUV: the base, a small crossover, a
seven-seater, a coupe-SUV, a taller boxier SUV. Minibus: high roof, medium roof, shorter
wheelbase. The headliner now follows the roof line, so a lower or sloping roof never leaves it
standing proud.

## Recommended order

1. Remove or replace the eight bad generated props (code rebuilds for the gloves, Swiss ball,
   plyo boxes, medals stand, ball bag; drop the generated framed shirt and trophy cabinet in
   favour of the coded ones).
2. Rebuild the eight removed ones in code (water carrier, foam rollers, resistance bands,
   bicycle, kit bag, tactics easel, flags, matchday litter).
3. Add detail to the six weak coded props.
4. Car body variants.
5. Upgrade the generation pipeline for organic props only (full Hunyuan3D-2.1 multi-view or
   TRELLIS, best-of-4 seeds, higher budgets), and redo the lion.
