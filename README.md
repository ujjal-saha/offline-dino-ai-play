# NEAT T-Rex Runner v3 (Powered by `t-rex-runner-v2`)

An upgraded NEAT neuroevolution AI for Chrome's offline T-Rex Runner game, built on top of the authentic **[t-rex-runner-v2](https://github.com/Abirett3/t-rex-runner-v2)** sprite sheet, multi-box collision system, and physics — with strict bird-ducking mechanics.

## What Changed in v3

1. **Authentic `t-rex-runner-v2` Engine & Sprites:**
   - Replaced the procedural rectangle drawings with the real Chromium `t-rex-runner-v2` sprite sheet (`assets/default_100_percent/100-offline-sprite.png`), including animated T-Rex running/ducking/jumping/crashed sprites, multi-cactus clusters (1–3), flapping Pterodactyls, scrolling bumpy horizon, clouds, and sprite-sheet distance meter.
   - Uses `t-rex-runner-v2`'s multi-box Axis-Aligned Bounding Box (`CollisionBox`) collision detection.
2. **True Bird-Ducking Mechanics (No Jumping Over Birds):**
   - Flying Pterodactyls spawn at head-height (`yPos` `70–75` on the `600x150` logical canvas) with an upper airspace collision barrier (`CollisionBox(4, -80, 38, 92)`).
   - **Standing upright** hits the bird's body/head.
   - **Jumping** hits the bird's upper airspace (the Dino **cannot** jump over flying birds).
   - **Ducking** lowers the T-Rex hitbox (`y = 112..137`) to pass cleanly underneath (`y = 98..103`).
   - **Mutually Exclusive Ground Actions:** Just like in `t-rex-runner-v2`, the Dino cannot initiate a jump while ducking, and training penalizes unnecessary ground-ducking when no bird is nearby — so the AI runs upright normally, jumps over cacti, ducks under flying birds, and immediately stands back up.
3. **100% Validation Survival (`dino_best_genome.pickle` & `best_genome_v3.json`):**
   - Tested on **100 unseen random courses** (6,000 frames / 100 seconds each): the v3 pretrained genome survives **100 / 100 courses (100%)** with **0.0 unnecessary ground-duck frames**.

## Run (Python / Pygame)

```bash
pip install -r requirements.txt

python dino_game.py --play        # watch the saved v3 best dino play
python dino_game.py               # train with a window, watch 100 dinos learn
python dino_game.py --headless    # train headless at full CPU speed
python dino_game.py --human       # play it yourself: Space/Up = jump, Down = duck
```

Window controls: **F** fast-forward, **L** debug hitboxes & sensor lines, **Esc** quit.
