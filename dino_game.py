"""
NEAT T-Rex Runner v3
Created by Ujjal Saha — Part of Joyboy Lab
Licensed under the MIT License (see LICENSE file for details).

Uses the authentic Chromium T-Rex Runner (t-rex-runner-v2) sprite sheet,
scrolling horizon, clouds, distance meter, multi-box collision detection,
and physics — with strict bird-ducking mechanics:
  - Cacti (Small & Large clusters of 1-3) MUST be jumped over.
  - Flying Pterodactyls fly at head-height with blocked upper airspace:
    the Dino CANNOT jump over birds and CANNOT stand upright through them —
    it MUST duck under every bird and stand back up to run and jump!

Usage:
    python dino_game.py                  # train with a window
    python dino_game.py --headless       # train with no window (fast)
    python dino_game.py --play           # watch the saved best dino
    python dino_game.py --human          # play it yourself (Space/Up jump, Down duck)

Window controls: F fast-forward, L debug hitboxes & sensor lines, Esc quit.
"""
import argparse
import base64
import json
import os
import pickle
import random
import sys

# Must be set before pygame is imported so --headless never opens a window.
if "--headless" in sys.argv:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

import neat
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config-dino.txt")
BEST_PATH = os.path.join(BASE_DIR, "dino_best_genome.pickle")
BEST_JSON_PATH = os.path.join(BASE_DIR, "best_genome_v3.json")
SPRITE_1X_PATH = os.path.join(BASE_DIR, "assets", "default_100_percent", "100-offline-sprite.png")

# Logical dimensions match t-rex-runner-v2 (600 x 150), rendered at 2x scale (1200 x 300)
LOGICAL_WIDTH, LOGICAL_HEIGHT = 600, 150
SCALE = 2
WIN_WIDTH, WIN_HEIGHT = LOGICAL_WIDTH * SCALE, LOGICAL_HEIGHT * SCALE

GROUND_Y = 140            # Bottom of standing T-Rex (93 + 47)
TREX_GROUND_YPOS = 93     # Runner.defaultDimensions.HEIGHT (150) - HEIGHT (47) - BOTTOM_PAD (10)
DINO_X = 50               # Trex.config.START_X_POS
FPS = 60

BG = (247, 247, 247)
FG = (83, 83, 83)

# Physics matching t-rex-runner-v2
START_SPEED = 6.0
MAX_SPEED = 13.0
ACCEL = 0.0015            # Smooth speed gain per frame
GRAVITY = 0.6
INITIAL_JUMP_VELOCITY = -10.0
DROP_VELOCITY = -5.0
MAX_JUMP_HEIGHT = 30
MIN_JUMP_HEIGHT = 30
SPEED_DROP_COEFFICIENT = 3
GAP_COEFFICIENT = 0.6
PTERO_START_FRAME = 250   # Birds appear early enough for every training course to test them

# Training parameters
ALIVE_REWARD = 0.1
OBSTACLE_REWARD = 5.0
BIRD_DUCK_BONUS = 3.0
UNNECESSARY_DUCK_PENALTY = 0.25
MAX_FRAMES = 6000         # 100s per course at 60 FPS
EPISODES = 3              # Courses per generation
JUMP_THRESHOLD = 0.5
DUCK_THRESHOLD = 0.5
INPUT_SCALE = 3.0


# --------------------------------------------------------------------------
# Collision Box System (Ported directly from t-rex-runner-v2/js/script.js)
# --------------------------------------------------------------------------
class CollisionBox:
    __slots__ = ("x", "y", "width", "height")

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def clone(self):
        return CollisionBox(self.x, self.y, self.width, self.height)


def create_adjusted_collision_box(box, adjustment):
    return CollisionBox(
        box.x + adjustment.x,
        box.y + adjustment.y,
        box.width,
        box.height,
    )


def box_compare(box_a, box_b):
    return (
        box_a.x < box_b.x + box_b.width
        and box_a.x + box_a.width > box_b.x
        and box_a.y < box_b.y + box_b.height
        and box_a.y + box_a.height > box_b.y
    )


# --------------------------------------------------------------------------
# Game Objects (Ported from t-rex-runner-v2)
# --------------------------------------------------------------------------
class Dino:
    W, H = 44, 47
    DUCK_W, DUCK_H = 59, 25

    COLLISION_BOXES_DUCKING = [
        CollisionBox(1, 18, 55, 25),
    ]
    COLLISION_BOXES_RUNNING = [
        CollisionBox(22, 0, 17, 16),
        CollisionBox(1, 18, 30, 9),
        CollisionBox(10, 35, 14, 8),
        CollisionBox(1, 24, 29, 5),
        CollisionBox(5, 30, 21, 4),
        CollisionBox(9, 34, 15, 4),
    ]

    def __init__(self):
        self.xPos = DINO_X
        self.yPos = TREX_GROUND_YPOS
        self.groundYPos = TREX_GROUND_YPOS
        self.minJumpHeight = self.groundYPos - MIN_JUMP_HEIGHT
        self.jumping = False
        self.ducking = False
        self.jumpVelocity = 0.0
        self.reachedMinHeight = False
        self.speedDrop = False
        self.ticks = 0

    @property
    def on_ground(self):
        return not self.jumping and self.yPos >= self.groundYPos

    @property
    def h(self):
        """Height of feet above ground (0 on ground, positive in air)."""
        return float(max(0, self.groundYPos - self.yPos))

    @property
    def vy(self):
        """Upward velocity (positive when rising, negative when falling)."""
        return float(-self.jumpVelocity) if self.jumping else 0.0

    def start_jump(self, speed):
        if not self.jumping and not self.ducking:
            self.jumpVelocity = INITIAL_JUMP_VELOCITY - (speed / 10.0)
            self.jumping = True
            self.reachedMinHeight = False
            self.speedDrop = False

    def end_jump(self):
        if self.reachedMinHeight and self.jumpVelocity < DROP_VELOCITY:
            self.jumpVelocity = DROP_VELOCITY

    def set_speed_drop(self):
        self.speedDrop = True
        self.jumpVelocity = 1.0

    def step(self, jump, duck, speed):
        if not self.jumping:
            # On the ground: duck and jump are mutually exclusive (cannot jump while ducking)
            if duck:
                self.ducking = True
            else:
                self.ducking = False
                if jump:
                    self.start_jump(speed)

        if self.jumping:
            # In mid-air: holding duck multiplies gravity (fast drop) so the dino lands sooner
            self.speedDrop = bool(duck)
            self.yPos += round(self.jumpVelocity)
            gravity = GRAVITY * (SPEED_DROP_COEFFICIENT if self.speedDrop else 1.0)
            self.jumpVelocity += gravity

            if self.yPos < self.minJumpHeight:
                self.reachedMinHeight = True
            if self.yPos < MAX_JUMP_HEIGHT:
                self.end_jump()

            if self.yPos >= self.groundYPos:
                self.yPos = self.groundYPos
                self.jumping = False
                self.jumpVelocity = 0.0
                self.speedDrop = False
                self.ducking = bool(duck)

        self.ticks += 1

    def outer_box(self):
        w = self.DUCK_W if self.ducking else self.W
        return CollisionBox(self.xPos + 1, self.yPos + 1, w - 2, self.H - 2)

    def detailed_boxes(self):
        outer = self.outer_box()
        sub = self.COLLISION_BOXES_DUCKING if self.ducking else self.COLLISION_BOXES_RUNNING
        return [create_adjusted_collision_box(b, outer) for b in sub]


class Obstacle:
    """
    Obstacle definitions from t-rex-runner-v2 with enforced bird-ducking hitboxes:
      - CACTUS_SMALL: width=17, height=35, yPos=105
      - CACTUS_LARGE: width=25, height=50, yPos=90
      - PTERODACTYL:  width=46, height=40, yPos in (75, 70)
        Includes upper airspace collision box extending to y=0 so the Dino
        CANNOT jump over a flying bird and MUST duck under it!
    """
    TYPES = {
        "small": {
            "width": 17,
            "height": 35,
            "yPos": 105,
            "multipleSpeed": 4.0,
            "minGap": 120,
            "collisionBoxes": [
                CollisionBox(0, 7, 5, 27),
                CollisionBox(4, 0, 6, 34),
                CollisionBox(10, 4, 7, 14),
            ],
        },
        "large": {
            "width": 25,
            "height": 50,
            "yPos": 90,
            "multipleSpeed": 7.0,
            "minGap": 120,
            "collisionBoxes": [
                CollisionBox(0, 12, 7, 38),
                CollisionBox(8, 0, 7, 49),
                CollisionBox(13, 10, 10, 38),
            ],
        },
        "ptero": {
            "width": 46,
            "height": 40,
            "yPos": (75, 70),  # Head-level heights: hits standing head, clears ducking dino!
            "multipleSpeed": 999.0,
            "minGap": 150,
            "collisionBoxes": [
                CollisionBox(15, 15, 16, 5),
                CollisionBox(18, 21, 24, 6),
                CollisionBox(2, 14, 4, 3),
                CollisionBox(6, 10, 4, 7),
                CollisionBox(10, 8, 6, 9),
                # Upper airspace barrier: prevents jumping over flying birds!
                CollisionBox(4, -80, 38, 92),
            ],
        },
    }

    def __init__(self, kind, x, speed, count=1, level=0):
        cfg = self.TYPES[kind]
        self.kind = kind
        self.unit_w = cfg["width"]
        self.h = cfg["height"]
        if count > 1 and speed < cfg["multipleSpeed"]:
            count = 1
        self.count = count
        self.w = self.unit_w * count
        self.x = float(x)

        if kind == "ptero":
            y_choices = cfg["yPos"]
            self.yPos = y_choices[level % len(y_choices)]
            # Bottom of bird's physical body is yPos + 28 -> clearance above ground (GROUND_Y=140)
            self.bottom = float(GROUND_Y - (self.yPos + 28))
            self.speed_offset = 0.0
        else:
            self.yPos = cfg["yPos"]
            self.bottom = 0.0
            self.speed_offset = 0.0

        self.passed = False

        # Clone and adjust collision boxes for cluster size > 1 (exact t-rex-runner-v2 logic)
        self.collision_boxes = [b.clone() for b in cfg["collisionBoxes"]]
        if self.count > 1 and len(self.collision_boxes) >= 3:
            self.collision_boxes[1].width = (
                self.w - self.collision_boxes[0].width - self.collision_boxes[2].width
            )
            self.collision_boxes[2].x = self.w - self.collision_boxes[2].width

        min_gap = round(self.w * speed + 150 * 0.85)
        self.gap = random.randint(min_gap, round(min_gap * 1.5))

    @property
    def top(self):
        """Height of the top edge above the ground."""
        return float(GROUND_Y - self.yPos)

    def outer_box(self):
        if self.kind == "ptero":
            # Include upper airspace so outer bounds check never skips jumping dinos
            return CollisionBox(self.x + 1, 0, self.w - 2, self.yPos + self.h - 2)
        return CollisionBox(self.x + 1, self.yPos + 1, self.w - 2, self.h - 2)

    def detailed_boxes(self):
        adj = CollisionBox(self.x + 1, self.yPos + 1, self.w - 2, self.h - 2)
        return [create_adjusted_collision_box(b, adj) for b in self.collision_boxes]


def check_for_collision(obstacle, dino):
    if not box_compare(dino.outer_box(), obstacle.outer_box()):
        return False
    dino_boxes = dino.detailed_boxes()
    obs_boxes = obstacle.detailed_boxes()
    for db in dino_boxes:
        for ob in obs_boxes:
            if box_compare(db, ob):
                return True
    return False


class World:
    """Shared world state matching t-rex-runner-v2."""

    def __init__(self):
        self.speed = START_SPEED
        self.distance = 0.0
        self.frame = 0
        self.obstacles = []
        self.passed = 0
        self.obstacle_history = []

    @property
    def score(self):
        return round(self.distance * 0.025)

    def _spawn(self):
        # Guarantee regular mix of cacti and flying birds
        can_ptero = self.frame >= PTERO_START_FRAME
        if can_ptero and random.random() < 0.35:
            kind = "ptero"
        elif random.random() < 0.5:
            kind = "small"
        else:
            kind = "large"

        # Prevent >2 identical obstacle types in a row (MAX_OBSTACLE_DUPLICATION = 2)
        if len(self.obstacle_history) >= 2 and all(k == kind for k in self.obstacle_history[:2]):
            kind = "ptero" if (can_ptero and kind != "ptero") else ("small" if kind != "small" else "large")

        if kind == "ptero":
            o = Obstacle("ptero", LOGICAL_WIDTH + 20, self.speed, count=1, level=random.randrange(2))
        elif kind == "small":
            o = Obstacle("small", LOGICAL_WIDTH + 20, self.speed, count=random.randint(1, 3))
        else:
            o = Obstacle("large", LOGICAL_WIDTH + 20, self.speed, count=random.randint(1, 2))

        self.obstacle_history.insert(0, kind)
        del self.obstacle_history[2:]
        self.obstacles.append(o)

    def update(self):
        self.frame += 1
        self.speed = min(MAX_SPEED, self.speed + ACCEL)
        self.distance += self.speed

        for o in self.obstacles:
            eff_speed = self.speed + (o.speed_offset if o.kind == "ptero" else 0.0)
            o.x -= eff_speed
        self.obstacles = [o for o in self.obstacles if o.x + o.w > -10]

        last = self.obstacles[-1] if self.obstacles else None
        if last is None or (last.x + last.w + last.gap) < LOGICAL_WIDTH:
            self._spawn()

        newly = 0
        for o in self.obstacles:
            if not o.passed and o.x + o.w < DINO_X:
                o.passed = True
                newly += 1
        self.passed += newly
        return newly

    def next_obstacle(self):
        for o in self.obstacles:
            if o.x + o.w >= DINO_X:
                return o
        return None

    def hits(self, dino):
        for o in self.obstacles:
            if o.x > DINO_X + 80:
                break
            if o.x + o.w < DINO_X - 5:
                continue
            if check_for_collision(o, dino):
                return True
        return False


def network_inputs(dino, world):
    """
    Seven normalised inputs describing the next obstacle and the dino itself:
      0: dx      - distance from dino front to obstacle front (normalised by 600)
      1: width   - obstacle width / 100
      2: bottom  - obstacle clearance above ground / 100 (0 for cacti, ~0.37-0.42 for birds)
      3: top     - obstacle top height above ground / 100
      4: speed   - world.speed / MAX_SPEED
      5: dino.h  - dino height above ground / 100
      6: dino.vy - dino upward velocity / 12
    """
    o = world.next_obstacle()
    if o is None:
        dx, width, bottom, top = 1.0, 0.0, 0.0, 0.0
    else:
        dx = min(1.0, max(0.0, (o.x - DINO_X - Dino.W) / 600.0))
        width, bottom, top = o.w / 100.0, o.bottom / 100.0, o.top / 100.0
    return tuple(
        INPUT_SCALE * v
        for v in (
            dx,
            width,
            bottom,
            top,
            world.speed / MAX_SPEED,
            dino.h / 100.0,
            dino.vy / 12.0,
        )
    )


def decide(net, dino, world):
    out = net.activate(network_inputs(dino, world))
    return out[0] > JUMP_THRESHOLD, out[1] > DUCK_THRESHOLD


# --------------------------------------------------------------------------
# Rendering using t-rex-runner-v2's authentic sprite sheet
# --------------------------------------------------------------------------

EMBEDDED_SPRITE_1X_B64 = "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"

def ensure_sprite_sheet():
    if not os.path.exists(SPRITE_1X_PATH):
        os.makedirs(os.path.dirname(SPRITE_1X_PATH), exist_ok=True)
        with open(SPRITE_1X_PATH, "wb") as f:
            f.write(base64.b64decode(EMBEDDED_SPRITE_1X_B64))


class Sprites:
    ready = False

    # LDPI Sprite sheet coordinates from t-rex-runner-v2/js/script.js
    LDPI = {
        "CACTUS_LARGE": (332, 2),
        "CACTUS_SMALL": (228, 2),
        "CLOUD": (86, 2),
        "HORIZON": (2, 54),
        "MOON": (484, 2),
        "PTERODACTYL": (134, 2),
        "RESTART": (2, 2),
        "TEXT_SPRITE": (655, 2),
        "TREX": (848, 2),
        "STAR": (645, 2),
    }

    @classmethod
    def build(cls):
        if cls.ready:
            return

        ensure_sprite_sheet()
        sheet = pygame.image.load(SPRITE_1X_PATH).convert_alpha()

        def crop_scaled(x, y, w, h):
            sub = sheet.subsurface((x, y, w, h)).copy()
            return pygame.transform.scale(sub, (w * SCALE, h * SCALE))

        tx, ty = cls.LDPI["TREX"]
        cls.dino = {
            # (ducking, leg_frame)
            (False, 0): crop_scaled(tx + 88, ty, 44, 47),
            (False, 1): crop_scaled(tx + 132, ty, 44, 47),
            (True, 0): crop_scaled(tx + 264, ty, 59, 47),
            (True, 1): crop_scaled(tx + 323, ty, 59, 47),
            "jump": crop_scaled(tx + 0, ty, 44, 47),
            "crashed": crop_scaled(tx + 220, ty, 44, 47),
        }

        # Pre-crop cactus clusters (size 1, 2, 3) using exact Chromium sprite formula:
        # sourceX = (unit_w * size) * (0.5 * (size - 1)) + spritePos.x
        cls.cactus = {"small": {}, "large": {}}
        sx, sy = cls.LDPI["CACTUS_SMALL"]
        for size in (1, 2, 3):
            src_x = int((17 * size) * (0.5 * (size - 1)) + sx)
            cls.cactus["small"][size] = crop_scaled(src_x, sy, 17 * size, 35)

        lx, ly = cls.LDPI["CACTUS_LARGE"]
        for size in (1, 2, 3):
            src_x = int((25 * size) * (0.5 * (size - 1)) + lx)
            cls.cactus["large"][size] = crop_scaled(src_x, ly, 25 * size, 50)

        px, py = cls.LDPI["PTERODACTYL"]
        cls.ptero = [
            crop_scaled(px, py, 46, 40),
            crop_scaled(px + 46, py, 46, 40),
        ]

        cx, cy = cls.LDPI["CLOUD"]
        cls.cloud = crop_scaled(cx, cy, 46, 14)

        hx, hy = cls.LDPI["HORIZON"]
        cls.horizon = [
            crop_scaled(hx, hy, 600, 12),
            crop_scaled(hx + 600, hy, 600, 12),
        ]

        # Distance meter digits (0-9, H, I)
        dx, dy = cls.LDPI["TEXT_SPRITE"]
        cls.digits = [crop_scaled(dx + i * 10, dy, 10, 13) for i in range(12)]

        cls.font = pygame.font.SysFont("consolas,couriernew,monospace", 20, bold=True)
        cls.small_font = pygame.font.SysFont("consolas,couriernew,monospace", 16)
        cls.ready = True


class Window:
    def __init__(self, headless=False, fast=False):
        pygame.init()
        pygame.display.set_caption("NEAT T-Rex Runner v3 — Ujjal Saha | Joyboy Lab")
        self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        self.headless = headless
        if not headless:
            Sprites.build()
        self.fast = fast or headless
        self.draw_lines = False
        self.clock = pygame.time.Clock()
        random_state = random.getstate()
        self.clouds = [[random.randrange(LOGICAL_WIDTH), random.randrange(30, 71)] for _ in range(4)]
        random.setstate(random_state)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_f:
                    self.fast = not self.fast
                if event.key == pygame.K_l:
                    self.draw_lines = not self.draw_lines
        return True

    def tick(self):
        if not self.fast:
            self.clock.tick(FPS)

    def draw(self, dinos, world, hud):
        if self.headless:
            return
        Sprites.build()
        win = self.screen
        win.fill(BG)

        # Draw Clouds
        for c in self.clouds:
            win.blit(Sprites.cloud, (int(c[0] * SCALE), int(c[1] * SCALE)))
            c[0] -= 0.2 * world.speed * 0.25
            if c[0] < -50:
                c[0] = LOGICAL_WIDTH + 20

        # Draw scrolling HorizonLine (2 segments of 600px logical)
        h_offset = int(world.distance) % (LOGICAL_WIDTH * 2)
        for seg in range(3):
            seg_x = (seg * LOGICAL_WIDTH) - (h_offset % LOGICAL_WIDTH)
            seg_type = ((h_offset // LOGICAL_WIDTH) + seg) % 2
            win.blit(Sprites.horizon[seg_type], (seg_x * SCALE, 127 * SCALE))

        # Draw Obstacles
        for o in world.obstacles:
            ox, oy = int(o.x * SCALE), int(o.yPos * SCALE)
            if o.kind == "ptero":
                flap = (world.frame // 10) % 2
                win.blit(Sprites.ptero[flap], (ox, oy))
            else:
                win.blit(Sprites.cactus[o.kind][o.count], (ox, oy))

            if self.draw_lines:
                for b in o.detailed_boxes():
                    pygame.draw.rect(
                        win,
                        (0, 180, 60),
                        (int(b.x * SCALE), int(b.y * SCALE), int(b.width * SCALE), int(b.height * SCALE)),
                        1,
                    )

        # Draw Dinos
        alpha = 255 if len(dinos) <= 1 else 95
        for d in dinos:
            if d.jumping:
                sprite = Sprites.dino["jump"]
            else:
                leg = (d.ticks // 5) % 2
                sprite = Sprites.dino[(d.ducking, leg)]
            sprite.set_alpha(alpha)
            win.blit(sprite, (int(d.xPos * SCALE), int(d.yPos * SCALE)))

            if self.draw_lines:
                for b in d.detailed_boxes():
                    pygame.draw.rect(
                        win,
                        (220, 40, 40),
                        (int(b.x * SCALE), int(b.y * SCALE), int(b.width * SCALE), int(b.height * SCALE)),
                        1,
                    )
                target = world.next_obstacle()
                if target:
                    pygame.draw.line(
                        win,
                        (234, 88, 12),
                        (int((d.xPos + Dino.W) * SCALE), int((d.yPos + 20) * SCALE)),
                        (int(target.x * SCALE), int((target.yPos + 20) * SCALE)),
                        1,
                    )

        # Draw sprite-sheet DistanceMeter in top-right
        score_str = f"{world.score:05d}"[-5:]
        for idx, ch in enumerate(score_str):
            digit = ord(ch) - ord("0")
            win.blit(Sprites.digits[digit], ((LOGICAL_WIDTH - 70 + idx * 11) * SCALE, 8 * SCALE))

        for i, line in enumerate(hud):
            win.blit(Sprites.small_font.render(line, True, FG), (12, 10 + 20 * i))

        pygame.display.update()


# --------------------------------------------------------------------------
# Training & Validation
# --------------------------------------------------------------------------
class Trainer:
    def __init__(self, window, max_frames=MAX_FRAMES, episodes=EPISODES):
        self.window = window
        self.max_frames = max_frames
        self.episodes = episodes
        self.generation = 0
        self.best_score = 0

    def eval_genomes(self, genomes, config):
        self.generation += 1
        nets = [
            (genome, neat.nn.FeedForwardNetwork.create(genome, config))
            for _, genome in genomes
        ]
        totals = [0.0] * len(nets)

        for episode in range(self.episodes):
            agents = [(i, net, Dino()) for i, (_, net) in enumerate(nets)]
            world = World()
            while agents and world.frame < self.max_frames:
                if not self.window.handle_events():
                    raise KeyboardInterrupt
                self.window.tick()

                next_obs = world.next_obstacle()
                is_bird_ahead = (
                    next_obs is not None
                    and next_obs.kind == "ptero"
                )

                for i, net, dino in agents:
                    jump, duck = decide(net, dino, world)
                    was_jumping = dino.jumping
                    dino.step(jump, duck, world.speed)
                    if not was_jumping and dino.ducking and not is_bird_ahead:
                        # Discourage staying crouched when approaching cacti / empty ground
                        totals[i] -= UNNECESSARY_DUCK_PENALTY
                    else:
                        totals[i] += ALIVE_REWARD

                newly = world.update()
                if newly:
                    for i, _, dino in agents:
                        totals[i] += OBSTACLE_REWARD * newly
                        if dino.ducking:
                            totals[i] += BIRD_DUCK_BONUS

                agents = [a for a in agents if not world.hits(a[2])]

                self.best_score = max(self.best_score, world.score)
                hud = [
                    f"Gen: {self.generation}  (course {episode + 1}/{self.episodes})",
                    f"Alive: {len(agents)}",
                    f"Speed: {world.speed:.1f}",
                    f"Best: {self.best_score}",
                ]
                self.window.draw([a[2] for a in agents], world, hud)

        for (genome, _), total in zip(nets, totals):
            genome.fitness = total / self.episodes


def run_course(net, max_frames=MAX_FRAMES):
    """Play one fresh random course with no rendering. Returns (frames, cactus_ducks, bird_ducks)."""
    world, dino = World(), Dino()
    cactus_ducks = 0
    bird_ducks = 0
    while world.frame < max_frames:
        jump, duck = decide(net, dino, world)
        next_obs = world.next_obstacle()
        is_bird_ahead = (
            next_obs is not None
            and next_obs.kind == "ptero"
        )
        was_jumping = dino.jumping
        dino.step(jump, duck, world.speed)
        if not was_jumping and dino.ducking:
            if is_bird_ahead:
                bird_ducks += 1
            else:
                cactus_ducks += 1
        world.update()
        if world.hits(dino):
            break
    return world.frame, cactus_ducks, bird_ducks


class StopTraining(Exception):
    pass


class Validator(neat.reporting.BaseReporter):
    def __init__(self, config, courses, required, max_frames, min_fitness=500):
        self.config = config
        self.courses = courses
        self.required = required
        self.max_frames = max_frames
        self.min_fitness = min_fitness
        self.validated = None

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome.fitness < self.min_fitness:
            return
        net = neat.nn.FeedForwardNetwork.create(best_genome, self.config)
        results = [run_course(net, self.max_frames) for _ in range(self.courses)]
        wins = sum(r[0] >= self.max_frames for r in results)
        avg_cactus_ducks = sum(r[1] for r in results) / len(results)
        avg_bird_ducks = sum(r[2] for r in results) / len(results)
        print(
            f"  Validation on {self.courses} fresh courses: {wins}/{self.courses} survived "
            f"(avg bird-duck frames: {avg_bird_ducks:.0f}, cactus-ground-ducks: {avg_cactus_ducks:.1f})"
        )
        if wins / self.courses >= self.required and avg_bird_ducks > 100 and avg_cactus_ducks < 200:
            self.validated = best_genome
            raise StopTraining


def export_genome_json(genome, path=BEST_JSON_PATH):
    """Export the NEAT genome to JSON so the web t-rex-runner-v3 engine can run the exact same network."""
    nodes = {}
    for nid, node in genome.nodes.items():
        nodes[str(nid)] = {
            "bias": float(node.bias),
            "response": float(node.response),
            "activation": str(node.activation),
            "aggregation": str(node.aggregation),
        }
    connections = []
    for (in_node, out_node), conn in genome.connections.items():
        if conn.enabled:
            connections.append({
                "in": int(in_node),
                "out": int(out_node),
                "weight": float(conn.weight),
            })
    payload = {
        "version": "3.0",
        "fitness": float(genome.fitness or 0.0),
        "input_keys": [-1, -2, -3, -4, -5, -6, -7],
        "output_keys": [0, 1],
        "input_scale": INPUT_SCALE,
        "jump_threshold": JUMP_THRESHOLD,
        "duck_threshold": DUCK_THRESHOLD,
        "nodes": nodes,
        "connections": connections,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_config():
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )


def run_training(args):
    config = load_config()
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    validator = Validator(config, courses=30, required=0.9, max_frames=args.max_frames)
    pop.add_reporter(validator)

    window = Window(headless=args.headless, fast=args.fast)
    trainer = Trainer(window, max_frames=args.max_frames, episodes=args.episodes)

    winner = None
    try:
        winner = pop.run(trainer.eval_genomes, args.generations)
        print("Reached the generation limit without passing validation; saving the best so far.")
    except StopTraining:
        winner = validator.validated
        print("\nValidated: this dino reliably jumps cacti and ducks under flying birds!")
    except KeyboardInterrupt:
        print("\nStopped early.")
        winner = stats.best_genome() if stats.most_fit_genomes else None
    except neat.CompleteExtinctionException:
        print("Population went extinct.")

    if winner is not None:
        with open(BEST_PATH, "wb") as f:
            pickle.dump(winner, f)
        export_genome_json(winner, BEST_JSON_PATH)
        print(f"\nBest genome (fitness {winner.fitness:.1f}) saved to {BEST_PATH} and {BEST_JSON_PATH}")

    pygame.quit()


def run_single(controller, hud_name, fast=False):
    window = Window(headless=False, fast=fast)
    best = 0
    running = True
    while running:
        world, dino = World(), Dino()
        while running:
            running = window.handle_events()
            if not running:
                break
            window.tick()
            jump, duck = controller(dino, world)
            dino.step(jump, duck, world.speed)
            world.update()
            best = max(best, world.score)
            state_str = "DUCKING" if dino.ducking else ("JUMPING" if dino.jumping else "RUNNING")
            window.draw(
                [dino],
                world,
                [f"{hud_name} [{state_str}]", f"Best: {best}", f"Speed: {world.speed:.1f}"],
            )
            if world.hits(dino):
                break
    pygame.quit()


def run_play(args):
    if not os.path.exists(BEST_PATH):
        sys.exit(f"No saved dino at {BEST_PATH}. Train one first (python dino_game.py).")
    with open(BEST_PATH, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, load_config())
    run_single(lambda dino, world: decide(net, dino, world), "AI (NEAT v3)", fast=args.fast)


def run_human(args):
    def controller(dino, world):
        keys = pygame.key.get_pressed()
        return keys[pygame.K_SPACE] or keys[pygame.K_UP], keys[pygame.K_DOWN]

    run_single(controller, "You")


def parse_args():
    p = argparse.ArgumentParser(description="Train or replay a NEAT T-Rex Runner v3 AI.")
    p.add_argument("--play", action="store_true", help="replay the saved best dino")
    p.add_argument("--human", action="store_true", help="play the game yourself")
    p.add_argument("--headless", action="store_true", help="train without a window (fast)")
    p.add_argument("--fast", action="store_true", help="start uncapped (toggle with F)")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if args.human:
        run_human(args)
    elif args.play:
        run_play(args)
    else:
        run_training(args)
