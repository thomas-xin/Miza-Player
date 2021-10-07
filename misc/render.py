import sys
write, sys.stdout.write = sys.stdout.write, lambda *args, **kwargs: None
import pygame
sys.stdout.write = write
import time, numpy, math, random, orjson, collections, colorsys, traceback, subprocess, itertools
from math import *
from traceback import print_exc
np = numpy
deque = collections.deque
from PIL import Image, ImageDraw, ImageFont

def pyg2pil(surf):
    mode = "RGBA" if surf.get_flags() & pygame.SRCALPHA else "RGB"
    b = pygame.image.tostring(surf, mode)
    return Image.frombuffer(mode, surf.get_size(), b)

def pil2pyg(im):
    mode = im.mode
    b = im.tobytes()
    return pygame.image.frombuffer(b, im.size, mode)

import pygame.gfxdraw as gfxdraw
pygame.font.init()
FONTS = {}

def round_random(x):
    y = int(x)
    if y == x:
        return y
    x -= y
    if random.random() <= x:
        y += 1
    return y


higher_bound = "C10"
highest_note = "C~D~EF~G~A~B".index(higher_bound[0].upper()) - 9 + ("#" in higher_bound)
while higher_bound[0] not in "0123456789-":
    higher_bound = higher_bound[1:]
    if not higher_bound:
        raise ValueError("Octave not found.")
highest_note += int(higher_bound) * 12 - 11

lower_bound = "C0"
lowest_note = "C~D~EF~G~A~B".index(lower_bound[0].upper()) - 9 + ("#" in lower_bound)
while lower_bound[0] not in "0123456789-":
    lower_bound = lower_bound[1:]
    if not lower_bound:
        raise ValueError("Octave not found.")
lowest_note += int(lower_bound) * 12 - 11

barcount = int(highest_note - lowest_note) + 1 + 2
freqscale2 = 2
barcount2 = barcount * freqscale2
barheight = 720

PARTICLES = set()
P_ORDER = 0
TICK = 0
TEXTS = {}

class Particle(collections.abc.Hashable):

    __slots__ = ("hash", "order")

    def __init__(self):
        global P_ORDER
        self.order = P_ORDER
        P_ORDER += 1
        self.hash = random.randint(-2147483648, 2147483647)

    __hash__ = lambda self: self.hash
    update = lambda self: None
    render = lambda self, surf: None

class Bar(Particle):

    __slots__ = ("x", "colour", "height", "height2", "surf", "line")

    fsize = 0

    def __init__(self, x, barcount):
        super().__init__()
        dark = False
        self.colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(x / barcount, 1, 1))
        note = highest_note - x + 9
        if note % 12 in (1, 3, 6, 8, 10):
            dark = True
        name = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[note % 12]
        octave = note // 12
        self.line = name + str(octave + 1)
        self.x = x
        if dark:
            self.colour = tuple(i + 1 >> 1 for i in self.colour)
            self.surf = Image.new("RGB", (1, 3), self.colour)
        else:
            self.surf = Image.new("RGB", (1, 2), self.colour)
        self.surf.putpixel((0, 0), 0)
        self.height = 0
        self.height2 = 0

    def update(self, dur=1):
        if specs == 3:
            rat = 1 / 60
        else:
            rat = 0.2
        if self.height:
            self.height = self.height * rat ** dur - 1
            if self.height < 0:
                self.height = 0
        if self.height2:
            self.height2 = self.height2 * rat ** dur - 1
            if self.height2 < 0:
                self.height2 = 0

    def ensure(self, value):
        if self.height < value:
            self.height = value
        if self.height2 < value:
            self.height2 = value

    def render(self, sfx, **void):
        size = min(2 * barheight, round_random(self.height))
        if size:
            dark = False
            self.colour = tuple(round_random(i * 255) for i in colorsys.hsv_to_rgb((pc_ / 3 + self.x / barcount) % 1, 1, 1))
            note = highest_note - self.x + 9
            if note % 12 in (1, 3, 6, 8, 10):
                dark = True
            if dark:
                self.colour = tuple(i + 1 >> 1 for i in self.colour)
                self.surf = Image.new("RGB", (1, 3), self.colour)
            else:
                self.surf = Image.new("RGB", (1, 2), self.colour)
            self.surf.putpixel((0, 0), 0)
            surf = self.surf.resize((1, size), resample=Image.BILINEAR)
            sfx.paste(surf, (barcount - 2 - self.x, barheight - size))
    
    def render2(self, sfx, **void):
        if not vertices > 0:
            return
        size = min(2 * barheight, round_random(self.height))
        if size:
            amp = size / barheight * 2
            val = min(1, amp)
            sat = 1 - min(1, max(0, amp - 1))
            self.colour = tuple(round_random(i * 255) for i in colorsys.hsv_to_rgb((pc_ / 3 + self.x / barcount / freqscale2) % 1, sat, val))
            x = barcount * freqscale2 - self.x - 2
            if vertices == 4:
                DRAW.rectangle(
                    (x, x, barcount * 2 * freqscale2 - x - 3, barcount * 2 * freqscale2 - x - 3),
                    None,
                    self.colour,
                    width=1,
                )
            elif vertices == 2:
                DRAW.line(
                    (0, x, barcount * 2 * freqscale2 - 2, x),
                    self.colour,
                    width=1,
                )
                DRAW.line(
                    (0, barcount * 2 * freqscale2 - x - 3, barcount * 2 * freqscale2 - 2, barcount * 2 * freqscale2 - x - 3),
                    self.colour,
                    width=1,
                )
            elif vertices <= 1:
                DRAW.line(
                    (0, barcount * 2 * freqscale2 - x * 2 - 4, barcount * 2 * freqscale2 - 2, barcount * 2 * freqscale2 - x * 2 - 4),
                    self.colour,
                    width=2,
                )
            elif vertices >= 32 or self.x <= 7 and not vertices & 1:
                DRAW.ellipse(
                    (x, x, barcount * 2 * freqscale2 - x - 3, barcount * 2 * freqscale2 - x - 3),
                    None,
                    self.colour,
                    width=2,
                )
            else:
                diag = 1
                # if vertices & 1:
                #     diag *= cos(pi / vertices)
                radius = self.x * diag + 1
                a = barcount * freqscale2 - 2
                b = a / diag + 1
                points = []
                for i in range(vertices + 1):
                    z = i / vertices * tau
                    p = (a + radius * sin(z), b - radius * cos(z))
                    points.append(p)
                DRAW.line(
                    points,
                    self.colour,
                    width=2,
                    joint="curve"
                )

    def post_render(self, sfx, scale, **void):
        size = self.height2
        if size > 8:
            ix = barcount - 1 - self.x - 1
            sx = ix / barcount * ssize2[0]
            w = (ix + 1) / barcount * ssize2[0] - sx
            alpha = round_random(255 * scale)
            t = (self.line, self.fsize)
            if t not in TEXTS:
                TEXTS[t] = self.font.render(self.line, True, (255,) * 3)
            surf = TEXTS[t]
            if alpha < 255:
                surf = surf.copy()
                surf.fill((255,) * 3 + (alpha,), special_flags=pygame.BLEND_RGBA_MULT)
            width, height = surf.get_size()
            x = round(sx + w / 2 - width / 2)
            y = round_random(max(84, ssize2[1] - size - width * (sqrt(5) + 1)))
            sfx.blit(surf, (x, y))

def schlafli(symbol):
    args = (sys.executable, "misc/schlafli.py")
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(symbol.encode("utf-8") + b"\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=2)
    except:
        proc.kill()
        print_exc()
        return ()
    resp = proc.stdout.read().decode("utf-8", "replace").splitlines()
    verts = deque()
    edges = deque()
    vert = True
    for line in resp:
        if ":" in line:
            continue
        if not line:
            if vert:
                vert = False
                continue
            else:
                break
        if vert:
            v = list(map(float, line.split()))
            if len(v) < 3:
                v.append(0)
            verts.append(v)
        else:
            e = list(map(float, line.split()))
            edges.append(e)
    verts = np.array(verts, dtype=np.float16)
    centre = np.mean(verts, axis=0)
    verts -= centre
    verts *= (0.9375 / sqrt(sum(x ** 2 for x in centre)))
    if verts.shape[-1] > 3:
        outs = verts.T[:3].T
        for dim in verts.T[3:]:
            dim *= 0.25
            dim += 1
            outs *= np.tile(dim, (3, 1)).T
        verts = outs
    verts = tuple(itertools.chain(*((verts[int(i)], verts[int(j)]) for i, j in edges)))
    out = np.array(verts, dtype=np.float16)

    return out

found_polytopes = {}

def animate_polyhedron(changed=False):
    if not vertices:
        return
    if type(vertices) is list:
        s = " ".join(map(str, vertices))
    else:
        s = str(vertices)
    try:
        if changed:
            raise KeyError
        poly = found_polytopes[s]
    except KeyError:
        poly = found_polytopes[s] = schlafli(s)
    # print(poly)
    # return
    bars = bars2[::-1]
    try:
        spec = globals()["poly-s"]
    except KeyError:
        class Poly_Spec:
            rx = 0
            ry = 0
            rz = 0
        spec = globals()["poly-s"] = Poly_Spec()
        glLoadIdentity()
    w, h = specsize
    glLineWidth(2.5)
    rx = 0.5 * (0.8 - abs(spec.rx % 90 - 45) / 90)
    ry = 1 / sqrt(2) * (0.8 - abs(spec.ry % 90 - 45) / 90)
    rz = 0.8 - abs(spec.rz % 90 - 45) / 90
    glRotatef(1, rx, ry, rz)
    spec.rx = (spec.rx + rx) % 360
    spec.ry = (spec.ry + ry) % 360
    spec.rz = (spec.rz + rz) % 360
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    try:
        radii = globals()["poly-r"]
        # if radii.shape[1] != len(poly):
        #     raise KeyError
    except KeyError:
        r = np.array([bar.x / barcount2 for bar in bars], dtype=np.float16)
        # r = np.tile(r, (len(poly), 3)).reshape((len(bars), len(poly), 3))
        radii = globals()["poly-r"] = r

    try:
        hsv = globals()["ripple-hsv"]
    except KeyError:
        hsv = globals()["ripple-hsv"] = np.empty((len(bars), 3), dtype=np.float16)
    try:
        hue = globals()["ripple-h"]
    except KeyError:
        hh = [1 - bar.x / barcount / freqscale2 for bar in bars]
        hue = globals()["ripple-h"] = np.array(hh, dtype=np.float16)
    H = hue + (pc_ / 4 + sin(pc_ * tau / 12) / 6) % 1
    hsv.T[0][:] = H % 1
    alpha = np.array([bar.height / barheight * 2 for bar in bars], dtype=np.float16)
    sat = np.clip(alpha - 1, None, 1)
    np.subtract(1, sat, out=sat)
    hsv.T[1][:] = sat
    np.clip(hsv, None, 1, out=hsv)
    hsv.T[:2] *= 255
    hsv = hsv.astype(np.uint8)
    hsv.T[2][:] = 255
    img = Image.frombuffer("HSV", (len(bars), 1), hsv.tobytes()).convert("RGBA")
    colours = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((len(bars), 4)).astype(np.float16)
    colours.T[:3] *= 1 / 255
    mult = np.linspace(1, 0, len(alpha))
    mult **= 2
    alpha *= mult
    colours.T[-1][:] = alpha
    # colours[0] = 1
    colours = np.tile(colours.astype(np.float32), (1, len(poly))).reshape((len(bars), len(poly), 4))
    verts = np.stack([poly.astype(np.float32)] * len(bars))
    for v, r in zip(verts, radii):
        v *= r
    glColorPointer(4, GL_FLOAT, 0, colours.ravel())
    glVertexPointer(3, GL_FLOAT, 0, verts.ravel())
    glDrawArrays(GL_LINES, 0, len(colours) * len(poly))
    glFlush()
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    sfx = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    return sfx

def animate_ripple(changed=False):
    if not vertices:
        return
    bars = bars2[::-1]
    try:
        if changed:
            raise KeyError
        spec = globals()["ripple-s"]
    except KeyError:
        class Ripple_Spec:
            angle = 0
            ry = 0
            rz = 0
        spec = globals()["ripple-s"] = Ripple_Spec()
        glLoadIdentity()
    w, h = specsize
    depth = 3
    glLineWidth(16 / depth)
    ry = 1 / sqrt(2) * (0.8 - abs(spec.ry % 90 - 45) / 90)
    rz = 0.8 - abs(spec.rz % 90 - 45) / 90
    glRotatef(1, 0, ry, rz)
    spec.ry = (spec.ry + ry) % 360
    spec.rz = (spec.rz + rz) % 360
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    try:
        radii = globals()["ripple-r"]
    except KeyError:
        r = np.array([bar.x / barcount2 for bar in bars], dtype=np.float16)
        r.T[-1] = 1
        radii = globals()["ripple-r"] = np.repeat(r, 3).reshape((barcount2, 3))

    try:
        hsv = globals()["ripple-hsv"]
    except KeyError:
        hsv = globals()["ripple-hsv"] = np.empty((len(bars), 3), dtype=np.float16)
    try:
        hue = globals()["ripple-h"]
    except KeyError:
        hh = [1 - bar.x / barcount / freqscale2 for bar in bars]
        hue = globals()["ripple-h"] = np.array(hh, dtype=np.float16)
    H = hue + (pc_ / 4 + sin(pc_ * tau / 8 / sqrt(2)) / 6) % 1
    hsv.T[0][:] = H % 1
    alpha = np.array([bar.height / barheight * 2 for bar in bars], dtype=np.float16)
    sat = np.clip(alpha - 1, None, 1)
    np.subtract(1, sat, out=sat)
    hsv.T[1][:] = sat
    np.clip(hsv, None, 1, out=hsv)
    hsv.T[:2] *= 255
    hsv = hsv.astype(np.uint8)
    hsv.T[2][:] = 255
    img = Image.frombuffer("HSV", (len(bars), 1), hsv.tobytes()).convert("RGBA")
    colours = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((len(bars), 4)).astype(np.float16)
    colours.T[:3] *= 1 / 255
    mult = np.linspace(1, 0, len(alpha))
    mult **= 2
    alpha *= mult
    colours.T[-1][:] = alpha
    colours = np.repeat(colours.astype(np.float32), 2, axis=0)[1:-1]
    colours = np.tile(colours, (vertices * depth, 1))
    hi = hue + pc_ / 3 % 1
    hi *= -tau
    np.sin(hi, out=hi)
    hi *= 0.25
    hi = np.repeat(hi.astype(np.float32), 2, axis=0)[1:-1]
    hi = np.tile(hi, (vertices * depth, 1)).T
    maxlen = ceil(384 / vertices)
    if linearray.maxlen != maxlen:
        globals()["linearray"] = deque(maxlen=maxlen)
    vertarray = np.empty((depth * vertices, len(bars), 3), dtype=np.float16)
    linearray.append([colours, None])
    i = 0
    for _ in range(depth):
        angle = spec.angle + tau - tau / vertices if vertices else spec.angle
        zs = np.linspace(spec.angle, angle, vertices)
        directions = ([cos(z), sin(z), 1] for z in zs)
        for d in directions:
            vertarray[i][:] = radii * d
            i += 1
        spec.angle += 1 / 360 / depth * tau
    linearray[-1][-1] = np.repeat(vertarray.astype(np.float32), 2, axis=1).swapaxes(0, 1)[1:-1].swapaxes(0, 1)
    r = 2 ** ((len(linearray) - 2) / len(linearray) - 1)
    for c, m in linearray:
        c.T[-1] *= r
        glColorPointer(4, GL_FLOAT, 0, c.ravel())
        verts = np.asanyarray(m, dtype=np.float32)
        verts.T[-1][:] = hi
        glVertexPointer(3, GL_FLOAT, 0, verts.ravel())
        glDrawArrays(GL_LINES, 0, len(c))
    glFlush()
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    sfx = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    return sfx

bars = [Bar(i - 1, barcount) for i in range(barcount)]
bars2 = [Bar(i - 1, barcount2) for i in range(barcount2)]
linearray = deque()
last = None
# r = 23 / 24
# s = 1 / 48
# colourmatrix = (
#     r, s, 0, 0,
#     0, r, s, 0,
#     s, 0, r, 0,
# )

def spectrogram_render(bars):
    global ssize2, specs, dur, last
    try:
        if specs == 1:
            sfx = Image.new("RGB", (barcount - 2, barheight), (0,) * 3)
            for bar in bars:
                bar.render(sfx=sfx)
            func = None
        elif specs == 2:
            func = animate_polyhedron
        else:
            func = animate_ripple
        if last != func:
            last = func
            changed = True
        else:
            changed = False
        if func:
            sfx = func(changed)

        if specs == 1:
            if sfx.size != ssize2:
                sfx = sfx.resize(ssize2, resample=Image.NEAREST)
            fsize = max(12, round(ssize2[0] / barcount * (sqrt(5) + 1) / 2))
            if Bar.fsize != fsize:
                Bar.fsize = fsize
                Bar.font = pygame.font.Font("misc/Pacifico.ttf", bar.fsize)
            highbars = sorted(bars, key=lambda bar: bar.height, reverse=True)[:48]
            high = highbars[0]
            sfx = pil2pyg(sfx)
            for bar in reversed(highbars):
                bar.post_render(sfx=sfx, scale=bar.height / max(1, high.height))

        if not sfx:
            sys.stdout.write("~s\n")
            return sys.stdout.flush()
        if type(sfx) is bytes:
            spectrobytes = sfx
        else:
            try:
                spectrobytes = sfx.tobytes()
            except AttributeError:
                spectrobytes = pygame.image.tostring(sfx, "RGB")

        write = specs == 3
        for bar in bars:
            if bar.height2:
                write = True
            bar.update(dur=dur)

        if write:
            if type(sfx) is bytes:
                size = specsize
            else:
                try:
                    size = sfx.size
                except AttributeError:
                    size = sfx.get_size()
            sys.stdout.buffer.write(b"~s" + "~".join(map(str, size)).encode("utf-8") + b"\n")
            sys.stdout.buffer.write(spectrobytes)
        else:
            sys.stdout.write("~s\n")
        sys.stdout.flush()
    except:
        print_exc()

ssize2 = (0, 0)
specs = 0
D = 7
specsize = (barcount * D - D,) * 2

import glfw
from glfw.GLFW import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

glfw.init()
glfw.window_hint(glfw.VISIBLE, False)
globals()["window"] = glfw.create_window(*specsize, "render.py", None, None)
glfw.make_context_current(window)
glutInitDisplayMode(GL_RGB)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(-1, 1, -1, 1, -1, 1)
glDisable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glBlendEquation(GL_FUNC_ADD)
glPixelStorei(GL_PACK_ALIGNMENT, 1)
gluPerspective(45, 1, 1/16, 16)
glTranslatef(0, 0, -2)

while True:
    try:
        line = sys.stdin.buffer.read(2)
        if line == b"~r":
            line = sys.stdin.buffer.readline().rstrip()
            ssize2, specs, vertices, dur, pc_ = map(orjson.loads, line.split(b"~"))
            bi = bars2 if specs > 1 else bars
            spectrogram_render(bi)
        elif line == b"~e":
            b = sys.stdin.buffer.readline()
            amp = np.frombuffer(sys.stdin.buffer.read(int(b)), dtype=np.float32)
            bi = bars2 if len(amp) > len(bars) else bars
            for i, pwr in enumerate(amp):
                bi[i].ensure(pwr / 2)
        elif not line:
            break
        glfwPollEvents()
    except:
        print_exc()

glfw.destroy_window(window)
glfw.terminate()