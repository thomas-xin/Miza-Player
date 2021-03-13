import os, sys, json, traceback, subprocess, psutil, copy, concurrent.futures

print = lambda *args, sep=" ", end="\n": sys.stdout.write(str(sep).join(map(str, args)) + end)

exc = concurrent.futures.ThreadPoolExecutor(max_workers=12)
submit = exc.submit
print_exc = traceback.print_exc

is_url = lambda url: "://" in url and url.split("://", 1)[0].removesuffix("s") in ("http", "hxxp", "ftp", "fxp")

downloader = concurrent.futures.Future()
def import_audio_downloader():
    try:
        audio_downloader = __import__("audio_downloader")
        globals().update(audio_downloader.__dict__)
        globals()["ytdl"] = ytdl = audio_downloader.AudioDownloader()
        downloader.set_result(ytdl)
    except Exception as ex:
        print_exc()
        downloader.set_exception(ex)


class cdict(dict):

    __slots__ = ()

    __init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
    __repr__ = lambda self: f"{self.__class__.__name__}({super().__repr__() if super().__len__() else ''})"
    __str__ = lambda self: super().__repr__()
    __iter__ = lambda self: iter(tuple(super().__iter__()))
    __call__ = lambda self, k: self.__getitem__(k)

    def __getattr__(self, k):
        try:
            return self.__getattribute__(k)
        except AttributeError:
            pass
        if not k.startswith("__") or not k.endswith("__"):
            try:
                return self.__getitem__(k)
            except KeyError as ex:
                raise AttributeError(*ex.args)
        raise AttributeError(k)

    def __setattr__(self, k, v):
        if k.startswith("__") and k.endswith("__"):
            return object.__setattr__(self, k, v)
        return self.__setitem__(k, v)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(self)
        return data

    @property
    def __dict__(self):
        return self

    ___repr__ = lambda self: super().__repr__()
    to_dict = lambda self: dict(**self)
    to_list = lambda self: list(super().values())


def as_str(s):
    if type(s) in (bytes, bytearray, memoryview):
        return bytes(s).decode("utf-8", "replace")
    return str(s)


hasmisc = os.path.exists("misc")
if hasmisc:
    submit(import_audio_downloader)
    mixer = psutil.Popen(("py", "misc/mixer.py"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
else:
    mixer = cdict()

def state(i):
    mixer.stdin.write(f"~state {int(i)}\n".encode("utf-8"))
    mixer.stdin.flush()

def clear():
    mixer.stdin.write(f"~clear\n".encode("utf-8"))
    mixer.stdin.flush()

def drop(i):
    mixer.stdin.write(f"~drop {i}\n".encode("utf-8"))
    mixer.stdin.flush()

laststart = set()
def mixer_submit(s, force):
    if not force:
        ts = pc()
        if laststart:
            diff = ts - min(laststart)
            if diff < 0.5:
                delay = 0.5 - diff
                laststart.add(ts)
                time.sleep(delay)
                if ts < max(laststart):
                    return
            laststart.clear()
        laststart.add(ts)
    s = as_str(s)
    if not s.endswith("\n"):
        s += "\n"
    mixer.stdin.write(s.encode("utf-8"))
    mixer.stdin.flush()

mixer.state = lambda i=0: state(i)
mixer.clear = lambda: clear()
mixer.drop = lambda i=0: drop(i)
mixer.submit = lambda s, force=False: submit(mixer_submit, s, force)

write, sys.stdout.write = sys.stdout.write, lambda *args, **kwargs: None
import pygame
sys.stdout.write = write


asettings = cdict(
    volume=(0, 5),
    speed=(-1, 3),
    pitch=(-12, 12, 0.5),
    pan=(0, 4),
    bassboost=(0, 7),
    reverb=(0, 3),
    compressor=(0, 6),
    chorus=(0, 5),
    nightcore=(-6, 18, 0.5),
)
aediting = dict.fromkeys(asettings)
config = "config.json"
options = None
if os.path.exists(config):
    try:
        with open(config, "r", encoding="utf-8") as f:
            options = json.load(f)
    except:
        print_exc()
if options:
    options = cdict(options)
else:
    options = cdict(
        screensize=[1280, 720],
        sidebar_width=256,
        toolbar_height=64,
        audio=cdict(
            volume=1,
            speed=1,
            pitch=0,
            pan=1,
            bassboost=0,
            reverb=0,
            compressor=0,
            chorus=0,
            nightcore=0,
        ),
        control=cdict(
            shuffle=1,
            loop=1
        ),
    )
options.audio = cdict(options.audio)
options.control = cdict(options.control)
orig_options = copy.deepcopy(options)
globals().update(options)


if os.name != "nt":
    raise NotImplementedError("This program is currently implemented to use Windows API only.")

import ctypes, ctypes.wintypes, io
appid = "Miza Player (" + str(os.getpid()) + ")"
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

rel = None
mouse_pointer = POINT()

def mouse_abs_pos():
    pt = mouse_pointer
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)

def mouse_rel_pos(force=True):
    global rel
    apos = mouse_abs_pos()
    if not rel or force and get_focused(True):
        rel = [x - y for x, y in zip(apos, pygame.mouse.get_pos())]
    return [x - y for x, y in zip(apos, rel)]

mouse_pos_check = None
def get_focused(replace=False):
    global mouse_pos_check
    if not pygame.mouse.get_focused():
        return
    mpc = pygame.mouse.get_pos()
    if replace and not mouse_pos_check:
        mouse_pos_check = mpc
    if mpc != mouse_pos_check:
        if replace:
            mouse_pos_check = mpc
        return True
    if any(i < 0 for i in mouse_rel_pos(False)):
        return
    return True

def get_pressed():
    mheld = [None] * 5
    for i, n in enumerate((1, 2, 4, 5, 6)):
        mheld[i] = bool(ctypes.windll.user32.GetAsyncKeyState(n) & 32768)
    return mheld

if hasmisc:
    icon = pygame.image.load("misc/icon.png")
    pygame.display.set_icon(icon)
pygame.display.set_caption("Miza Player")
DISP = pygame.display.set_mode(screensize, pygame.RESIZABLE, vsync=True)
screensize2 = list(screensize)

hwnd = pygame.display.get_wm_info()["window"]
pygame.display.set_allow_screensaver(True)
pygame.font.init()
if hasmisc:
    s = io.StringIO()
    s.write(("%" + str(hwnd) + "\n"))
    for k, v in audio.items():
        s.write(f"~setting {k} {v}\n")
    s.write(f"~setting shuffle {control.shuffle}\n")
    s.seek(0)
    mixer.stdin.write(s.read().encode("utf-8"))
    mixer.stdin.flush()

in_rect = lambda point, rect: point[0] >= rect[0] and point[0] < rect[0] + rect[2] and point[1] >= rect[1] and point[1] < rect[1] + rect[3]
in_circ = lambda point, dest, radius=1: hypot(dest[0] - point[0], dest[1] - point[1]) <= radius

class WR(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]

wr = WR()

def get_window_rect():
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(wr))
    return wr.left, wr.top, wr.right - wr.left, wr.bottom - wr.top

class WP(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("showCmd", ctypes.c_uint),
        ("ptMinPosition", ctypes.c_void_p),
        ("ptMaxPosition", ctypes.c_void_p),
        ("rcNormalPosition", ctypes.c_void_p),
        ("rcDevice", ctypes.c_void_p),
    ]

wp = WP()
ptMinPosition = WR()
ptMaxPosition = WR()
rcNormalPosition = WR()
rcDevice = WR()

wp.length = 44
wp.ptMinPosition = ctypes.cast(ctypes.byref(ptMinPosition), ctypes.c_void_p)
wp.ptMaxPosition = ctypes.cast(ctypes.byref(ptMaxPosition), ctypes.c_void_p)
wp.rcNormalPosition = ctypes.cast(ctypes.byref(rcNormalPosition), ctypes.c_void_p)
wp.rcDevice = ctypes.cast(ctypes.byref(rcDevice), ctypes.c_void_p)

def get_window_flags():
    ctypes.windll.user32.GetWindowPlacement(hwnd, ctypes.byref(wp))
    return wp.showCmd

is_minimised = lambda: ctypes.windll.user32.IsIconic(hwnd)

if options.get("maximised"):
    ctypes.windll.user32.ShowWindow(hwnd, 3)
elif options.get("screenpos"):# and not options.get("maximised"):
    x, y = screenpos
    ctypes.windll.user32.SetWindowPos(hwnd, 0, x, y, -1, -1, 0x4561)
# else:
#     ctypes.windll.user32.SetWindowPos(hwnd, -1, -1, -1, -1, -1, 0x4563)
screenpos2 = get_window_rect()[:2]

flash_window = lambda bInvert=True: ctypes.windll.user32.FlashWindow(hwnd, bInvert)

spath = "misc/Shobjidl.dll"
shobjidl_core = ctypes.cdll.LoadLibrary(spath)
# shobjidl_core.SetWallpaper(0, os.path.abspath("misc/icon.png"))

proglast = (0, 0)
def taskbar_progress_bar(ratio=1, colour=0):
    global proglast
    if ratio <= 0 and not colour & 1 or not colour:
        ratio = colour = 0
    r = round(min(1, ratio) * 256)
    t = (r, colour)
    if t != proglast:
        proglast = t
        shobjidl_core.SetProgressState(hwnd, colour)
        if colour:
            shobjidl_core.SetProgressValue(hwnd, r, 256)

# above = True
# @ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int)
# def callback(h, lp):
#     global above, pts
#     if not above or not pts:
#         return
#     if h == hwnd:
#         above = False
#         return
#     ctypes.windll.user32.GetWindowPlacement(h, ctypes.byref(wp))
#     if wp.showCmd != 1 and wp.showCmd != 3:
#         return
#     if not ctypes.windll.user32.IsWindowVisible(h):
#         return
#     ctypes.windll.user32.GetWindowRect(h, ctypes.byref(wr))
#     rect = (wr.left, wr.top, wr.right - wr.left, wr.bottom - wr.top)
#     if not rect[2] or not rect[3]:
#         return
#     pops = []
#     for i, p in enumerate(pts):
#         if in_rect(p, rect):
#             pops.append(i)
#     if pops:
#         pts = [p for i, p in enumerate(pts) if i not in pops]
#         print(pts)
#     # print(*rect)

# def is_covered():
#     global pts
#     ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(wr))
#     rect = (wr.left, wr.top, wr.right, wr.bottom)
#     pts = [(rect[:2]), (rect[2], rect[1]), (rect[0], rect[3]), rect[2:]]
#     print(pts)
#     ctypes.windll.user32.EnumWindows(callback, -1)
#     print(pts)

# def get_window_clip():
#     hdc = ctypes.windll.user32.GetWindowDC(hwnd)
#     clip = ctypes.windll.gdi32.GetClipBox(hdc, ctypes.byref(wr))
#     ctypes.windll.user32.ReleaseDC(hdc)
#     return hdc, clip, wr.left, wr.top, wr.right - wr.left, wr.bottom - wr.top


import PIL, easygui, numpy, time, math, random, itertools, collections, re, colorsys, ast, contextlib, pyperclip, pyaudio
from PIL import Image, ImageChops
from math import *
np = numpy
deque = collections.deque
suppress = contextlib.suppress
d2r = pi / 180
utc = time.time

pt = None
def pc():
    global pt
    t = time.perf_counter()
    if not pt:
        pt = t
        return 0
    return t - pt

math.round = round
afut = submit(pyaudio.PyAudio)

def round(x, y=None):
    try:
        if isfinite(x):
            try:
                if x == int(x):
                    return int(x)
                if y is None:
                    return int(math.round(x))
            except:
                pass
            return round_min(math.round(x, y))
        else:
            return x
    except:
        pass
    if type(x) is complex:
        return round(x.real, y) + round(x.imag, y) * 1j
    try:
        return math.round(x, y)
    except:
        pass
    return x

def round_min(x):
    if type(x) is str:
        if "." in x:
            x = x.strip("0")
            if len(x) > 8:
                x = mpf(x)
            else:
                x = float(x)
        else:
            try:
                return int(x)
            except ValueError:
                return float(x)
    if type(x) is int:
        return x
    if type(x) is not complex:
        if isfinite(x):
            if type(x) is globals().get("mpf", None):
                y = int(x)
                if x == y:
                    return y
                f = float(x)
                if str(x) == str(f):
                    return f
            else:
                y = math.round(x)
                if x == y:
                    return int(y)
        return x
    else:
        if x.imag == 0:
            return round_min(x.real)
        else:
            return round_min(complex(x).real) + round_min(complex(x).imag) * (1j)

def bit_crush(dest, b=0, f=round):
    if type(b) == int:
        a = 1 << b
    else:
        a = 2 ** b
    try:
        len(dest)
        dest = list(dest)
        for i in range(len(dest)):
            dest[i] = f(dest[i] / a) * a
    except TypeError:
        dest = f(dest / a) * a
    return dest

def shuffle(it):
    if not isinstance(it, list):
        it = list(it)
    random.shuffle(it)
    return it

from pygame.locals import *
from pygame import gfxdraw

verify_colour = lambda c: [max(0, min(255, abs(i))) for i in c]

def adj_colour(colour, brightness=0, intensity=1, hue=0):
    if hue != 0:
        h = colorsys.rgb_to_hsv(i / 255 for i in colour)
        c = adj_colour(colorsys.hsv_to_rgb((h[0] + hue) % 1, h[1], h[2]), intensity=255)
    else:
        c = list(colour)
    for i in range(len(c)):
        c[i] = round(c[i] * intensity + brightness)
    return verify_colour(c)

gsize = (1920, 1)
gradient = ((np.arange(1, 0, -1 / gsize[0], dtype=np.float64)) ** 2 * 256).astype(np.uint8).reshape(tuple(reversed(gsize)))
qhue = Image.fromarray(gradient, "L")
qsat = qval = Image.new("L", gsize, 255)
quadratics = [None] * 256

def quadratic_gradient(size=gsize, t=None):
    size = tuple(size)
    if t is None:
        t = pc()
    x = int(t * 128) & 255
    if not quadratics[x]:
        hue = qhue.point(lambda i: i + x & 255)
        img = Image.merge("HSV", (hue, qsat, qval)).convert("RGB")
        b = img.tobytes()
        quadratics[x] = pygame.image.frombuffer(b, gsize, "RGB")
    surf = quadratics[x]
    if surf.get_size() != size:
        surf = pygame.transform.scale(surf, size)
    return surf

rgw = 256
mid = (rgw - 1) / 2
row = np.arange(rgw, dtype=np.float64) - mid
data = [None] * rgw
for i in range(rgw):
    data[i] = a = np.arctan2(i - mid, row)
    np.around(np.multiply(a, 256 / tau, out=a), 0, out=a)
data = np.uint8(data)
rhue = Image.fromarray(data, "L")
rsat = rval = Image.new("L", (rgw,) * 2, 255)
radials = [None] * 256

def radial_gradient(size=(rgw,) * 2, t=None):
    size = tuple(size)
    if t is None:
        t = pc()
    x = int(t * 128) & 255
    if not radials[x]:
        hue = rhue.point(lambda i: i + x & 255)
        img = Image.merge("HSV", (hue, rsat, rval)).convert("RGB")
        b = img.tobytes()
        radials[x] = pygame.image.frombuffer(b, (rgw,) * 2, "RGB")
    surf = radials[x]
    if surf.get_size() != size:
        surf = pygame.transform.scale(surf, size)
    return surf

draw_line = pygame.draw.line
draw_aaline = pygame.draw.aaline
draw_hline = gfxdraw.hline
draw_vline = gfxdraw.vline
draw_polygon = pygame.draw.polygon
draw_tpolygon = gfxdraw.textured_polygon

def custom_scale(source, size, dest=None, antialias=False):
    dsize = list(map(round, size))
    ssize = source.get_size()
    if antialias > 1 or (ssize[0] >= dsize[0] and ssize[1] >= dsize[1]):
        if dest:
            return pygame.transform.smoothscale(source, dsize, dest)
        return pygame.transform.smoothscale(source, dsize)
    else:
        while ssize[0] < dsize[0] or ssize[1] < dsize[1]:
            source = pygame.transform.scale2x(source)
            ssize = source.get_size()
    if antialias:
        scalef = pygame.transform.smoothscale
    else:
        scalef = pygame.transform.scale
    if dest:
        return scalef(source, dsize, dest)
    return scalef(source, dsize)

def blit_complex(dest, source, position=(0, 0), alpha=255, angle=0, scale=1, colour=(255,) * 3, area=None, copy=True):
    pos = position
    s1 = source.get_size()
    if dest:
        s2 = dest.get_size()
        if pos[0] >= s2[0] or pos[1] >= s2[1] or pos[0] <= -s1[0] or pos[1] <= -s1[1]:
            return
    alpha = round(min(alpha, 255))
    if alpha <= 0:
        return
    s = source
    if alpha != 255 or any(i != 255 for i in colour) or dest is None:
        if copy:
            try:
                s = source.convert_alpha()
            except:
                s = source.copy()
        if alpha != 255:
            s.fill(tuple(colour) + (alpha,), special_flags=BLEND_RGBA_MULT)
        elif any(i != 255 for i in colour):
            s.fill(tuple(colour), special_flags=BLEND_RGB_MULT)
    if angle:
        ckf = [s.get_colorkey(), s.get_flags()]
        s = pygame.transform.rotate(s, -angle / d2r)
        s.set_colorkey(*ckf)
        s3 = s.get_size()
        pos = [z - (y - x >> 1) for x, y, z in zip(s1, s3, pos)]
    if scale != 1:
        s = custom_scale(s, list(map(lambda i: round(i * scale), s.get_size())))
    if area is not None:
        area = list(map(lambda i: round(i * scale), area))
    if dest:
        return dest.blit(s, pos, area, special_flags=BLEND_ALPHA_SDL2)
    return s

def draw_rect(dest, colour, rect, width=0, alpha=255, angle=0):
    alpha = max(0, min(255, round(alpha)))
    width = round(abs(width))
    if width > 0:
        if angle != 0 or alpha != 255:
            ssize = [i + width for i in rect[2:]]
            s = pygame.Surface(ssize)
            srec = [i + width // 2 for i in rect[2:]]
            pygame.draw.rect(s, colour, srec, width)
            blit_complex(dest, s, rect[:2], alpha, angle)
            #raise NotImplementedError("Alpha blending and rotation of rectangles with variable width is not implemented.")
        else:
            pygame.draw.rect(dest, colour, width)
    else:
        if angle != 0:
            bevel_rectangle(dest, colour, rect, 0, alpha, angle)
        else:
            rect = list(rect)
            if rect[0] < 0:
                rect[2] += rect[0]
                rect[0] = 0
            if rect[1] < 0:
                rect[3] += rect[1]
                rect[1] = 0
            if alpha != 255:
                dest.fill((255 - alpha,) * 4, rect, special_flags=BLEND_RGBA_MULT)
                dest.fill([min(i + alpha / 255, 255) for i in colour] + [alpha], rect, special_flags=BLEND_RGBA_ADD)
            else:
                dest.fill(colour, rect)

def bevel_rectangle(dest, colour, rect, bevel=0, alpha=255, angle=0, grad_col=None, grad_angle=0, filled=True, cache=True):
    rect = list(map(round, rect))
    if min(alpha, rect[2], rect[3]) > 0:
        br_surf = globals().setdefault("br_surf", {})
        colour = list(map(lambda i: min(i, 255), colour))
        if alpha == 255 and angle == 0 and not (colour[0] == colour[1] == colour[2]):
            if dest is None:
                dest = pygame.Surface(rect[2:], SRCALPHA)
                rect[:2] = (0, 0)
                surf = True
            else:
                surf = False
            for c in range(bevel):
                p = [rect[0] + c, rect[1] + c]
                q = [a + b - c - 1 for a, b in zip(rect[:2], rect[2:])]
                v1 = 128 - c / bevel * 128
                v2 = c / bevel * 96 - 96
                col1 = col2 = colour
                if v1:
                    col1 = [min(i + v1, 255) for i in col1]
                if v2:
                    col2 = [max(i + v2, 0) for i in col1]
                try:
                    draw_hline(dest, p[0], q[0], p[1], col1)
                    draw_vline(dest, p[0], p[1], q[1], col1)
                    draw_hline(dest, p[0], q[0], q[1], col2)
                    draw_vline(dest, q[0], p[1], q[1], col2)
                except:
                    print_exc()
            if filled:
                if grad_col is None:
                    draw_rect(dest, colour, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
                else:
                    gradient_rectangle(dest, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
            return dest if surf else rect
        ctr = max(colour)
        contrast = min(round(ctr) + 2 >> 2 << 2, 255)
        data = tuple(rect[2:]) + (grad_col, grad_angle, contrast)
        s = br_surf.get(data)
        if s is None:
            colour2 = (contrast,) * 3
            s = pygame.Surface(rect[2:], SRCALPHA)
            s.fill((1, 2, 3))
            s.set_colorkey((1, 2, 3))
            for c in range(bevel):
                p = [c, c]
                q = [i - c - 1 for i in rect[2:]]
                v1 = 128 - c / bevel * 128
                v2 = c / bevel * 96 - 96
                col1 = col2 = colour2
                if v1:
                    col1 = [min(i + v1, 255) for i in col1]
                if v2:
                    col2 = [max(i + v2, 0) for i in col1]
                draw_hline(s, p[0], q[0], p[1], col1)
                draw_vline(s, p[0], p[1], q[1], col1)
                draw_hline(s, p[0], q[0], q[1], col2)
                draw_vline(s, q[0], p[1], q[1], col2)
            if grad_col is None:
                draw_rect(s, colour2, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
            else:
                gradient_rectangle(s, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
            if cache:
                br_surf[data] = s
        colour = tuple(round(i * 255 / ctr) for i in colour)
        return blit_complex(dest, s, rect[:2], angle=angle, alpha=alpha, colour=colour)

def reg_polygon_complex(dest, centre, colour, sides, width, height, angle=pi / 4, alpha=255, thickness=0, repetition=1, filled=False, rotation=0, soft=False, attempts=128):
    width = max(round(width), 0)
    height = max(round(height), 0)
    try:
        newS = pygame.Surface((width << 1, height << 1), SRCALPHA)
    except:
        print_exc()
        return
    newS.fill((0,) * 4)
    repetition = int(repetition)
    draw_direction = 1 if repetition >= 0 else -1
    if draw_direction >= 0:
        a = draw_direction
        b = repetition + 1
    else:
        a = repetition + 1
        b = -draw_direction
    if sides > 32:
        sides = 0
    elif sides < 0:
        sides = 0
    draw_direction *= max(thickness, 3) - 2
    loop = a
    setted = filled
    att = 0
    while loop < b + draw_direction:
        if att >= attempts:
            break
        att += 1
        if loop - b > 0:
            loop = b
        move_direction = loop / repetition + 0.2
        points = []
        if soft:
            colourU = tuple(colour) + (min(round(255 * move_direction + 8), 255),)
        else:
            colourU = (colour[0] * move_direction + 8, colour[1] * move_direction + 8, colour[2] * move_direction + 8)
            colourU = list(map(lambda c: min(c, 255), colourU))
        try:
            size = (min(width, height) - loop)
            thickness = int(min(thickness, size))
            if setted:
                thickness = 0
                setted = False
            elif not filled:
                thickness = thickness + 4 >> 1
            if sides:
                for p in range(sides):
                    points.append((
                        width + (width - loop) * cos(angle + p * tau / sides),
                        height + (height - loop) * sin(angle + p * tau / sides),
                    ))
                pygame.draw.polygon(newS, colourU, points, thickness)
            else:
                if thickness > loop:
                    thickness = 0
                pygame.draw.ellipse(newS, colourU, (loop, loop, (width - loop) << 1, (height - loop) << 1), thickness)
        except:
            print_exc()
        loop += draw_direction
    pos = [centre[0] - width, centre[1] - height]
    return blit_complex(dest, newS, pos, alpha, rotation, copy=False)

def concentric_circle(dest, colour, pos, radius, width=0, fill_ratio=1, alpha=255, gradient=False, filled=False, cache=True):
    reverse = fill_ratio < 0
    radius = max(0, round(radius * 2) / 2)
    if min(alpha, radius) > 0:
        cc_surf = globals().setdefault("cc_surf", {})
        width = max(0, min(width, radius))
        tw = width / radius
        fill_ratio = min(1, abs(fill_ratio))
        cr = bit_crush(round(fill_ratio * 64), 3)
        wr = bit_crush(round(tw * 64), 3)
        colour = verify_colour(colour)
        data = (radius, wr, cr, gradient, filled)
        s = cc_surf.get(data)
        if s == 0:
            cache = False
        if not s:
            radius2 = min(128, bit_crush(radius, 5, ceil))
            width2 = max(2, round(radius2 * tw))
            colour2 = (255,) * 3
            data2 = tuple(colour2) + (radius2 * 2, wr, cr, gradient, filled)
            s2 = cc_surf.get(data2)
            if not s2:
                # print(str(data2) + " concircle created!")
                width2 = round(width2)
                size = [radius2 * 2] * 2
                size2 = [round(radius2 * 4), round(radius2 * 4) + 1]
                s2 = pygame.Surface(size2, SRCALPHA)
                s2.fill((0,) * 4)
                circles = round(radius2 * 2 * fill_ratio / width2)
                col = colour2
                r = radius2 * 2
                for i in range(circles):
                    if reverse:
                        it = (i + 1) / circles
                    else:
                        it = 1 - i / circles
                    if filled and i == circles - 1:
                        width2 = 0
                    if gradient:
                        col = adj_colour(colour2, 0, it)
                    c = col + (round(255 * min(1, (it + gradient))),)
                    pygame.draw.circle(s2, c, [i - 1 for i in size], r, min(r, width2 + (width2 > 0)))
                    r -= width2
                cc_surf[data2] = s2
            size3 = [round(radius * 2) for i in range(2)]
            s = custom_scale(s2, size3, antialias=1)
            if cache:
                cc_surf[data] = s
            # print(str(data2) + " concircle copied to " + str(data) + " concircle!")
        p = [i - radius for i in pos]
        return blit_complex(dest, s, p, alpha=alpha, colour=colour)

def anima_rectangle(surface, colour, rect, frame, count=2, speed=1, flash=1, ratio=0, reduction=0.3):
    if flash:
        n = 4
        a = (ratio * speed * n) % (flash * n)
        if a < speed:
            pos = round((a * 4 / flash - 1) * rect[3])
            bevel_rectangle(surface, (255,) * 3, (rect[0], rect[1] + max(pos, 0), rect[2], min(rect[3] + pos, rect[3]) - max(pos, 0)), 0, 159)
            bevel_rectangle(surface, (255,) * 3, (rect[0], rect[1] + max(pos + 8, 0), rect[2], min(rect[3] + pos, rect[3]) - max(pos + 16, 0)), 0, 159)
    perimeter = rect[2] * 2 + rect[3] * 2
    increment = 0
    f = frame
    while frame > 0:
        c = list(colour)
        for i in range(count):
            pos = perimeter * ((i / count - increment / perimeter + ratio * speed) % 1)
            side = 0
            if pos >= rect[2]:
                pos -= rect[2]
                side = 1
                if pos >= rect[3]:
                    pos -= rect[3]
                    side = 2
                    if pos >= rect[2]:
                        pos -= rect[2]
                        side = 3
            if side == 0:
                r = [round(rect[0] + pos), round(rect[1] + 0.5)]
            elif side == 1:
                r = [round(rect[0] + rect[2]), round(rect[1] + pos + 0.5)]
            elif side == 2:
                r = [round(rect[0] + rect[2] - pos), round(rect[1] + rect[3] + 0.5)]
            else:
                r = [round(rect[0]), round(rect[1] + rect[3] - pos + 0.5)]
            r_rect = [r[0] - round(frame), r[1] - round(frame), round(frame) << 1, round(frame) << 1]
            pygame.draw.rect(surface, adj_colour(c, (frame * 8) - f * 4), r_rect)
        frame -= reduction
        increment += 3

def text_objects(text, font, colour):
    text_surface = font.render(text, True, colour)
    if not text_surface.get_flags() & SRCALPHA:
        try:
            text_surface = text_surface.convert_alpha()
        except:
            pass
    return text_surface, text_surface.get_rect()

def surface_font(text, colour, size, font):
    size = round(size)
    ct_font = globals().setdefault("ct_font", {})
    data = (size, font)
    f = ct_font.get(data, None)
    if not f:
        f = ct_font[data] = pygame.font.SysFont(font, size)
    for i in range(4):
        try:
            return text_objects(text, f, colour)
        except:
            if i >= 3:
                raise
            f = ct_font[data] = pygame.font.SysFont(font, size)

def message_display(text, size, pos, colour=(255,) * 3, surface=None, font="Comic Sans MS", alpha=255, align=1):
    TextSurf, TextRect = surface_font(text, colour, size, font)
    if align == 1:
        TextRect.center = pos
    elif align == 0:
        TextRect = list(pos) + TextRect[2:]
    elif align == 2:
        TextRect = [y - x for x, y in zip(TextRect[2:], pos)] + TextRect[2:]
    if surface:
        blit_complex(surface, TextSurf, TextRect, alpha, copy=False)
        return TextRect
    else:
        return TextSurf

def char_display(char, size, font="Comic Sans MS"):
    size = round(size)
    cs_font = globals().setdefault("cs_font", {})
    data = (char, size, font)
    f = cs_font.get(data, None)
    if not f:
        f = surface_font(char, (255,) * 3, size, font)[0]
    return f


class KeyList(list):

    def __getitem__(self, k):
        return super().__getitem__(k & -1073741825)


class alist(collections.abc.MutableSequence, collections.abc.Callable):

    maxoff = (1 << 24) - 1
    minsize = 9
    __slots__ = ("hash", "block", "offs", "size", "data", "frozenset", "queries", "_index")

    # For thread-safety: Waits until the list is not busy performing an operation.
    def waiting(self):
        func = self
        def call(self, *args, force=False, **kwargs):
            if not force and type(self.block) is concurrent.futures.Future:
                self.block.result(timeout=12)
            return func(self, *args, **kwargs)
        return call

    # For thread-safety: Blocks the list until the operation is complete.
    def blocking(self):
        func = self
        def call(self, *args, force=False, **kwargs):
            if not force and type(self.block) is concurrent.futures.Future:
                self.block.result(timeout=12)
            self.block = concurrent.futures.Future()
            self.hash = None
            self.frozenset = None
            try:
                del self.queries
            except AttributeError:
                pass
            try:
                output = func(self, *args, **kwargs)
            except:
                try:
                    self.block.set_result(None)
                except concurrent.futures.InvalidStateError:
                    pass
                raise
            try:
                self.block.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
            return output
        return call

    # Init takes arguments and casts to a deque if possible, else generates as a single value. Allocates space equal to 3 times the length of the input iterable.
    def __init__(self, *args, fromarray=False, **void):
        fut = getattr(self, "block", None)
        self.block = concurrent.futures.Future()
        self.hash = None
        self.frozenset = None
        if fut:
            try:
                del self.queries
            except AttributeError:
                pass
            try:
                del self._index
            except AttributeError:
                pass
        if not args:
            self.offs = 0
            self.size = 0
            self.data = None
            try:
                self.block.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
            if fut:
                try:
                    fut.set_result(None)
                except concurrent.futures.InvalidStateError:
                    pass
            return
        elif len(args) == 1:
            iterable = args[0]
        else:
            iterable = args
        if issubclass(type(iterable), self.__class__) and iterable:
            self.offs = iterable.offs
            self.size = iterable.size
            if fromarray:
                self.data = iterable.data
            else:
                self.data = iterable.data.copy()
        elif fromarray:
            self.offs = 0
            self.size = len(iterable)
            self.data = iterable
        else:
            if not issubclass(type(iterable), collections.abc.Sequence) or issubclass(type(iterable), collections.abc.Mapping) or type(iterable) in (str, bytes):
                try:
                    iterable = deque(iterable)
                except TypeError:
                    iterable = [iterable]
            self.size = len(iterable)
            size = max(self.minsize, self.size * 3)
            self.offs = size // 3
            self.data = np.empty(size, dtype=object)
            self.view[:] = iterable
        if not fut or fut.done():
            try:
                self.block.set_result(None)
            except concurrent.futures.InvalidStateError:
                pass
            if fut:
                try:
                    fut.set_result(None)
                except concurrent.futures.InvalidStateError:
                    pass

    def __getstate__(self):
        return self.data, self.offs, self.size

    def __setstate__(self, s):
        if type(s) is tuple:
            if len(s) == 2:
                if s[0] is None:
                    for k, v in s[1].items():
                        setattr(self, k, v)
                    self.block = None
                    return
            elif len(s) == 3:
                self.data, self.offs, self.size = s
                self.hash = None
                self.frozenset = None
                try:
                    del self.queries
                except AttributeError:
                    pass
                self.block = None
                return
        raise TypeError("Unpickling failed:", s)

    def __getattr__(self, k):
        try:
            return self.__getattribute__(k)
        except AttributeError:
            pass
        return getattr(self.__getattribute__("view"), k)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(dir(self.view))
        return data

    # Returns a numpy array representing the items currently "in" the list.
    @property
    def view(self):
        data = self.__getattribute__("data")
        if data is None:
            return []
        offs, size = [self.__getattribute__(i) for i in ("offs", "size")]
        return data[offs:offs + size]

    @waiting
    def __call__(self, arg=1, *void1, **void2):
        if arg == 1:
            return self.copy()
        return self * arg

    # Returns the hash value of the data in the list.
    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.view.tobytes())
        return self.hash

    def to_frozenset(self):
        if self.frozenset is None:
            self.frozenset = frozenset(self)
        return self.frozenset

    # Basic functions
    __str__ = lambda self: "[" + ", ".join(repr(i) for i in iter(self)) + "]"
    __repr__ = lambda self: f"{self.__class__.__name__}({tuple(self) if self.__bool__() else ''})"
    __bool__ = lambda self: self.size > 0

    # Arithmetic functions

    @blocking
    def __iadd__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.add(arr, iterable, out=arr)
        return self

    @blocking
    def __isub__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.subtract(arr, iterable, out=arr)
        return self

    @blocking
    def __imul__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.multiply(arr, iterable, out=arr)
        return self

    @blocking
    def __imatmul__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        temp = np.matmul(arr, iterable)
        self.size = len(temp)
        arr[:self.size] = temp
        return self

    @blocking
    def __itruediv__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.true_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __ifloordiv__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.floor_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __imod__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.mod(arr, iterable, out=arr)
        return self

    @blocking
    def __ipow__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.power(arr, iterable, out=arr)
        return self

    @blocking
    def __ilshift__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        try:
            np.left_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.multiply(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __irshift__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        try:
            np.right_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.divide(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __iand__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_and(arr, iterable, out=arr)
        return self

    @blocking
    def __ixor__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_xor(arr, iterable, out=arr)
        return self

    @blocking
    def __ior__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_or(arr, iterable, out=arr)
        return self

    @waiting
    def __neg__(self):
        return self.__class__(-self.view)

    @waiting
    def __pos__(self):
        return self

    @waiting
    def __abs__(self):
        d = self.data
        return self.__class__(np.abs(self.view))

    @waiting
    def __invert__(self):
        return self.__class__(np.invert(self.view))

    @waiting
    def __add__(self, other):
        temp = self.copy()
        temp += other
        return temp

    @waiting
    def __sub__(self, other):
        temp = self.copy()
        temp -= other
        return temp

    @waiting
    def __mul__(self, other):
        temp = self.copy()
        temp *= other
        return temp

    @waiting
    def __matmul__(self, other):
        temp1 = self.view
        temp2 = self.to_iterable(other)
        result = temp1 @ temp2
        return self.__class__(result)

    @waiting
    def __truediv__(self, other):
        temp = self.copy()
        temp /= other
        return temp

    @waiting
    def __floordiv__(self, other):
        temp = self.copy()
        temp //= other
        return temp

    @waiting
    def __mod__(self, other):
        temp = self.copy()
        temp %= other
        return temp

    @waiting
    def __pow__(self, other):
        temp = self.copy()
        temp **= other
        return temp

    @waiting
    def __lshift__(self, other):
        temp = self.copy()
        temp <<= other
        return temp

    @waiting
    def __rshift__(self, other):
        temp = self.copy()
        temp >>= other
        return temp

    @waiting
    def __and__(self, other):
        temp = self.copy()
        temp &= other
        return temp

    @waiting
    def __xor__(self, other):
        temp = self.copy()
        temp ^= other
        return temp

    @waiting
    def __or__(self, other):
        temp = self.copy()
        temp |= other
        return temp

    @waiting
    def __round__(self, prec=0):
        temp = np.round(self.view, prec)
        return self.__class__(temp)

    @waiting
    def __trunc__(self):
        temp = np.trunc(self.view)
        return self.__class__(temp)

    @waiting
    def __floor__(self):
        temp = np.floor(self.view)
        return self.__class__(temp)

    @waiting
    def __ceil__(self):
        temp = np.ceil(self.view)
        return self.__class__(temp)

    __index__ = lambda self: self.view
    __radd__ = __add__
    __rsub__ = lambda self, other: -self + other
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

    @waiting
    def __rtruediv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.true_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rfloordiv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.floor_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rmod__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.mod(iterable, arr, out=arr)
        return temp

    @waiting
    def __rpow__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.power(iterable, arr, out=arr)
        return temp

    @waiting
    def __rlshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        try:
            np.left_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.multiply(iterable, np.power(2, arr), out=arr)
        return temp

    @waiting
    def __rrshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        try:
            np.right_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.divide(iterable, np.power(2, arr), out=arr)
        return temp

    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    # Comparison operations

    @waiting
    def __lt__(self, other):
        it = self.to_iterable(other)
        return self.view < other

    @waiting
    def __le__(self, other):
        it = self.to_iterable(other)
        return self.view <= other

    @waiting
    def __eq__(self, other):
        try:
            it = self.to_iterable(other)
            return self.view == other
        except (TypeError, IndexError):
            return

    @waiting
    def __ne__(self, other):
        try:
            it = self.to_iterable(other)
            return self.view != other
        except (TypeError, IndexError):
            return

    @waiting
    def __gt__(self, other):
        it = self.to_iterable(other)
        return self.view > other

    @waiting
    def __ge__(self, other):
        it = self.to_iterable(other)
        return self.view >= other

    # Takes ints, floats, slices and iterables for indexing
    @waiting
    def __getitem__(self, *args):
        if len(args) == 1:
            key = args[0]
            if type(key) in (float, complex):
                return get(self.view, key, 1)
            if type(key) is int:
                try:
                    key = key % self.size
                except ZeroDivisionError:
                    raise IndexError("Array List index out of range.")
                return self.view.__getitem__(key)
            if type(key) is slice:
                if key.step in (None, 1):
                    start = key.start
                    if start is None:
                        start = 0
                    stop = key.stop
                    if stop is None:
                        stop = self.size
                    if start >= self.size or stop <= start and (stop >= 0 or stop + self.size <= start):
                        return self.__class__()
                    temp = self.__class__(self, fromarray=True)
                    if start < 0:
                        if start < -self.size:
                            start = 0
                        else:
                            start %= self.size
                    if stop < 0:
                        stop %= self.size
                    elif stop > self.size:
                        stop = self.size
                    temp.offs += start
                    temp.size = stop - start
                    if not temp.size:
                        return self.__class__()
                    return temp
            return self.__class__(self.view.__getitem__(key), fromarray=True)
        return self.__class__(self.view.__getitem__(*args), fromarray=True)

    # Takes ints, slices and iterables for indexing
    @blocking
    def __setitem__(self, *args):
        if len(args) == 2:
            key = args[0]
            if type(key) is int:
                try:
                    key = key % self.size
                except ZeroDivisionError:
                    raise IndexError("Array List index out of range.")
            return self.view.__setitem__(key, args[1])
        return self.view.__setitem__(*args)

    # Takes ints and slices for indexing
    @blocking
    def __delitem__(self, key):
        if type(key) is slice:
            s = key.indices(self.size)
            return self.pops(xrange(*s))
        try:
            len(key)
        except TypeError:
            return self.pop(key, force=True)
        return self.pops(key)

    # Basic sequence functions
    __len__ = lambda self: self.size
    __length_hint__ = __len__
    __iter__ = lambda self: iter(self.view)
    __reversed__ = lambda self: iter(np.flip(self.view))

    def next(self):
        try:
            self._index = (self._index + 1) % self.size
        except AttributeError:
            self._index = 0
        return self[self._index]

    @waiting
    def __bytes__(self):
        return bytes(round(i) & 255 for i in self.view)

    def __contains__(self, item):
        try:
            if self.queries >= 8:
                return item in self.to_frozenset()
            if self.frozenset is not None:
                return item in self.frozenset
            self.queries += 1
        except AttributeError:
            self.queries = 1
        return item in self.view

    __copy__ = lambda self: self.copy()

    # Creates an iterable from an iterator, making sure the shape matches.
    def to_iterable(self, other, force=False):
        if not issubclass(type(other), collections.abc.Sequence) or issubclass(type(other), collections.abc.Mapping):
            try:
                other = list(other)
            except TypeError:
                other = [other]
        if len(other) not in (1, self.size) and not force:
            raise IndexError(f"Unable to perform operation on objects with size {self.size} and {len(other)}.")
        return other

    @blocking
    def clear(self):
        self.size = 0
        if self.data is not None:
            self.offs = len(self.data) // 3
        else:
            self.offs = 0
        return self

    @waiting
    def copy(self):
        return self.__class__(self.view)

    @waiting
    def sort(self, *args, **kwargs):
        return self.__class__(sorted(self.view, *args, **kwargs))

    @waiting
    def shuffle(self, *args, **kwargs):
        return self.__class__(shuffle(self.view, *args, **kwargs))

    @waiting
    def reverse(self):
        return self.__class__(np.flip(self.view))

    # Rotates the list a certain amount of steps, using np.roll for large rotate operations.
    @blocking
    def rotate(self, steps):
        s = self.size
        if not s:
            return self
        steps %= s
        if steps > s >> 1:
            steps -= s
        if abs(steps) < self.minsize:
            while steps > 0:
                self.appendleft(self.popright(force=True), force=True)
                steps -= 1
            while steps < 0:
                self.appendright(self.popleft(force=True), force=True)
                steps += 1
            return self
        self.offs = (len(self.data) - self.size) // 3
        self.view[:] = np.roll(self.view, steps)
        return self

    @blocking
    def rotateleft(self, steps):
        return self.rotate(-steps, force=True)

    rotateright = rotate

    # Re-initializes the list if the positional offsets are too large or if the list is empty.
    @blocking
    def isempty(self):
        if self.size:
            if abs(len(self.data) // 3 - self.offs) > self.maxoff:
                self.reconstitute(force=True)
            return False
        if len(self.data) > 4096:
            self.data = None
            self.offs = 0
        elif self.data is not None:
            self.offs = len(self.data) // 3
        return True

    # For compatibility with dict.get
    @waiting
    def get(self, key, default=None):
        try:
            return self[key]
        except LookupError:
            return default

    @blocking
    def popleft(self):
        temp = self.data[self.offs]
        self.offs += 1
        self.size -= 1
        self.isempty(force=True)
        return temp

    @blocking
    def popright(self):
        temp = self.data[self.offs + self.size - 1]
        self.size -= 1
        self.isempty(force=True)
        return temp

    # Removes an item from the list. O(n) time complexity.
    @blocking
    def pop(self, index=None, *args):
        try:
            if index is None:
                return self.popright(force=True)
            if index >= len(self.data):
                return self.popright(force=True)
            elif index == 0:
                return self.popleft(force=True)
            index %= self.size
            temp = self.data[index + self.offs]
            if index > self.size >> 1:
                self.view[index:-1] = self.data[self.offs + index + 1:self.offs + self.size]
            else:
                self.view[1:index + 1] = self.data[self.offs:self.offs + index]
                self.offs += 1
            self.size -= 1
            return temp
        except LookupError:
            if not args:
                raise
            return args[0]

    # Inserts an item into the list. O(n) time complexity.
    @blocking
    def insert(self, index, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if index >= self.size:
            return self.append(value, force=True)
        elif index == 0:
            return self.appendleft(value, force=True)
        index %= self.size
        if index > self.size >> 1:
            if self.size + self.offs + 1 >= len(self.data):
                self.reconstitute(force=True)
            self.size += 1
            self.view[index + 1:] = self.view[index:-1]
        else:
            if self.offs < 1:
                self.reconstitute(force=True)
            self.size += 1
            self.offs -= 1
            self.view[:index] = self.view[1:index + 1]
        self.view[index] = value
        return self

    # Insertion sort using a binary search to find target position. O(n) time complexity.
    @blocking
    def insort(self, value, key=None, sorted=True):
        if self.data is None:
            self.__init__((value,))
            return self
        if not sorted:
            self.__init__(sorted(self, key=key))
        if key is None:
            return self.insert(np.searchsorted(self.view, value), value, force=True)
        v = key(value)
        x = self.size
        index = (x >> 1) + self.offs
        gap = 3 + x >> 2
        seen = {}
        d = self.data
        while index not in seen and index >= self.offs and index < self.offs + self.size:
            check = key(d[index])
            if check < v:
                seen[index] = True
                index += gap
            else:
                seen[index] = False
                index -= gap
            gap = 1 + gap >> 1
        index -= self.offs - seen.get(index, 0)
        if index <= 0:
            return self.appendleft(value, force=True)
        return self.insert(index, value, force=True)

    # Removes all instances of a certain value from the list.
    @blocking
    def remove(self, value, key=None, sorted=False):
        pops = self.search(value, key, sorted, force=True)
        if pops:
            self.pops(pops, force=True)
        return self

    discard = remove

    # Removes all duplicate values from the list.
    @blocking
    def removedups(self, sorted=True):
        if sorted:
            try:
                temp = np.unique(self.view)
            except:
                temp = sorted(set(self.view))
        elif sorted is None:
            temp = tuple(set(self.view))
        else:
            temp = {}
            for x in self.view:
                if x not in temp:
                    temp[x] = None
            temp = tuple(temp.keys())
        self.size = len(temp)
        self.offs = (len(self.data) - self.size) // 3
        self.view[:] = temp
        return self

    uniq = unique = removedups

    # Returns first matching value in list.
    @waiting
    def index(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[0]

    # Returns last matching value in list.
    @waiting
    def rindex(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[-1]

    # Returns indices representing positions for all instances of the target found in list, using binary search when applicable.
    @waiting
    def search(self, value, key=None, sorted=False):
        if key is None:
            if sorted and self.size > self.minsize:
                i = np.searchsorted(self.view, value)
                if self.view[i] != value:
                    raise IndexError(f"{value} not found.")
                pops = self.__class__()
                pops.append(i)
                for x in range(i + self.offs - 1, -1, -1):
                    if self.data[x] == value:
                        pops.appendleft(x - self.offs)
                    else:
                        break
                for x in range(i + self.offs + 1, self.offs + self.size):
                    if self.data[x] == value:
                        pops.append(x - self.offs)
                    else:
                        break
                return pops
            else:
                return self.__class__(np.arange(self.size, dtype=np.uint32)[self.view == value])
        if sorted:
            v = value
            d = self.data
            pops = self.__class__()
            x = len(d)
            index = (x >> 1) + self.offs
            gap = x >> 2
            seen = {}
            while index not in seen and index >= self.offs and index < self.offs + self.size:
                check = key(d[index])
                if check < v:
                    seen[index] = True
                    index += gap
                elif check == v:
                    break
                else:
                    seen[index] = False
                    index -= gap
                gap = 1 + gap >> 1
            i = index + seen.get(index, 0)
            while i in d and key(d[i]) == v:
                pops.append(i - self.offs)
                i += 1
            i = index + seen.get(index, 0) - 1
            while i in d and key(d[i]) == v:
                pops.append(i - self.offs)
                i -= 1
        else:
            pops = self.__class__(i for i, x in enumerate(self.view) if key(x) == value)
        if not pops:
            raise IndexError(f"{value} not found.")
        return pops

    find = findall = search

    # Counts the amount of instances of the target within the list.
    @waiting
    def count(self, value, key=None):
        if key is None:
            return sum(self.view == value)
        return sum(1 for i in self if key(i) == value)

    concat = lambda self, value: self.__class__(np.concatenate([self.view, value]))

    # Appends item at the start of the list, reallocating when necessary.
    @blocking
    def appendleft(self, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if self.offs <= 0:
            self.reconstitute(force=True)
        self.offs -= 1
        self.size += 1
        self.data[self.offs] = value
        return self

    # Appends item at the end of the list, reallocating when necessary.
    @blocking
    def append(self, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if self.offs + self.size >= len(self.data):
            self.reconstitute(force=True)
        self.data[self.offs + self.size] = value
        self.size += 1
        return self

    add = lambda self, value: object.__getattribute__(self, ("append", "appendleft")[random.randint(0, 1)])(value)
    appendright = append

    # Appends iterable at the start of the list, reallocating when necessary.
    @blocking
    def extendleft(self, value):
        if self.data is None or not self.size:
            self.__init__(reversed(value))
            return self
        value = self.to_iterable(reversed(value), force=True)
        if self.offs >= len(value):
            self.data[self.offs - len(value):self.offs] = value
            self.offs -= len(value)
            self.size += len(value)
            return self
        self.__init__(np.concatenate([value, self.view]), fromarray=True)
        return self

    # Appends iterable at the end of the list, reallocating when necessary.
    @blocking
    def extend(self, value):
        if self.data is None or not self.size:
            self.__init__(value)
            return self
        value = self.to_iterable(value, force=True)
        if len(self.data) - self.offs - self.size >= len(value):
            self.data[self.offs + self.size:self.offs + self.size + len(value)] = value
            self.size += len(value)
            return self
        self.__init__(np.concatenate([self.view, value]), fromarray=True)
        return self

    extendright = extend

    # Similar to str.join
    @waiting
    def join(self, iterable):
        iterable = self.to_iterable(iterable)
        temp = deque()
        for i, v in enumerate(iterable):
            try:
                temp.extend(v)
            except TypeError:
                temp.append(v)
            if i != len(iterable) - 1:
                temp.extend(self.view)
        return self.__class__(temp)

    # Similar to str.replace().
    @blocking
    def replace(self, original, new):
        view = self.view
        for i, v in enumerate(view):
            if v == original:
                view[i] = new
        return self

    # Fills list with value(s).
    @blocking
    def fill(self, value):
        self.offs = (len(self.data) - self.size) // 3
        self.view[:] = value
        return self

    # For compatibility with dict() attributes.
    keys = lambda self: range(len(self))
    values = lambda self: iter(self)
    items = lambda self: enumerate(self)

    # For compatibility with set() attributes.
    @waiting
    def isdisjoint(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().isdisjoint(other)

    @waiting
    def issubset(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().issubset(other)

    @waiting
    def issuperset(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().issuperset(other)

    @waiting
    def union(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().union(*args)

    @waiting
    def intersection(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().intersection(*args)

    @waiting
    def difference(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().difference(*args)

    @waiting
    def symmetric_difference(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().symmetric_difference(other)

    @blocking
    def update(self, *others, uniq=True):
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            self.extend(other, force=True)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def intersection_update(self, *others, uniq=True):
        pops = set()
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            for i, v in enumerate(self):
                if v not in other:
                    pops.add(i)
        self.pops(pops)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def difference_update(self, *others, uniq=True):
        pops = set()
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            for i, v in enumerate(self):
                if v in other:
                    pops.add(i)
        self.pops(pops)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def symmetric_difference_update(self, other):
        data = set(self)
        if issubclass(other, collections.abc.Mapping):
            other = other.values()
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        data.symmetric_difference_update(other)
        self.__init__(data)
        return self

    # Clips all values in list to input boundaries.
    @blocking
    def clip(self, a, b=None):
        if b is None:
            b = -a
        if a > b:
            a, b = b, a
        arr = self.view
        np.clip(arr, a, b, out=arr)
        return self

    # Casting values to various types.

    @waiting
    def real(self):
        return self.__class__(np.real(self.view))

    @waiting
    def imag(self):
        return self.__class__(np.imag(self.view))

    @waiting
    def float(self):
        return self.__class__(float(i.real) for i in self.view)

    @waiting
    def complex(self):
        return self.__class__(complex(i) for i in self.view)

    @waiting
    def mpf(self):
        return self.__class__(mpf(i.real) for i in self.view)

    @waiting
    def sum(self):
        return np.sum(self.view)

    @waiting
    def mean(self):
        return np.mean(self.view)

    @waiting
    def product(self):
        return np.prod(self.view)

    prod = product

    # Reallocates list.
    @blocking
    def reconstitute(self, data=None):
        self.__init__(data if data is not None else self.view)
        return self

    # Removes items according to an array of indices.
    @blocking
    def delitems(self, iterable):
        iterable = self.to_iterable(iterable, force=True)
        if len(iterable) < 1:
            return self
        if len(iterable) == 1:
            return self.pop(iterable[0], force=True)
        temp = np.delete(self.view, np.asarray(iterable, dtype=np.int32))
        self.size = len(temp)
        if self.data is not None:
            self.offs = (len(self.data) - self.size) // 3
            self.view[:] = temp
        else:
            self.reconstitute(temp, force=True)
        return self

    pops = delitems

hlist = alist

arange = lambda a, b=None, c=None: alist(range(a, b, c))

azero = lambda size: alist(repeat(0, size))


# Runs ffprobe on a file or url, returning the duration if possible.
def _get_duration(filename, _timeout=12):
    command = (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "format=duration,bit_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        filename,
    )
    resp = None
    try:
        proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        fut = submit(proc.wait, timeout=_timeout)
        res = fut.result(timeout=_timeout)
        resp = proc.stdout.read().split()
    except:
        with suppress():
            proc.kill()
        print_exc()
    try:
        dur = float(resp[0])
    except (IndexError, ValueError):
        dur = None
    bps = None
    if len(resp) > 1:
        with suppress(ValueError):
            bps = float(resp[1])
    return dur, bps

def get_duration(filename):
    if not is_url(filename) and filename.endswith(".pcm"):
        return os.path.getsize(filename) / (48000 * 2 * 2)
    if filename:
        dur, bps = _get_duration(filename, 4)
        if not dur and is_url(filename):
            with requests.get(filename, headers=Request.header(), stream=True) as resp:
                head = fcdict(resp.headers)
                if "Content-Length" not in head:
                    return _get_duration(filename, 20)[0]
                if bps:
                    return (int(head["Content-Length"]) << 3) / bps
                ctype = [e.strip() for e in head.get("Content-Type", "").split(";") if "/" in e][0]
                if ctype.split("/", 1)[0] not in ("audio", "video"):
                    return nan
                if ctype == "audio/midi":
                    return nan
                it = resp.iter_content(65536)
                data = next(it)
            ident = str(magic.from_buffer(data))
            try:
                bitrate = regexp("[0-9]+\\s.bps").findall(ident)[0].casefold()
            except IndexError:
                return _get_duration(filename, 16)[0]
            bps, key = bitrate.split(None, 1)
            bps = float(bps)
            if key.startswith("k"):
                bps *= 1e3
            elif key.startswith("m"):
                bps *= 1e6
            elif key.startswith("g"):
                bps *= 1e9
            return (int(head["Content-Length"]) << 3) / bps
        return dur

def time_disp(s, rounded=True):
    if not isfinite(s):
        return str(s)
    if rounded:
        s = round(s)
    output = str(s % 60)
    if len(output) < 2:
        output = "0" + output
    if s >= 60:
        temp = str((s // 60) % 60)
        if len(temp) < 2 and s >= 3600:
            temp = "0" + temp
        output = temp + ":" + output
        if s >= 3600:
            temp = str((s // 3600) % 24)
            if len(temp) < 2 and s >= 86400:
                temp = "0" + temp
            output = temp + ":" + output
            if s >= 86400:
                output = str(s // 86400) + ":" + output
    else:
        output = "0:" + output
    return output

is_youtube_stream = lambda url: re.findall("^https?:\\/\\/r[0-9]+---.{2}-\\w+-\\w{4,}\\.googlevideo\\.com", url)
# Regex moment - Smudge