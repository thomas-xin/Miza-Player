import os, sys, json, traceback, subprocess, copy, time, concurrent.futures

utc = time.time
print = lambda *args, sep=" ", end="\n": sys.stdout.write(str(sep).join(map(str, args)) + end)
from concurrent.futures import thread

def _adjust_thread_count(self):
    # if idle threads are available, don't spin new threads
    if self._idle_semaphore.acquire(timeout=0):
        return

    # When the executor gets lost, the weakref callback will wake up
    # the worker threads.
    def weakref_cb(_, q=self._work_queue):
        q.put(None)

    num_threads = len(self._threads)
    if num_threads < self._max_workers:
        thread_name = '%s_%d' % (self._thread_name_prefix or self, num_threads)
        t = thread.threading.Thread(
            name=thread_name,
            target=thread._worker,
            args=(
                thread.weakref.ref(self, weakref_cb),
                self._work_queue,
                self._initializer,
                self._initargs,
            ),
            daemon=True
        )
        t.start()
        self._threads.add(t)
        thread._threads_queues[t] = self._work_queue

concurrent.futures.ThreadPoolExecutor._adjust_thread_count = lambda self: _adjust_thread_count(self)

exc = concurrent.futures.ThreadPoolExecutor(max_workers=48)
submit = exc.submit
print_exc = traceback.print_exc

is_url = lambda url: "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")

downloader = concurrent.futures.Future()
lyrics_scraper = concurrent.futures.Future()
def import_audio_downloader():
    try:
        audio_downloader = __import__("audio_downloader")
        # globals().update(audio_downloader.__dict__)
        globals()["ytdl"] = ytdl = audio_downloader.AudioDownloader()
        downloader.set_result(ytdl)
        lyrics_scraper.set_result(audio_downloader.get_lyrics)
    except Exception as ex:
        print_exc()
        downloader.set_exception(ex)
        lyrics_scraper.set_exception(ex)


from rainbow_print import *


def as_str(s):
    if type(s) in (bytes, bytearray, memoryview):
        return bytes(s).decode("utf-8", "replace")
    return str(s)


collections2f = "misc/collections2.tmp"
try:
    update_collections = utc() - os.path.getmtime(collections2f) > 3600
except FileNotFoundError:
    update_collections = True

hasmisc = os.path.exists("misc")
argp = ["py"]
pyv = sys.version_info[1]
if hasmisc:
    argp = ["py", f"-3.{pyv}"]
    if update_collections:
        from install_update_p import *

import psutil

if hasmisc:
    submit(import_audio_downloader)
    mixer = psutil.Popen(argp + ["misc/mixer.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
def mixer_submit(s, force, debug):
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
    if debug:
        sys.stdout.write(s)
    mixer.stdin.write(s.encode("utf-8"))
    mixer.stdin.flush()

mixer.state = lambda i=0: state(i)
mixer.clear = lambda: clear()
mixer.drop = lambda i=0: drop(i)
mixer.submit = lambda s, force=True, debug=False: submit(mixer_submit, s, force, debug)

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
audio_default = dict(
    volume=1,
    speed=1,
    pitch=0,
    pan=1,
    bassboost=0,
    reverb=0,
    compressor=0,
    chorus=0,
    nightcore=0,
)
control_default = cdict(
    shuffle=1,
    loop=1,
    silenceremove=0,
    presearch=0,
    ripples=1,
    autoupdate=0,
)
ssettings = cdict(
    shape=(0, 3),
    amplitude=(-1, 8),
    phase=(0, 1),
    pulse=(0, 1),
    shrink=(0, 1),
    exponent=(0, 3),
    unison=(1, 8, 1),
)
synth_default = cdict(
    shape=0,
    amplitude=1,
    phase=0,
    pulse=0.5,
    shrink=0,
    exponent=1,
    unison=1,
)
aediting = dict.fromkeys(asettings)
sediting = dict.fromkeys(ssettings)
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
        toolbar_height=80,
        audio=cdict(audio_default),
        control=cdict(control_default),
        spectrogram=1,
        oscilloscope=1,
    )
_audio = options.get("audio")
oaudio = cdict(audio_default)
if _audio:
    oaudio.update(_audio)
options.audio = oaudio
_control = options.get("control")
control = cdict(control_default)
if _control:
    control.update(_control)
options.control = control
orig_options = copy.deepcopy(options)


if os.name != "nt":
    raise NotImplementedError("This program is currently implemented to use Windows API only.")

import ctypes, struct, io
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
appid = "Miza Player (" + str(os.getpid()) + ")"
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)

psize = struct.calcsize("P")
if psize == 8:
    win = "win_amd64"
else:
    win = "win32"

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

rel = None
mouse_pointer = POINT()

def mouse_abs_pos():
    pt = mouse_pointer
    user32.GetCursorPos(ctypes.byref(pt))
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
    if is_unfocused():
        return
    mpc = pygame.mouse.get_pos()
    # if not in_rect(mpc, (0, 0, *window.get_size())):
    #     return
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
        mheld[i] = bool(user32.GetAsyncKeyState(n) & 32768)
    return mheld

if hasmisc:
    icon = pygame.image.load("misc/icon.png")
    pygame.display.set_icon(icon)
pygame.display.set_caption("Miza Player")
DISP = pygame.display.set_mode(options.screensize, pygame.RESIZABLE, vsync=True)
screensize2 = list(options.screensize)

# displaysize = [user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)]
# print(displaysize)

hwnd = pygame.display.get_wm_info()["window"]
pygame.display.set_allow_screensaver(True)
if hasmisc:
    s = io.StringIO()
    s.write(("%" + str(hwnd) + "\n"))
    for k, v in options.audio.items():
        s.write(f"~setting #{k} {v}\n")
    s.write(f"~setting #shuffle {options.control.setdefault('shuffle', 0)}\n")
    s.write(f"~setting silenceremove {options.control.setdefault('silenceremove', 0)}\n")
    s.write(f"~setting spectrogram {options.setdefault('spectrogram', 1)}\n")
    s.write(f"~setting oscilloscope {options.setdefault('oscilloscope', 1)}\n")
    s.write(f"~synth 1.5 1 0 0.75 0 1\n")
    s.seek(0)
    mixer.stdin.write(s.read().encode("utf-8"))
    try:
        mixer.stdin.flush()
    except OSError:
        print(mixer.stderr.read(), end="")
        raise

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
    user32.GetWindowRect(hwnd, ctypes.byref(wr))
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
    user32.GetWindowPlacement(hwnd, ctypes.byref(wp))
    return wp.showCmd

is_minimised = lambda: user32.IsIconic(hwnd)
is_unfocused = lambda: hwnd != user32.GetForegroundWindow(hwnd)

if options.get("maximised"):
    user32.ShowWindow(hwnd, 3)
elif options.get("screenpos"):# and not options.get("maximised"):
    x, y = options.screenpos
    user32.SetWindowPos(hwnd, 0, x, y, -1, -1, 0x4561)
# else:
#     user32.SetWindowPos(hwnd, -1, -1, -1, -1, -1, 0x4563)
screenpos2 = get_window_rect()[:2]

flash_window = lambda bInvert=True: user32.FlashWindow(hwnd, bInvert)

# shobjidl_core.SetWallpaper(0, os.path.abspath("misc/icon.png"))

proglast = (0, 0)
def taskbar_progress_bar(ratio=1, colour=0):
    if "shobjidl_core" not in globals():
        if win == "win32":
            spath = "misc/Shobjidl-32.dll"
        else:
            spath = "misc/Shobjidl.dll"
        try:
            globals()["shobjidl_core"] = ctypes.cdll.LoadLibrary(spath)
        except OSError:
            globals()["shobjidl_core"] = None
            print_exc()
    elif not shobjidl_core:
        return
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
#     user32.GetWindowPlacement(h, ctypes.byref(wp))
#     if wp.showCmd != 1 and wp.showCmd != 3:
#         return
#     if not user32.IsWindowVisible(h):
#         return
#     user32.GetWindowRect(h, ctypes.byref(wr))
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
#     user32.GetWindowRect(hwnd, ctypes.byref(wr))
#     rect = (wr.left, wr.top, wr.right, wr.bottom)
#     pts = [(rect[:2]), (rect[2], rect[1]), (rect[0], rect[3]), rect[2:]]
#     print(pts)
#     user32.EnumWindows(callback, -1)
#     print(pts)

# def get_window_clip():
#     hdc = user32.GetWindowDC(hwnd)
#     clip = ctypes.windll.gdi32.GetClipBox(hdc, ctypes.byref(wr))
#     user32.ReleaseDC(hdc)
#     return hdc, clip, wr.left, wr.top, wr.right - wr.left, wr.bottom - wr.top


import PIL, easygui, easygui_qt, numpy, math, random, itertools, collections, re, colorsys, ast, contextlib, pyperclip, pyaudio, zipfile, pickle, hashlib, base64, urllib, requests, datetime
import PyQt5
from PyQt5 import QtCore, QtWidgets
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
from PIL import Image, ImageOps, ImageChops
from math import *
np = numpy
easygui.__dict__.update(easygui_qt.easygui_qt.__dict__)
deque = collections.deque
suppress = contextlib.suppress
d2r = pi / 180
ts_us = lambda: time.time_ns() // 1000


commitf = "misc/commit.tmp"

def update_repo():
    print("Checking for updates...")
    try:
        with requests.get("https://github.com/thomas-xin/Miza-Player") as resp:
            s = resp.text
        try:
            search = '<include-fragment src="/thomas-xin/Miza-Player/tree-commit/'
            s = s[s.index(search) + len(search):]
        except ValueError:
            search = '<a data-pjax="true" data-test-selector="commit-tease-commit-message"'
            s = s[s.index(search) + len(search):]
            search = 'href="/thomas-xin/Miza-Player/commit/'
            s = s[s.index(search) + len(search):]
        commit = s.split('"', 1)[0]
        try:
            try:
                with open(commitf, "r") as f:
                    s = f.read()
            except FileNotFoundError:
                print("First run, treating as latest update...")
                raise EOFError
            if commit != s:
                print("Update found!")
                if not options.control.autoupdate:
                    globals()["repo-update"] = fut = concurrent.futures.Future()
                with requests.get("https://codeload.github.com/thomas-xin/Miza-Player/zip/refs/heads/main") as resp:
                    b = resp.content
                if not options.control.autoupdate:
                    r = fut.result()
                else:
                    r = True
                if r:
                    with zipfile.ZipFile(io.BytesIO(b), allowZip64=True, strict_timestamps=False) as z:
                        nl = z.namelist()
                        globals()["updating"] = updating = cdict(target=len(nl), progress=0)
                        for i, fn in enumerate(nl):
                            updating.progress = i
                            fn2 = fn[len("Miza-Player-main/"):]
                            if fn2 and not fn2.endswith("/") and not fn2.endswith(".ttf"):
                                try:
                                    folder = fn2.rsplit("/", 1)[0]
                                    if not os.path.exists(folder):
                                        os.mkdir(folder)
                                    with open(fn2, "wb") as f2:
                                        with z.open(fn, force_zip64=True) as f:
                                            f2.write(f.read())
                                except PermissionError:
                                    pass
                        updating.progress = len(nl)
                    globals().pop("updating", None)
                    if not options.control.autoupdate:
                        globals()["repo-update"] = True
                if r is not None:
                    raise EOFError
            else:
                print("No updates found.")
        except EOFError:
            with open(commitf, "w") as f:
                f.write(commit)
    except:
        print_exc()

def update_collections2():
    with requests.get("https://raw.githubusercontent.com/thomas-xin/Python-Extra-Classes/main/full.py") as resp:
        b = resp.content
    with open(collections2f, "wb") as f:
        f.write(b)
    exec(compile(b, "collections2.tmp", "exec"), globals())
    print("collections2.tmp updated.")

repo_fut = None
if not os.path.exists(collections2f):
    update_collections2()
    repo_fut = submit(update_repo)
with open(collections2f, "rb") as f:
    b = f.read()
exec(compile(b, "collections2.tmp", "exec"), globals())
if utc() - os.path.getmtime(collections2f) > 3600:
    submit(update_collections2)
repo_fut = submit(update_repo)

options.history = alist(options.get("history", ()))
globals().update(options)

shash = lambda s: as_str(base64.urlsafe_b64encode(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).replace(b"/", b"-").rstrip(b"="))

def quote(s):
    if s.isascii():
        return urllib.parse.quote_plus(s)
    a = urllib.parse.quote_plus(s)
    b = base64.urlsafe_b64encode(s.encode("utf-8")).rstrip(b"=")
    if len(a) < len(b):
        return a
    return "\x7f" + as_str(b)

def unquote(s):
    if s.startswith("\x7f"):
        s = s[1:].encode("utf-8")
        s += b"=="
        if (len(s) - 1) & 3 == 0:
            s += b"="
        return as_str(base64.urlsafe_b64decode(s))
    return urllib.parse.unquote_plus(s)


afut = submit(pyaudio.PyAudio)

pt = None
def pc():
    global pt
    t = time.perf_counter()
    if not pt:
        pt = t
        return 0
    return t - pt

math.round = round

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

def limit_size(w, h, wm, hm):
    r = h / w
    w2 = min(wm, hm / r)
    h2 = w2 * r
    return tuple(map(round, (w2, h2)))

from pygame.locals import *
import pygame.ftfont, pygame.gfxdraw
gfxdraw = pygame.gfxdraw

a = submit(pygame.ftfont.init)
b = submit(pygame.font.init)
a.result()
b.result()

def load_surface(fn, greyscale=False, size=None):
    im = image = Image.open(fn)
    if im.mode == "P":
        im = im.convert("RGBA")
    if size:
        im = im.resize(size)
    if greyscale:
        if "A" in im.mode:
            A = im.getchannel("A")
        im2 = ImageOps.grayscale(im)
        if "A" in im.mode:
            im2.putalpha(A)
        if "RGB" not in im2.mode:
            im2 = im2.convert("RGB" + ("A" if "A" in im.mode else ""))
        im = im2
    b = im.tobytes()
    surf = pygame.image.frombuffer(b, im.size, im.mode)
    image.close()
    return surf.convert_alpha() if "A" in im.mode else surf.convert()

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
row = np.arange(rgw, dtype=np.float64)
row -= mid
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

def draw_arc(surf, colour, pos, radius, start_angle=0, stop_angle=0):
    start_angle = int(start_angle % 360)
    stop_angle = int(stop_angle % 360)
    if radius <= 1:
        gfxdraw.filled_circle(surf, *pos, 1, colour)
    if start_angle == stop_angle:
        gfxdraw.circle(surf, *pos, radius, colour)
    else:
        gfxdraw.arc(surf, *pos, radius, start_angle, stop_angle, colour)

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
    if len(colour) > 3:
        colour, alpha = colour[:-1], colour[-1]
    if min(alpha, rect[2], rect[3]) > 0:
        br_surf = globals().setdefault("br_surf", {})
        colour = verify_colour(colour)
        if alpha == 255 and angle == 0 and (any(i > 160 for i in colour) or all(i in (0, 16, 32, 48, 64, 96, 127, 159, 191, 223, 255) for i in colour)):
            if cache:
                data = tuple(rect[2:]) + (grad_col, grad_angle, tuple(colour), filled)
            else:
                data = None
            try:
                surf = br_surf[data]
            except KeyError:
                surf = pygame.Surface(rect[2:]).convert()
                if not filled:
                    surf.fill((1, 2, 3))
                    surf.set_colorkey((1, 2, 3))
                r = rect
                rect = [0] * 2 + rect[2:]
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
                        draw_hline(surf, p[0], q[0], p[1], col1)
                        draw_vline(surf, p[0], p[1], q[1], col1)
                        draw_hline(surf, p[0], q[0], q[1], col2)
                        draw_vline(surf, q[0], p[1], q[1], col2)
                    except:
                        print_exc()
                if filled:
                    if grad_col is None:
                        draw_rect(surf, colour, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
                    else:
                        gradient_rectangle(surf, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
                rect = r
                if data:
                    br_surf[data] = surf
            if dest:
                dest.blit(surf, rect[:2])
                return rect
            return surf.convert()
        ctr = max(colour)
        contrast = min(round(ctr) + 2 >> 2 << 2, 255)
        data = tuple(rect[2:]) + (grad_col, grad_angle, contrast, filled)
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
            if filled:
                if grad_col is None:
                    draw_rect(s, colour2, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
                else:
                    gradient_rectangle(s, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
            if cache:
                br_surf[data] = s
        if ctr > 0:
            colour = tuple(round(i * 255 / ctr) for i in colour)
        else:
            colour = (0,) * 3
        return blit_complex(dest, s, rect[:2], angle=angle, alpha=alpha, colour=colour)

def rounded_bev_rect(dest, colour, rect, bevel=0, alpha=255, angle=0, grad_col=None, grad_angle=0, filled=True, cache=True):
    rect = list(map(round, rect))
    if len(colour) > 3:
        colour, alpha = colour[:-1], colour[-1]
    if min(alpha, rect[2], rect[3]) > 0:
        rb_surf = globals().setdefault("rb_surf", {})
        colour = list(map(lambda i: min(i, 255), colour))
        if alpha == 255 and angle == 0 and (any(i > 160 for i in colour) or all(i in (0, 16, 32, 48, 64, 96, 127, 159, 191, 223, 255) for i in colour)):
            if cache:
                data = tuple(rect[2:]) + (grad_col, grad_angle, tuple(colour), filled)
            else:
                data = None
            try:
                surf = rb_surf[data]
            except KeyError:
                surf = pygame.Surface(rect[2:]).convert()
                surf.fill((1, 2, 3))
                surf.set_colorkey((1, 2, 3))
                r = rect
                rect = [0] * 2 + rect[2:]
                s = surf
                for c in range(bevel):
                    p = [rect[0] + c, rect[1] + c]
                    q = [a + b - c - 1 for a, b in zip(rect[:2], rect[2:])]
                    b = bevel - c
                    v1 = 128 - c / bevel * 128
                    v2 = c / bevel * 96 - 96
                    col1 = col2 = colour
                    if v1:
                        col1 = [min(i + v1, 255) for i in col1]
                    if v2:
                        col2 = [max(i + v2, 0) for i in col1]
                    n = b <= 1
                    draw_hline(s, p[0] + b - n, q[0] - b, p[1], col1)
                    draw_vline(s, p[0], p[1] + b, q[1] - b + n, col1)
                    draw_hline(s, p[0] + b, q[0] - b + n, q[1], col2)
                    draw_vline(s, q[0], p[1] + b - n, q[1] - b, col2)
                    if b > 1:
                        draw_arc(s, col1, [p[0] + b, p[1] + b], b, 180, 270)
                        draw_arc(s, colour, [q[0] - b, p[1] + b], b, 270, 360)
                        draw_arc(s, colour, [p[0] + b, q[1] - b], b, 90, 180)
                        draw_arc(s, col2, [q[0] - b, q[1] - b], b, 0, 90)
                if filled:
                    if grad_col is None:
                        draw_rect(surf, colour, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
                    else:
                        gradient_rectangle(surf, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
                rect = r
                if data:
                    rb_surf[data] = surf
            if dest:
                dest.blit(surf, rect[:2])
                return rect
            return surf.convert()
        ctr = max(colour)
        contrast = min(round(ctr) + 2 >> 2 << 2, 255)
        data = tuple(rect[2:]) + (grad_col, grad_angle, contrast, filled)
        s = rb_surf.get(data)
        if s is None:
            colour2 = (contrast,) * 3
            s = pygame.Surface(rect[2:], SRCALPHA)
            s.fill((1, 2, 3))
            s.set_colorkey((1, 2, 3))
            for c in range(bevel):
                p = [c, c]
                q = [i - c - 1 for i in rect[2:]]
                b = bevel - c
                v1 = 128 - c / bevel * 128
                v2 = c / bevel * 96 - 96
                col1 = col2 = colour2
                if v1:
                    col1 = [min(i + v1, 255) for i in col1]
                if v2:
                    col2 = [max(i + v2, 0) for i in col1]
                n = b <= 1
                draw_hline(s, p[0] + b - n, q[0] - b, p[1], col1)
                draw_vline(s, p[0], p[1] + b, q[1] - b + n, col1)
                draw_hline(s, p[0] + b, q[0] - b + n, q[1], col2)
                draw_vline(s, q[0], p[1] + b - n, q[1] - b, col2)
                if b > 1:
                    draw_arc(s, col1, [p[0] + b, p[1] + b], b, 180, 270)
                    draw_arc(s, colour2, [q[0] - b, p[1] + b], b, 270, 360)
                    draw_arc(s, colour2, [p[0] + b, q[1] - b], b, 90, 180)
                    draw_arc(s, col2, [q[0] - b, q[1] - b], b, 0, 90)
            if filled:
                if grad_col is None:
                    draw_rect(s, colour2, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
                else:
                    gradient_rectangle(s, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
            if cache:
                rb_surf[data] = s
        if ctr > 0:
            colour = tuple(round(i * 255 / ctr) for i in colour)
        else:
            colour = (0,) * 3
        return blit_complex(dest, s, rect[:2], angle=angle, alpha=alpha, colour=colour)

reg_polygon_cache = {}

def reg_polygon_complex(dest, centre, colour, sides, width, height, angle=pi / 4, alpha=255, thickness=0, repetition=1, filled=False, rotation=0, soft=False, attempts=128, cache=False):
    width = max(round(width), 0)
    height = max(round(height), 0)
    repetition = int(repetition)
    if sides:
        angle %= tau / sides
    else:
        angle = 0
    cache |= angle % (pi / 4) == 0
    if cache:
        colour = tuple(min(255, round(i / 9) * 9) for i in colour)
        h = (colour, sides, width, height, round(angle / tau * 256), thickness, repetition, filled, soft)
        try:
            newS = reg_polygon_cache[h]
        except KeyError:
            pass
        else:
            pos = [centre[0] - width, centre[1] - height]
            return blit_complex(dest, newS, pos, alpha, rotation, copy=True)
    try:
        newS = pygame.Surface((width << 1, height << 1), SRCALPHA)
    except:
        print_exc()
        return
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
    if cache:
        reg_polygon_cache[h] = newS
        # print(len(reg_polygon_cache), h)
    return blit_complex(dest, newS, pos, alpha, rotation, copy=cache)

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
    increment = 3
    orig = frame
    f = orig - reduction
    while frame > 1:
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
            pygame.draw.rect(surface, adj_colour(c, (frame - f) * 16), r_rect)
        frame -= reduction
        increment += 3
    frame = orig
    for i in range(count):
        pos = perimeter * ((i / count + ratio * speed) % 1)
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
        r_rect = [r[0] - round(frame) - 1, r[1] - round(frame) - 1, round(frame) + 1 << 1, round(frame) + 1 << 1]
        bevel_rectangle(surface, colour, r_rect, 3)

def text_objects(text, font, colour, background):
    text_surface = font.render(text, True, colour, background)
    if text_surface.get_flags() & SRCALPHA:
        try:
            text_surface = text_surface.convert_alpha()
        except:
            pass
    else:
        try:
            text_surface = text_surface.convert()
        except:
            pass
    return text_surface, text_surface.get_rect()

def get_font(font):
    try:
        fn = "misc/" + font + ".ttf"
        if font == "Rockwell":
            with requests.get("https://drive.google.com/u/0/uc?id=1QFIyn5YPVCPitA5wZjqzJD3tRLe3sKW9&export=download") as resp:
                with open(fn, "wb") as f:
                    f.write(resp.content)
        elif font == "OpenSansEmoji":
            with requests.get("https://drive.google.com/u/0/uc?id=1sUZqTAOLd4mSW2i2xn7dqWxwRenvBkQa&export=download") as resp:
                with open(fn, "wb") as f:
                    f.write(resp.content)
        if "ct_font" in globals():
            ct_font.clear()
        if "ft_font" in globals():
            ft_font.clear()
        md_font.clear()
        globals()["font_reload"] = True
    except:
        print_exc()

loaded_fonts = set()
font_reload = False

def sysfont(font, size, unicode=False):
    func = pygame.ftfont if unicode else pygame.font
    fn = "misc/" + font + ".ttf"
    if not os.path.exists(fn) and font not in loaded_fonts:
        if font in ("Rockwell", "OpenSansEmoji"):
            loaded_fonts.add(font)
            submit(get_font, font)
    if os.path.exists(fn):
        return func.Font(fn, size)
    return func.SysFont(font, size)

def surface_font(text, colour, background, size, font):
    size = round(size)
    unicode = any(ord(c) >= 65536 for c in text)
    if not unicode:
        ct_font = globals().setdefault("ct_font", {})
    else:
        ct_font = globals().setdefault("ft_font", {})
    data = (size, font)
    f = ct_font.get(data, None)
    if not f:
        f = ct_font[data] = sysfont(font, size, unicode=unicode)
    for i in range(4):
        try:
            return text_objects(text, f, colour, background)
        except:
            if i >= 3:
                raise
            f = ct_font[data] = sysfont(font, size, unicode=unicode)

def text_size(text, size, font="OpenSansEmoji"):
    size = round(size)
    asc = text.isascii()
    if asc:
        ct_font = globals().setdefault("ct_font", {})
    else:
        ct_font = globals().setdefault("ft_font", {})
    data = (size, font)
    f = ct_font.get(data, None)
    if not f:
        f = ct_font[data] = sysfont(font, size, unicode=not asc)
    for i in range(4):
        try:
            return f.size(text)
        except:
            if i >= 3:
                raise
            f = ct_font[data] = sysfont(font, size, unicode=not asc)

md_font = {}
def message_display(text, size, pos=(0, 0), colour=(255,) * 3, background=None, surface=None, font="OpenSansEmoji", alpha=255, align=1, cache=False):
    # text = "".join(c if ord(c) < 65536 else "\x7f" for c in text)
    colour = tuple(verify_colour(colour))
    data = (text, colour, background, size, font)
    try:
        resp = md_font[data]
    except KeyError:
        resp = surface_font(*data)
    TextSurf, TextRect = resp
    if cache:
        md_font[data] = resp
        while len(md_font) > 4096:
            try:
                md_font.pop(next(iter(md_font)))
            except (KeyError, RuntimeError):
                pass
    if surface:
        if align == 1:
            TextRect.center = pos
        elif align == 0:
            TextRect = list(pos) + TextRect[2:]
        elif align == 2:
            TextRect = [y - x for x, y in zip(TextRect[2:], pos)] + TextRect[2:]
        blit_complex(surface, TextSurf, TextRect, alpha, copy=False)
        return TextRect
    else:
        return TextSurf

def char_display(char, size, font="OpenSansEmoji"):
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


PRINT.start()


# Runs ffprobe on a file or url, returning the duration if possible.
def _get_duration_2(filename, _timeout=12):
    command = (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,",
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
        resp = proc.stdout.read().splitlines()
    except:
        with suppress():
            proc.kill()
        print_exc()
    try:
        cdc = as_str(resp[0].rstrip())
    except (IndexError, ValueError, TypeError):
        cdc = "auto"
    try:
        dur = float(resp[1])
    except (IndexError, ValueError, TypeError):
        dur = None
    bps = None
    if resp and len(resp) > 2:
        with suppress(ValueError):
            bps = float(resp[2])
    return dur, bps, cdc

def get_duration_2(filename):
    if not is_url(filename):
        if filename.endswith(".pcm"):
            return os.path.getsize(filename) / (48000 * 2 * 2), "pcm"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                if f.read(4) == b"MThd":
                    return None, "N/A"
    if filename:
        dur, bps, cdc = _get_duration_2(filename, 4)
        if not dur and is_url(filename):
            with requests.get(filename, stream=True) as resp:
                head = fcdict(resp.headers)
                if "content-length" not in head:
                    dur, bps, cdc = _get_duration_2(filename, 20)
                    return dur, cdc
                if bps:
                    return (int(head["content-length"]) << 3) / bps, cdc
                ctype = [e.strip() for e in head.get("content-type", "").split(";") if "/" in e][0]
                if ctype.split("/", 1)[0] not in ("audio", "video"):
                    return nan, cdc
                if ctype == "audio/midi":
                    return nan, cdc
                it = resp.iter_content(65536)
                data = next(it)
            ident = str(magic.from_buffer(data))
            try:
                bitrate = regexp("[0-9]+\\s.bps").findall(ident)[0].casefold()
            except IndexError:
                dur, bps, cdc = _get_duration_2(filename, 16)
                return dur, cdc
            bps, key = bitrate.split(None, 1)
            bps = float(bps)
            if key.startswith("k"):
                bps *= 1e3
            elif key.startswith("m"):
                bps *= 1e6
            elif key.startswith("g"):
                bps *= 1e9
            return (int(head["content-length"]) << 3) / bps, cdc
        return dur, cdc


# runs org2xm on a file, with an optional custom sample bank.
def org2xm(org, dat=None):
    if os.name != "nt":
        raise OSError("org2xm is only available on Windows.")
    r_org = None
    if not org or type(org) is not bytes:
        if is_url(org):
            with requests.get(org) as r:
                data = r.content
        else:
            r_org = org
            with open(r_org, "rb") as f:
                data = f.read(6)
        if not data:
            raise FileNotFoundError("Error downloading file content.")
    else:
        if not org.startswith(b"Org-"):
            raise ValueError("Invalid file header.")
        data = org
    # Set compatibility option if file is not of org2 format.
    compat = not data.startswith(b"Org-02")
    ts = ts_us()
    # Write org data to file.
    if not r_org:
        r_org = "cache/" + str(ts) + ".org"
        with open(r_org, "wb") as f:
            f.write(data)
    r_dat = "cache/" + str(ts) + ".dat"
    orig = False
    # Load custom sample bank if specified
    if dat is not None and is_url(dat):
        with open(r_dat, "wb") as f:
            with requests.get(dat) as r:
                f.write(r.content)
    else:
        if type(dat) is bytes and dat:
            with open(r_dat, "wb") as f:
                f.write(dat)
        else:
            r_dat = "misc/ORG210EN.DAT"
            orig = True
    args = ["misc/org2xm.exe", r_org, r_dat]
    if compat:
        args.append("c")
    print(args)
    subprocess.check_output(args)
    r_xm = f"cache/{ts}.xm"
    if not os.path.exists("cache/" + str(ts) + ".xm"):
        raise FileNotFoundError("Unable to locate converted file.")
    if not os.path.getsize(r_xm):
        raise RuntimeError("Converted file is empty.")
    for f in (r_org, r_dat)[:2 - orig]:
        with suppress():
            os.remove(f)
    return r_xm

def mid2mp3(mid):
    url = requests.post(
        "https://hostfast.onlineconverter.com/file/send",
        files={
            "class": (None, "audio"),
            "from": (None, "midi"),
            "to": (None, "mp3"),
            "source": (None, "file"),
            "file": mid,
            "audio_quality": (None, "192"),
        },
    ).text
    fn = url.rsplit("/", 1)[-1].strip("\x00")
    for i in range(360):
        t = utc()
        test = requests.get(f"https://hostfast.onlineconverter.com/file/{fn}").content
        if test == b"d":
            break
        delay = utc() - t
        if delay < 1:
            time.sleep(1 - delay)
    ts = ts_us()
    r_mp3 = f"cache/{ts}.mp3"
    with open(r_mp3, "wb") as f:
        f.write(requests.get(f"https://hostfast.onlineconverter.com/file/{fn}/download").content)
    return r_mp3

def png2wav(png):
    ts = ts_us()
    r_png = f"cache/{ts}"
    r_wav = f"cache/{ts}.wav"
    args = ["py", f"-3.{sys.version_info[1]}", "png2wav.py", "../" + r_png, "../" + r_wav]
    with open(r_png, "wb") as f:
        f.write(png)
    print(args)
    subprocess.run(args, cwd="misc", stderr=subprocess.PIPE)
    return r_wav


CONVERTERS = {
    b"MThd": mid2mp3,
    b"Org-": org2xm,
}

def select_and_convert(stream):
    if is_url(stream):
        with requests.get(stream, timeout=8, stream=True) as resp:
            it = resp.iter_content(4096)
            b = bytes()
            while len(b) < 4:
                b += next(it)
            try:
                convert = CONVERTERS[b[:4]]
            except KeyError:
                convert = png2wav
            b += resp.content
    else:
        with open(stream, "rb") as file:
            b = file.read(4096)
            try:
                convert = CONVERTERS[b[:4]]
            except KeyError:
                convert = png2wav
            b += file.read()
    print(convert, stream)
    return convert(b)


eval_const = {
    "none": None,
    "null": None,
    "NULL": None,
    "true": True,
    "false": False,
    "TRUE": True,
    "FALSE": False,
    "inf": inf,
    "nan": nan,
    "Infinity": inf,
}
safe_eval = lambda s: eval(s, {}, eval_const)

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

def time_parse(ts):
    data = ts.split(":")
    if len(data) >= 5: 
        raise TypeError("Too many time arguments.")
    mults = (1, 60, 3600, 86400)
    return round_min(sum(float(count) * mult for count, mult in zip(data, reversed(mults[:len(data)]))))

def expired(stream):
    if is_youtube_url(stream):
        return True
    if stream.startswith("https://www.yt-download.org/download/"):
        if int(stream.split("/download/", 1)[1].split("/", 4)[3]) < utc() + 60:
            return True
    elif is_youtube_stream(stream):
        if int(stream.replace("/", "=").split("expire=", 1)[-1].split("=", 1)[0].split("&", 1)[0]) < utc() + 60:
            return True

is_youtube_stream = lambda url: url and re.findall("^https?:\\/\\/r[0-9]+---.{2}-\\w+-\\w{4,}\\.googlevideo\\.com", url)
is_youtube_url = lambda url: url and re.findall("^https?:\\/\\/(?:www\\.)?youtu(?:\\.be|be\\.com)\\/[^\\s<>`|\"']+", url)
# Regex moment - Smudge