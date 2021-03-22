# ONE SMALL STEP FOR MAN, ONE GIANT LEAP FOR SMUDGE KIND! Invaded once again on the 6th March >:D

import sys
sys.stdout.write = lambda *args, **kwargs: None
import pygame

import os, sys, pyaudio, numpy, math, random, base64, hashlib, time, subprocess, psutil, traceback, contextlib, colorsys, ctypes, collections, concurrent.futures
from math import *
from PIL import Image, ImageDraw, ImageFont
np = numpy
deque = collections.deque
suppress = contextlib.suppress
hwnd = int(sys.stdin.readline()[1:])
is_minimised = lambda: ctypes.windll.user32.IsIconic(hwnd)

pt = None
pt2 = None
def pc():
    global pt
    t = time.perf_counter()
    if not pt:
        pt = t
        return 0
    return t - pt

def as_str(s):
    if type(s) in (bytes, bytearray, memoryview):
        return bytes(s).decode("utf-8", "replace")
    return str(s)

is_url = lambda url: "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")
shash = lambda s: as_str(base64.b64encode(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).replace(b"/", b"-").rstrip(b"="))

exc = concurrent.futures.ThreadPoolExecutor(max_workers=24)
submit = exc.submit

afut = submit(pyaudio.PyAudio)

def pya_init():
    try:
        pya = afut.result()
        return pya.open(
            rate=48000,
            channels=2,
            format=pyaudio.paInt16,
            output=True,
            frames_per_buffer=800,
        )
    except:
        print_exc()
        raise

channel2 = None
aout = submit(pya_init)
pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=1024)


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


def point(s):
    b = str(s).encode("utf-8") if type(s) is not bytes else s
    if not b.endswith(b"\n"):
        b += b"\n"
    sys.__stdout__.buffer.write(b)
    sys.__stdout__.flush()

def bsend(*args):
    for s in args:
        b = str(s).encode("utf-8") if type(s) is not bytes else s
        sys.__stderr__.buffer.write(b)
    sys.__stderr__.flush()

print = lambda *args, sep=" ", end="\n": point(repr(str(sep).join(map(str, args)) + end))
print_exc = lambda: point(repr(traceback.format_exc()))


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
            with requests.get(filename, stream=True) as resp:
                head = {k.casefold(): v for k, v in resp.headers.items()}
                if "content-length" not in head:
                    return _get_duration(filename, 20)[0]
                if bps:
                    return (int(head["content-length"]) << 3) / bps
                ctype = [e.strip() for e in head.get("content-type", "").split(";") if "/" in e][0]
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
            return (int(head["content-length"]) << 3) / bps
        return dur

probe_cache = {}
def probe(stream):
    try:
        return probe_cache[stream]
    except KeyError:
        pass
    command = (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,",
        "-show_entries",
        "format=format_name",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        stream,
    )
    print(command)
    resp = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = list(s.strip() for s in reversed(as_str(resp.stdout).splitlines()))
    probe_cache[stream] = out
    return out

def kill(proc):
    try:
        return proc.kill2()
    except:
        pass
    proc.kill()

def duration_est():
    global duration
    last_fn = ""
    last_mt = 0
    last_fs = 0
    while True:
        try:
            if stream and not is_url(stream) and (stream[0] != "<" or stream[-1] != ">") and os.path.exists(stream):
                stat = None
                if last_fn == stream:
                    stat = os.stat(stream)
                    if stat.st_mtime == last_mt and stat.st_size == last_fs:
                        time.sleep(1)
                        continue
                if not stat:
                    stat = os.stat(stream)
                last_fn = stream
                last_mt = stat.st_mtime
                last_fs = stat.st_size
                duration = get_duration(stream)
        except:
            print_exc()
        time.sleep(0.5)


removing = set()
def remover():
    try:
        while True:
            for fn in removing:
                try:
                    os.remove(fn)
                except PermissionError:
                    continue
                except FileNotFoundError:
                    pass
                removing.discard(file)
            time.sleep(0.1)
    except:
        print_exc()

def remove(file):
    fn = file.name
    if "\x7f" in fn:
        removing.add(fn)

def stdclose(p):
    try:
        if not p.stdin.closed:
            # fut = submit(p.stdin.write, emptymem[:BSIZE])
            # fut.result(timeout=1)
            fut = submit(p.stdin.close)
            fut.result(timeout=1)
        time.sleep(2)
    except concurrent.futures.TimeoutError:
        pass
    except:
        print_exc()
    print("Closing", p)
    p.kill()

shuffling = False
transfer = False
BSIZE = 1600
RSIZE = BSIZE << 1
TSIZE = BSIZE // 3
def reader(f, pos=None, reverse=False, shuffling=False):
    global proc, transfer
    try:
        if f.closed:
            raise StopIteration
        if pos is None:
            pos = f.tell()
        if reverse:
            pos -= RSIZE
        if pos:
            f.seek(pos)
        opos = pos
        while True:
            lpos = pos
            while shuffling:
                for i in range(1024):
                    pos = random.randint(0, fsize >> 2) << 2
                    if abs(pos - lpos) > 65536:
                        break
                f.seek(pos)
                pos += BSIZE
                b = f.read(BSIZE)
                if len(b) & 1:
                    b = memoryview(b)[:-1]
                a = np.frombuffer(b, dtype=np.int16)
                u, c = np.unique(a, return_counts=True)
                s = np.sort(c)
                x = np.sum(s[-3:])
                if x >= TSIZE:
                    while True:
                        b = f.read(BSIZE)
                        pos += BSIZE
                        if not b:
                            if type(proc) is cdict:
                                pos = 0
                                f.seek(0)
                            break
                        if len(b) & 1:
                            b = memoryview(b)[:-1]
                        a = np.frombuffer(b, dtype=np.int16)
                        u, c = np.unique(a, return_counts=True)
                        s = np.sort(c)
                        x = np.sum(s[-3:])
                        if not x >= TSIZE:
                            pos = round(pos / 4) << 2
                            if type(proc) is cdict and pos >= fsize - 65536:
                                pos = 0
                            f.seek(pos)
                            break
                    globals()["pos"] = pos / fsize * duration
                    globals()["frame"] = globals()["pos"] * 30
                    print(f"Autoshuffle {pos}/{fsize}")
                    shuffling = False
                    p = proc
                    print(p.args)
                    proc = psutil.Popen(p.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    try:
                        proc.kill2 = p.kill2
                    except AttributeError:
                        pass
                    submit(stdclose, p)
                    opos = pos
                    transfer = True
            try:
                b = f.read(RSIZE)
            except ValueError:
                b = b""
            if settings.shuffle == 2 and abs(pos - opos) / fsize * duration >= 60:
                if b:
                    if len(b) & 1:
                        b = memoryview(b)[:-1]
                    a = np.frombuffer(b, dtype=np.int16)
                    u, c = np.unique(a, return_counts=True)
                    s = np.sort(c)
                    x = np.sum(s[-3:])
                    if x >= TSIZE << 1:
                        if pos != 0:
                            print(x, TSIZE << 1)
                        shuffling = True
                        continue
            if not b:
                break
            if reverse:
                if len(b) & 1:
                    b = memoryview(b)[:-1]
                b = np.flip(np.frombuffer(b, dtype=np.uint16)).tobytes()
                pos -= RSIZE
                if pos <= 0:
                    proc.stdin.write(b)
                    size = -pos
                    if size:
                        f.seek(0)
                        b = f.read(size)
                        if len(b) & 1:
                            b = memoryview(b)[:-1]
                        b = np.flip(np.frombuffer(b, dtype=np.uint16)).tobytes()
                        proc.stdin.write(b)
                    break
                f.seek(pos)
            else:
                pos += RSIZE
            p = proc
            try:
                fut = submit(proc.stdin.write, b)
                try:
                    fut.result(timeout=12)
                    proc.stdin.flush()
                except concurrent.futures.TimeoutError:
                    if not paused:
                        raise ValueError
                    paused.result()
            except (OSError, BrokenPipeError, ValueError, AttributeError):
                if p and p.is_running():
                    try:
                        p.kill()
                    except:
                        pass
                if not proc or not proc.is_running():
                    break
    except StopIteration:
        pass
    except:
        print_exc()
    try:
        if proc and proc.is_running():
            submit(stdclose, proc)
        f.close()
    except:
        pass

def construct_options(full=True):
    stats = cdict(settings)
    pitchscale = 2 ** ((stats.pitch + stats.nightcore) / 12)
    chorus = min(16, abs(stats.chorus))
    reverb = stats.reverb
    volume = 1
    if reverb:
        args = ["-i", "misc/SNB3,0all.wav"]
    else:
        args = []
    options = deque()
    if not isfinite(stats.compressor):
        options.extend(("anoisesrc=a=.001953125:c=brown", "amerge"))
    if pitchscale != 1 or stats.speed != 1:
        speed = abs(stats.speed) / pitchscale
        speed *= 2 ** (stats.nightcore / 12)
        if round(speed, 9) != 1:
            speed = max(0.005, speed)
            if speed >= 64:
                raise OverflowError
            opts = ""
            while speed > 3:
                opts += "atempo=3,"
                speed /= 3
            while speed < 0.5:
                opts += "atempo=0.5,"
                speed /= 0.5
            opts += "atempo=" + str(speed)
            options.append(opts)
    if pitchscale != 1:
        if abs(pitchscale) >= 64:
            raise OverflowError
        if full:
            options.append("aresample=48k")
        options.append("asetrate=" + str(48000 * pitchscale))
    if chorus:
        A = B = C = D = ""
        for i in range(ceil(chorus)):
            neg = ((i & 1) << 1) - 1
            i = 1 + i >> 1
            i *= stats.chorus / ceil(chorus)
            if i:
                A += "|"
                B += "|"
                C += "|"
                D += "|"
            delay = (25 + i * tau * neg) % 39 + 18
            A += str(round(delay, 3))
            decay = (0.125 + i * 0.03 * neg) % 0.25 + 0.25
            B += str(round(decay, 3))
            speed = (2 + i * 0.61 * neg) % 4.5 + 0.5
            C += str(round(speed, 3))
            depth = (i * 0.43 * neg) % max(4, stats.chorus) + 0.5
            D += str(round(depth, 3))
        b = 0.5 / sqrt(ceil(chorus + 1))
        options.append(
            "chorus=0.5:" + str(round(b, 3)) + ":"
            + A + ":"
            + B + ":"
            + C + ":"
            + D
        )
        volume *= 2
    if stats.compressor:
        comp = min(8000, abs(stats.compressor * 10 + (1 if stats.compressor >= 0 else -1)))
        while abs(comp) > 1:
            c = min(20, comp)
            try:
                comp /= c
            except ZeroDivisionError:
                comp = 1
            mult = str(round((c * math.sqrt(2)) ** 0.5, 4))
            options.append(
                "acompressor=mode=" + ("upward" if stats.compressor < 0 else "downward")
                + ":ratio=" + str(c) + ":level_in=" + mult + ":threshold=0.0625:makeup=" + mult
            )
    if stats.bassboost:
        opt = "anequalizer="
        width = 4096
        x = round(sqrt(1 + abs(stats.bassboost)), 5)
        coeff = width * max(0.03125, (0.25 / x))
        ch = " f=" + str(coeff if stats.bassboost > 0 else width - coeff) + " w=" + str(coeff / 2) + " g=" + str(max(0.5, min(48, 4 * math.log2(x * 5))))
        opt += "c0" + ch + "|c1" + ch
        options.append(opt)
    if reverb:
        coeff = abs(reverb)
        wet = min(3, coeff) / 3
        if wet != 1:
            options.append("asplit[2]")
        volume *= 1.2
        options.append("afir=dry=10:wet=10")
        if wet != 1:
            dry = 1 - wet
            options.append("[2]amix=weights=" + str(round(dry, 6)) + " " + str(round(wet, 6)))
        if coeff > 1:
            decay = round(1 - 4 / (3 + coeff), 4)
            options.append(f"aecho=1:1:400|600:{decay}|{decay / 2}")
            if decay >= 0.25:
                options.append(f"aecho=1:1:800|1100:{decay / 4}|{decay / 8}")
    if stats.pan != 1:
        pan = min(10000, max(-10000, stats.pan))
        while abs(abs(pan) - 1) > 0.001:
            p = max(-10, min(10, pan))
            try:
                pan /= p
            except ZeroDivisionError:
                pan = 1
            options.append("extrastereo=m=" + str(p) + ":c=0")
            volume *= 1 / max(1, round(math.sqrt(abs(p)), 4))
    if volume != 1:
        options.append("volume=" + str(round(volume, 7)))
    if options:
        if stats.compressor:
            options.append("alimiter")
        # elif volume > 1:
        #     options.append("asoftclip=atan")
        args.append(("-af", "-filter_complex")[bool(reverb)])
        args.append(",".join(options))
    return args

stderr_lock = None
def oscilloscope(buffer):
    global stderr_lock
    try:
        if packet:
            arr = buffer[::2] + buffer[1::2]
            osci = np.empty(osize[0])
            r = len(arr) / len(osci)
            for i in range(len(osci)):
                x = round(i * r)
                y = round(i * r + r)
                osci[i] = np.mean(arr[x:y])
            osci = np.clip(osci * (1 / 65536), -1, 1, out=osci)
            if packet:
                size = osize
                OSCI = pygame.Surface(size)
                time.sleep(0.01)
                if packet:
                    point = (0, osize[1] / 2 + osci[0] * osize[1] / 2)
                    for i in range(1, len(osci)):
                        if not packet:
                            return
                        prev = point
                        point = (i, osize[1] / 2 + osci[i] * osize[1] / 2)
                        hue = (osci[i] + osci[i - 1]) / 4 % 1
                        col = tuple(map(lambda x: round(x * 255), colorsys.hsv_to_rgb(1 - hue, 1, 1)))
                        pygame.draw.line(
                            OSCI,
                            col,
                            point,
                            prev,
                        )
                    time.sleep(0.01)
                    if packet:
                        b = pygame.image.tostring(OSCI, "RGB")
                        while stderr_lock:
                            stderr_lock.result()
                        stderr_lock = concurrent.futures.Future()
                        bsend(b"o" + "~".join(map(str, size)).encode("utf-8") + b"\n", b)
                        lock, stderr_lock = stderr_lock, None
                        if lock:
                            lock.set_result(None)
    except:
        print_exc()

higher_bound = "F#10"
highest_note = "C~D~EF~G~A~B".index(higher_bound[0].upper()) - 9 + ("#" in higher_bound)
while higher_bound[0] not in "0123456789-":
    higher_bound = higher_bound[1:]
    if not higher_bound:
        raise ValueError("Octave not found.")
highest_note += int(higher_bound) * 12 + 1

lower_bound = "A0"
lowest_note = "C~D~EF~G~A~B".index(lower_bound[0].upper()) - 9 + ("#" in lower_bound)
while lower_bound[0] not in "0123456789-":
    lower_bound = lower_bound[1:]
    if not lower_bound:
        raise ValueError("Octave not found.")
lowest_note += int(lower_bound) * 12 - 1

maxfreq = 27.5 * 2 ** ((highest_note + 0.5) / 12)
minfreq = 27.5 * 2 ** ((lowest_note - 0.5) / 12)
barcount = int(highest_note - lowest_note) + 1 + 2
freqmul = 1 / (1 - log(minfreq, maxfreq))

barheight = 720
res_scale = 65536
dfts = res_scale // 2 + 1
fff = np.fft.fftfreq(res_scale, 1 / 48000)[:dfts]
fftrans = np.zeros(dfts, dtype=int)

for i, x in enumerate(fff):
    if x <= 0:
        continue
    else:
        x = round((1 - log(x, maxfreq)) * freqmul * (barcount - 1))
    if x > barcount - 1 or x < 0:
        continue
    fftrans[i] = x

PARTICLES = set()
P_ORDER = 0
TICK = 0

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

    font = ImageFont.truetype("misc/Pacifico.ttf", 12)

    def __init__(self, x):
        super().__init__()
        dark = False
        self.colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(x / barcount, 1, 1))
        note = highest_note - x + 9
        if note % 12 in (1, 3, 6, 8, 10):
            dark = True
        name = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[note % 12]
        octave = note // 12
        self.line = name + str(octave)
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
        if self.height:
            self.height = self.height * 0.2 ** dur - 1
            if self.height < 0:
                self.height = 0
        if self.height2:
            self.height2 = self.height2 * 0.4 ** dur - 1
            if self.height2 < 0:
                self.height2 = 0

    def ensure(self, value):
        if self.height < value:
            self.height = value
        if self.height2 < value:
            self.height2 = value

    def render(self, sfx, **void):
        size = min(2 * barheight, round(self.height))
        if size:
            dark = False
            self.colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb((pc() / 3 + self.x / barcount) % 1, 1, 1))
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
        size = min(2 * barheight, round(self.height))
        if size:
            amp = size / barheight * 2
            val = min(1, amp)
            sat = 1 - min(1, max(0, amp - 1))
            self.colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb((pc() / 3 + self.x / barcount) % 1, sat, val))
            x = barcount - self.x - 2
            DRAW.rectangle(
                (x, x, barcount * 2 - x - 3, barcount * 2 - x - 3),
                None,
                self.colour,
                width=1,
            )

    def post_render(self, sfx, scale, **void):
        size = round(self.height2)
        if size > 8:
            ix = barcount - 1 - self.x - 1
            sx = round(ix / barcount * ssize2[0])
            width = round((ix + 1) / barcount * ssize2[0]) - sx
            x = sx + (width >> 1) - 8
            try:
                width = DRAW.textlength(self.line, self.font)
            except (TypeError, AttributeError):
                width = 8 * len(self.line)
            pos = max(64, ssize2[1] - size - width)
            factor = round(255 * scale)
            col = sum(factor << (i << 3) for i in range(3))
            DRAW.text((x, pos), self.line, col, self.font)

bars = [Bar(i - 1) for i in range(barcount)]
spec_update_fut = None
lastspec = 0
lastspec2 = 0

def spectrogram_render():
    global stderr_lock, ssize2, spectrobytes, lastspec2, spec_update_fut, packet_advanced2
    try:
        t = pc()
        dur = max(0.001, min(0.125, t - lastspec2))
        lastspec2 = t

        ssize2 = ssize
        specs = settings.spectrogram
        if specs != 2:
            sfx = Image.new("RGB", (barcount - 2, barheight), (0,) * 3)
            time.sleep(0.01)
            for bar in bars:
                bar.render(sfx=sfx)
        else:
            sfx = Image.new("RGB", (barcount * 2 - 2,) * 2, (0,) * 3)
            globals()["DRAW"] = ImageDraw.Draw(sfx)
            for bar in bars:
                bar.render2(sfx=sfx)
        sfx = sfx.resize(ssize2, resample=Image.NEAREST)

        if specs != 2:
            globals()["DRAW"] = ImageDraw.Draw(sfx)
            time.sleep(0.01)
            highbars = sorted(bars, key=lambda bar: bar.height, reverse=True)[:24]
            high = highbars[0]
            for bar in reversed(highbars):
                bar.post_render(sfx=sfx, scale=bar.height / max(1, high.height))
        time.sleep(0.01)
        spectrobytes = sfx.tobytes()
        
        write = False
        for bar in bars:
            if bar.height2:
                write = True
            bar.update(dur=dur)

        if write:
            while stderr_lock:
                stderr_lock.result()
            stderr_lock = concurrent.futures.Future()
            bsend(b"s" + "~".join(map(str, ssize2)).encode("utf-8") + b"\n", spectrobytes)
            lock, stderr_lock = stderr_lock, None
            if lock:
                lock.set_result(None)
        if packet_advanced2 and not is_minimised() and (not spec_update_fut or spec_update_fut.done()):
            spec_update_fut = submit(spectrogram_render)
            packet_advanced2 = False
    except:
        print_exc()

def spectrogram_update():
    global lastspec, spec_update_fut, spec2_fut, packet_advanced2, packet_advanced3
    try:
        if packet_advanced2 and not is_minimised() and (not spec_update_fut or spec_update_fut.done()):
            spec_update_fut = submit(spectrogram_render)
            packet_advanced2 = False
        t = pc()
        dur = max(0.001, min(0.125, t - lastspec))
        lastspec = t
        dft = np.fft.rfft(spec_buffer)
        np.multiply(spec_buffer, 0.25 ** dur, out=spec_buffer)
        arr = np.zeros(barcount, dtype=np.complex64)
        np.add.at(arr, fftrans, dft)
        arr[0] = 0
        amp = np.abs(arr, dtype=np.float32)
        for i, pwr in enumerate(amp):
            bars[i].ensure(pwr / 8)
        if packet_advanced3 and ssize[0] and ssize[1] and not is_minimised() and (not spec2_fut or spec2_fut.done()):
            spec2_fut = submit(spectrogram_update)
            packet_advanced3 = False
    except:
        print_exc()

spec_empty = np.zeros(res_scale, dtype=np.float32)
spec_buffer = spec_empty
def spectrogram(buffer):
    global spec_buffer, spec2_fut, packet_advanced3
    try:
        if packet:
            buffer = np.clip(sample.astype(np.float32) / 32767, -1, None)
            spec_buffer = np.concatenate((spec_buffer[-res_scale + len(buffer):], buffer))
            if packet_advanced3 and ssize[0] and ssize[1] and not is_minimised() and (not spec2_fut or spec2_fut.done()):
                spec2_fut = submit(spectrogram_update)
                packet_advanced3 = None
    except:
        print_exc()

packet_advanced = None
packet_advanced2 = None
packet_advanced3 = None
emptyamp = np.zeros(barcount, dtype=np.float32)
osci_fut = None
spec_fut = None
spec2_fut = None
def render():
    global lastpacket, osci_fut, spec_fut, spec2_fut, packet_advanced, sbuffer
    try:
        while True:
            if lastpacket != packet:
                lastpacket = packet
                buffer = sbuffer

                ampm1 = buffer > 0
                ampm2 = buffer < 0
                amp1 = sum(buffer[ampm1])
                amp2 = -sum(buffer[ampm2])
                amp = (amp1 + amp2) / len(buffer)
                p_amp = sqrt(amp / 32767)

                if is_minimised():
                    point(f"~y {p_amp}")
                else:
                    if settings.oscilloscope and osize[0] and osize[1] and (not osci_fut or osci_fut.done()):
                        osci_fut = submit(oscilloscope, buffer)
                    out = []
                    out.append(round(max(np.max(buffer), -np.min(buffer)) / 32767 * 100, 3))
                    out.append(round(amp / 32767 * 100, 2))

                    if packet:
                        vel1 = buffer[::2][1:] - buffer[::2][:-1]
                        vel2 = buffer[1::2][1:] - buffer[1::2][:-1]
                        amp1 = np.mean(np.abs(vel1))
                        amp2 = np.mean(np.abs(vel2))
                        vel = (amp1 + amp2)
                        out.append(round(vel / 131068 * 100, 3))

                        if packet:
                            amp1 = np.mean(np.abs(vel1[::2][1:] - vel1[::2][:-1]))
                            amp2 = np.mean(np.abs(vel2[1::2][1:] - vel2[1::2][:-1]))
                            out.append(round((amp1 + amp2) / 262136 * 100, 3))

                            out = [min(i, 100) for i in out]
                            out.append(p_amp)

                            if packet:
                                point("~x " + " ".join(map(str, out)))

                    if settings.spectrogram and ssize[0] and ssize[1] and (not spec2_fut or spec2_fut.done()):
                        spec2_fut = submit(spectrogram_update)
            elif packet_advanced:
                if settings.spectrogram and ssize[0] and ssize[1] and (not spec2_fut or spec2_fut.done()):
                    spec2_fut = submit(spectrogram_update)
            packet_advanced = False
            time.sleep(0.01)
    except:
        print_exc()

emptybuff = b"\x00" * (1600 * 2 * 2)
emptymem = memoryview(emptybuff)
emptysample = np.frombuffer(emptybuff, dtype=np.uint16)

def play(pos):
    global file, fn, proc, drop, frame, packet, lastpacket, sample, transfer, point_fut, spec_fut, sbuffer, packet_advanced, packet_advanced2, packet_advanced3
    skipzeros = False
    try:
        frame = pos * 30
        while True:
            if paused and drop <= 0:
                paused.result()
            p = proc
            if fn:
                if not file:
                    try:
                        if os.path.getsize(fn) < req:
                            raise FileNotFoundError
                        file = open(fn, "rb") # *cough* "FM", Miza is secretly a radio host and Txin's house is a radio station - Smudge
                    except (OSError, FileNotFoundError, PermissionError):
                        if proc and not proc.is_running():
                            raise
                        time.sleep(0.01)
                        continue
                try:
                    b = file.read(req)
                except:
                    b = b""
            else:
                while not getattr(proc, "readable", lambda: True)():
                    time.sleep(0.01)
                try:
                    fut = submit(proc.stdout.read, req)
                    b = fut.result(timeout=4)
                except:
                    b = b""
            if not b:
                if transfer and proc and proc.is_running():
                    transfer = False
                    fut = submit(proc.stdout.read, req)
                    b = fut.result(timeout=4)
                    drop = 0
                else:
                    print(f"{proc} {fn}")
                    if proc:
                        if proc.is_running():
                            try:
                                kill(proc)
                            except:
                                pass
                        point("~s")
                    if file:
                        file.close()
                        remove(file)
                    fn = file = proc = None
                    packet = None
                    drop = 0
                    return
            if p and p != proc and p.is_running():
                try:
                    p.kill()
                except:
                    pass
            if drop > 0:
                drop -= 1
            else:
                r = b
                if len(b) < req:
                    b += bytes(emptymem[:req - len(b)])
                lastpacket = packet
                packet = b
                packet_advanced = True
                packet_advanced2 = True
                packet_advanced3 = True
                if len(b) & 1:
                    b = memoryview(b)[:-1]
                smp = np.frombuffer(b, dtype=np.int16)
                if settings.volume != 1:
                    s = smp * settings.volume
                    if abs(settings.volume) > 1:
                        s = np.clip(s, -32767, 32767)
                    sample = s.astype(np.int16)
                    b = sample.tobytes()
                else:
                    sample = smp
                sbuffer = sample.astype(np.float32)
                if channel2:
                    b = bytes(b)
                    fut = submit(channel2.write, b)
                    try:
                        fut.result(timeout=0.12)
                    except:
                        print("Pyaudio timed out.")
                        globals()["aout"] = submit(pya_init)
                        globals()["channel2"] = None
                if not channel2:
                    buffer = sample.reshape((1600, 2))
                    sound = pygame.sndarray.make_sound(buffer)
                    channel = pygame.mixer.Channel(0)
                    for i in range(12):
                        if not channel.get_queue():
                            break
                        time.sleep(0.01)
                    if channel.get_queue():
                        print("Pygame timed out.")
                        channel.stop()
                        if aout.done():
                            globals()["channel2"] = aout.result()
                if not point_fut or point_fut.done():
                    point_fut = submit(point, f"~{frame} {duration}")
                if not channel2:
                    channel.queue(sound)
                if settings.spectrogram and ssize[0] and ssize[1] and not is_minimised():
                    if spec_fut:
                        spec_fut.result()
                    spec_fut = submit(spectrogram, sbuffer)
            frame += settings.speed * 2 ** (settings.nightcore / 12)
    except:
        try:
            kill(proc)
        except:
            pass
        print_exc()

def ensure_parent():
    proc = psutil.Process()
    par = psutil.Process(os.getppid())
    while True:
        if par.is_running():
            time.sleep(0.1)
        else:
            proc.kill()


ffmpeg_start = ("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+discardcorrupt+fastseek+genpts+igndts+flush_packets", "-err_detect", "ignore_err", "-hwaccel", "auto", "-vn")
settings = cdict()
osize = (0, 0)
ssize = (0, 0)
lastpacket = None
packet = None
sample = None
cdc = "auto"
duration = inf
stream = ""
fut = None
reading = None
point_fut = None
proc = None
fn = None
file = None
paused = None
req = 1600 * 2 * 2
frame = 0
drop = 0
failed = 0
submit(render)
submit(remover)
submit(duration_est)
submit(ensure_parent)
pc()
while not sys.stdin.closed and failed < 16:
    try:
        command = sys.stdin.readline().rstrip().rstrip("\x00")
        if command:
            failed = 0
            if command == "~clear":
                if proc:
                    temp, proc = proc, None
                    try:
                        temp.kill()
                    except:
                        pass
                if fn and file:
                    file.close()
                    submit(remove, file)
                fn = file = proc = None
                continue
            if command.startswith("~state"):
                i = int(command[6:])
                if i and not paused:
                    paused = concurrent.futures.Future()
                elif paused:
                    paused.set_result(None)
                    paused = None
                packet_advanced = paused
                if paused:
                    pt2 = time.perf_counter() - pt
                else:
                    if pt2:
                        pt = time.perf_counter() - pt2
                    pt2 = None
                continue
            if command.startswith("~osize"):
                osize = tuple(map(int, command[7:].split()))
                continue
            if command.startswith("~ssize"):
                ssize = tuple(map(int, command[7:].split()))
                continue
            if command.startswith("~setting"):
                setting, value = command[9:].split()
                settings[setting] = float(value)
                if setting in ("volume", "shuffle", "spectrogram", "oscilloscope") or not stream:
                    continue
                pos = frame / 30
            elif command.startswith("~drop"):
                drop += float(command[5:]) * 30
                if drop <= 60 * 30:
                    continue
                pos = (frame + drop) / 30
            else:
                pos, duration, cdc = sys.stdin.readline().rstrip().split(" ", 2)
                pos, duration = map(float, (pos, duration))
                stream = command
            shuffling = False
            if aout.done():
                channel2 = aout.result()
            if proc:
                temp, proc = proc, None
                try:
                    temp.kill()
                except:
                    pass
                if fn and file:
                    file.close()
                    remove(file)
            if fut:
                if paused:
                    paused.set_result(None)
                fut.result(timeout=4)
                if paused:
                    paused = concurrent.futures.Future()
            if reading:
                reading.result()
                reading = None
            ext = construct_options()
            if is_url(stream):
                fn = "cache/~" + shash(stream) + ".pcm"
                if os.path.exists(fn):
                    stream = fn
                    fn = None
                    file = None
                elif pos or not duration < inf or ext:
                    ts = time.time_ns() // 1000
                    fn = "cache/\x7f" + str(ts) + ".pcm"
            else:
                fn = None
                file = None
            if not is_url(stream) and stream.endswith(".pcm") and not ext and settings.speed >= 0:
                f = open(stream, "rb")
                proc = cdict(
                    stdout=cdict(
                        read=f.read,
                    ),
                    stderr=cdict(
                        read=lambda: b"",
                    ),
                    is_running=lambda: True,
                    kill=f.close,
                )
                if pos:
                    i = round(pos * 48000 * 2) * 2
                    f.seek(i)
            elif stream[0] == "<" and stream[-1] == ">":
                pya = afut.result()
                i = int(stream.strip("<>"))
                d = pya.get_device_info_by_index(i)
                ai = pya.open(
                    48000,
                    2,
                    pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=800,
                    input_device_index=i,
                )
                proc = cdict(
                    stdout=cdict(
                        read=lambda n: ai.read(n >> 2, exception_on_overflow=False),
                    ),
                    stderr=cdict(
                        read=lambda: b"",
                    ),
                    is_running=lambda: True,
                    kill=ai.close,
                )
                proc.readable = lambda: ai.get_read_available() >= req >> 2
                if ext:
                    fsize = inf
                    f = proc.stdout
                    f.seek = f.tell = lambda *args: 0
                    f.close = ai.close
                    cmd = list(ffmpeg_start + ("-f", "s16le", "-ar", "48k", "-ac", "2", "-i", "-"))
                    cmd.extend(ext)
                    cmd.extend(("-f", "s16le", "-ar", "48k", "-ac", "2", "-"))
                    print(cmd)
                    proc = psutil.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    reading = submit(reader, f, pos=0)
            else:
                f = None
                if not fn and (cdc == "mp3" and pos >= 960 or settings.shuffle == 2 and duration >= 960) or settings.speed < 0:
                    if (fn or not stream.endswith(".pcm")) and (settings.speed < 0 or cdc != "mp3"):
                        ostream = stream
                        stream = "cache/~" + shash(ostream) + ".pcm"
                        if not os.path.exists(stream):
                            cmd = ffmpeg_start + ("-i", ostream, "-f", "s16le", "-ar", "48k", "-ac", "2", stream)
                            print(cmd)
                            resp = subprocess.run(cmd)
                    fn = None
                    f = open(stream, "rb")
                cmd = list(ffmpeg_start)
                if not fn:
                    if stream.endswith(".pcm"):
                        cmd.extend(("-f", "s16le", "-ar", "48k", "-ac", "2"))
                    elif cdc == "mp3":
                        cmd.extend(("-c:a", "mp3"))
                cmd.extend(("-i", "-" if f else stream))
                cmd.extend(ext)
                cmd.extend(("-f", "s16le", "-ar", "48k", "-ac", "2", fn or "-"))
                if pos and not f:
                    i = cmd.index("-i")
                    ss = "-ss"
                    cmd = cmd[:i] + [ss, str(pos)] + cmd[i:]
                if not fn and f and pos == 0 and settings.shuffle == 2 and duration >= 960:
                    proc = cdict(
                        args=cmd,
                        is_running=lambda: True,
                        stdin=cdict(
                            write=lambda b: None,
                            close=lambda: None,
                            closed=None,
                        ),
                        stdout=cdict(
                            read=lambda n: emptymem[:n],
                        ),
                        stderr=cdict(
                            read=lambda n: emptymem[:n],
                        ),
                        kill=lambda: None,
                        kill2=f.close,
                    )
                else:
                    print(cmd)
                    proc = psutil.Popen(cmd, stdin=subprocess.PIPE if f else subprocess.DEVNULL, stdout=subprocess.DEVNULL if fn else subprocess.PIPE)
                if fn and not pos:
                    proc.kill = lambda: None
                elif f:
                    fsize = os.path.getsize(stream)
                    if pos:
                        fp = round(pos / duration * fsize / 4) << 2
                        if fp > fsize - 2:
                            fp = fsize - 2
                    else:
                        fp = 0 if settings.speed >= 0 else fsize - 2
                    reading = submit(reader, f, pos=fp, reverse=settings.speed < 0, shuffling=pos == 0 and settings.shuffle == 2)
            if point_fut and not point_fut.done():
                point_fut.result()
            point(f"~{pos * 30} {duration}")
            fut = submit(play, pos)
        else:
            failed += 1
    except:
        print_exc()
    time.sleep(0.01)