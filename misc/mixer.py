# ONE SMALL STEP FOR MAN, ONE GIANT LEAP FOR SMUDGE KIND! Invaded once again on the 6th March >:D

import sys
sys.stdout.write = lambda *args, **kwargs: None
import pygame

import os, sys, pyaudio, numpy, math, base64, hashlib, time, subprocess, psutil, traceback, contextlib, colorsys, ctypes, concurrent.futures
from math import *
np = numpy
suppress = contextlib.suppress
hwnd = int(sys.stdin.readline()[1:])
is_minimised = lambda: ctypes.windll.user32.IsIconic(hwnd)

def as_str(s):
    if type(s) in (bytes, bytearray, memoryview):
        return bytes(s).decode("utf-8", "replace")
    return str(s)

is_url = lambda url: "://" in url and url.split("://", 1)[0].removesuffix("s") in ("http", "hxxp", "ftp", "fxp")
shash = lambda s: as_str(base64.b64encode(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).replace(b"/", b"-").rstrip(b"="))

exc = concurrent.futures.ThreadPoolExecutor(max_workers=12)
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
            frames_per_buffer=1600,
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

def bsend(s):
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

def duration_est():
    global duration
    last_fn = ""
    last_mt = 0
    last_fs = 0
    try:
        while True:
            if stream and not is_url(stream) and (stream[0] != "<" or stream[-1] != ">"):
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
            time.sleep(0.5)
    except:
        print_exc()


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

def communicate():
    try:
        while True:
            if out[0] != prev[0] and not is_minimised():
                prev[:] = out
                # sample(out[1])
                point(f"~{out[0]} {duration}")
            time.sleep(0.001)
    except:
        print_exc()

def reader(f, proc):
    while True:
        b = f.read(65536)
        if not b:
            break
        proc.stdin.write(b)
    proc.stdin.close()
    f.close()

osize = (152, 61)
OSCI = pygame.Surface(osize)
def oscilloscope(buffer):
    try:
        arr = buffer[::2] + buffer[1::2]
        osci = np.empty(osize[0])
        r = len(arr) / len(osci)
        for i in range(len(osci)):
            x = round(i * r)
            y = round(i * r + r)
            osci[i] = np.mean(arr[x:y])
        osci = np.clip(osci * (1 / 65536), -1, 1, out=osci)
        time.sleep(0.001)
        if packet:
            OSCI.fill((0,) * 4)
            time.sleep(0.001)
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
                time.sleep(0.001)
                if packet:
                    b = pygame.image.tostring(OSCI, "RGB")
                    bsend(b)
    except:
        print_exc()

def render():
    global lastpacket
    try:
        while True:
            if lastpacket != packet and not is_minimised():
                lastpacket = packet
                buffer = sample.astype(np.float64)
                submit(oscilloscope, buffer)

                out = []
                out.append(round(max(np.max(buffer), -np.min(buffer)) / 32768 * 100, 3))

                msk1 = buffer >= 0
                msk2 = buffer < 0
                b1 = buffer[msk1]
                b2 = buffer[msk2]
                h = np.mean(b1) if len(b1) else 0
                l = np.mean(b2) if len(b2) else 0
                amp = (h - l) / 2
                out.append(round(amp / 32768 * 100, 2))

                time.sleep(0.001)
                if packet:
                    vel1 = buffer[::2][1:] - buffer[::2][:-1]
                    vel2 = buffer[1::2][1:] - buffer[1::2][:-1]
                    amp1 = np.mean(np.abs(vel1))
                    amp2 = np.mean(np.abs(vel2))
                    vel = (amp1 + amp2)
                    out.append(round(vel / 131072 * 100, 3))

                    time.sleep(0.001)
                    if packet:
                        amp1 = np.mean(np.abs(vel1[::2][1:] - vel1[::2][:-1]))
                        amp2 = np.mean(np.abs(vel2[1::2][1:] - vel2[1::2][:-1]))
                        out.append(round((amp1 + amp2) / 262144 * 100, 3))

                        nrg = (amp1 + amp2)
                        adj = nrg / 32768 + vel / 65536 + amp / 49152
                        out.append(min(1, sqrt(adj)))

                        if packet:
                            point("~x " + " ".join(map(str, out)))
            time.sleep(0.001)
    except:
        print_exc()

def play(pos):
    global file, fn, proc, drop, frame, packet, lastpacket, sample
    try:
        frame = round(pos * 30)
        while True:
            if paused and drop <= 0:
                paused.result()
            if fn:
                if not file:
                    try:
                        if os.path.getsize(fn) < req:
                            raise FileNotFoundError
                        file = open(fn, "rb") # *cough* "FM", Miza is secretly a radio host and Txin's house is a radio station - Smudge
                    except (OSError, FileNotFoundError, PermissionError):
                        if proc and not proc.is_running():
                            raise
                        time.sleep(0.001)
                        continue
                try:
                    b = file.read(req)
                except:
                    b = b""
            else:
                b = b""
                while not getattr(proc, "readable", lambda: True)():
                    time.sleep(0.001)
                try:
                    b = proc.stdout.read(req)
                except:
                    pass
            if not b:
                out[:] = frame, b"\x00" * 4
                print(f"{proc} {fn}")
                if proc:
                    if proc.is_running():
                        try:
                            proc.kill()
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
            if drop > 0:
                drop -= 1
            else:
                r = b
                if len(b) < req:
                    b += b"\x00" * (req - len(b))
                lastpacket = packet
                packet = b
                sample = np.frombuffer(b, dtype=np.int16)
                if channel2:
                    channel2.write(b)
                else:
                    buffer = sample.reshape((1600, 2))
                    sound = pygame.sndarray.make_sound(buffer)
                    channel = pygame.mixer.Channel(0)
                    while channel.get_queue():
                        time.sleep(0.001)
                out[:] = frame, r
                if not channel2:
                    channel.queue(sound)
            frame += 1
    except:
        try:
            proc.kill()
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


lastpacket = None
packet = None
out = [-1, b""]
prev = list(out)
duration = inf
stream = ""
fut = None
proc = None
fn = None
file = None
paused = None
req = 1600 * 2 * 2
frame = 0
drop = 0
submit(communicate)
submit(render)
submit(remover)
submit(duration_est)
submit(ensure_parent)
while True:
    try:
        command = sys.stdin.readline().rstrip()
        if command:
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
                continue
            if command.startswith("~drop"):
                drop += float(command[5:]) * 30
                if drop <= 180 * 30:
                    continue
                pos = (frame + drop) / 30
            else:
                pos, duration = map(float, sys.stdin.readline().split(" ", 1))
                stream = command
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
                fut.result()
            if is_url(stream):
                fn = "cache/~" + shash(stream) + ".pcm"
                if os.path.exists(fn):
                    stream = fn
                    fn = None
                    file = None
                elif pos or not duration < inf:
                    ts = time.time_ns() // 1000
                    fn = "cache/\x7f" + str(ts) + ".pcm"
            else:
                fn = None
                file = None
            if not is_url(stream) and stream.endswith(".pcm"):
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
                f = pya.open(
                    48000,
                    2,
                    pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=req >> 2,
                    input_device_index=i,
                )
                proc = cdict(
                    stdout=cdict(
                        read=lambda n: f.read(n >> 2, exception_on_overflow=False),
                    ),
                    stderr=cdict(
                        read=lambda: b"",
                    ),
                    is_running=lambda: True,
                    kill=f.close,
                )
                proc.readable = lambda: f.get_read_available() >= req >> 2
            else:
                f = None
                if pos >= 960 and not fn:
                    f = open(stream, "rb")
                cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-vn", "-i", "-" if f else stream, "-map_metadata", "-1", "-f", "s16le", "-ar", "48k", "-ac", "2", fn or "-"]
                if pos and not f:
                    i = cmd.index("-i")
                    ss = "-ss"
                    cmd = cmd[:i] + [ss, str(pos)] + cmd[i:]
                print(cmd)
                proc = psutil.Popen(cmd, stdin=subprocess.PIPE if f else subprocess.DEVNULL, stdout=subprocess.DEVNULL if fn else subprocess.PIPE, stderr=subprocess.PIPE)
                if fn and not pos:
                    proc.kill = lambda: None
                elif f:
                    if pos:
                        f.seek(round(pos / duration * os.path.getsize(stream)))
                    kill = proc.kill
                    proc.kill = lambda: (kill(), f.close())
                    submit(reader, f, proc)
            fut = submit(play, pos)
    except:
        print_exc()