# ONE SMALL STEP FOR MAN, ONE GIANT LEAP FOR SMUDGE KIND! Invaded once again on the 6th March >:D

import sys
sys.stdout.write = lambda *args, **kwargs: None
import pygame

import os, sys, pyaudio, numpy, math, random, base64, hashlib, json, time, subprocess, psutil, traceback, contextlib, colorsys, ctypes, collections, samplerate, itertools
import concurrent.futures, scipy.special
from math import *
np = numpy
deque = collections.deque
suppress = contextlib.suppress
hwnd = int(sys.stdin.readline()[1:])

async_wait = lambda: time.sleep(0.001)

is_strict_minimised = lambda: ctypes.windll.user32.IsIconic(hwnd)
globals()["unfocus-time"] = 0
def is_minimised():
    if ctypes.windll.user32.IsIconic(hwnd):
        return True
    if not settings.get("unfocus"):
        return
    if hwnd == ctypes.windll.user32.GetForegroundWindow(hwnd):
        globals()["unfocus-time"] = time.time()
        return
    return time.time() - globals()["unfocus-time"] > 3

rproc = psutil.Popen((sys.executable, "misc/render.py"), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

pt = None
pt2 = None
def pc():
    global pt
    t = time.perf_counter()
    if not pt:
        pt = t
        return 0
    return t - pt

def round_random(x):
    y = int(x)
    if y == x:
        return y
    x %= 1
    if random.random() <= x:
        y += 1
    return y

def as_str(s):
    if type(s) in (bytes, bytearray, memoryview):
        return bytes(s).decode("utf-8", "replace")
    return str(s)

is_url = lambda url: "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")


def supersample(a, size, hq=False):
    n = len(a)
    if n == size:
        return a
    if n < size:
        if hq:
            a = samplerate.resample(a, size / len(a), "sinc_fastest")
            return supersample(a, size)
        interp = np.linspace(0, n - 1, size)
        return np.interp(interp, range(n), a)
    dtype = a.dtype
    x = ceil(n / size)
    interp = np.linspace(0, n - 1, x * size)
    a = np.interp(interp, range(n), a)
    return np.mean(a.reshape(-1, x), 1, dtype=dtype)


from concurrent.futures import thread

def _adjust_thread_count(self):
    # if idle threads are available, don't spin new threads
    try:
        if self._idle_semaphore.acquire(timeout=0):
            return
    except AttributeError:
        pass

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

exc = concurrent.futures.ThreadPoolExecutor(max_workers=24)
submit = exc.submit

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

    def union(self, other=None, **kwargs):
        temp = self.copy()
        if other:
            temp.update(other)
        if kwargs:
            temp.update(kwargs)
        return temp

    @property
    def __dict__(self):
        return self

    ___repr__ = lambda self: super().__repr__()
    copy = __copy__ = lambda self: self.__class__(self)
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


def aff():
    try:
        pya = pyaudio.PyAudio()
        globals()["OUTPUT_DEVICE"] = pya.get_default_output_device_info()["name"]
        point(f"~o {OUTPUT_DEVICE}")
        globals()["aout"] = submit(pya_init)
        return pya
    except:
        print_exc()

afut = submit(aff)
aout = concurrent.futures.Future()
aout.set_result(None)

OUTPUT_FILE = None

def get_device_by_name(name):
    pya = afut.result()
    for i in range(pya.get_device_count()):
        d = pya.get_device_info_by_index(i)
        if d.get("name") == name:
            return i

def pya_init():
    try:
        i = get_device_by_name(OUTPUT_DEVICE)
        if i is None:
            print(OUTPUT_DEVICE, "not found.")
            i = afut.result().get_default_output_device_info()["index"]
            point(f"~w {OUTPUT_DEVICE}")
        else:
            point("~W")
        try:
            out = afut.result().open(
                output_device_index=i,
                rate=48000,
                channels=2,
                format=pyaudio.paInt16,
                output=True,
                frames_per_buffer=800,
            )
        except OSError:
            out = None
        if not out:
            point(f"~w {OUTPUT_DEVICE}")
            while not out:
                try:
                    out = afut.result().open(
                        rate=48000,
                        channels=2,
                        format=pyaudio.paInt16,
                        output=True,
                        frames_per_buffer=800,
                    )
                except:
                    print_exc()
                    time.sleep(1)
        globals()["channel2"] = out
        return out
    except:
        print_exc()

channel2 = None
pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=1024)

ffmpeg = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
ffprobe = "ffprobe.exe" if os.name == "nt" else "ffprobe"

def _get_duration(filename, _timeout=12):
    command = (
        ffprobe,
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
        ffprobe,
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
            while is_minimised():
                time.sleep(0.5)
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
            for fn in deque(removing):
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
    if file:
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
RSIZE = BSIZE << 2
TSIZE = BSIZE // 3
def reader(f, pos=None, reverse=False, shuffling=False, pcm=False):
    global proc, transfer
    try:
        if pcm:
            rsize = RSIZE << 3
        else:
            rsize = RSIZE
        if getattr(f, "closed", None):
            raise StopIteration
        if pos is None:
            pos = f.tell()
        if reverse:
            pos -= rsize
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
                    if aout.done():
                        globals()["channel2"] = aout.result()
            try:
                b = f.read(rsize)
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
                    if x >= rsize // 3:
                        if pos != 0:
                            print(x, rsize // 3)
                        shuffling = True
                        continue
            if not b:
                break
            if reverse:
                if len(b) & 1:
                    b = memoryview(b)[:-1]
                b = np.flip(np.frombuffer(b, dtype=np.uint16)).tobytes()
                pos -= rsize
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
                pos += rsize
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
    if stats.chorus:
        chorus = abs(stats.chorus)
        ch = min(16, chorus)
        A = B = C = D = ""
        for i in range(ceil(ch)):
            neg = ((i & 1) << 1) - 1
            i = 1 + i >> 1
            i *= stats.chorus / ceil(chorus)
            if i:
                A += "|"
                B += "|"
                C += "|"
                D += "|"
            delay = (8 + 5 * i * tau * neg) % 39 + 19
            A += str(round(delay, 3))
            decay = (0.36 + i * 0.47 * neg) % 0.65 + 1.7
            B += str(round(decay, 3))
            speed = (0.27 + i * 0.573 * neg) % 0.3 + 0.02
            C += str(round(speed, 3))
            depth = (0.55 + i * 0.25 * neg) % max(1, stats.chorus) + 0.15
            D += str(round(depth, 3))
        b = 0.5 / sqrt(ceil(ch + 1))
        options.append(
            "chorus=0.5:" + str(round(b, 3)) + ":"
            + A + ":"
            + B + ":"
            + C + ":"
            + D
        )
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
        opt = "firequalizer=gain_entry="
        entries = []
        high = 24000
        low = 13.75
        bars = 4
        small = 0
        for i in range(bars):
            freq = low * (high / low) ** (i / bars)
            bb = -(i / (bars - 1) - 0.5) * stats.bassboost * 64
            dB = log(abs(bb) + 1, 2)
            if bb < 0:
                dB = -dB
            if dB < small:
                small = dB
            entries.append(f"entry({round(freq, 5)},{round(dB, 5)})")
        entries.insert(0, f"entry(0,{round(small, 5)})")
        entries.append(f"entry(24000,{round(small, 5)})")
        opt += repr(";".join(entries))
        options.append(opt)
    if reverb:
        coeff = abs(reverb)
        wet = min(3, coeff) / 3
        if wet != 1:
            options.append("asplit[2]")
        volume *= 1.2
        if reverb < 0:
            volume = -volume
        options.append("afir=dry=10:wet=10")
        if wet != 1:
            dry = 1 - wet
            options.append("[2]amix=weights=" + str(round(dry, 6)) + " " + str(round(-wet, 6)))
        d = [round(1 - i ** 1.6 / (i ** 1.6 + coeff), 4) for i in range(2, 18, 2)]
        options.append(f"aecho=1:1:400|630:{d[0]}|{d[1]}")
        if d[2] >= 0.05:
            options.append(f"aecho=1:1:920|1450:{d[2]}|{d[3]}")
            if d[4] >= 0.06:
                options.append(f"aecho=1:1:1760|2190:{d[4]}|{d[5]}")
                if d[6] >= 0.07:
                    options.append(f"aecho=1:1:2520|3000:{d[6]}|{d[7]}")
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
        # elif volume > 1 or abs(stats.bassboost):
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
                v = arr[x:y]
                a = np.max(v)
                b = np.min(v)
                osci[i] = a if a >= -b else b
            osci = np.clip(osci * (1 / 65536), -1, 1, out=osci)
            if packet:
                size = osize
                OSCI = pygame.Surface(size)
                time.sleep(0.005)
                if packet:
                    point = (0, osize[1] / 2 + osci[0] * osize[1] / 2)
                    for i in range(1, len(osci)):
                        if not packet:
                            return
                        prev = point
                        point = (i, osize[1] / 2 + osci[i] * osize[1] / 2)
                        hue = ((osci[i] + osci[i - 1]) / 4 + pc() / 3 - i / len(osci)) % 1
                        col = tuple(map(lambda x: round(x * 255), colorsys.hsv_to_rgb(1 - hue, 1, 1)))
                        pygame.draw.line(
                            OSCI,
                            col,
                            point,
                            prev,
                        )
                    time.sleep(0.005)
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

maxfreq = 27.5 * 2 ** ((highest_note + 1.5) / 12)
minfreq = 27.5 * 2 ** ((lowest_note - 1.5) / 12)
barcount = int(highest_note - lowest_note) + 1 + 2
freqmul = 1 / (1 - log(minfreq, maxfreq))

res_scale = 84000
dfts = res_scale // 2 + 1
fff = np.fft.fftfreq(res_scale, 1 / 48000)[:dfts]
fftrans = np.zeros(dfts, dtype=int)
freqscale = 7

bins = barcount * freqscale - 1
for i, x in enumerate(fff):
    if x <= 0:
        continue
    x = round((1 - log(x, maxfreq)) * freqmul * bins)
    if x > bins or x < 0:
        continue
    fftrans[i] = x

spec_update_fut = None
lastspec = 0
lastspec2 = 0

def spectrogram_render():
    global stderr_lock, ssize2, lastspec2, spec_update_fut, packet_advanced2
    try:
        t = pc()
        dur = max(0.001, min(0.125, t - lastspec2))
        lastspec2 = t
        ssize2 = ssize
        specs = settings.spectrogram
        packet_advanced2 = False
        if specs <= 1:
            vertices = 0
        elif specs == 2:
            vertices = settings.get("gradient-vertices", 4)
        elif specs == 3:
            vertices = settings.get("spiral-vertices", 6)
        if vertices > 64:
            vertices = 64
        binfo = b"~r" + b"~".join(map(lambda a: json.dumps(a).encode("utf-8"), (ssize2, specs, vertices, dur, pc()))) + b"\n"
        rproc.stdin.write(binfo)
        rproc.stdin.flush()
        line = rproc.stdout.readline().rstrip()
        while not line.startswith(b"~s"):
            if line:
                print("\x00" + line.decode("utf-8", "replace"))
            line = rproc.stdout.readline().rstrip()
        if line[2:]:
            bsize = 3 * np.prod(deque(map(int, line[2:].split(b"~"))))
            spectrobytes = rproc.stdout.read(bsize)
            while stderr_lock:
                stderr_lock.result()
            stderr_lock = concurrent.futures.Future()
            bsend(line[1:] + b"\n", spectrobytes)
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
        np.multiply(spec_buffer, (1 / 3) ** dur, out=spec_buffer)
        async_wait()
        arr = np.zeros(barcount * freqscale, dtype=np.complex64)
        np.add.at(arr, fftrans, dft)
        arr[0] = 0
        async_wait()
        amp = np.abs(arr, dtype=np.float32)
        x = barcount - np.argmax(amp) / freqscale - 0.5
        point(f"~n {x}")
        amp = supersample(amp, barcount)
        b = amp.tobytes()
        rproc.stdin.write(b"~e" + b)
        rproc.stdin.flush()
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
            spec_buffer = np.append(spec_buffer[-res_scale + len(buffer):], buffer)
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
            t = pc()
            sleep = 0.005
            if lastpacket != packet and sbuffer is not None:
                lastpacket = packet
                if len(sbuffer) > 3200:
                    buffer = sbuffer[:3200]
                    sbuffer = sbuffer[3200:]
                    sleep = 1 / 30
                else:
                    buffer = sbuffer

                amp = sum(np.abs(buffer)) / len(buffer)
                # ampm1 = buffer > 0
                # ampm2 = buffer < 0
                # amp1 = sum(buffer[ampm1])
                # amp2 = -sum(buffer[ampm2])
                # amp = (amp1 + amp2) / len(buffer)
                p_amp = sqrt(amp / 32767)

                if is_strict_minimised():
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

                    if settings.spectrogram > 0 and ssize[0] and ssize[1] and (not spec2_fut or spec2_fut.done()):
                        if is_strict_minimised():
                            spec2_fut = submit(spectrogram_update)
            elif packet_advanced:
                if settings.spectrogram > 0 and ssize[0] and ssize[1] and (not spec2_fut or spec2_fut.done()):
                    if is_strict_minimised():
                        spec2_fut = submit(spectrogram_update)
            packet_advanced = False
            dur = sleep + t - pc()
            if dur > 0:
                time.sleep(dur)
    except:
        print_exc()

emptybuff = b"\x00" * (1600 * 2 * 2)
emptymem = memoryview(emptybuff)
emptysample = np.frombuffer(emptybuff, dtype=np.uint16)

def play(pos):
    global file, fn, proc, drop, quiet, frame, packet, lastpacket, sample, transfer, point_fut, spec_fut, sbuffer, packet_advanced, packet_advanced2, packet_advanced3, sfut
    skipzeros = False
    try:
        frame = pos * 30
        while True:
            if paused and drop <= 0:
                paused.result()
            if stopped:
                break
            p = proc
            if fn:
                if not file:
                    try:
                        if os.path.getsize(fn) < req:
                            raise FileNotFoundError
                        file = open(fn, "rb") # *cough* "FM", Miza is secretly a radio host and Txin's house is a radio station - Smudge
                    except (OSError, FileNotFoundError, PermissionError):
                        if proc and not proc.is_running():
                            point("~r")
                            raise
                        time.sleep(0.005)
                        continue
                try:
                    b = file.read(req)
                except ValueError:
                    if proc_waiting:
                        break
                    else:
                        b = b""
                except:
                    print_exc()
                    b = b""
            else:
                while not getattr(proc, "readable", lambda: True)():
                    time.sleep(0.005)
                try:
                    fut = submit(proc.stdout.read, req)
                    b = fut.result(timeout=4)
                except (AttributeError, concurrent.futures.TimeoutError):
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
                        temp, file = file, None
                        temp.close()
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
            try:
                if drop > 0:
                    drop -= 1
                    raise StopIteration
                r = b
                if len(b) < req:
                    b += bytes(emptymem[:req - len(b)])
                if len(b) & 1:
                    b = memoryview(b)[:-1]
                smp = np.frombuffer(b, dtype=np.int16)
                if sfut:
                    s = sfut.result()
                else:
                    s = synthesize()
                sfut = submit(synthesize)
                wave = s
                if settings.volume != 1 or wave is not None:
                    if settings.volume != 1:
                        s = smp.astype(np.float32)
                        s *= settings.volume
                    else:
                        s = smp
                    if wave is not None:
                        wave += s
                        s = wave
                    if wave is not None or abs(settings.volume) > 1 or settings.volume < -32767 / 32768:
                        s = np.clip(s, -32767, 32767)
                    sample = s.astype(np.int16)
                    b = sample.tobytes()
                else:
                    sample = smp
                sbuffer = sample.astype(np.float32)
                if settings.silenceremove and np.mean(np.abs(sample)) < 64:
                    if quiet >= 15:
                        raise StopIteration
                    else:
                        quiet += 1
                else:
                    quiet = 0
                lastpacket = packet
                packet = b
                packet_advanced = True
                packet_advanced2 = True
                packet_advanced3 = True
                if OUTPUT_FILE:
                    submit(OUTPUT_FILE.write, b)
                while waiting:
                    waiting.result()
                if channel2:
                    b = bytes(b)
                    fut = submit(channel2.write, b)
                    try:
                        fut.result(timeout=1)
                    except:
                        print("Pyaudio timed out.")
                        globals()["waiting"] = concurrent.futures.Future()
                        aout.result().close()
                        afut.result().terminate()
                        globals()["afut"] = submit(pyaudio.PyAudio)
                        globals()["aout"] = submit(pya_init)
                        globals()["channel2"] = None
                        waiting.set_result(None)
                        globals()["waiting"] = None
                if not channel2:
                    buffer = sample.reshape((1600, 2))
                    sound = pygame.sndarray.make_sound(buffer)
                    channel = pygame.mixer.Channel(0)
                    for i in range(24):
                        if not channel.get_queue():
                            break
                        time.sleep(0.005)
                    if channel.get_queue():
                        print("Pygame timed out.")
                        channel.stop()
                        if aout.done():
                            globals()["channel2"] = aout.result()
                if not point_fut or point_fut.done():
                    point_fut = submit(point, f"~{frame} {duration}")
                if not channel2:
                    channel.queue(sound)
                if settings.spectrogram > 0 and ssize[0] and ssize[1] and not is_minimised():
                    if spec_fut:
                        spec_fut.result()
                    spec_fut = submit(spectrogram, sbuffer)
            except StopIteration:
                pass
            frame += settings.speed * 2 ** (settings.nightcore / 12)
    except:
        try:
            kill(proc)
        except:
            pass
        print_exc()


SR = 48000
FR = 1600

baserange = np.arange(FR, dtype=np.float64)
basewave = baserange / FR
c = SR // 48
cm = np.linspace(-2, 2, c, endpoint=True)
cm = scipy.special.erf(cm)
cm += 1
cm /= 2
den = 2
da = 0
db = 0
dc = 8

wavecache = {}
sel_instrument = 0

class Sample(collections.abc.Hashable):

    def __init__(self, data, opt):
        self.data = np.frombuffer(base64.b64decode(data), dtype=np.float32)
        self.cache = {}
        self.opt = opt

    def __hash__(self):
        data = self.data
        x = ceil(log2(len(data)))
        y = [round(i) for i in np.linspace(0, len(data) - 1, num=x)]
        return hash(np.sum(data[y]))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return np.all(self.data == other.data)

    __len__ = lambda self: len(self.data)

    def get(self, period=None):
        if period is None:
            return self.data
        period = round(period / 8) << 3
        if period not in self.cache:
            self.cache[period] = supersample(self.data, period)
            print(period)
            print(len(self.cache))
        return self.cache[period]


sfut = None

def synthesize():
    global prevkeys, den, da, db, dc, synth_buff
    if not wavecache:
        return
    totalkeys = alphakeys.union(prevkeys) if alphakeys or prevkeys else ()
    dt = max(1, len(totalkeys)) + 1
    if db != dt:
        if da != 0:
            dc = 0
        da = den
        db = dt
    instrument = wavecache[sel_instrument]
    u = instrument.opt
    s = None
    for i in totalkeys:
        partials = [(x / (u[0] - 1) * u[1] * 2) - u[1] for x in range(u[0])] if u[0] > 1 and u[1] else [0] * u[0]
        for p in partials:
            freq = 110 * 2 ** ((i + p + 3 + settings.get("pitch", 0) + settings.get("nightcore", 0)) / 12)
            period = SR / freq
            synth_period = period * ceil(4096 / period)
            offs = buffoffs % synth_period
            wave = instrument.get(synth_period)
            if len(partials) > 1:
                wave = wave * (1 / len(partials))
            c = SR // 48
            if i in alphakeys:
                xa = np.linspace(0, period, len(wave), endpoint=False)
                wave = np.interp(np.linspace(offs, FR + offs, FR, endpoint=False) % period, xa, wave)
                if i not in prevkeys:
                    x = min(int((period - offs) % period), FR - c)
                    wave[:x] = 0
                    wave[x:x + c] *= cm
                if s is not None:
                    s += wave
                else:
                    s = wave
            elif i in prevkeys:
                x = min(int((period - offs - c) % period), FR - c)
                if s is None:
                    s = np.zeros(FR, dtype=np.float32)
                xa = np.linspace(0, period, len(wave), endpoint=False)
                wave = np.interp(np.linspace(offs, x + c + offs, x + c, endpoint=False) % period, xa, wave)
                samplespace = np.linspace(-2, 2, x + c)
                samplespace = scipy.special.erf(samplespace)
                samplespace -= 1
                wave *= samplespace
                wave *= -0.5
                s[:x + c] += wave
    time.sleep(0.005)
    if s is not None:
        globals()["buffoffs"] += FR
        m = 32767
        if dc < 5:
            lin = basewave + dc
            lin -= 2
            lin = scipy.special.erf(lin)
            lin += 1
            lin *= (db - da) / 2
            lin += da
            den = lin[-1]
            s /= lin
            dc += 1
        else:
            den = db
            m /= den
            da = db
        m *= settings.get("volume", 1)
        s *= m
        s = np.round(s, out=s)
        s = np.repeat(s, 2)
    else:
        globals()["buffoffs"] = 0
        den = db
        dc = 8
        da = 0
    prevkeys = alphakeys
    return s


def piano_player():
    global sample, sbuffer, spec_fut, point_fut, ssize, lastpacket, packet, packet_advanced, packet_advanced2, packet_advanced3, sfut
    try:
        while True:
            if channel2 and channel2.get_write_available() or not channel2 and not pygame.mixer.Channel(0).get_queue():
                if sfut:
                    s = sfut.result()
                else:
                    s = synthesize()
                sfut = submit(synthesize)
                if s is not None and len(s):
                    s = np.clip(s, -32767, 32767)
                    sample = s.astype(np.int16)
                    b = sample.tobytes()
                    lastpacket = packet
                    packet = b
                    packet_advanced = True
                    packet_advanced2 = True
                    packet_advanced3 = True
                    sbuffer = sample.astype(np.float32)
                    while waiting:
                        waiting.result()
                    if channel2:
                        fut = submit(channel2.write, b)
                        try:
                            fut.result(timeout=1)
                        except:
                            print("Pyaudio timed out.")
                            globals()["waiting"] = concurrent.futures.Future()
                            aout.result().close()
                            afut.result().terminate()
                            globals()["afut"] = submit(pyaudio.PyAudio)
                            globals()["aout"] = submit(pya_init)
                            globals()["channel2"] = None
                            waiting.set_result(None)
                            globals()["waiting"] = None
                    if not channel2:
                        buffer = sample.reshape((1600, 2))
                        sound = pygame.sndarray.make_sound(buffer)
                        channel = pygame.mixer.Channel(0)
                        for i in range(12):
                            if not channel.get_queue():
                                break
                            time.sleep(0.005)
                        if channel.get_queue():
                            print("Pygame timed out.")
                            channel.stop()
                            if aout.done():
                                globals()["channel2"] = aout.result()
                    if not point_fut or point_fut.done():
                        point_fut = submit(point, "~0 inf")
                    if not channel2:
                        channel.queue(sound)
                if settings.get("spectrogram") and ssize[0] and ssize[1] and not is_minimised():
                    if spec_fut:
                        spec_fut.result()
                    spec_fut = submit(spectrogram, sbuffer)
            time.sleep(0.005)
    except:
        print_exc()


PROC = psutil.Process()
def ensure_parent():
    par = psutil.Process(os.getppid())
    while True:
        if par.is_running():
            time.sleep(0.1)
        else:
            PROC.kill()


n_measure = lambda n: n[0]
n_pos = lambda n: n[1]
n_end = lambda n: n[1] + n[4]
n_instrument = lambda n: n[2]
n_pitch = lambda n: n[3]
n_length = lambda n: n[4]
n_volume = lambda n: n[5] if len(n) > 5 else 0.25
n_pan = lambda n: n[6] if len(n) > 6 else 0
n_effects = lambda n: n[7] if len(n) > 7 else ()


synth_samples = np.zeros(0, dtype=np.float32)
def render_notes(i, notes):
    global synth_samples
    timesig = editor.timesig
    note_length = editor.note_length
    samplecount = round_random(timesig[0] * note_length)
    if len(synth_samples) < samplecount << 1:
        if not i:
            synth_samples = np.zeros(samplecount << 1, dtype=np.float32)
        else:
            synth_samples = np.append(synth_samples, np.zeros(samplecount * 2 - len(synth_samples), dtype=np.float32))
    left, right = synth_samples[::2], synth_samples[1::2]
    bufofs = bar * editor.timesig[0] * note_length
    for n in notes:
        v = n_volume(n)
        if v <= 0:
            continue
        v *= 32768
        instrument = wavecache[n_instrument(n)]
        u = instrument.opt
        wave = instrument.data
        offs = n_pos(n) * note_length
        length = n_length(n) * note_length
        pitch = n_pitch(n)
        partials = [(x / (u[0] - 1) * u[1] * 2) - u[1] for x in range(u[0])] if u[0] > 1 and u[1] else [0] * u[0]
        if len(partials) > 1:
            v /= len(partials)
        for p in partials:
            freq = 440 * 2 ** ((p + pitch - 57) / 12)
            slength = 48000 / freq
            pos = round_random(ceil(offs / slength) * slength + (bufofs + offs) % slength)
            if n_measure(n) < bar:
                pos -= slength
            if length > 2 * slength:
                over = length % slength
                if over:
                    length += slength - over
            if v != 1:
                wave = wave * v
            pos = round_random(pos)
            length = round_random(length)
            if pos + length > len(left):
                synth_samples = np.append(synth_samples, np.zeros(pos + length - len(left) << 1, dtype=np.float32))
                left, right = synth_samples[::2], synth_samples[1::2]
                # length = len(right) - pos
                # length -= length % slength
            if length <= 0:
                continue
            positions = np.linspace(0, length / slength * len(wave), length, endpoint=False)
            positions %= len(wave)
            sample = np.interp(positions, np.arange(len(wave)), wave)
            if pos < 0:
                sample = sample[-pos:]
                pos = 0
            sl = slice(pos, pos + len(sample))
            pan = n_pan(n)
            if pan <= -1:
                left[sl] += sample
            elif pan >= 1:
                right[sl] += sample
            elif pan == 0:
                left[sl] += sample
                right[sl] += sample
            else:
                p = 0.5 + (pan / 2)
                lsamp = sample * (1 - p)
                sample *= p
                rsamp = sample
                left[sl] += lsamp
                right[sl] += rsamp
    samples, synth_samples = synth_samples[:samplecount << 1], synth_samples[samplecount << 1:]
    np.clip(samples, -32768, 32767, out=samples)
    return samples


ffmpeg_start = (ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-err_detect", "ignore_err", "-hwaccel", "auto", "-vn")
ffmpeg_stream = ("-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "60")
settings = cdict()
alphakeys = prevkeys = ()
buffoffs = 0
osize = (0, 0)
ssize = (0, 0)
lastpacket = None
packet = None
sample = None
sbuffer = None
cdc = "auto"
duration = inf
stream = ""
sh = ""
fut = None
sf = None
reading = None
stopped = False
proc_waiting = False
point_fut = None
proc = None
fn = None
file = None
paused = None
req = 1600 * 2 * 2
frame = 0
drop = 0
quiet = 0
failed = 0
editor = None
submit(render)
submit(remover)
submit(duration_est)
submit(piano_player)
submit(ensure_parent)
pc()
waiting = None
while not sys.stdin.closed and failed < 8:
    try:
        command = sys.stdin.readline()
        if command:
            command = command.rstrip().rstrip("\x00")
            failed = 0
            pos = frame / 30
            if command.startswith("~render"):
                s = command[8:]
                p, b, s = s.split(" ", 2)
                bar = int(b)
                notes = json.loads(s)
                samples = render_notes(b, notes)
                out = samples.astype(np.int16).tobytes()
                with open(f"cache/&p{p}b{b}.pcm", "wb") as f:
                    f.write(out)
                continue
            if command.startswith("~wave"):
                b = command[6:]
                if b == "clear":
                    wavecache.clear()
                    continue
                sid, opt, data = b.split(None, 2)
                sid = int(sid)
                s = Sample(data.encode("ascii"), json.loads(opt))
                wavecache[sid] = s
                sel_instrument = sid
                continue
            if command.startswith("~select"):
                s = int(command[8:])
                assert wavecache[s]
                sel_instrument = s
                continue
            if command.startswith("~keys"):
                s = command[6:]
                alphakeys = set(map(int, s.split(","))) if s else set()
                if aout.done() and alphakeys or prevkeys:
                    channel2 = aout.result()
                continue
            if command.startswith("~editor"):
                s = command[8:]
                editor = cdict(json.loads(s))
                editor.note_length = 60 / editor.tempo * 48000
                continue
            if command == "~clear":
                if proc:
                    temp, proc = proc, None
                    try:
                        temp.kill()
                    except:
                        pass
                if fn and file:
                    temp, file = file, None
                    temp.close()
                    submit(remove, file)
                fn = file = proc = None
                synth_samples = np.zeros(0, dtype=np.float32)
                continue
            if command.startswith("~state"):
                i = int(command[6:])
                if i and not paused:
                    paused = concurrent.futures.Future()
                elif not i and paused:
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
            if command.startswith("~download"):
                st, fn2 = command[10:].split(" ", 1)
                # fn2 = "cache/~" + h + ".pcm"
                if not os.path.exists(fn2) or time.time() - os.path.getmtime(fn2) > 86400 * 7:
                    cmd = ffmpeg_start
                    if is_url(st):
                        cmd += ffmpeg_stream
                    cmd += ("-nostdin", "-i", st)
                    if fn2.endswith(".pcm"):
                        cmd += ("-f", "s16le")
                    else:
                        cmd += ("-b:a", "192k")
                    cmd += ("-ar", "48k", "-ac", "2", fn2)
                    print(cmd)
                    subprocess.Popen(cmd)
                continue
            if command.startswith("~output"):
                OUTPUT_DEVICE = command[8:]
                waiting = concurrent.futures.Future()
                if aout.result():
                    aout.result().close()
                afut.result().terminate()
                afut = submit(pyaudio.PyAudio)
                aout = submit(pya_init)
                channel2 = None
                waiting.set_result(None)
                waiting = None
                continue
            if command.startswith("~record"):
                s = command[8:]
                if s:
                    OUTPUT_FILE = open(s, "ab")
                else:
                    OUTPUT_FILE.close()
                    OUTPUT_FILE = None
                continue
            if command.startswith("~setting"):
                setting, value = command[9:].split()
                if setting.startswith("#"):
                    setting = setting[1:]
                    nostart = True
                else:
                    nostart = False
                settings[setting] = eval(value, {}, {})
                if nostart or setting in ("volume", "shuffle", "spectrogram", "oscilloscope", "unfocus") or not stream:
                    continue
            elif command.startswith("~drop"):
                drop += float(command[5:]) * 30
                if drop <= 60 * 30:
                    continue
                pos = (frame + drop) / 30
                drop = 0
            elif command != "~replay":
                s = sys.stdin.readline().rstrip().split(" ", 3)
                pos, duration, cdc, sh = s
                pos, duration = map(float, (pos, duration))
                stream = command
            shuffling = False
            if aout.done():
                channel2 = aout.result()
            if proc:
                proc_waiting = True
                temp, proc = proc, None
                try:
                    temp.kill()
                except:
                    pass
                if fn and file:
                    temp, file = file, None
                    temp.close()
                    remove(file)
            if fut:
                stopped = True
                if paused:
                    paused.set_result(None)
                try:
                    fut.result(timeout=5)
                except:
                    print_exc()
                    print("Previously playing song timed out, killing relevant subprocesses and skipping...")
                    drop = 0
                    fut = None
                if paused:
                    paused = concurrent.futures.Future()
            stopped = False
            if reading:
                reading.result()
                reading = None
            ext = construct_options()
            if is_url(stream):
                fn = "cache/~" + sh + ".pcm"
                if os.path.exists(fn) and abs(os.path.getsize(fn) / 48000 / 2 / 2 - duration) < 1:
                    stream = fn
                    fn = None
                    file = None
                elif pos or not duration < inf or ext:
                    ts = time.time_ns() // 1000
                    fn = "cache/\x7f" + str(ts) + ".pcm"
            else:
                fn = None
                file = None
            if not stream:
                continue
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
                    reading = submit(reader, f, pos=0, pcm=True)
            else:
                f = None
                if not fn and (cdc == "mp3" and pos >= 960 or settings.shuffle == 2 and duration >= 960) or settings.speed < 0:
                    if (fn or not stream.endswith(".pcm")) and (settings.speed < 0 or cdc != "mp3"):
                        ostream = stream
                        stream = "cache/~" + sh + ".pcm"
                        if not os.path.exists(stream) or os.path.getsize(stream) / 48000 / 2 / 2 < duration - 1:
                            cmd = ffmpeg_start
                            if is_url(stream):
                                cmd += ffmpeg_stream
                            cmd += ("-nostdin", "-i", ostream, "-f", "s16le", "-ar", "48k", "-ac", "2", stream)
                            print(cmd)
                            resp = subprocess.run(cmd)
                    fn = None
                    f = open(stream, "rb")
                cmd = ffmpeg_start
                if is_url(stream):
                    cmd += ffmpeg_stream
                cmd = list(cmd)
                pcm = False
                if not fn:
                    if stream.endswith(".pcm"):
                        cmd.extend(("-f", "s16le", "-ar", "48k", "-ac", "2"))
                        pcm = True
                    elif cdc == "mp3":
                        cmd.extend(("-c:a", "mp3"))
                cmd.extend(("-nostdin", "-i", "-" if f else stream))
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
                    reading = submit(reader, f, pos=fp, reverse=settings.speed < 0, shuffling=pos == 0 and settings.shuffle == 2, pcm=pcm)
            if point_fut and not point_fut.done():
                point_fut.result()
            point(f"~{pos * 30} {duration}")
            fut = submit(play, pos)
            proc_waiting = False
        else:
            failed += 1
    except:
        print_exc()
    time.sleep(0.005)