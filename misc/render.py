import sys, time, numpy, math, random, json, collections, colorsys, traceback, concurrent.futures
from math import *
from traceback import print_exc
np = numpy
from PIL import Image, ImageDraw, ImageFont

async_wait = lambda: time.sleep(0.001)


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


exc = concurrent.futures.ThreadPoolExecutor(max_workers=4)
submit = exc.submit


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
    # font = ImageFont.truetype("misc/Pacifico.ttf", fsize)

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
        size = min(2 * barheight, round(self.height))
        if size:
            dark = False
            self.colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb((pc_ / 3 + self.x / barcount) % 1, 1, 1))
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
        size = min(2 * barheight, round(self.height))
        if size:
            amp = size / barheight * 2
            val = min(1, amp)
            sat = 1 - min(1, max(0, amp - 1))
            self.colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb((pc_ / 3 + self.x / barcount / freqscale2) % 1, sat, val))
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
        size = round(self.height2)
        if size > 8:
            ix = barcount - 1 - self.x - 1
            sx = round(ix / barcount * ssize2[0])
            w = round((ix + 1) / barcount * ssize2[0]) - sx
            try:
                width = DRAW.textlength(self.line, self.font)
            except (TypeError, AttributeError):
                width = self.fsize / (sqrt(5) + 1) * len(self.line)
            x = sx + w / 2 - width / 2
            pos = max(84, ssize2[1] - size - width * (sqrt(5) + 1))
            factor = round(255 * scale)
            col = sum(factor << (i << 3) for i in range(3))
            DRAW.text((x, pos), self.line, col, self.font)

bars = [Bar(i - 1, barcount) for i in range(barcount)]
bars2 = [Bar(i - 1, barcount2) for i in range(barcount2)]
CIRCLE_SPEC = None

def spectrogram_render(bars):
    global ssize2, specs, dur
    try:
        if specs != 3:
            globals()["CIRCLE_SPEC"] = None
        if specs == 1:
            sfx = Image.new("RGB", (barcount - 2, barheight), (0,) * 3)
            for bar in bars:
                bar.render(sfx=sfx)
        elif specs == 2:
            sfx = Image.new("RGB", (barcount * freqscale2 * 2 - 2,) * 2, (0,) * 3)
            globals()["DRAW"] = ImageDraw.Draw(sfx)
            for bar in bars:
                bar.render2(sfx=sfx)
        else:
            spec = CIRCLE_SPEC
            D = 5
            R = D / 2 / freqscale2
            for i in range(2):
                if not spec:
                    class Circle_Spec:
                        angle = 0
                        rotation = 0
                    globals()["CIRCLE_SPEC"] = spec = Circle_Spec()
                    spec.image = Image.new("RGB", (barcount * D - D,) * 2)
                else:
                    spec.angle += 1 / 720 * tau
                    spec.rotation += 0.25
                    r = 31 / 32
                    s = 1 / 96
                    colourmatrix = (
                        r, s, 0, 0,
                        0, r, s, 0,
                        s, 0, r, 0,
                    )
                    spec.image = spec.image.convert("RGB", colourmatrix)
                    # spec.image = im
                sfx = spec.image
                globals()["DRAW"] = ImageDraw.Draw(sfx)
                c = (sfx.width + 1 >> 1, sfx.height + 1 >> 1)
                for bar in sorted(bars, key=lambda bar: bar.height):
                    size = min(2 * barheight, round(bar.height))
                    amp = size / barheight * 2
                    val = min(1, amp)
                    sat = 1 - min(1, max(0, amp - 1))
                    colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb((pc_ / 3 + bar.x / barcount / freqscale2) % 1, sat, val))
                    x = bar.x + 1
                    r1 = x * R - 1
                    r2 = x * R + 1
                    for i in range(vertices):
                        z = spec.angle + i / vertices * tau
                        p1 = (c[0] + cos(z) * r1, c[1] + sin(z) * r1)
                        p2 = (c[0] + cos(z) * r2, c[1] + sin(z) * r2)
                        DRAW.line((p1, p2), colour, 3)
                async_wait()
            sfx = sfx.rotate(spec.rotation, resample=Image.NEAREST)

        if specs == 1:
            if sfx.size != ssize2:
                sfx = sfx.resize(ssize2, resample=Image.NEAREST)
            fsize = max(12, round(ssize2[0] / barcount * (sqrt(5) + 1) / 2))
            if Bar.fsize != fsize:
                Bar.fsize = fsize
                Bar.font = ImageFont.truetype("misc/Pacifico.ttf", Bar.fsize)
            globals()["DRAW"] = ImageDraw.Draw(sfx)
            highbars = sorted(bars, key=lambda bar: bar.height, reverse=True)[:24]
            high = highbars[0]
            for bar in reversed(highbars):
                bar.post_render(sfx=sfx, scale=bar.height / max(1, high.height))
        async_wait()
        spectrobytes = sfx.tobytes()

        write = specs == 3
        for bar in bars:
            if bar.height2:
                write = True
            bar.update(dur=dur)

        if write:
            sys.stdout.buffer.write(b"~s" + "~".join(map(str, sfx.size)).encode("utf-8") + b"\n")
            sys.stdout.buffer.write(spectrobytes)
        else:
            sys.stdout.write("~s\n")
        sys.stdout.flush()
    except:
        # with open("log.txt", "w") as f:
        #     f.write(traceback.format_exc())
        print_exc()

fut = None
ssize2 = (0, 0)
specs = 0
while True:
    try:
        line = sys.stdin.buffer.read(2)
        if line == b"~r":
            line = sys.stdin.buffer.readline().rstrip()
            # print(line)
            ssize2, specs, vertices, dur, pc_ = map(json.loads, line.split(b"~"))
            if fut:
                fut.result()
            bi = bars2 if specs > 1 else bars
            fut = submit(spectrogram_render, bi)
        elif line == b"~e":
            b = sys.stdin.buffer.read(1)
            while b[-1:] != b"\n":
                b += sys.stdin.buffer.read(1)
            amp = np.frombuffer(sys.stdin.buffer.read(int(b)), dtype=np.float32)
            bi = bars2 if len(amp) > len(bars) else bars
            for i, pwr in enumerate(amp):
                bi[i].ensure(pwr / 2)
        elif not line:
            break
    except:
        print_exc()