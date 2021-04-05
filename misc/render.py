import sys, numpy, math, random, json, collections, colorsys, concurrent.futures
from math import *
from traceback import print_exc
np = numpy
from PIL import Image, ImageDraw, ImageFont
exc = concurrent.futures.ThreadPoolExecutor(max_workers=4)
submit = exc.submit


higher_bound = "C10"
highest_note = "C~D~EF~G~A~B".index(higher_bound[0].upper()) - 9 + ("#" in higher_bound)
while higher_bound[0] not in "0123456789-":
    higher_bound = higher_bound[1:]
    if not higher_bound:
        raise ValueError("Octave not found.")
highest_note += int(higher_bound) * 12 + 1

lower_bound = "C0"
lowest_note = "C~D~EF~G~A~B".index(lower_bound[0].upper()) - 9 + ("#" in lower_bound)
while lower_bound[0] not in "0123456789-":
    lower_bound = lower_bound[1:]
    if not lower_bound:
        raise ValueError("Octave not found.")
lowest_note += int(lower_bound) * 12 + 1

maxfreq = 27.5 * 2 ** ((highest_note + 0.5) / 12)
minfreq = 27.5 * 2 ** ((lowest_note - 0.5) / 12)
barcount = int(highest_note - lowest_note) + 1 + 2
freqmul = 1 / (1 - log(minfreq, maxfreq))

barheight = 720
res_scale = 98304
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

    fsize = 0
    # font = ImageFont.truetype("misc/Pacifico.ttf", fsize)

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
        size = min(2 * barheight, round(self.height))
        if size:
            amp = size / barheight * 2
            val = min(1, amp)
            sat = 1 - min(1, max(0, amp - 1))
            self.colour = tuple(round(i * 255) for i in colorsys.hsv_to_rgb((pc_ / 3 + self.x / barcount) % 1, sat, val))
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

def spectrogram_render():
    global ssize2, specs, dur
    try:
        if specs != 2:
            sfx = Image.new("RGB", (barcount - 2, barheight), (0,) * 3)
            for bar in bars:
                bar.render(sfx=sfx)
        else:
            sfx = Image.new("RGB", (barcount * 2 - 2,) * 2, (0,) * 3)
            globals()["DRAW"] = ImageDraw.Draw(sfx)
            for bar in bars:
                bar.render2(sfx=sfx)
        if sfx.size != ssize2:
            sfx = sfx.resize(ssize2, resample=Image.NEAREST)

        if specs != 2:
            fsize = max(1, round(ssize2[0] / barcount * 3 / 2))
            if Bar.fsize != fsize:
                Bar.fsize = fsize
                Bar.font = ImageFont.truetype("misc/Pacifico.ttf", Bar.fsize)
            globals()["DRAW"] = ImageDraw.Draw(sfx)
            highbars = sorted(bars, key=lambda bar: bar.height, reverse=True)[:24]
            high = highbars[0]
            for bar in reversed(highbars):
                bar.post_render(sfx=sfx, scale=bar.height / max(1, high.height))
        spectrobytes = sfx.tobytes()
        
        write = False
        for bar in bars:
            if bar.height2:
                write = True
            bar.update(dur=dur)

        if write:
            sys.stdout.buffer.write(b"~s" + "~".join(map(str, ssize2)).encode("utf-8") + b"\n")
            sys.stdout.buffer.write(spectrobytes)
        else:
            sys.stdout.write("~s\n")
        sys.stdout.flush()
    except:
        print_exc()

fut = None
ssize2 = (0, 0)
specs = 0
while True:
    line = sys.stdin.buffer.read(2)
    if line == b"~r":
        line = sys.stdin.buffer.readline().rstrip()
        # print(line)
        ssize2, specs, dur, pc_ = map(json.loads, line.split(b"~"))
        if fut:
            fut.result()
        fut = submit(spectrogram_render)
    elif line == b"~e":
        amp = np.frombuffer(sys.stdin.buffer.read(4 * len(bars)), dtype=np.float32)
        for i, pwr in enumerate(amp):
            bars[i].ensure(pwr / 8)
    elif not line:
        break