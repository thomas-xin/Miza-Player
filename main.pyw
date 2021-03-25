import sys
sys.path.append("misc")
import common
globals().update(common.__dict__)


atypes = "wav mp3 ogg opus flac aac m4a webm wma f4a mka mp2 riff".split()
ftypes = [[f"*.{f}" for f in atypes + "mp4 mov qt mkv avi f4v flv wmv raw".split()]]
ftypes[0].append("All supported audio files")


# if 1 or not options.get("init"):
#     yn = easygui.indexbox(
#         "Assign Miza Player as the default application for audio?",
#         "Welcome!",
#         ("Yes", "No", "No and don't ask again")
#     )
#     if yn not in (None, "No"):
#         options.init = True
#     if yn == 0:
#         s = "\n".join(f"assoc .{f}=Miza-Player.{f}" for f in atypes)
#         if s:
#             s += "\n"
#         s += "\n".join(f'ftype Miza-Player.{f}="py" "{os.path.abspath("")}/{sys.argv[0]}" %%*' for f in atypes)
#         with open("assoc.bat", "w", encoding="utf-8") as f:
#             f.write(s)
#         print(ctypes.windll.shell32.ShellExecuteW(None, "runas", "cmd", "/k assoc.bat", None, 1))
        # subprocess.run(("runas", "/user:Administrator", "temp.bat"), stderr=subprocess.PIPE)

with open("assoc.bat", "w", encoding="utf-8") as f:
    f.write(f"cd {os.path.abspath('')}\nstart /MIN py -3.{pyv} {sys.argv[0]} %*")
if not os.path.exists("cache"):
    os.mkdir("cache")


player = cdict(
    paused=False,
    index=0,
    pos=0,
    end=inf,
    amp=0,
    stats=cdict(
        peak=0,
        amplitude=0,
        velocity=0,
        energy=0,
    )
)
sidebar = cdict(
    queue=alist(),
    entries=alist(),
    buttons=alist(),
    particles=alist(),
    ripples=alist(),
)
toolbar = cdict(
    pause=cdict(
        speed=0,
        angle=0,
        maxspeed=4,
    ),
    progress=cdict(
        vis=0,
        angle=0,
        spread=0,
        alpha=0,
        num=0,
        particles=alist(),
    ),
    buttons=alist(),
    ripples=alist(),
)
queue = sidebar.queue
entries = sidebar.entries
progress = toolbar.progress
modified = set()


def setup_buttons():
    try:
        gears = pygame.image.load("misc/gears.bmp").convert_alpha()
        img = Image.frombuffer("RGBA", gears.get_size(), pygame.image.tostring(gears, "RGBA"))
        B, A = img.getchannel("B"), img.getchannel("A")
        I = ImageChops.invert(B)
        R = Image.new("L", I.size, 0)
        inv = pygame.image.frombuffer(Image.merge("RGBA", (R, I, I, A)).tobytes(), I.size, "RGBA")
        def settings_toggle():
            sidebar.abspos ^= 1
        sidebar.buttons.append(cdict(
            sprite=gears,
            invert=inv,
            click=settings_toggle,
        ))
        reset_menu(full=False)
        folder = pygame.image.load("misc/folder.bmp").convert_alpha()
        sidebar.buttons.append(cdict(
            sprite=folder,
            click=enqueue_local,
        ))
        reset_menu(full=False)
        hyperlink = pygame.image.load("misc/hyperlink.bmp").convert_alpha()
        sidebar.buttons.append(cdict(
            sprite=hyperlink,
            click=enqueue_search,
        ))
        reset_menu(full=False)
        repeat = pygame.image.load("misc/repeat.bmp").convert_alpha()
        def repeat_1():
            control.loop = (control.loop + 1) % 3
        toolbar.buttons.append(cdict(
            image=repeat,
            click=repeat_1,
        ))
        reset_menu(full=False)
        shuffle = pygame.image.load("misc/shuffle.bmp").convert_alpha()
        def shuffle_1():
            control.shuffle = (control.shuffle + 1) % 3
            if control.shuffle in (0, 2):
                mixer.submit(f"~setting shuffle {control.shuffle}", force=True)
            if control.shuffle == 2 and player.get("needs_shuffle"):
                seek_abs(player.pos)
        toolbar.buttons.append(cdict(
            image=shuffle,
            click=shuffle_1,
        ))
        reset_menu(full=False)
        back = pygame.image.load("misc/back.bmp").convert_alpha()
        def rleft():
            mixer.clear()
            queue.rotate(1)
            start()
        toolbar.buttons.append(cdict(
            image=back,
            click=rleft,
        ))
        front = pygame.transform.flip(back, True, False)
        def rright():
            mixer.clear()
            queue.rotate(-1)
            start()
        toolbar.buttons.append(cdict(
            image=front,
            click=rright,
        ))
        reset_menu(full=False)
        microphone = pygame.image.load("misc/microphone.bmp").convert_alpha()
        globals()["pya"] = afut.result()
        sidebar.buttons.append(cdict(
            sprite=microphone,
            click=enqueue_device,
        ))
        reset_menu(full=False)
    except:
        print_exc()

cached_fns = {}
def _enqueue_local(*files):
    try:
        if files:
            sidebar.loading = True
            for fn in files:
                if fn[0] == "<" and fn[-1] == ">":
                    pya = afut.result()
                    dev = pya.get_device_info_by_index(int(fn.strip("<>")))
                    entry = cdict(
                        url=fn,
                        stream=fn,
                        name=dev.get("name"),
                        duration=inf,
                    )
                else:
                    fn = fn.replace("\\", "/")
                    if "/" not in fn:
                        fn = "/" + fn
                    options.path, name = fn.rsplit("/", 1)
                    name = name.rsplit(".", 1)[0]
                    try:
                        try:
                            dur, cdc = cached_fns[fn]
                        except KeyError:
                            dur, cdc = cached_fns[fn] = get_duration_2(fn)
                    except:
                        print_exc()
                        dur, cdc = (None, "N/A")
                    # if dur is None and cdc == "N/A":
                    #     fi = fn
                    #     fn = "cache/~" + shash(fi) + ".pcm"
                    #     if not os.path.exists(fn):
                    #         args = ["py", f"-3.{sys.version_info[1]}", "misc/png2wav.py", fi, fn]
                    #         proc = psutil.Popen(args, stderr=subprocess.PIPE)
                    #         while True:
                    #             if os.path.exists(fn) and os.path.getsize(fn) >= 96000:
                    #                 break
                    #             if not proc.is_running():
                    #                 raise RuntimeError(as_str(proc.stderr.read()))
                    #     dur, cdc = get_duration_2(fn)
                    entry = cdict(
                        url=fn,
                        stream=fn,
                        name=name,
                        duration=dur,
                        cdc=cdc,
                    )
                queue.append(entry)
            sidebar.loading = False
    except:
        sidebar.loading = False
        print_exc()

def enqueue_local():
    default = None
    if options.get("path"):
        default = options.path.rstrip("/") + "/"
    files = easygui.fileopenbox(
        "Open an audio or video file here!",
        "Miza Player",
        default=default,
        filetypes=ftypes,
        multiple=True,
    )
    if files:
        submit(_enqueue_local, *files)

eparticle = dict(colour=(255,) * 3)
def _enqueue_search(query):
    try:
        if query:
            sidebar.loading = True
            ytdl = downloader.result()
            try:
                entries = ytdl.search(query)
            except:
                print_exc()
                sidebar.particles.append(cdict(eparticle))
            else:
                if entries:
                    queue.extend(cdict(e) for e in entries)
                else:
                    sidebar.particles.append(cdict(eparticle))
            sidebar.loading = False
    except:
        sidebar.loading = False
        print_exc()

def enqueue_search():
    query = easygui.enterbox(
        "Search for one or more songs online!",
        "Miza Player",
        "",
    )
    if query:
        submit(_enqueue_search, query)

def enqueue_device():
    globals()["pya"] = afut.result()
    count = pya.get_device_count()
    devices = alist()
    for i in range(count):
        d = cdict(pya.get_device_info_by_index(i))
        if d.maxInputChannels > 0 and d.get("hostAPI", 0) >= 0:
            try:
                if not pya.is_format_supported(
                    48000,
                    i,
                    2,
                    pyaudio.paInt16,
                ):
                    continue
                pya.open(
                    48000,
                    2,
                    pyaudio.paInt16,
                    input=True,
                    frames_per_buffer=48000 >> 2,
                    input_device_index=i,
                    start=False,
                ).close()
            except:
                continue
            d.id = i
            devices.add(d)
    selected = easygui.choicebox(
        "Transfer audio from a sound input device!",
        "Miza Player",
        sorted(str(d.id) + ": " + d.name for d in devices),
    )
    if selected:
        submit(_enqueue_local, "<" + selected.split(":", 1)[0] + ">")

def enqueue_auto(*queries):
    for query in queries:
        q = query.strip()
        if q:
            if is_url(q) or not os.path.exists(q):
                submit(_enqueue_search, q)
            else:
                submit(_enqueue_local, q)


if len(sys.argv) > 1:
    submit(enqueue_auto, *sys.argv[1:])


sidebar.abspos = 0
osize = None
ssize = None
def reset_menu(full=True, reset=False):
    global osize, ssize
    if full:
        globals().update(options)
        common.__dict__.update(options)
        if reset:
            DISP.fill(0)
            modified.add(tuple(screensize))
    ssize2 = (screensize[0] - sidebar_width, screensize[1] - toolbar_height)
    if ssize != ssize2:
        ssize = ssize2
        mixer.submit(f"~ssize {' '.join(map(str, ssize))}", True)
    player.rect = (0,) * 2 + ssize
    sidebar.colour = None
    sidebar.updated = False
    sidebar.rect = (screensize[0] - sidebar_width, 0, sidebar_width, screensize[1] - toolbar_height)
    sidebar.rect2 = (screensize[0] - sidebar_width, 0, sidebar_width, screensize[1] - toolbar_height + 4)
    for i, button in enumerate(sidebar.buttons, -1):
        if i < 0:
            button.pos = (screensize[0] - 48, sidebar.rect[1] + 8)
        else:
            button.pos = (sidebar.rect[0] + 8 + 44 * i, sidebar.rect[1] + 8)
        button.rect = button.pos + (40, 40)
    sidebar.resizing = False
    sidebar.resizer = False
    toolbar.colour = None
    toolbar.updated = False
    toolbar.rect = (0, screensize[1] - toolbar_height, screensize[0], toolbar_height)
    toolbar.pause.radius = toolbar_height // 2 - 2
    toolbar.pause.pos = (toolbar.pause.radius + 2, screensize[1] - toolbar.pause.radius - 2)
    progress.pos = (round(toolbar.pause.pos[0] + toolbar.pause.radius * 1.5 + 4), screensize[1] - toolbar_height * 2 // 3 + 1)
    progress.box = toolbar_height * 3 // 2 + 8
    progress.length = max(0, screensize[0] - progress.pos[0] - toolbar.pause.radius // 2 - progress.box)
    progress.width = min(16, toolbar_height // 6)
    progress.rect = (progress.pos[0] - progress.width // 2 - 3, progress.pos[1] - progress.width // 2 - 3, progress.length + 6, progress.width + 6)
    progress.seeking = False
    bsize = min(40, toolbar_height // 3)
    for i, button in enumerate(toolbar.buttons):
        button.pos = (toolbar_height + 8 + (bsize + 4) * i, screensize[1] - toolbar_height // 3 - 8)
        rect = button.pos + (bsize,) * 2
        sprite = button.get("sprite", button.image)
        isize = (bsize - 6,) * 2
        if sprite.get_size() != isize:
            sprite = pygame.transform.smoothscale(button.image, isize)
        button.sprite = sprite
        if i < 2:
            button.on = button.sprite.convert_alpha()
            button.on.fill((0, 255, 255), special_flags=BLEND_RGB_MULT)
            button.off = button.sprite.convert_alpha()
            button.off.fill((0,) * 3, special_flags=BLEND_RGB_MULT)
        button.rect = rect
    toolbar.resizing = False
    toolbar.resizer = False
    osize2 = (progress.box, toolbar_height * 2 // 3 - 3)
    if osize != osize2:
        osize = osize2
        mixer.submit(f"~osize {' '.join(map(str, osize))}", True)


submit(setup_buttons)


is_active = lambda: pc() - player.get("last", 0) <= max(player.get("lastframe", 0), 1 / 30) * 4

e_dur = lambda d: float(d) if type(d) is str else (d if d is not None else nan)

def prepare(entry, force=False):
    stream = entry.get("stream")
    if not stream or stream.startswith("ytsearch:") or force and (stream.startswith("https://cf-hls-media.sndcdn.com/") or stream.startswith("https://www.yt-download.org/download/") and int(stream.split("/download/", 1)[1].split("/", 4)[3]) < utc() + 60) or is_youtube_stream(stream) and int(stream.split("expire=", 1)[-1].split("&", 1)[0]) < utc() + 60:
        ytdl = downloader.result()
        try:
            data = ytdl.search(entry.url)[0]
            stream = ytdl.get_stream(entry, force=True, download=False)
        except:
            print_exc()
            return
        else:
            entry.update(data)
    else:
        stream = entry.stream
    return stream.strip()

def start_player(entry, pos=None):
    player.last = 0
    player.amp = 0
    player.pop("osci", None)
    stream = prepare(entry, force=True)
    if not stream:
        player.fut = None
        return None, inf
    duration = entry.duration
    if not duration:
        info = get_duration_2(stream)
        duration = info[0]
        if info[0] in (None, nan) and info[1] in ("N/A", "auto"):
            fi = stream
            fn = "cache/~" + shash(fi) + ".pcm"
            if not os.path.exists(fn):
                args = ["py", f"-3.{sys.version_info[1]}", "png2wav.py", fi, "../" + fn]
                print(args)
                proc = psutil.Popen(args, cwd="misc", stderr=subprocess.PIPE)
                while True:
                    if os.path.exists(fn) and os.path.getsize(fn) >= 96000:
                        break
                    if not proc.is_running():
                        raise RuntimeError(as_str(proc.stderr.read()))
            duration = get_duration_2(fn)[0]
            stream = entry.stream = fn
    entry.duration = duration
    if duration is None:
        player.fut = None
        return None, inf
    if pos is None:
        if audio.speed >= 0:
            pos = 0
        else:
            pos = entry.duration
    elif pos >= entry.duration:
        if audio.speed > 0:
            return skip()
        pos = entry.duration
    elif pos <= 0:
        if audio.speed < 0:
            return skip()
        pos = 0
    if control.shuffle == 2:
        player.needs_shuffle = False
    else:
        player.needs_shuffle = not is_url(stream)
    mixer.submit(stream + "\n" + str(pos) + " " + str(duration) + " " + str(entry.get("cdc", "auto")))
    player.pos = pos
    player.index = player.pos * 30
    player.end = duration or inf
    return stream, duration

def start():
    if queue:
        return enqueue(queue[0])
    player.last = 0
    player.pos = 0
    player.end = inf
    return None, inf

def skip():
    if queue:
        if control.loop == 2:
            pass
        elif control.loop == 1:
            e = queue.popleft()
            if control.shuffle:
                for i, entry in enumerate(queue):
                    if i >= sidebar.maxitems:
                        break
                    entry.pos = sidebar.maxitems
                queue[:] = queue.shuffle()
            queue.append(e)
        else:
            sidebar.particles.append(queue.popleft())
            if control.shuffle:
                for i, entry in enumerate(queue):
                    if i >= sidebar.maxitems:
                        break
                    entry.pos = sidebar.maxitems
                queue[:] = queue.shuffle()
        if queue:
            return enqueue(queue[0])
        mixer.clear()
    player.last = 0
    player.pos = 0
    player.end = inf
    return None, inf

def seek_abs(pos):
    start_player(queue[0], pos) if queue else (None, inf)

def seek_rel(pos):
    if not pos:
        return
    player.last = 0
    player.amp = 0
    player.pop("osci", None)
    if pos + player.pos >= player.end:
        if audio.speed > 0:
            return skip()
        pos = player.end
    if pos + player.pos <= 0:
        if audio.speed < 0:
            return skip()
        pos = 0
    progress.num += pos
    progress.alpha = 255
    if audio.speed > 0 and pos > 0 and pos <= 180:
        mixer.drop(pos)
    elif audio.speed < 0 and pos < 0 and pos >= -180:
        mixer.drop(pos)
    else:
        seek_abs(max(0, player.pos + pos))

def play():
    try:
        while True:
            b = as_str(mixer.stderr.readline().rstrip())
            if "~" not in b:
                if b:
                    print(b)
                continue
            if b[0] == "o":
                b = b[1:]
                osize = list(map(int, b.split("~")))
                req = int(np.prod(osize) * 3)
                b = mixer.stderr.read(req)
                while len(b) < req:
                    if not mixer.is_running():
                        raise StopIteration
                    b += mixer.stderr.read(req - len(b))
                osci = pygame.image.frombuffer(b, osize, "RGB")
                osci.set_colorkey((0,) * 3)
                player.osci = osci
            else:
                b = b[1:]
                ssize = list(map(int, b.split("~")))
                req = int(np.prod(ssize) * 3)
                b = mixer.stderr.read(req)
                while len(b) < req:
                    if not mixer.is_running():
                        raise StopIteration
                    b += mixer.stderr.read(req - len(b))
                spec = pygame.image.frombuffer(b, ssize, "RGB")
                player.spec = spec
    except:
        if not mixer.is_running():
            print(mixer.stderr.read().decode("utf-8", "replace"))
        print_exc()

def pos():
    try:
        while True:
            s = None
            while not s and mixer.is_running():
                s = mixer.stdout.readline().decode("utf-8", "replace").rstrip()
                if s and s[0] != "~":
                    if s[0] in "'\"":
                        s = ast.literal_eval(s)
                    print(s, end="")
                    s = ""
            if not s:
                if not mixer.is_running():
                    raise StopIteration
                continue
            player.last = pc()
            s = s[1:]
            if s == "s":
                submit(skip)
                player.last = 0
                continue
            if s[0] == "x":
                spl = s[2:].split()
                player.stats.peak = spl[0]
                player.stats.amplitude = spl[1]
                player.stats.velocity = spl[2]
                player.stats.energy = spl[3]
                player.amp = float(spl[4])
                continue
            elif s[0] == "y":
                player.amp = float(s[2:])
                continue
            i, dur = map(float, s.split(" ", 1))
            if not progress.seeking:
                player.index = i
                player.pos = round(player.index / 30, 4)
            if dur >= 0:
                player.end = dur or inf
    except:
        if not mixer.is_running():
            print(mixer.stderr.read().decode("utf-8", "replace"))
        print_exc()

submit(play)
submit(pos)

def enqueue(entry, start=True):
    try:
        if len(queue) > 1:
            submit(prepare, queue[1])
        flash_window()
        stream, duration = start_player(entry)
        progress.num = 0
        progress.alpha = 0
        return stream, duration
    except:
        print_exc()


def update_menu():
    ts = toolbar.pause.setdefault("timestamp", 0)
    t = pc()
    player.lastframe = duration = max(0.001, min(t - ts, 0.125))
    player.flash_s = max(0, player.get("flash_s", 0) - duration * 60)
    player.flash_i = max(0, player.get("flash_i", 0) - duration * 60)
    player.flash_o = max(0, player.get("flash_o", 0) - duration * 60)
    toolbar.pause.timestamp = pc()
    ratio = 1 / (duration * 8)
    progress.vis = (progress.vis * (ratio - 1) + player.pos) / ratio
    progress.alpha *= 0.998 ** (duration * 480)
    if progress.alpha < 16:
        progress.alpha = progress.num = 0
    progress.angle = -t * pi
    if progress.seeking:
        player.amp = 0.5
    elif not is_active():
        player.amp = 0
        player.pop("osci", None)
    progress.spread = min(1, (progress.spread * (ratio - 1) + player.amp) / ratio)
    toolbar.pause.angle = (toolbar.pause.angle + (toolbar.pause.speed + 1) * duration * 2) % tau
    toolbar.pause.speed *= 0.995 ** (duration * 480)
    sidebar.maxitems = int(screensize[1] - options.toolbar_height - 36 >> 5)
    for i, entry in enumerate(queue):
        if i >= sidebar.maxitems:
            break
        entry.pos = (entry.get("pos", 0) * (ratio - 1) + i) / ratio
    sidebar.scroll = max(0, min(len(sidebar.queue) - sidebar.maxitems // 2, sidebar.get("scroll", 0)))
    if kspam[K_SPACE]:
        player.paused ^= True
        mixer.state(player.paused)
        toolbar.pause.speed = toolbar.pause.maxspeed
    if toolbar.resizing:
        toolbar_height = min(96, max(48, screensize[1] - mpos2[1] + 2))
        if options.toolbar_height != toolbar_height:
            options.toolbar_height = toolbar_height
            reset_menu()
            toolbar.resizing = True
            modified.add(toolbar.rect)
    if progress.seeking:
        orig = player.pos
        if player.end < inf:
            player.pos = max(0, min(1, (mpos2[0] - progress.pos[0] + progress.width // 2) / progress.length) * player.end)
            progress.num += (player.pos - orig)
        progress.alpha = 255
        player.index = player.pos * 30
        if not mheld[0]:
            progress.seeking = False
            if queue and isfinite(e_dur(queue[0].duration)):
                submit(seek_abs, player.pos)
    if sidebar.resizing:
        sidebar_width = min(512, max(144, screensize[0] - mpos2[0] + 2))
        if options.sidebar_width != sidebar_width:
            options.sidebar_width = sidebar_width
            reset_menu()
            sidebar.resizing = True
            modified.add(sidebar.rect)
    if queue and isfinite(e_dur(queue[0].duration)):
        if kspam[K_PAGEUP]:
            submit(seek_rel, 300)
        elif kspam[K_PAGEDOWN]:
            submit(seek_rel, -300)
        elif kspam[K_UP]:
            submit(seek_rel, 30)
        elif kspam[K_DOWN]:
            submit(seek_rel, -30)
        elif kspam[K_RIGHT]:
            submit(seek_rel, 5)
        elif kspam[K_LEFT]:
            submit(seek_rel, -5)
    if in_rect(mpos, toolbar.rect[:3] + (5,)):
        if mclick[0]:
            toolbar.resizing = True
        else:
            toolbar.resizer = True
    if in_circ(mpos, toolbar.pause.pos, max(4, toolbar.pause.radius - 2)):
        if mclick[0]:
            player.paused ^= True
            mixer.state(player.paused)
            toolbar.pause.speed = toolbar.pause.maxspeed
        toolbar.pause.outer = 255
        toolbar.pause.inner = 191
        toolbar.updated = False
        sidebar.updated = False
    else:
        toolbar.pause.outer = 191
        toolbar.pause.inner = 127
    if in_rect(mpos, progress.rect):
        if mclick[0]:
            progress.seeking = True
            if queue and isfinite(e_dur(queue[0].duration)):
                mixer.clear()
    if toolbar.resizing or in_rect(mpos, toolbar.rect):
        c = (64, 32, 96)
    else:
        c = (64, 0, 96)
    toolbar.updated = toolbar.colour != c
    toolbar.colour = c
    if mclick[0]:
        for button in toolbar.buttons:
            if in_rect(mpos, button.rect):
                button.flash = 64
                button.click()
    else:
        for button in toolbar.buttons:
            if "flash" in button:
                button.flash = max(0, button.flash - duration * 64)
    maxb = (options.sidebar_width - 12) // 44
    if mclick[0]:
        for button in sidebar.buttons[:maxb]:
            if in_rect(mpos, button.rect):
                button.flash = 32
                button.click()
    else:
        for button in sidebar.buttons:
            if "flash" in button:
                button.flash = max(0, button.flash - duration * 64)
    if in_rect(mpos, sidebar.rect[:2] + (5, sidebar.rect[3])):
        if not toolbar.resizing and mclick[0]:
            sidebar.resizing = True
        else:
            sidebar.resizer = True
    if sidebar.resizing or in_rect(mpos, sidebar.rect):
        c = (64, 32, 96)
    else:
        c = (64, 0, 96)
    sidebar.updated = sidebar.colour != c
    sidebar.colour = c
    sidebar.relpos = min(1, (sidebar.get("relpos", 0) * (ratio - 1) + sidebar.abspos) / ratio)

ripple_colours = (
    (191, 127, 255),
    (255, 0, 127),
    (0, 255, 255),
    (0, 0, 0),
    (255, 255, 255),
)

def draw_menu():
    ts = toolbar.progress.setdefault("timestamp", 0)
    t = pc()
    dur = max(0.001, min(t - ts, 0.125))
    if not tick & 7:
        toolbar.progress.timestamp = pc()
        pops = set()
        for i, ripple in enumerate(toolbar.ripples):
            ripple.radius += dur * 128
            ripple.alpha -= dur * 128
            if ripple.alpha <= 2:
                pops.add(i)
        if pops:
            toolbar.ripples.pops(pops)
        pops = set()
        for i, ripple in enumerate(sidebar.ripples):
            ripple.radius += dur * 128
            ripple.alpha -= dur * 128
            if ripple.alpha <= 2:
                pops.add(i)
        if pops:
            sidebar.ripples.pops(pops)
    crosshair = False
    hovertext = None
    if (sidebar.updated or not tick & 7 or in_rect(mpos2, sidebar.rect) and (any(mclick) or any(kclick))) and sidebar.colour:
        if any(mclick) and in_rect(mpos, sidebar.rect):
            sidebar.ripples.append(cdict(
                pos=mpos,
                radius=0,
                colour=ripple_colours[mclick.index(1)],
                alpha=255,
            ))
        modified.add(sidebar.rect)
        offs = round(sidebar.setdefault("relpos", 0) * -sidebar_width)
        if offs > -sidebar_width + 4:
            Z = 52 + 16
            DISP2 = pygame.Surface(sidebar.rect2[2:])
            DISP2.fill(sidebar.colour)
            for ripple in sidebar.ripples:
                concentric_circle(
                    DISP2,
                    ripple.colour,
                    (ripple.pos[0] - screensize[0] + sidebar_width, ripple.pos[1]),
                    ripple.radius,
                    fill_ratio=1 / 3,
                    alpha=sqrt(max(0, ripple.alpha)) * 16,
                )
            if (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_v]:
                enqueue_auto(*pyperclip.paste().splitlines())
            n = len(queue)
            t = sum(e.get("duration") or 300 for e in queue) - (player.pos or 0)
            message_display(
                f"{n} item{'s' if n != 1 else ''}, estimated time remaining: {time_disp(t)}",
                12,
                (6, 48),
                surface=DISP2,
                align=0,
            )
            if in_rect(mpos, sidebar.rect) and mclick[0] or not mheld[0]:
                sidebar.pop("dragging", None)
            if sidebar.get("last_selected") and not any(entry.get("selected") for entry in queue):
                sidebar.pop("last_selected")
            copies = deque()
            pops = set()
            try:
                if not sidebar.last_selected.selected:
                    raise ValueError
                lq = queue.index(sidebar.last_selected)
            except (AttributeError, ValueError, IndexError):
                sidebar.pop("last_selected", None)
                lq = nan
            lq2 = lq
            swap = None
            maxitems = sidebar.maxitems
            etarget = round((mpos[1] - Z - 16) / 32) if in_rect(mpos, (screensize[0] - sidebar_width + 8, Z, sidebar_width - 16, screensize[1] - toolbar_height - Z)) else nan
            target = min(max(0, round((mpos[1] - Z - 16) / 32)), len(queue) - 1)
            if in_rect(mpos, sidebar.rect) and mclick[0] and not kheld[K_LSHIFT] and not kheld[K_RSHIFT] and not kheld[K_LCTRL] and not kheld[K_RCTRL]:
                if etarget not in range(len(queue)) or not queue[etarget].get("selected"):
                    for entry in queue:
                        entry.pop("selected", None)
                    sidebar.pop("last_selected", None)
                    lq = nan
            noparticles = set()
            for i, entry in enumerate(queue):
                if entry.get("selected"):
                    if kclick[K_DELETE] or kclick[K_BACKSPACE] or (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_x]:
                        pops.add(i)
                        if sidebar.get("last_selected") == entry:
                            sidebar.pop("last_selected", None)
                        if i >= maxitems:
                            noparticles.add(i)
                    if (kheld[K_LCTRL] or kheld[K_RCTRL]) and (kclick[K_c] or kclick[K_x]):
                        entry.flash = 16
                        copies.append(entry.url)
                elif (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_a]:
                    entry.selected = True
                    sidebar.last_selected = entry
                    lq = i
                if i >= maxitems:
                    entry.pop("flash", None)
                    entry.pos = i
                    continue
                if not isfinite(lq):
                    lq2 = nan
                x = 8 + offs
                if entry.get("selected") and sidebar.get("dragging"):
                    y = round(Z + entry.get("pos", 0) * 32)
                    rect = (x, y, sidebar_width - 16, 32)
                    sat = 0.875
                    val = 1
                    secondary = True
                    if pc() % 0.25 < 0.125:
                        entry.colour = col = [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
                    else:
                        col = (255,) * 3
                    bevel_rectangle(
                        DISP2,
                        col,
                        rect,
                        4,
                        alpha=round(255 / (1 + abs(entry.get("pos", 0) - i) / 4)),
                        filled=False,
                    )
                    y = mpos2[1] - 16
                    if isfinite(lq2):
                        y += (i - lq2) * 32
                    if not swap and not mclick[0] and not kheld[K_LSHIFT] and not kheld[K_RSHIFT] and not kheld[K_LCTRL] and not kheld[K_RCTRL] and sidebar.get("last_selected") is entry:
                        if target != i:
                            swap = target - i
                else:
                    y = round(Z + entry.get("pos", 0) * 32)
                rect = (x, y, sidebar_width - 16, 32)
                t = 255
                selectable = i == etarget
                if not selectable and sidebar.get("last_selected") and (kheld[K_LSHIFT] or kheld[K_RSHIFT]):
                    b = lq
                    if b >= 0:
                        a = target
                        a, b = sorted((a, b))
                        if a <= i <= b:
                            selectable = True
                if selectable or entry.get("selected"):
                    if mclick[0] and selectable:
                        if entry.get("selected") and (kheld[K_LCTRL] or kheld[K_RCTRL]):
                            entry.selected = False
                            sidebar.dragging = False
                            sidebar.pop("last_selected", None)
                            lq = nan
                        else:
                            entry.selected = True
                            sidebar.dragging = True
                            if i == target:
                                sidebar.last_selected = entry
                                lq2 = i
                    if entry.get("selected"):
                        flash = entry.get("flash", 16)
                        if flash:
                            entry.flash = flash - 1
                        continue
                    sat = 0.875
                    val = 1
                    secondary = True
                else:
                    sat = 1
                    val = 0.75
                    secondary = False
                flash = entry.get("flash", 16)
                if flash:
                    if flash < 0:
                        entry.flash = 0
                    else:
                        sat = max(0, sat - flash / 16)
                        val = min(1, val + flash / 16)
                        entry.flash = flash - 1
                entry.colour = col = [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
                bevel_rectangle(
                    DISP2,
                    col,
                    rect,
                    4,
                    alpha=255 if secondary else round(255 / (1 + abs(entry.get("pos", 0) - i) / 4)),
                    filled=not secondary,
                )
                if secondary:
                    sat = 0.875
                    val = 0.4375
                    if flash:
                        sat = max(0, sat - flash / 16)
                        val = min(1, val + flash / 16)
                    pygame.draw.rect(
                        DISP2,
                        [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)],
                        [rect[0] + 4, rect[1] + 4, rect[2] - 8, rect[3] - 8],
                    )
                if not entry.get("surf"):
                    entry.surf = message_display(
                        "".join(c if ord(c) < 65536 else "\x7f" for c in entry.name[:128]),
                        12,
                        (0,) * 2,
                        align=0,
                    )
                if not t:
                    blit_complex(
                        DISP2,
                        entry.surf,
                        (x + 6, y + 4),
                        area=(0, 0, sidebar_width - 32, 24),
                        colour=(t,) * 3,
                    )
                else:
                    DISP2.blit(
                        entry.surf,
                        (x + 6, y + 4),
                        (0, 0, sidebar_width - 32, 24),
                    )
                message_display(
                    time_disp(entry.duration) if entry.duration else "N/A",
                    10,
                    [x + sidebar_width - 20, y + 28],
                    (t,) * 3,
                    surface=DISP2,
                    align=2,
                )
            if copies:
                pyperclip.copy("\n".join(copies))
            if pops:
                sidebar.particles.extend(queue[i] for i in pops if i not in noparticles)
                skipping = 0 in pops
                queue.pops(pops)
                if skipping:
                    mixer.clear()
                    submit(start)
            for i, entry in enumerate(queue):
                if i >= maxitems:
                    break
                if not entry.get("selected"):
                    continue
                x = 8 + offs
                y = round(Z + entry.get("pos", 0) * 32)
                sat = 0.875
                val = 1
                rect = (x, y, sidebar_width - 16, 32)
                entry.colour = col = [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
                secondary = sidebar.get("dragging")
                if secondary:
                    y = mpos2[1] - 16
                    if isfinite(lq2):
                        y += (i - lq2) * 32
                    rect = (x, y, sidebar_width - 16, 32)
                sat = 0.875
                val = 0.4375
                flash = entry.get("flash", 16)
                if flash:
                    sat = max(0, sat - flash / 16)
                    val = min(1, val + flash / 16)
                pygame.draw.rect(
                    DISP2,
                    [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)],
                    [rect[0] + 4, rect[1] + 4, rect[2] - 8, rect[3] - 8],
                )
                bevel_rectangle(
                    DISP2,
                    col,
                    rect,
                    4,
                    alpha=255,
                    filled=False,
                )
                if not entry.get("surf"):
                    entry.surf = message_display(
                        "".join(c if ord(c) < 65536 else "\x7f" for c in entry.name[:128]),
                        12,
                        (0,) * 2,
                        align=0,
                    )
                if not t:
                    blit_complex(
                        DISP2,
                        entry.surf,
                        (x + 6, y + 4),
                        area=(0, 0, sidebar_width - 32, 24),
                        colour=(t,) * 3,
                    )
                else:
                    DISP2.blit(
                        entry.surf,
                        (x + 6, y + 4),
                        (0, 0, sidebar_width - 32, 24),
                    )
                message_display(
                    time_disp(entry.duration) if entry.duration else "N/A",
                    10,
                    [x + sidebar_width - 20, y + 28],
                    (t,) * 3,
                    surface=DISP2,
                    align=2,
                )
                anima_rectangle(
                    DISP2,
                    [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12 + 1 / 12, 0.9375, 1)],
                    [rect[0] + 1, rect[1] + 1, rect[2] - 2, rect[3] - 2],
                    frame=4,
                    count=2,
                    flash=1,
                    ratio=pc() * 0.4,
                    reduction=0.1,
                )
            if sidebar.get("loading"):
                x = 8 + offs
                y = round(Z + len(queue) * 32)
                rect = (x, y, sidebar_width - 16, 32)
                bevel_rectangle(
                    DISP2,
                    (191,) * 3,
                    rect,
                    4,
                    alpha=255,
                )
                anima_rectangle(
                    DISP2,
                    (255,) * 3,
                    [rect[0] + 1, rect[1] + 1, rect[2] - 2, rect[3] - 2],
                    frame=4,
                    count=2,
                    flash=1,
                    ratio=pc() * 0.4,
                    reduction=0.1,
                )
                if not sidebar.get("loading_text"):
                    sidebar.loading_text = message_display(
                        "Loading...",
                        12,
                        [0] * 2,
                        align=0,
                    )
                DISP2.blit(
                    sidebar.loading_text,
                    (x + 6, y + 4),
                    (0, 0, sidebar_width - 32, 24),
                )
            if swap:
                orig = queue[0]
                dest = deque()
                targets = {}
                for i, entry in enumerate(queue):
                    if i + swap < 0 or i + swap >= len(queue):
                        continue
                    if entry.get("selected"):
                        targets[i + swap] = entry
                        entry.moved = True
                i = 0
                for entry in queue:
                    while i in targets:
                        dest.append(targets.pop(i))
                        i += 1
                    if not entry.get("moved"):
                        dest.append(entry)
                        i += 1
                    else:
                        entry.pop("moved", None)
                if targets:
                    dest.extend(targets.values())
                    for entry in targets.values():
                        entry.pop("moved", None)
                queue[:] = dest
                if queue[0] is not orig:
                    submit(start_player, queue[0])
            DISP.blit(
                DISP2,
                (screensize[0] - sidebar_width, 0),
            )
        bevel_rectangle(
            DISP,
            sidebar.colour or (64, 0, 96),
            sidebar.rect2,
            4,
            filled=offs <= -sidebar_width + 4
        )
        if offs <= -4:
            if sidebar.ripples:
                DISP2 = pygame.Surface(sidebar.rect[2:], SRCALPHA)
                for ripple in sidebar.ripples:
                    concentric_circle(
                        DISP2,
                        ripple.colour,
                        (ripple.pos[0] - screensize[0] + sidebar_width, ripple.pos[1]),
                        ripple.radius,
                        fill_ratio=1 / 3,
                        alpha=sqrt(max(0, ripple.alpha)) * 16,
                    )
                DISP.blit(
                    DISP2,
                    sidebar.rect[:2],
                )
            offs2 = offs + sidebar_width
            for i, opt in enumerate(asettings):
                message_display(
                    opt.capitalize(),
                    11,
                    (screensize[0] + offs + 8, 52 + i * 32),
                    surface=DISP,
                    align=0,
                )
                # numrect = (screensize[0] + offs + sidebar_width - 8, 68 + i * 32)
                s = str(round(options.audio.get(opt, 0) * 100, 2)) + "%"
                message_display(
                    s,
                    11,
                    (screensize[0] + offs + sidebar_width - 8, 68 + i * 32),
                    surface=DISP,
                    align=2,
                )
                srange = asettings[opt]
                w = (sidebar_width - 16)
                x = (options.audio.get(opt, 0) - srange[0]) / (srange[1] - srange[0])
                if opt in ("speed", "pitch", "nightcore"):
                    x = min(1, max(0, x))
                else:
                    x = min(1, abs(x))
                x = round(x * w)
                brect = (screensize[0] + offs + 6, 67 + i * 32, sidebar_width - 12, 13)
                hovered = in_rect(mpos, brect) or aediting[opt]
                crosshair |= bool(hovered) << 1
                v = max(0, min(1, (mpos2[0] - (screensize[0] + offs + 8)) / (sidebar_width - 16))) * (srange[1] - srange[0]) + srange[0]
                if len(srange) > 2:
                    v = round_min(math.round(v / srange[2]) * srange[2])
                else:
                    rv = round_min(math.round(v * 32) / 32)
                    if type(rv) is int and rv not in srange:
                        v = rv
                if hovered and not hovertext:
                    hovertext = str(round(v * 100, 2)) + "%"
                    if aediting[opt]:
                        if not mheld[0]:
                            aediting[opt] = False
                    elif mclick[0]:
                        aediting[opt] = True
                    if aediting[opt]:
                        orig, options.audio[opt] = options.audio[opt], v
                        if orig != v:
                            mixer.submit(f"~setting {opt} {v}", force=opt == "volume")
                z = max(0, x - 4)
                rect = (screensize[0] + offs + 8 + z, 69 + i * 32, sidebar_width - 16 - z, 9)
                col = (48 if hovered else 32,) * 3
                bevel_rectangle(
                    DISP,
                    col,
                    rect,
                    3,
                )
                rainbow = quadratic_gradient((w, 9), pc() / 2 + i / 4)
                DISP.blit(
                    rainbow,
                    (screensize[0] + offs + 8 + x, 69 + i * 32),
                    (x, 0, w - x, 9),
                    special_flags=BLEND_RGB_MULT,
                )
                rect = (screensize[0] + offs + 8, 69 + i * 32, x, 9)
                col = (223 if hovered else 191,) * 3
                bevel_rectangle(
                    DISP,
                    col,
                    rect,
                    3,
                )
                rainbow = quadratic_gradient((w, 9), pc() + i / 4)
                DISP.blit(
                    rainbow,
                    (screensize[0] + offs + 8, 69 + i * 32),
                    (0, 0, x, 9),
                    special_flags=BLEND_RGB_MULT,
                )
                if hovered:
                    bevel_rectangle(
                        DISP,
                        (191, 127, 255),
                        brect,
                        2,
                        filled=False,
                    )
        maxb = (sidebar_width - 12) // 44
        for i, button in enumerate(sidebar.buttons[:maxb]):
            if button.get("rect"):
                lum = 223 if in_rect(mpos, button.rect) else 191
                lum += button.get("flash", 0)
                if not i:
                    lum -= 32
                    lum += button.get("flash", 0)
                bevel_rectangle(
                    DISP,
                    (lum,) * 3,
                    button.rect,
                    4,
                )
                sprite = button.sprite
                if not i and sidebar.abspos:
                    sprite = button.invert
                DISP.blit(
                    sprite,
                    (button.rect[0] + 5, button.rect[1] + 5),
                    special_flags=BLEND_ALPHA_SDL2,
                )
        if offs > -sidebar_width + 4:
            pops = set()
            for i, entry in enumerate(sidebar.particles):
                if entry.get("life") is None:
                    entry.life = 1
                else:
                    entry.life -= dur
                    if entry.life <= 0:
                        pops.add(i)
                col = [round(i * entry.life) for i in entry.get("colour", (223, 0, 0))]
                y = round(Z + entry.get("pos", 0) * 32)
                ext = round(32 - 32 * entry.life)
                rect = (screensize[0] - sidebar_width + 8 - ext + offs, y - ext * 3, sidebar_width - 16 + ext * 2, 32 + ext * 2)
                bevel_rectangle(
                    DISP,
                    col,
                    rect,
                    4,
                    alpha=round(255 * entry.life),
                )
            sidebar.particles.pops(pops)
        else:
            sidebar.particles.clear()
    if any(mclick) and in_rect(mpos, toolbar.rect):
        toolbar.ripples.append(cdict(
            pos=mpos,
            radius=0,
            colour=ripple_colours[mclick.index(1)],
            alpha=255,
        ))
    highlighted = progress.seeking or in_rect(mpos, progress.rect)
    crosshair |= highlighted
    osci_rect = (screensize[0] - 8 - progress.box, screensize[1] - toolbar_height, progress.box, toolbar_height * 2 // 3 - 3)
    if (toolbar.updated or not tick & 7) and toolbar.colour:
        bevel_rectangle(
            DISP,
            toolbar.colour,
            toolbar.rect,
            4,
        )
        modified.add(toolbar.rect)
        if toolbar.ripples:
            DISP2 = pygame.Surface(toolbar.rect[2:], SRCALPHA)
            for ripple in toolbar.ripples:
                concentric_circle(
                    DISP2,
                    ripple.colour,
                    (ripple.pos[0], ripple.pos[1] - screensize[1] + toolbar_height),
                    ripple.radius,
                    fill_ratio=1 / 3,
                    alpha=sqrt(max(0, ripple.alpha)) * 16,
                )
            DISP.blit(
                DISP2,
                toolbar.rect[:2],
            )
        pos = progress.pos
        length = progress.length
        width = progress.width
        ratio = player.pos / player.end
        if highlighted:
            bevel_rectangle(
                DISP,
                (191, 127, 255),
                progress.rect,
                3,
                filled=False,
            )
        xv = round(length * progress.vis / player.end) if not progress.seeking or player.end < inf else mpos2[0] - pos[0] + width // 2
        xv = max(0, min(xv, length))
        xv2 = max(0, xv - 4)
        if highlighted:
            c = (48,) * 3
        else:
            c = (32,) * 3
        bevel_rectangle(
            DISP,
            c,
            (xv2 + pos[0] - width // 2, pos[1] - width // 2, length - xv2, width),
            min(4, width >> 1),
        )
        rainbow = quadratic_gradient((length, width), pc() / 2)
        DISP.blit(
            rainbow,
            (pos[0] - width // 2 + xv, pos[1] - width // 2),
            (xv, 0, length - xv, width),
            special_flags=BLEND_RGB_MULT,
        )
        if progress.vis or not player.end < inf:
            if highlighted:
                c = (223,) * 3
            else:
                c = (191,) * 3
            bevel_rectangle(
                DISP,
                c,
                (pos[0] - width // 2, pos[1] - width // 2, xv, width),
                min(4, width >> 1),
            )
        rainbow = quadratic_gradient((length, width))
        DISP.blit(
            rainbow,
            (pos[0] - width // 2, pos[1] - width // 2),
            (0, 0, xv, width),
            special_flags=BLEND_RGB_MULT,
        )
        for i, button in enumerate(toolbar.buttons):
            if button.get("rect"):
                lum = 191 if in_rect(mpos, button.rect) else 127
                lum += button.get("flash", 0)
                bevel_rectangle(
                    DISP,
                    (lum,) * 3,
                    button.rect,
                    3,
                )
                if i == 1:
                    val = control.shuffle
                elif i == 0:
                    val = control.loop
                else:
                    val = -1
                if val == 2:
                    if i:
                        sprite = quadratic_gradient(button.sprite.get_size(), pc()).convert_alpha()
                    else:
                        sprite = radial_gradient(button.sprite.get_size(), -pc()).convert_alpha()
                    sprite.blit(
                        button.sprite,
                        (0, 0),
                        special_flags=BLEND_RGBA_MULT,
                    )
                elif val == 1:
                    sprite = button.on
                elif val == 0:
                    sprite = button.off
                else:
                    sprite = button.sprite
                DISP.blit(
                    sprite,
                    (button.rect[0] + 3, button.rect[1] + 3),
                    special_flags=BLEND_ALPHA_SDL2,
                )
                if val == 2:
                    message_display(
                        "1",
                        12,
                        (button.rect[0] + button.rect[2] - 4, button.rect[1] + button.rect[3] - 8),
                        colour=(0,) * 3,
                        surface=DISP,
                    )
        pos = toolbar.pause.pos
        radius = toolbar.pause.radius
        spl = max(4, radius >> 2)
        lum = round(toolbar.pause.speed / toolbar.pause.maxspeed * toolbar.pause.outer)
        if player.paused:
            c = (toolbar.pause.outer, lum, lum)
        elif is_active() and player.amp > 1 / 64:
            c = (lum, toolbar.pause.outer, lum)
        else:
            c = (toolbar.pause.outer, toolbar.pause.outer, lum)
        reg_polygon_complex(
            DISP,
            pos,
            c,
            6,
            radius,
            radius,
            thickness=2,
            repetition=spl,
            angle=toolbar.pause.angle,
        )
        if player.paused:
            c = (toolbar.pause.inner, 0, 0)
        elif is_active():
            c = (lum, toolbar.pause.inner, lum)
        else:
            c = (toolbar.pause.inner, toolbar.pause.inner, lum)
        reg_polygon_complex(
            DISP,
            pos,
            c,
            6,
            radius - spl,
            radius - spl,
            thickness=2,
            repetition=radius - spl,
            angle=toolbar.pause.angle,
        )
        lum = toolbar.pause.outer + 224 >> 1
        rad = max(4, radius // 2)
        col = (lum,) * 3
        if player.paused:
            w = 4
            for i in range(w):
                r = rad + w - i
                x = (w - i) / 2
                x1 = pos[0] - r * (2 - sqrt(3)) // 2
                A = (x1 + r, pos[1])
                B = (x1 - r // 2, pos[1] - r * sqrt(3) // 2)
                C = (x1 - r // 2, pos[1] + r * sqrt(3) // 2)
                c1 = (min(255, (lum + 64) // x),) * 3
                c2 = (min(255, (lum + 256) // x),) * 3
                c3 = (max(0, (lum - 128) // x),) * 3
                pygame.draw.aaline(DISP, c1, A, B)
                pygame.draw.aaline(DISP, c2, B, C)
                pygame.draw.aaline(DISP, c3, A, C)
            x2 = pos[0] - rad * (2 - sqrt(3)) // 2
            pts = (
                (x2 + rad, pos[1]),
                (x2 - rad // 2, pos[1] - rad * sqrt(3) // 2),
                (x2 - rad // 2, pos[1] + rad * sqrt(3) // 2),
            )
            pygame.draw.polygon(DISP, col, pts)
        else:
            bevel_rectangle(
                DISP,
                col,
                (pos[0] - rad, pos[1] - rad, rad * 4 // 5, rad << 1),
                3,
            )
            bevel_rectangle(
                DISP,
                col,
                (pos[0] + (rad + 3) // 5, pos[1] - rad, rad * 4 // 5, rad << 1),
                3,
            )
        if options.get("oscilloscope") and is_active() and player.get("osci"):
            surf = player.osci
            if tuple(osci_rect[2:]) != surf.get_size():
                player.osci = surf = pygame.transform.scale(surf, osci_rect[2:])
            DISP.blit(
                surf,
                osci_rect[:2],
            )
        else:
            if options.get("oscilloscope"):
                c = (255, 0, 0)
            else:
                c = (127, 0, 127)
            y = screensize[1] - toolbar_height * 2 // 3 - 2
            pygame.draw.line(
                DISP,
                c,
                (screensize[0] - 8 - progress.box, y),
                (screensize[0] - 8 - 1, y)
            )
        if player.flash_o > 0:
            bevel_rectangle(
                DISP,
                (191,) * 3,
                osci_rect,
                4,
                alpha=player.flash_o * 8 - 1,
            )
        if not toolbar.resizer and in_rect(mpos, osci_rect):
            bevel_rectangle(
                DISP,
                (191,) * 3,
                osci_rect,
                4,
                filled=False,
            )
        s = f"{time_disp(player.pos)}/{time_disp(player.end)}"
        message_display(
            s,
            min(24, toolbar_height // 3),
            (screensize[0] - 4, screensize[1] - 2),
            surface=DISP,
            align=2,
        )
        x = progress.pos[0] + round(progress.length * progress.vis / player.end) - width // 2 if not progress.seeking or player.end < inf else mpos2[0]
        x = min(progress.pos[0] - width // 2 + progress.length, max(progress.pos[0] - width // 2, x))
        r = ceil(progress.spread * toolbar_height) >> 1
        if r:
            concentric_circle(
                DISP,
                (127, 127, 255),
                (x, progress.pos[1]),
                r,
                fill_ratio=0.5,
            )
        pops = set()
        for i, p in enumerate(progress.particles):
            ri = round(p.life)
            p.hsv[0] = (p.hsv[0] + dur / 12) % 1
            p.hsv[1] = max(p.hsv[0] - dur / 12, 1)
            col = [round(i * 255) for i in colorsys.hsv_to_rgb(*p.hsv)]
            a = round(min(255, (p.life - 2) * 20))
            point = [cos(p.angle) * p.rad, sin(p.angle) * p.rad]
            pos = (p.centre[0] + point[0], p.centre[1] + point[1])
            reg_polygon_complex(
                DISP,
                pos,
                col,
                0,
                ri,
                ri,
                alpha=a,
                thickness=2,
                repetition=ri - 2,
                soft=True,
            )
            p.life -= dur * 2.5
            if p.life <= 6:
                p.angle += dur
                p.rad = max(0, p.rad - 16 * dur)
            if p.life < 3:
                pops.add(i)
        progress.particles.pops(pops)
        for i in shuffle(range(3)):
            hsv = [0.5, 1, 1]
            col = [round(i * 255) for i in colorsys.hsv_to_rgb(*hsv)]
            a = progress.angle + i / 3 * tau
            point = [cos(a) * r, sin(a) * r]
            p = (x + point[0], progress.pos[1] + point[1])
            if r and not tick & 7:
                progress.particles.append(cdict(
                    centre=(x, progress.pos[1]),
                    angle=a,
                    rad=r,
                    life=7,
                    hsv=hsv,
                ))
            ri = max(8, progress.width // 2 + 2)
            reg_polygon_complex(
                DISP,
                p,
                col,
                0,
                ri,
                ri,
                alpha=127 if r else 255,
                thickness=2,
                repetition=ri - 2,
                soft=True,
            )
        a = int(progress.alpha)
        if a >= 16:
            n = round(progress.num)
            if n >= 0:
                s = "+" + str(n)
                c = (0, 255, 0)
            else:
                s = str(n)
                c = (255, 0, 0)
            message_display(
                s,
                min(20, toolbar_height // 3),
                (x, progress.pos[1] - 16),
                c,
                surface=DISP,
                alpha=a,
            )
    if not tick & 7 or toolbar.rect in modified:
        if toolbar.resizing:
            pygame.draw.rect(DISP, (255, 0, 0), toolbar.rect[:3] + (4,))
            if not mheld[0]:
                toolbar.resizing = False
        elif toolbar.resizer:
            pygame.draw.rect(DISP, (191, 127, 255), toolbar.rect[:3] + (4,))
            toolbar.resizer = False
    if not tick & 7 or sidebar.rect in modified:
        if sidebar.resizing:
            pygame.draw.rect(DISP, (255, 0, 0), sidebar.rect[:2] + (4, sidebar.rect[3]))
            if not mheld[0]:
                sidebar.resizing = False
        elif sidebar.resizer:
            pygame.draw.rect(DISP, (191, 127, 255), sidebar.rect[:2] + (4, sidebar.rect[3]))
            sidebar.resizer = False
    if mclick[0]:
        text_rect = (0, 0, 128, 64)
        if in_rect(mpos, text_rect):
            player.flash_i = 32
            options.insights = (options.get("insights", 0) + 1) % 2
        elif in_rect(mpos, player.rect):
            player.flash_s = 32
            options.spectrogram = (options.get("spectrogram", 0) + 1) % 3
            mixer.submit(f"~setting spectrogram {options.spectrogram}", force=True)
        elif in_rect(mpos, osci_rect) and not toolbar.resizer:
            player.flash_o = 32
            options.oscilloscope = (options.get("oscilloscope", 0) + 1) % 2
            mixer.submit(f"~setting oscilloscope {options.oscilloscope}", force=True)
    if crosshair & 1 and (not tick & 7 or toolbar.rect in modified) or crosshair & 2 and (not tick + 4 & 7 or sidebar.rect in modified):
        pygame.draw.line(DISP, (255, 0, 0), (mpos2[0] - 13, mpos2[1] - 1), (mpos2[0] + 11, mpos2[1] - 1), width=2)
        pygame.draw.line(DISP, (255, 0, 0), (mpos2[0] - 1, mpos2[1] - 13), (mpos2[0] - 1, mpos2[1] + 11), width=2)
        pygame.draw.circle(DISP, (255, 0, 0), mpos2, 9, width=2)
        if crosshair & 1:
            p = max(0, min(1, (mpos2[0] - progress.pos[0] + progress.width // 2) / progress.length) * player.end)
            s = time_disp(p)
            message_display(
                s,
                min(20, toolbar_height // 3),
                (mpos2[0], mpos2[1] - 17),
                (255, 255, 127),
                surface=DISP,
            )
        if hovertext:
            message_display(
                hovertext,
                10,
                (mpos2[0], mpos2[1] - 17),
                (255, 255, 127),
                surface=DISP,
            )


for i in range(26):
    globals()[f"K_{chr(i + 97)}"] = i + 4
K_SPACE = 44
K_DELETE = 76
reset_menu(True)
foc = True
minimised = False
mpos = mpos2 = (-inf,) * 2
mheld = mclick = mrelease = mprev = (None,) * 5
kheld = pygame.key.get_pressed()
kprev = kclick = KeyList((None,)) * len(kheld)
last_tick = 0
try:
    for tick in itertools.count(0):
        lpos = mpos
        mprev = mheld
        mheld = get_pressed()
        foc = get_focused()
        if foc:
            minimised = False
        else:
            minimised = is_minimised()
        if not tick & 15:
            if player.paused:
                colour = 4
            elif is_active() and player.amp > 1 / 64:
                colour = 2
            else:
                colour = 8
            submit(taskbar_progress_bar, player.pos / player.end, colour | (player.end >= inf))
        if not minimised:
            mclick = [x and not y for x, y in zip(mheld, mprev)]
            mrelease = [not x and y for x, y in zip(mheld, mprev)]
            mpos2 = mouse_rel_pos()
            mpos3 = pygame.mouse.get_pos()
            if foc:
                mpos = mpos3
            else:
                mpos = (nan,) * 2
            kprev = kheld
            kheld = KeyList(x + y if y else 0 for x, y in zip(kheld, pygame.key.get_pressed()))
            kclick = KeyList(x and not y for x, y in zip(kheld, kprev))
            kspam = kclick
            # if any(kspam):
            #     print(" ".join(map(str, (i for i, v in enumerate(kspam) if v))))
            if not tick & 15:
                kspam = KeyList(x or y >= 240 for x, y in zip(kclick, kheld))
            if not tick & 3 or mpos != lpos or (mpos2 != lpos and any(mheld)) or any(mclick) or any(kclick) or any(mrelease) or any(isnan(x) != isnan(y) for x, y in zip(mpos, lpos)):
                try:
                    update_menu()
                except:
                    print_exc()
                draw_menu()
            if not player.get("fut"):
                if queue:
                    player.fut = submit(start)
            elif not queue:
                player.pop("fut").result()
            if not queue:
                player.pos = 0
                player.end = inf
                player.last = 0
                progress.num = 0
                progress.alpha = 0
            if not tick + 2 & 7:
                if player.get("spec"):
                    if options.get("spectrogram"):
                        rect = player.rect
                        surf = player.spec
                        if tuple(rect[2:]) != surf.get_size():
                            player.spec = surf = pygame.transform.scale(surf, rect[2:])
                        DISP.blit(
                            surf,
                            rect[:2],
                        )
                if player.flash_s > 0:
                    bevel_rectangle(
                        DISP,
                        (191,) * 3,
                        player.rect,
                        4,
                        alpha=player.flash_s * 8 - 1,
                    )
                text_rect = (0, 0, 128, 64)
                if player.flash_i > 0:
                    bevel_rectangle(
                        DISP,
                        (191,) * 3,
                        text_rect,
                        4,
                        alpha=player.flash_i * 8 - 1,
                    )
                if in_rect(mpos, text_rect):
                    bevel_rectangle(
                        DISP,
                        (191,) * 3,
                        text_rect,
                        4,
                        filled=False,
                    )
                elif in_rect(mpos, player.rect):
                    bevel_rectangle(
                        DISP,
                        (191,) * 3,
                        player.rect,
                        4,
                        filled=False,
                    )
            if not tick + 6 & 7 or tuple(screensize) in modified:
                if options.get("insights"):
                    for i, k in enumerate(("peak", "amplitude", "velocity", "energy")):
                        v = player.stats.get(k, 0) if is_active() else 0
                        message_display(
                            f"{k.capitalize()}: {v}%",
                            13,
                            (4, 14 * i),
                            align=0,
                            surface=DISP,
                        )
                if modified:
                    modified.add(tuple(screensize))
                else:
                    modified.add(player.rect)
            if modified:
                if tuple(screensize) in modified:
                    pygame.display.flip()
                else:
                    pygame.display.update(tuple(modified))
                if not tick + 6 & 7 and (tuple(screensize) in modified or player.rect in modified):
                    DISP.fill(
                        (0,) * 3,
                        player.rect,
                    )
                modified.clear()
        else:
            sidebar.particles.clear()
            progress.particles.clear()
        delay = max(0, last_tick - pc() + 1 / 480)
        last_tick = pc()
        time.sleep(delay)
        for event in pygame.event.get():
            if event.type == QUIT:
                raise StopIteration
            elif event.type == VIDEORESIZE:
                flags = get_window_flags()
                if flags == 3:
                    options.maximised = True
                else:
                    options.pop("maximised", None)
                    screensize2[:] = event.w, event.h
                screensize[:] = event.w, event.h
                DISP = pygame.display.set_mode(screensize, RESIZABLE)
                reset_menu(reset=True)
                mpos = (-inf,) * 2
            elif event.type == VIDEOEXPOSE:
                rect = get_window_rect()
                if screenpos2 != rect[:2] and not is_minimised():
                    options.screenpos = rect[:2]
                    screenpos2 = None
        pygame.event.clear()
except Exception as ex:
    pygame.quit()
    if type(ex) is not StopIteration:
        print_exc()
    print("Exiting...")
    options.screensize = screensize2
    if options != orig_options:
        with open(config, "w", encoding="utf-8") as f:
            json.dump(dict(options), f, indent=4)
    if mixer.is_running():
        mixer.clear()
        time.sleep(0.1)
        try:
            mixer.kill()
        except:
            pass
    PROC = psutil.Process()
    for c in PROC.children(recursive=True):
        try:
            c.kill()
        except:
            pass
    futs = set()
    for fn in os.listdir("cache"):
        if fn[0] == "\x7f" and fn.endswith(".pcm"):
            futs.add(submit(os.remove, "cache/" + fn))
    for fut in futs:
        try:
            fut.result()
        except:
            pass
    PROC.kill()