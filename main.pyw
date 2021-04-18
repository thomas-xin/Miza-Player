import sys
sys.path.append("misc")
import common
globals().update(common.__dict__)


atypes = "wav mp3 ogg opus flac aac m4a webm wma f4a mka mp2 riff".split()
ftypes = [[f"*.{f}" for f in atypes + "mp4 mov qt mkv avi f4v flv wmv raw".split()]]
ftypes[0].append("All supported audio files")

with open("assoc.bat", "w", encoding="utf-8") as f:
    f.write(f"cd {os.path.abspath('')}\nstart /MIN py -3.{pyv} {sys.argv[0]} %*")
if not os.path.exists("cache"):
    os.mkdir("cache")
if not os.path.exists("playlists"):
    os.mkdir("playlists")


teapot_fut = submit(load_surface, "misc/teapot.png")

project = cdict(
    settings=cdict(
        time=(4, 4),
        speed=144,
    ),
    instruments=alist(),
    measures=alist(),
)
instruments = project.instruments
measures = project.measures

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
    scroll=cdict(),
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
    downloading=cdict(
        target=0,
    ),
    buttons=alist(),
    ripples=alist(),
    editor=0,
)
queue = sidebar.queue
entries = sidebar.entries
progress = toolbar.progress
downloading = toolbar.downloading
modified = set()


button_sources = (
    "gears",
    "notes",
    "folder",
    "hyperlink",
    "playlist",
    "history",
    "plus",
    "edit",
    "waves",
    "repeat",
    "shuffle",
    "back",
    "flip",
    "scramble",
    "microphone",
)
button_images = cdict((i, submit(load_surface, "misc/" + i + ".png")) for i in button_sources)

def load_pencil():
    globals()["pencilw"] = load_surface("misc/pencil.png")
    globals()["pencilb"] = pencilw.convert_alpha()
    pencilb.fill((0, 0, 0), special_flags=BLEND_RGB_MULT)
submit(load_pencil)

def setup_buttons():
    try:
        gears = button_images.gears.result()
        notes = button_images.notes.result()
        # img = Image.frombuffer("RGBA", gears.get_size(), pygame.image.tostring(gears, "RGBA"))
        # B, A = img.getchannel("B"), img.getchannel("A")
        # I = ImageChops.invert(B)
        # R = Image.new("L", I.size, 0)
        # inv = pygame.image.frombuffer(Image.merge("RGBA", (R, I, I, A)).tobytes(), I.size, "RGBA")
        def settings_toggle():
            sidebar.abspos ^= 1#= (sidebar.abspos + 1) % 3
        def settings_reset():
            sidebar.abspos = 1
            options.audio.update(audio_default)
            s = io.StringIO()
            for k, v in options.audio.items():
                s.write(f"~setting #{k} {v}\n")
            s.write(f"~replay\n")
            s.seek(0)
            mixer.stdin.write(s.read().encode("utf-8"))
            mixer.stdin.flush()
        sidebar.buttons.append(cdict(
            name="Settings",
            sprite=(gears, notes),
            click=(settings_toggle, settings_reset),
        ))
        reset_menu(full=False)
        folder = button_images.folder.result()
        def load_project():
            raise BaseException
        sidebar.buttons.append(cdict(
            name="Open",
            sprite=folder,
            click=enqueue_local,
            click2=load_project,
        ))
        reset_menu(full=False)
        hyperlink = button_images.hyperlink.result()
        plus = button_images.plus.result()
        def add_instrument():
            x = len(instruments)
            project.instruments.append(cdict(
                name=f"Instrument {x}",
                colour=tuple(round(i * 255) for i in colorsys.hsv_to_rgb(x / 12 % 1, 1, 1)),
                wave=cdict(synth_default),
            ))
        sidebar.buttons.append(cdict(
            name="Search",
            name2="New instrument",
            sprite=hyperlink,
            sprite2=plus,
            click=enqueue_search,
            click2=add_instrument,
        ))
        reset_menu(full=False)
        playlist = button_images.playlist.result()
        waves = button_images.waves.result()
        def get_playlist():
            items = [unquote(item[:-5]) for item in os.listdir("playlists") if item.endswith(".json")]
            if not items:
                return easygui.show_message("Right click this button to create, edit, or remove a playlist!", "Playlist folder is empty.")
            choice = easygui.get_choice("Select a locally saved playlist here!", "Miza Player", items)
            if choice:
                sidebar.loading = True
                start = len(queue)
                fn = "playlists/" + quote(choice)[:245] + ".json"
                if os.path.exists(fn) and os.path.getsize(fn):
                    with open(fn, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    q = data.get("queue", ())
                    options.history.appendleft((choice, tuple(e["url"] for e in q)))
                    options.history = options.history.uniq(sort=False)[:64]
                    queue.extend(ensure_duration(cdict(**e, pos=start)) for e in q)
                if control.shuffle and len(queue) > 1:
                    random.shuffle(queue.view[bool(start):])
                sidebar.loading = False
        def edit_playlist():
            method = easygui.buttonbox(
                "Select an operation to perform on a playlist!",
                "Miza Player",
                ("Create", "Edit", "Delete"),
            ).casefold()
            if method == "create":
                text = (easygui.textbox(
                    "Enter a list of URLs or file paths to include in the playlist!",
                    "Miza Player",
                ) or "").strip()
                if text:
                    urls = text.splitlines()
                    entries = deque()
                    for url in urls:
                        if url:
                            name = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
                            entries.append(dict(name=name, url=url))
                    if entries:
                        entries = list(entries)
                        ytdl = downloader.result()
                        url = entries[0]["url"]
                        if is_url(url):
                            name = None
                            if url in ytdl.searched:
                                resp = ytdl.searched[url].data
                                if len(resp) == 1:
                                    name = resp[0].get("name")
                            if not name:
                                resp = ytdl.downloader.extract_info(url, download=False, process=False)
                                name = resp.get("title") or entries[0].name
                            if name:
                                entries[0]["name"] = name
                        else:
                            name = entries[0].name
                        if len(entries) > 1:
                            name += f" +{len(entries) - 1}"
                        text = (easygui.get_string(
                            "Enter a name for your new playlist!",
                            "Miza Player",
                            name,
                        ) or "").strip()
                        if text:
                            with open("playlists/" + quote(text)[:245] + ".json", "w", encoding="utf-8") as f:
                                json.dump(dict(queue=entries, stats={}), f)
                            easygui.show_message(
                                f"Playlist {repr(text)} with {len(entries)} item{'s' if len(entries) != 1 else ''} has been added!",
                                "Success!",
                            )
            elif method == "edit":
                items = [unquote(item[:-5]) for item in os.listdir("playlists") if item.endswith(".json")]
                if not items:
                    return easygui.show_message("Right click this button to create, edit, or remove a playlist!", "Playlist folder is empty.")
                choice = easygui.get_choice("Select a playlist to edit", "Miza Player", items)
                if choice:
                    fn = "playlists/" + quote(choice)[:245] + ".json"
                    if os.path.exists(fn) and os.path.getsize(fn):
                        with open(fn, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        s = "\n".join(e["url"] for e in data.get("queue", ()) if e.get("url"))
                    else:
                        print(fn)
                        s = ""
                    text = easygui.textbox(
                        "Enter a list of URLs or file paths to include in the playlist!",
                        "Miza Player",
                        s,
                    )
                    if text is not None:
                        if not text:
                            os.remove(fn)
                            easygui.show_message(
                                f"Playlist {repr(choice)} has been removed!",
                                "Success!",
                            )
                        else:
                            urls = text.splitlines()
                            entries = deque()
                            for url in urls:
                                if url:
                                    name = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
                                    entries.append(dict(name=name, url=url))
                            if entries:
                                entries = list(entries)
                                with open(fn, "w", encoding="utf-8") as f:
                                    json.dump(dict(queue=entries, stats={}), f)
                                easygui.show_message(
                                    f"Playlist {repr(choice)} has been updated!",
                                    "Success!",
                                )
            elif method == "delete":
                items = [unquote(item[:-5]) for item in os.listdir("playlists") if item.endswith(".json")]
                if not items:
                    return easygui.show_message("Right click this button to create, edit, or remove a playlist!", "Playlist folder is empty.")
                choice = easygui.get_choice("Select a playlist to delete", "Miza Player", items)
                if choice:
                    fn = "playlists/" + quote(choice)[:245] + ".json"
                    os.remove(fn)
                    easygui.show_message(
                        f"Playlist {repr(choice)} has been removed!",
                        "Success!",
                    )
        def waves_1():
            raise BaseException
        sidebar.buttons.append(cdict(
            name="Playlist",
            name2="Audio Clip",
            sprite=playlist,
            sprite2=waves,
            click=(get_playlist, edit_playlist),
            click2=waves_1,
        ))
        reset_menu(full=False)
        history = button_images.history.result()
        def player_history():
            f = f"%0{len(str(len(options.history)))}d"
            choices = [f % i + ": " + e[0] for i, e in enumerate(options.history)]
            if not choices:
                return easygui.show_message("Play some music to fill up this menu!", "Player history is empty.")
            selected = easygui.get_choice(
                "Player History",
                "Miza Player",
                choices,
            )
            if selected:
                entry = options.history.pop(int(selected.split(":", 1)[0]))
                options.history.appendleft(entry)
                queue.extend(cdict(name=entry[0], url=url, duration=None, pos=len(queue)) for url in entry[1])
        def project_history():
            raise BaseException
        sidebar.buttons.append(cdict(
            name="History",
            sprite=history,
            click=player_history,
            click2=project_history,
        ))
        reset_menu(full=False)
        edit = button_images.edit.result()
        def edit_1():
            toolbar.editor ^= 1
            if toolbar.editor:
                mixer.submit(f"~setting spectrogram 0", force=True)
            else:
                mixer.submit(f"~setting spectrogram {options.spectrogram}", force=True)
        toolbar.buttons.append(cdict(
            name="Editor",
            image=edit,
            click=edit_1,
        ))
        reset_menu(full=False)
        repeat = button_images.repeat.result()
        def repeat_1():
            control.loop = (control.loop + 1) % 3
        def repeat_2():
            control.loop = (control.loop - 1) % 3
        toolbar.buttons.append(cdict(
            name="Repeat",
            image=repeat,
            click=(repeat_1, repeat_2),
        ))
        reset_menu(full=False)
        shuffle = button_images.shuffle.result()
        def shuffle_1():
            control.shuffle = (control.shuffle + 1) % 3
            if control.shuffle in (0, 2):
                mixer.submit(f"~setting shuffle {control.shuffle}", force=True)
            if control.shuffle == 2 and player.get("needs_shuffle") and player.end >= 960:
                seek_abs(player.pos)
        def shuffle_2():
            control.shuffle = (control.shuffle - 1) % 3
            if control.shuffle in (0, 2):
                mixer.submit(f"~setting shuffle {control.shuffle}", force=True)
            if control.shuffle == 2 and player.get("needs_shuffle"):
                seek_abs(player.pos)
        toolbar.buttons.append(cdict(
            name="Shuffle",
            image=shuffle,
            click=(shuffle_1, shuffle_2),
        ))
        reset_menu(full=False)
        back = button_images.back.result()
        def rleft():
            mixer.clear()
            queue.rotate(1)
            start()
        toolbar.buttons.append(cdict(
            name="Back",
            image=back,
            click=rleft,
        ))
        front = pygame.transform.flip(back, True, False)
        def rright():
            mixer.clear()
            queue.rotate(-1)
            start()
        toolbar.buttons.append(cdict(
            name="Next",
            image=front,
            click=rright,
        ))
        reset_menu(full=False)
        flip = button_images.flip.result()
        def flip_1():
            mixer.clear()
            queue.__init__(queue.view[::-1], fromarray=True)
            start()
        toolbar.buttons.append(cdict(
            name="Flip",
            image=flip,
            click=flip_1,
        ))
        reset_menu(full=False)
        scramble = button_images.scramble.result()
        def scramble_1():
            mixer.clear()
            random.shuffle(queue.view)
            start()
        toolbar.buttons.append(cdict(
            name="Scramble",
            image=scramble,
            click=scramble_1,
        ))
        reset_menu(full=False)
        microphone = button_images.microphone.result()
        globals()["pya"] = afut.result()
        sidebar.buttons.append(cdict(
            name="Audio input",
            sprite=microphone,
            click=enqueue_device,
        ))
        reset_menu(full=False)
    except:
        print_exc()

cached_fns = {}
def _enqueue_local(*files, probe=True):
    try:
        if files:
            sidebar.loading = True
            start = len(queue)
            title = None
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
                            if probe:
                                dur, cdc = cached_fns[fn] = get_duration_2(fn)
                            else:
                                dur, cdc = None, None
                    except:
                        print_exc()
                        dur, cdc = (None, "N/A")
                    entry = cdict(
                        url=fn,
                        stream=fn,
                        name=name,
                        duration=dur,
                        cdc=cdc,
                        pos=start,
                    )
                if not title:
                    title = entry.name
                    if len(files) > 1:
                        title += f" +{len(files) - 1}"
                    options.history.appendleft((title, files))
                    options.history = options.history.uniq(sort=False)[:64]
                queue.append(entry)
            if control.shuffle and len(queue) > 1:
                random.shuffle(queue.view[bool(start):])
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
            start = len(queue)
            ytdl = downloader.result()
            try:
                entries = ytdl.search(query)
            except:
                print_exc()
                sidebar.particles.append(cdict(eparticle))
            else:
                if entries:
                    entry = entries[0]
                    name = entry.name
                    if len(entries) > 1:
                        name += f" +{len(entries) - 1}"
                    url = query if is_url(query) and len(entries) > 1 else entry.url
                    options.history.appendleft((name, (url,)))
                    options.history = options.history.uniq(sort=False)[:64]
                    entries = [cdict(**e, pos=start) for e in entries]
                    queue.extend(entries)
                else:
                    sidebar.particles.append(cdict(eparticle))
            if control.shuffle and len(queue) > 1:
                random.shuffle(queue.view[bool(start):])
            sidebar.loading = False
    except:
        sidebar.loading = False
        print_exc()

def enqueue_search():
    query = easygui.get_string(
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
    f = f"%0{len(str(len(devices)))}d"
    selected = easygui.get_choice(
        "Transfer audio from a sound input device!",
        "Miza Player",
        sorted(f % d.id + ": " + d.name for d in devices),
    )
    if selected:
        submit(_enqueue_local, "<" + selected.split(":", 1)[0].lstrip("0") + ">")

def enqueue_auto(*queries):
    futs = deque()
    start = len(queue)
    for i, query in enumerate(queries):
        q = query.strip()
        if q:
            if is_url(q) or not os.path.exists(q):
                if i < 1:
                    futs.append(submit(_enqueue_search, q))
                else:
                    for fut in futs:
                        fut.result()
                    name = q.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
                    queue.append(cdict(name=name, url=q, duration=None, pos=start))
            else:
                futs.append(submit(_enqueue_local, q, probe=i < 1))


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
    sidebar.scrolling = False
    sidebar.scroll.setdefault("pos", 0)
    sidebar.scroll.setdefault("target", 0)
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
        if i:
            button.pos = (toolbar_height + 8 + (bsize + 4) * (i - 1), screensize[1] - toolbar_height // 3 - 8)
        else:
            button.pos = (screensize[0] - bsize - 4, screensize[1] - toolbar_height // 3 - 4)
        rect = button.pos + (bsize,) * 2
        sprite = button.get("sprite", button.image)
        isize = (bsize - 6,) * 2
        if sprite.get_size() != isize:
            sprite = pygame.transform.smoothscale(button.image, isize)
        button.sprite = sprite
        if 0 < i < 3:
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
shash = lambda s: as_str(base64.b64encode(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).replace(b"/", b"-").rstrip(b"="))
e_dur = lambda d: float(d) if type(d) is str else (d if d is not None else nan)

def ensure_duration(e):
    e.duration = e.get("duration")
    return e

lyrics_surf = None
lyrics_cache = {}
lyrics_renders = {}
def render_lyrics(entry):
    global lyrics_surf
    try:
        name = entry.name
        try:
            lyrics = lyrics_cache[name]
        except KeyError:
            try:
                lyrics = lyrics_scraper.result()(name)
            except LookupError:
                lyrics = None
            except:
                print_exc()
                lyrics = None
            if lyrics:
                lyrics_cache[name] = lyrics
            else:
                entry.lyrics = None
                return
        if options.spectrogram:
            return
        rect = (player.rect[2] - 8, player.rect[3] - 64)
        if entry.get("lyrics") and entry.lyrics[1].get_size() == rect:
            return entry.pop("lyrics_loading", None)
        try:
            render = lyrics_renders[name]
            if render[1].get_size() != rect:
                raise KeyError
        except KeyError:
            if not lyrics_surf or lyrics_surf.get_size() != rect:
                lyrics_surf = pygame.Surface(rect)
            render = [lyrics[0], lyrics_surf.convert()]
            mx = 0
            x = 0
            y = 0
            for para in lyrics[1].split("\n\n"):
                lines = para.splitlines()
                if mx and y + 42 > render[1].get_height():
                    y = 0
                    x += mx
                    mx = 0
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if mx and y + 28 > render[1].get_height():
                        y = 0
                        x += mx
                        mx = 0
                    col = (255,) * 3
                    if line[0] in "([" and line[-1] in "])":
                        if line[0] != "(":
                            name = line.split("(", 1)[0][1:]
                        else:
                            name = line[1:]
                        name = name.rstrip("]) ").casefold().strip()
                        if name.startswith("pre-") or name.startswith("pre "):
                            pre = True
                            name = name[4:]
                        elif name.startswith("post-") or name.startswith("post "):
                            pre = True
                            name = name[5:]
                        else:
                            pre = False
                        if name.startswith("intro") or name.startswith("outro"):
                            col = (255, 0, 0)
                        elif name.startswith("verse"):
                            col = (191, 127, 255)
                        elif name.startswith("chorus"):
                            col = (0, 255, 255)
                        elif name.startswith("bridge"):
                            col = (255, 127, 191)
                        elif name.startswith("breakdown"):
                            col = (255, 0, 255)
                        elif name.startswith("refrain"):
                            col = (255, 255, 0)
                        elif name.startswith("interlude"):
                            col = (0, 255, 0)
                        elif name.startswith("instrumental"):
                            col = (255, 127, 0)
                        elif line[0] == "[":
                            col = (0, 0, 255)
                        if pre:
                            col = tuple(i >> 1 for i in col)
                    s = text_size(line, 12)
                    if s[0] <= render[1].get_width() >> 1:
                        rect = message_display(
                            line,
                            12,
                            (x, y),
                            col,
                            align=0,
                            surface=render[1],
                        )
                        mx = max(mx, rect[2] + 8)
                        y += 14
                    else:
                        p = 0
                        words = line.split()
                        curr = ""
                        while words:
                            w = words.pop(0)
                            orig = curr
                            curr = curr + " " + w if curr else w
                            if orig:
                                s = text_size(curr, 12)
                                if s[0] > render[1].get_width() // 2 - p:
                                    rect = message_display(
                                        orig,
                                        12,
                                        (x + p, y),
                                        col,
                                        align=0,
                                        surface=render[1],
                                    )
                                    mx = max(mx, rect[2] + 8 + p)
                                    y += 14
                                    p = 8
                                    curr = w
                        if curr:
                            rect = message_display(
                                curr,
                                12,
                                (x + p, y),
                                col,
                                align=0,
                                surface=render[1],
                            )
                            mx = max(mx, rect[2] + 8 + p)
                            y += 14
                if y:
                    y += 7
        entry.lyrics = render
        entry.pop("lyrics_loading", None)
    except:
        print_exc()

def prepare(entry, force=False, download=False):
    if not entry.url:
        return
    stream = entry.get("stream", "")
    if not stream or stream.startswith("ytsearch:") or force and (stream.startswith("https://cf-hls-media.sndcdn.com/") or expired(stream)):
        ytdl = downloader.result()
        try:
            resp = ytdl.search(entry.url)
            data = resp[0]
            if force:
                stream = ytdl.get_stream(entry, force=True, download=False)
        except:
            entry.url = ""
            print_exc()
            return
        else:
            if entry.name != data.get("name"):
                entry.pop("lyrics", None)
                entry.pop("surf", None)
                entry.pop("lyrics_loading", None)
                if len(resp) == 1:
                    submit(render_lyrics, queue[0])
            entry.update(data)
            if len(resp) > 1:
                try:
                    i = queue.index(entry) + 1
                except:
                    i = len(queue)
                q1, q3 = queue.view[:i - 1], queue.view[i:]
                q2 = alist(cdict(**e, pos=i) for e in resp)
                if control.shuffle and len(q2) > 1:
                    random.shuffle(q2.view)
                queue.__init__(np.concatenate((q1, q2, q3)), fromarray=True)
                submit(render_lyrics, queue[0])
    elif force and is_url(entry.get("url")):
        ytdl = downloader.result()
        stream = ytdl.get_stream(entry, force=True, download=False)
    else:
        stream = entry.stream
    stream = stream.strip()
    if stream and is_url(stream) and download:
        mixer.submit(f"~download {stream} cache/~{shash(entry.url)}.pcm")
    return stream

def start_player(pos=None):
    player.last = 0
    player.amp = 0
    player.pop("osci", None)
    stream = prepare(queue[0], force=True)
    if not queue[0].url:
        return skip()
    stream = prepare(queue[0], force=True)
    entry = queue[0]
    if not entry.url:
        return skip()
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
    h = shash(entry.url)
    mixer.submit(stream + "\n" + str(pos) + " " + str(duration) + " " + str(entry.get("cdc", "auto")) + " " + h)
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
        e = queue.popleft()
        if control.shuffle:
            # for entry in queue[sidebar.base + 1:sidebar.base + sidebar.maxitems]:
            #     entry.pos = sidebar.base + sidebar.maxitems
            random.shuffle(queue.view[1:])
        if control.loop == 2:
            queue.appendleft(e)
        elif control.loop == 1:
            queue.append(e)
        else:
            sidebar.particles.append(e)
        if queue:
            return enqueue(queue[0])
        mixer.clear()
    player.last = 0
    player.pos = 0
    player.end = inf
    return None, inf

def seek_abs(pos):
    start_player(pos) if queue else (None, inf)

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

def ensure_next(i=1):
    if i <= 1 or all(e.done() for e in enext):
        enext.clear()
        if len(queue) > i:
            e = queue[i]
            e.duration = e.get("duration") or False
            e.pop("research", None)
            enext.add(submit(prepare, e, force=i == 1, download=i == 1))
            if i <= 1 and not e.get("lyrics_loading") and not e.get("lyrics"):
                e.lyrics_loading = True
                enext.add(submit(render_lyrics, e))

def enqueue(entry, start=True):
    try:
        queue[0].lyrics_loading = True
        submit(render_lyrics, queue[0])
        flash_window()
        stream, duration = start_player()
        progress.num = 0
        progress.alpha = 0
        return stream, duration
    except:
        print_exc()
    return None, inf

ffmpeg_start = ("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-err_detect", "ignore_err", "-hwaccel", "auto", "-vn")
concat_buff = b"\x00" * (48000 * 2 * 2)

def download(entries, fn):
    try:
        downloading.fn = fn
        downloading.target = sum(e.duration or 300 for e in entries)
        if len(entries) > 1:
            downloading.target += len(entries)
        cmd = ffmpeg_start + ("-f", "s16le", "-ar", "48k", "-ac", "2", "-i", "-")
        if fn.startswith(".pcm"):
            cmd += ("-f", "s16le")
        else:
            cmd += ("-b:a", "192k")
        cmd += ("-ar", "48k", "-ac", "2", fn)
        print(cmd)
        saving = psutil.Popen(cmd, stdin=subprocess.PIPE)
        ots = ts = time.time_ns() // 1000
        procs = deque()
        for entry in entries:
            if len(procs) > 2:
                while procs:
                    p = procs.popleft()
                    if len(entries) > 1:
                        saving.stdin.write(concat_buff)
                    p.wait()
                    if os.path.exists(p.fn):
                        if os.path.getsize(p.fn):
                            with open(p.fn, "rb") as sf:
                                while True:
                                    b = sf.read(1048576)
                                    if not b:
                                        break
                                    saving.stdin.write(b)
                        submit(os.remove(p.fn))
            st = prepare(entry, force=True)
            downloading.target = sum(e.duration or 300 for e in entries)
            if len(entries) > 1:
                downloading.target += len(entries)
            fn3 = f"cache/\x7f{ts}.pcm"
            cmd = ffmpeg_start + ("-nostdin", "-i", st)
            if len(entries) > 1:
                cmd += ("-af", "silenceremove=start_periods=1:start_duration=0.015625:start_threshold=-50dB:start_silence=0.015625:stop_periods=-9000:stop_threshold=-50dB:window=0.015625")
            cmd += ("-f", "s16le", "-ar", "48k", "-ac", "2", fn3)
            print(cmd)
            p = psutil.Popen(cmd)
            p.fn = fn3
            procs.append(p)
            ts += 1
        while procs:
            p = procs.popleft()
            saving.stdin.write(concat_buff)
            p.wait()
            if os.path.exists(p.fn):
                if os.path.getsize(p.fn):
                    with open(p.fn, "rb") as sf:
                        while True:
                            b = sf.read(1048576)
                            if not b:
                                break
                            saving.stdin.write(b)
                submit(os.remove(p.fn))
        saving.stdin.close()
        saving.wait()
        downloading.target = 0
    except:
        print_exc()


def update_menu():
    global sidebar_width, toolbar_height
    ts = toolbar.pause.setdefault("timestamp", 0)
    t = pc()
    player.lastframe = duration = max(0.001, min(t - ts, 0.125))
    player.flash_s = max(0, player.get("flash_s", 0) - duration * 60)
    player.flash_i = max(0, player.get("flash_i", 0) - duration * 60)
    player.flash_o = max(0, player.get("flash_o", 0) - duration * 60)
    toolbar.pause.timestamp = pc()
    ratio = 1 + 1 / (duration * 8)
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
    sidebar.scroll.target = max(0, min(sidebar.scroll.target, len(queue) * 32 - screensize[1] + options.toolbar_height + 36 + 32))
    r = ratio if sidebar.scrolling else (ratio - 1) / 3 + 1
    sidebar.scroll.pos = (sidebar.scroll.pos * (r - 1) + sidebar.scroll.target) / r
    sidebar.maxitems = ceil(screensize[1] - options.toolbar_height - 36 + sidebar.scroll.pos % 32) // 32
    sidebar.base = max(0, int(sidebar.scroll.pos // 32))
    # print(sidebar.scroll.target, sidebar.scroll.pos, sidebar.maxitems, sidebar.base)
    for i, entry in enumerate(queue[sidebar.base:sidebar.base + sidebar.maxitems], sidebar.base):
        entry.pos = (entry.get("pos", 0) * (ratio - 1) + i) / ratio
    if kspam[K_SPACE]:
        player.paused ^= True
        mixer.state(player.paused)
        toolbar.pause.speed = toolbar.pause.maxspeed
        if player.paused:
            c = (255, 0, 0)
        else:
            c = (0, 255, 0)
        toolbar.ripples.append(cdict(
            pos=toolbar.pause.pos,
            radius=0,
            colour=c,
            alpha=255,
        ))
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
        if any(mclick):
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
        elif mclick[1]:
            enter = easygui.get_string(
                "Seek to position",
                "Miza Player",
                time_disp(player.pos),
            )
            if enter:
                pos = time_parse(enter)
                submit(seek_abs, pos)
    if toolbar.resizing or in_rect(mpos, toolbar.rect):
        c = (64, 32, 96)
    else:
        c = (64, 0, 96)
    toolbar.updated = False#toolbar.colour != c
    toolbar.colour = c
    if any(mclick):
        for button in toolbar.buttons:
            if in_rect(mpos, button.rect):
                button.flash = 64
                if callable(button.click):
                    button.click()
                else:
                    button.click[min(mclick.index(1), len(button.click) - 1)]()
            if toolbar.editor:
                break
    else:
        for button in toolbar.buttons:
            if "flash" in button:
                button.flash = max(0, button.flash - duration * 64)
    maxb = (options.sidebar_width - 12) // 44
    if any(mclick):
        for button in sidebar.buttons[:maxb]:
            if in_rect(mpos, button.rect):
                button.flash = 32
                click = button.click if not toolbar.editor else button.get("click2") or button.click
                if callable(click):
                    click()
                else:
                    click[min(mclick.index(1), len(click) - 1)]()
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
    sidebar.updated = False#sidebar.colour != c
    sidebar.colour = c
    sidebar.relpos = (sidebar.get("relpos", 0) * (ratio - 1) + sidebar.abspos) / ratio
    scroll_height = screensize[1] - toolbar_height - 72
    sidebar.scroll.rect = (screensize[0] - 20, 52 + 16, 12, scroll_height)
    scroll_rat = max(12, min(scroll_height, scroll_height / max(1, len(queue)) * (screensize[1] - toolbar_height - 36) / 32))
    scroll_pos = sidebar.scroll.pos / (32 * max(1, len(queue)) - screensize[1] + toolbar_height + 52 + 16) * (scroll_height - scroll_rat) + 52 + 16
    sidebar.scroll.select_rect = (sidebar.scroll.rect[0], scroll_pos, sidebar.scroll.rect[2], scroll_rat)
    if sidebar.scrolling or in_rect(mpos, sidebar.scroll.rect):
        c1 = (191, 127, 255, 191)
        c2 = (255, 127, 191)
        if mclick[0]:
            sidebar.scrolling = True
        elif mclick[1]:
            enter = easygui.get_string(
                "Scroll to position",
                "Miza Player",
                str(round(sidebar.scroll.target / 32)),
            )
            if enter:
                pos = float(enter) * 32
                sidebar.scroll.target = pos
        if sidebar.scrolling:
            if not mheld[0]:
                sidebar.scrolling = False
            r = min(max(0, mpos2[1] - 52 - 16 - scroll_rat / 2) / max(1, scroll_height - scroll_rat), 1)
            sidebar.scroll.target = r * (32 * max(1, len(queue)) - screensize[1] + toolbar_height + 52 + 16)
    else:
        c1 = (191, 127, 255, 127)
        c2 = (255, 127, 191, 191)
    sidebar.scroll.background = c1
    sidebar.scroll.colour = c2

ripple_colours = (
    (191, 127, 255),
    (255, 127, 191),
    (0, 255, 255),
    (0, 0, 0),
    (255, 255, 255),
)


with open("misc/sidebar.py", "rb") as f:
    b = f.read()
exec(compile(b, "sidebar.py", "exec"))
with open("misc/sidebar2.py", "rb") as f:
    b = f.read()
exec(compile(b, "sidebar2.py", "exec"))


def draw_menu():
    global crosshair, hovertext
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
    if any(mclick) and in_rect(mpos, sidebar.rect):
        sidebar.ripples.append(cdict(
            pos=mpos,
            radius=0,
            colour=ripple_colours[mclick.index(1)],
            alpha=255,
        ))
    if (sidebar.updated or not tick & 7 or in_rect(mpos2, sidebar.rect) and (any(mclick) or any(kclick))) and sidebar.colour:
        if toolbar.editor:
            render_sidebar_2(dur)
        else:
            render_sidebar(dur)
        offs = round(sidebar.setdefault("relpos", 0) * -sidebar_width)
        Z = -sidebar.scroll.pos
        maxb = (sidebar_width - 12) // 44
        for i, button in enumerate(sidebar.buttons[:maxb]):
            if button.get("rect"):
                if in_rect(mpos, button.rect):
                    lum = 223
                    cm = abs(pc() % 1 - 0.5) * 0.328125
                    c = [round(i * 255) for i in colorsys.hls_to_rgb(cm + 0.75, 0.75, 1)]
                    hovertext = cdict(
                        text=button.name,
                        size=15,
                        colour=c,
                        offset=19,
                        font="Rockwell",
                    )
                    crosshair |= 4
                else:
                    lum = 191
                lum += button.get("flash", 0)
                if not i:
                    lum -= 32
                    lum += button.get("flash", 0)
                col = [round(i * 255) for i in colorsys.hls_to_rgb(0.75, lum / 255, 1)]
                bevel_rectangle(
                    DISP,
                    col,
                    button.rect,
                    4,
                )
                sprite = button.sprite if not toolbar.editor else button.get("sprite2") or button.sprite
                if type(sprite) in (tuple, list, alist):
                    sprite = sprite[bool(sidebar.abspos)]
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
                y = round(Z + 52 + 16 + entry.get("pos", 0) * 32)
                ext = round(32 - 32 * entry.life)
                rect = (screensize[0] - sidebar_width + 8 - ext + offs, y - ext * 3, sidebar_width - 16 - 16 + ext * 2, 32 + ext * 2)
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
            if i and toolbar.editor:
                break
            if button.get("rect"):
                if in_rect(mpos, button.rect):
                    lum = 191
                    cm = abs(pc() % 1 - 0.5) * 0.328125
                    c = [round(i * 255) for i in colorsys.hls_to_rgb(cm + 0.75, 0.75, 1)]
                    hovertext = cdict(
                        text=button.name,
                        size=15,
                        colour=c,
                        font="Rockwell",
                        offset=-19,
                    )
                    crosshair |= 4
                else:
                    lum = 127
                lum += button.get("flash", 0)
                col = [round(i * 255) for i in colorsys.hls_to_rgb(0.5 + bool(i) / 4, lum / 255, 0.25)]
                rounded_bev_rect(
                    DISP,
                    col,
                    button.rect,
                    3,
                )
                if i == 2:
                    val = control.shuffle
                elif i == 1:
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
                        cache=True,
                        font="Comic Sans MS",
                    )
        if downloading.target:
            pgr = os.path.exists(downloading.fn) and os.path.getsize(downloading.fn) / 192000 * 8
            ratio = min(1, pgr / max(1, downloading.target))
            percentage = round(ratio * 100, 3)
            message_display(
                f"Downloading: {percentage}%",
                16,
                (screensize[0] / 2, screensize[1] - 16),
                colour=[round(i * 255) for i in colorsys.hsv_to_rgb(ratio / 3, 1, 1)],
                surface=DISP,
                cache=True,
                font="Rockwell",
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
        bsize = min(40, toolbar_height // 3)
        s = f"{time_disp(player.pos)}/{time_disp(player.end)}"
        message_display(
            s,
            min(24, toolbar_height // 3),
            (screensize[0] - 8 - bsize, screensize[1]),
            surface=DISP,
            align=2,
            font="Comic Sans MS",
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
        for i, p in sorted(enumerate(progress.particles), key=lambda t: t[1].life):
            ri = max(1, round(p.life ** 1.2 / 7 ** 0.2))
            col = [round(i * 255) for i in colorsys.hsv_to_rgb(*p.hsv)]
            a = round(min(255, (p.life - 2) * 20))
            point = [cos(p.angle) * p.rad, sin(p.angle) * p.rad]
            pos = (p.centre[0] + point[0], p.centre[1] + point[1])
            if ri > 2:
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
            else:
                pygame.draw.circle(
                    DISP,
                    col + [a],
                    pos,
                    ri,
                )
            p.life -= dur * 2.5
            if p.life <= 6:
                p.angle += dur
                p.rad = max(0, p.rad - 12 * dur)
                # p.hsv[0] = (p.hsv[0] + dur / 24) % 1
                # p.hsv[1] = max(p.hsv[1] - dur / 6, 0)
                p.hsv[2] = max(p.hsv[2] - dur / 5, 0)
            if p.life < 3:
                pops.add(i)
        progress.particles.pops(pops)
        d = abs(pc() % 2 - 1)
        hsv = [0.5 + d / 4, 1 - 0.75 + abs(d - 0.75), 1]
        col = [round(i * 255) for i in colorsys.hsv_to_rgb(*hsv)]
        for i in shuffle(range(3)):
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
                font="Comic Sans MS",
            )
    if (mclick[0] or not tick & 7) and sidebar.get("dragging"):
        if toolbar.editor:
            render_dragging_2()
        else:
            render_dragging()
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
            if not options.spectrogram and queue:
                submit(render_lyrics, queue[0])
        elif in_rect(mpos, osci_rect) and not toolbar.resizer:
            player.flash_o = 32
            options.oscilloscope = (options.get("oscilloscope", 0) + 1) % 2
            mixer.submit(f"~setting oscilloscope {options.oscilloscope}", force=True)
    if crosshair & 1 and (not tick & 7 or toolbar.rect in modified) or crosshair & 2 and (not tick + 4 & 7 or sidebar.rect in modified) or crosshair & 4:
        if crosshair & 3:
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
                font="Comic Sans MS",
            )
        if hovertext:
            message_display(
                hovertext.text,
                hovertext.size,
                (mpos2[0], mpos2[1] + hovertext.get("offset", -17)),
                hovertext.colour,
                surface=DISP,
                font=hovertext.get("font", "Comic Sans MS"),
            )


for i in range(26):
    globals()[f"K_{chr(i + 97)}"] = i + 4
for i in range(1, 11):
    globals()[f"K_{i % 10}"] = i + 29
code = ""
K_SPACE = 44
K_EQUALS = 46
K_LEFTBRACKET = 47
K_RIGHTBRACKET = 48
K_SEMICOLON = 51
K_BACKQUOTE = 53
K_COMMA = 54
K_PERIOD = 55
K_SLASH = 56
K_DELETE = 76
reset_menu(True)
enext = set()
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
            unfocused = False
        else:
            minimised = is_minimised()
            unfocused = is_unfocused()
        if not tick & 15:
            if not downloading.target:
                if player.paused:
                    colour = 4
                elif is_active() and player.amp > 1 / 64:
                    colour = 2
                else:
                    colour = 8
                submit(taskbar_progress_bar, player.pos / player.end, colour | (player.end >= inf))
            else:
                pgr = os.path.exists(downloading.fn) and os.path.getsize(downloading.fn) / 192000 * 8
                ratio = min(1, pgr / max(1, downloading.target))
                submit(taskbar_progress_bar, ratio, 4 if ratio < 1 / 3 else 8 if ratio < 2 / 3 else 2)
            if minimised:
                toolbar.ripples.clear()
                sidebar.ripples.clear()
        if not player.get("fut"):
            if queue:
                player.fut = submit(start)
        elif not queue:
            player.pop("fut").result()
        if not minimised and (not unfocused or not tick % 48):
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
            krelease = [not x and y for x, y in zip(kheld, kprev)]
            kspam = kclick
            # if any(kclick):
            #     print(" ".join(map(str, (i for i, v in enumerate(kclick) if v))))
            if kclick[K_BACKQUOTE]:
                output = easygui.textbox(
                    "Debug mode",
                    "Miza Player",
                    code or "",
                )
                while output:
                    orig = output
                    try:
                        c = None
                        try:
                            c = compile(output, "<debug>", "eval")
                        except SyntaxError:
                            pass
                        if not c:
                            c = compile(output, "<debug>", "exec")
                        output = str(eval(c))
                    except:
                        output = traceback.format_exc()
                    output = output.strip()
                    code = output
                    output = easygui.textbox(
                        orig,
                        "Miza Player",
                        code or "",
                    )
            if not tick & 15:
                kspam = KeyList(x or y >= 240 for x, y in zip(kclick, kheld))
            if any(kclick) or any(krelease):
                alphakeys = [0] * 32
                if not kheld[K_LCTRL] and not kheld[K_LSHIFT] and not kheld[K_RCTRL] and not kheld[K_RSHIFT]:
                    notekeys = "zsxdcvgbhnjmq2w3er5t6y7ui9o0p"
                    alphakeys = [kheld[globals()[f"K_{c}"]] for c in notekeys] + [0] * 3
                    alphakeys[-3] = kheld[K_LEFTBRACKET]
                    alphakeys[-2] = kheld[K_EQUALS]
                    alphakeys[-1] = kheld[K_RIGHTBRACKET]
                    alphakeys[12] |= kheld[K_COMMA]
                    alphakeys[13] |= kheld[K_l]
                    alphakeys[14] |= kheld[K_PERIOD]
                    alphakeys[15] |= kheld[K_SEMICOLON]
                    alphakeys[16] |= kheld[K_SLASH]
                mixer.submit("~keys " + repr(alphakeys), force=True)
            if not tick & 3 or mpos != lpos or (mpos2 != lpos and any(mheld)) or any(mclick) or any(kclick) or any(mrelease) or any(isnan(x) != isnan(y) for x, y in zip(mpos, lpos)):
                try:
                    update_menu()
                except:
                    print_exc()
                draw_menu()
            if not queue and not is_active() and not any(kheld):
                player.pos = 0
                player.end = inf
                player.last = 0
                progress.num = 0
                progress.alpha = 0
                # player.pop("spec", None)
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
                if queue and not options.spectrogram:
                    size = max(16, min(32, (screensize[0] - sidebar_width) // 36))
                    if "lyrics" not in queue[0]:
                        if pc() % 0.25 < 0.125:
                            col = (255,) * 3
                        else:
                            col = (255, 0, 0)
                        message_display(
                            f"Loading lyrics for {queue[0].name}...",
                            size,
                            (player.rect[2] >> 1, size),
                            col,
                            surface=DISP,
                            cache=True,
                            background=(0,) * 3,
                            font="Rockwell",
                        )
                    elif queue[0].lyrics:
                        rect = (player.rect[2] - 8, player.rect[3] - 64)
                        if not queue[0].get("lyrics_loading") and rect != queue[0].lyrics[1].get_size():
                            queue[0].lyrics_loading = True
                            submit(render_lyrics, queue[0])
                        DISP.blit(
                            queue[0].lyrics[1],
                            (8, 64),
                        )
                        message_display(
                            queue[0].lyrics[0],
                            size,
                            (player.rect[2] >> 1, size),
                            (255,) * 3,
                            surface=DISP,
                            cache=True,
                            background=(0,) * 3,
                            font="Rockwell",
                        )
                    else:
                        teapot_source = teapot_fut.result()
                        teapot_size = limit_size(*teapot_source.get_size(), *player.rect[2:])
                        teapot = globals().get("teapot")
                        if not teapot or teapot.get_size() != teapot_size:
                            teapot = globals()["teapot"] = pygame.transform.scale(teapot_source, teapot_size)
                        DISP.blit(
                            teapot,
                            (player.rect[2] - teapot.get_width() >> 1, player.rect[3] - teapot.get_height() >> 1),
                        )
                        message_display(
                            f"No lyrics found for {queue[0].name}.",
                            size,
                            (player.rect[2] >> 1, size),
                            (255, 0, 0),
                            surface=DISP,
                            cache=True,
                            background=(0,) * 3,
                            font="Rockwell",
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
                            font="Comic Sans MS",
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
                if not tick + 6 & 7 and (tuple(screensize) in modified or player.rect in modified) and (not options.spectrogram or not player.get("spec")):
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
            elif event.type == MOUSEWHEEL:
                sidebar.scroll.target -= event.y * 8
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
    options.history = list(options.history)
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
    if os.path.exists("misc/temp.tmp"):
        futs.add(submit(os.remove, "misc/temp.tmp"))
    for fut in futs:
        try:
            fut.result()
        except:
            pass
    if type(ex) is not StopIteration:
        easygui.exceptionbox()
    PROC.kill()