def render_sidebar(dur=0):
    global crosshair, hovertext, lq2
    modified.add(sidebar.rect)
    offs = round(sidebar.setdefault("relpos", 0) * -sidebar_width)
    sc = sidebar.colour or (64, 0, 96)
    if sidebar.ripples or offs > -sidebar_width + 4:
        DISP2 = pygame.Surface((sidebar.rect[2], sidebar.rect[3] + 4), SRCALPHA)
        bevel_rectangle(
            DISP2,
            sc,
            (0, 0) + sidebar.rect2[2:],
            4,
        )
        ripple_f = globals().get("h-ripple", concentric_circle)
        for ripple in sidebar.ripples:
            ripple_f(
                DISP2,
                colour=ripple.colour,
                pos=(ripple.pos[0] - screensize[0] + sidebar_width, ripple.pos[1]),
                radius=ripple.radius,
                fill_ratio=1 / 3,
                alpha=sqrt(max(0, ripple.alpha)) * 16,
            )
        if offs > -sidebar_width + 4:
            n = len(queue)
            t = sum(e.get("duration") or 300 for e in queue) - (player.pos or 0)
            message_display(
                f"{n} item{'s' if n != 1 else ''}, estimated time remaining: {time_disp(t)}",
                12,
                (6 + offs, 48),
                surface=DISP2,
                align=0,
                font="Comic Sans MS",
            )
        if queue and sidebar.scroll.get("colour"):
            rounded_bev_rect(
                DISP2,
                sidebar.scroll.background,
                (sidebar.scroll.rect[0] + offs - screensize[0] + sidebar_width, sidebar.scroll.rect[1]) + sidebar.scroll.rect[2:],
                4,
            )
            rounded_bev_rect(
                DISP2,
                sidebar.scroll.colour,
                (sidebar.scroll.select_rect[0] + offs - screensize[0] + sidebar_width, sidebar.scroll.select_rect[1]) + sidebar.scroll.select_rect[2:],
                4,
            )
        DISP.blit(
            DISP2,
            sidebar.rect[:2],
        )
    else:
        bevel_rectangle(
            DISP,
            sc,
            sidebar.rect2,
            4,
        )
    if offs > 4 - sidebar_width:
        if queue and (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_s]:
            name = None
            entries = deque()
            for entry in queue:
                if entry.get("selected"):
                    if not name:
                        name = entry.name
                    entries.append(entry)
            if not entries:
                name = queue[0].name
                entries = queue[:1]
            if len(entries) > 1:
                name += f" +{len(entries) - 1}"
            fn = easygui.filesavebox(
                "Save As",
                "Miza Player",
                name + ".ogg",
                filetypes=ftypes,
            )
            if fn:
                submit(download, entries, fn)
        Z = -sidebar.scroll.pos
        DISP2 = pygame.Surface((sidebar.rect2[2], sidebar.rect2[3] - 52 - 16), SRCALPHA)
        DISP2.fill((0, 0, 0, 0))
        # DISP2.set_colorkey(sc)
        if (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_v]:
            submit(enqueue_auto, *pyperclip.paste().splitlines())
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
        base, maxitems = sidebar.base, sidebar.maxitems
        otarget = round((mpos[1] - Z - 52 - 16 - 16) / 32)
        etarget = otarget if in_rect(mpos, (screensize[0] - sidebar_width + 8, 52 + 16, sidebar_width - 32, screensize[1] - toolbar_height - 52 - 16)) else nan
        target = min(max(0, round((mpos2[1] - Z - 52 - 16 - 16) / 32)), len(queue) - 1)
        if mclick[0] and not sidebar.scrolling and in_rect(mpos, sidebar.rect) and not in_rect(mpos, sidebar.scroll.rect) and not kheld[K_LSHIFT] and not kheld[K_RSHIFT] and not kheld[K_LCTRL] and not kheld[K_RCTRL]:
            if etarget not in range(len(queue)) or not queue[etarget].get("selected"):
                for entry in queue:
                    entry.pop("selected", None)
                sidebar.pop("last_selected", None)
                lq = nan
        if len(queue) > 1:
            entry = queue[1]
            if (entry.duration is None or entry.get("research")):
                ensure_next(1)
        for i, entry in enumerate(queue[base:base + maxitems], base):
            if i > 1 and (entry.duration is None or entry.get("research")) and (queue[i - 1].duration or i == base):
                ensure_next(i)
            elif i == 1 and not entry.get("lyrics") and not entry.get("lyrics_loading"):
                ensure_next(i)
            if entry.get("selected") and sidebar.get("dragging"):
                x = 4 + offs
                y = round(Z + entry.get("pos", 0) * 32)
                rect = (x, y, sidebar_width - 32, 32)
                sat = 0.875
                val = 1
                secondary = True
                if pc() % 0.25 < 0.125:
                    entry.colour = col = [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
                else:
                    col = (255,) * 3
                rounded_bev_rect(
                    DISP2,
                    col,
                    rect,
                    4,
                    alpha=round(255 / (1 + abs(entry.get("pos", 0) - i) / 4)),
                    filled=False,
                )
                if not swap and not mclick[0] and not kheld[K_LSHIFT] and not kheld[K_RSHIFT] and not kheld[K_LCTRL] and not kheld[K_RCTRL] and sidebar.get("last_selected") is entry:
                    if target != i:
                        swap = target - i
        for i, entry in enumerate(queue):
            if not entry.url:
                pops.add(i)
                continue
            if entry.get("selected"):
                if kclick[K_DELETE] or kclick[K_BACKSPACE] or (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_x]:
                    pops.add(i)
                    if sidebar.get("last_selected") == entry:
                        sidebar.pop("last_selected", None)
                if (kheld[K_LCTRL] or kheld[K_RCTRL]) and (kclick[K_c] or kclick[K_x]):
                    entry.flash = 16
                    copies.append(entry.url)
            elif (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_a]:
                entry.selected = True
                sidebar.last_selected = entry
                lq = i
            if i < base or i >= base + maxitems:
                entry.flash = 8
                entry.pos = i
                continue
            if not isfinite(lq):
                lq2 = nan
            x = 4 + offs
            y = round(Z + entry.get("pos", 0) * 32)
            rect = (x, y, sidebar_width - 32, 32)
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
                            sidebar.selection_offset = np.array(mpos2) - rect[:2]
                elif mclick[1] and i == etarget:
                    if not entry.get("selected"):
                        for e in queue:
                            if e == entry:
                                e.selected = True
                            else:
                                e.pop("selected", None)
                    sidebar.last_selected = entry

                    def play_now():
                        if queue:
                            a = deque()
                            b = deque()
                            for e in queue:
                                if e == sidebar.get("last_selected"):
                                    a.appendleft(e)
                                elif e.get("selected"):
                                    a.append(e)
                                else:
                                    b.append(e)
                            a.extend(b)
                            queue[:] = a
                            mixer.clear()
                            submit(start)

                    def play_next():
                        if len(queue) > 1:
                            s = queue.popleft()
                            a = deque()
                            b = deque()
                            for e in queue:
                                if e == sidebar.get("last_selected"):
                                    a.appendleft(e)
                                elif e.get("selected"):
                                    a.append(e)
                                else:
                                    b.append(e)
                            a.extend(b)
                            queue[:] = a
                            queue.appendleft(s)

                    def add_to_playlist():
                        entries = list(dict(name=e.name, url=e.url) for e in queue if e.get("selected"))
                        if not entries:
                            entries = (entry,)
                        url = entries[0]["url"]
                        if is_url(url):
                            ytdl = downloader.result()
                            name = None
                            if url in ytdl.searched:
                                resp = ytdl.searched[url].data
                                if len(resp) == 1:
                                    name = resp[0].get("name")
                            if not name:
                                resp = ytdl.downloader.extract_info(url, download=False, process=False)
                                name = resp.get("title") or entries[0].name
                        else:
                            name = entries[0]["name"]
                        if len(entries) > 1:
                            name += f" +{len(entries) - 1}"
                        playlists = os.listdir("playlists")
                        text = (easygui.get_string(
                            "Enter a name for your new playlist!",
                            "Miza Player",
                            name,
                        ) or "").strip()
                        if text:
                            with open("playlists/" + quote(text)[:245] + ".json", "w", encoding="utf-8") as f:
                                json.dump(dict(queue=entries, stats={}), f)
                            easygui.show_message(
                                f"Success! Playlist {repr(text)} with {len(entries)} item{'s' if len(entries) != 1 else ''} has been added!",
                            )
                    
                    def save_as():
                        name = None
                        entries = deque()
                        for entry in queue:
                            if entry.get("selected"):
                                if not name:
                                    name = entry.name
                                entries.append(entry)
                        if not entries:
                            name = queue[0].name
                            entries = queue[:1]
                        if len(entries) > 1:
                            name += f" +{len(entries) - 1}"
                        fn = easygui.filesavebox(
                            "Save As",
                            "Miza Player",
                            name + ".ogg",
                            filetypes=ftypes,
                        )
                        if fn:
                            submit(download, entries, fn)
                    
                    def delete():
                        if queue:
                            pops = set()
                            for j, e in enumerate(queue):
                                if e.get("selected"):
                                    pops.add(j)
                            queue.pops(pops)
                            if 0 in pops:
                                mixer.clear()
                                submit(start)

                    sidebar.menu = cdict(
                        buttons=(
                            ("Play now", play_now),
                            ("Play next", play_next),
                            ("Add to playlist", add_to_playlist),
                            ("Save as", save_as),
                            ("Delete", delete),
                        ),
                    )
                if entry.get("selected"):
                    flash = entry.get("flash", 16)
                    if flash >= 0:
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
            rounded_bev_rect(
                DISP2,
                col,
                rect,
                4,
                alpha=255 if secondary else round(255 / (1 + abs(entry.get("pos", 0) - i) / 4)),
                filled=not secondary,
            )
            if secondary:
                sat = 0.875
                val = 0.5
                if flash:
                    sat = max(0, sat - flash / 16)
                    val = min(1, val + flash / 16)
                bevel_rectangle(
                    DISP2,
                    [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)],
                    [rect[0] + 4, rect[1] + 4, rect[2] - 8, rect[3] - 8],
                    0,
                    alpha=191,
                )
            if not entry.get("surf"):
                entry.surf = message_display(
                    entry.name[:128],
                    12,
                    (0,) * 2,
                    align=0,
                    cache=True,
                )
            DISP2.blit(
                entry.surf,
                (x + 6, y + 4),
                (0, 0, sidebar_width - 48, 24),
            )
            message_display(
                time_disp(entry.duration) if entry.duration else "N/A",
                10,
                [x + sidebar_width - 36, y + 28],
                (255,) * 3,
                surface=DISP2,
                align=2,
                cache=True,
                font="Comic Sans MS",
            )
        if copies:
            pyperclip.copy("\n".join(copies))
        if pops:
            r = range(base, base + maxitems + 1)
            sidebar.particles.extend(queue[i] for i in pops if i in r)
            skipping = 0 in pops
            queue.pops(pops)
            if skipping:
                mixer.clear()
                submit(start)
        if not sidebar.get("dragging"):
            for i, entry in enumerate(queue[base:base + maxitems], base):
                if not entry.get("selected"):
                    continue
                x = 4 + offs
                y = round(Z + entry.get("pos", 0) * 32)
                sat = 0.875
                val = 1
                rect = (x, y, sidebar_width - 32, 32)
                entry.colour = col = [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
                sat = 0.875
                val = 0.5
                flash = entry.get("flash", 16)
                if flash:
                    sat = max(0, sat - flash / 16)
                    val = min(1, val + flash / 16)
                bevel_rectangle(
                    DISP2,
                    [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)],
                    [rect[0] + 4, rect[1] + 4, rect[2] - 8, rect[3] - 8],
                    0,
                    alpha=191,
                )
                rounded_bev_rect(
                    DISP2,
                    col,
                    rect,
                    4,
                    alpha=255,
                    filled=False,
                )
                if not entry.get("surf"):
                    entry.surf = message_display(
                        entry.name[:128],
                        12,
                        (0,) * 2,
                        align=0,
                        cache=True,
                    )
                DISP2.blit(
                    entry.surf,
                    (x + 6, y + 4),
                    (0, 0, sidebar_width - 48, 24),
                )
                message_display(
                    time_disp(entry.duration) if entry.duration else "N/A",
                    10,
                    [x + sidebar_width - 36, y + 28],
                    (255,) * 3,
                    surface=DISP2,
                    align=2,
                    cache=True,
                    font="Comic Sans MS",
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
            x = 4 + offs
            y = round(Z + (len(queue) - base) * 32)
            rect = (x, y, sidebar_width - 32, 32)
            rounded_bev_rect(
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
                    cache=True,
                )
            DISP2.blit(
                sidebar.loading_text,
                (x + 6, y + 4),
                (0, 0, sidebar_width - 48, 24),
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
                submit(enqueue, queue[0])
            try:
                if not sidebar.last_selected.selected:
                    raise ValueError
                lq2 = queue.index(sidebar.last_selected)
            except (AttributeError, ValueError, IndexError):
                sidebar.pop("last_selected", None)
                lq2 = nan
        DISP.blit(
            DISP2,
            (screensize[0] - sidebar_width + 4, 52 + 16),
        )
    if offs <= -4:
        DISP2 = pygame.Surface((sidebar.rect2[2], sidebar.rect2[3] - 52))
        DISP2.fill(sc)
        DISP2.set_colorkey(sc)
        in_sidebar = in_rect(mpos, sidebar.rect)
        offs2 = offs + sidebar_width
        for i, opt in enumerate(asettings):
            message_display(
                opt.capitalize(),
                11,
                (offs2 + 8, i * 32),
                surface=DISP2,
                align=0,
                cache=True,
                font="Comic Sans MS",
            )
            # numrect = (screensize[0] + offs + sidebar_width - 8, 68 + i * 32)
            s = str(round(options.audio.get(opt, 0) * 100, 2)) + "%"
            message_display(
                s,
                11,
                (offs2 + sidebar_width - 8, 16 + i * 32),
                surface=DISP2,
                align=2,
                cache=True,
                font="Comic Sans MS",
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
            brect2 = (offs2 + 6, 17 + i * 32, sidebar_width - 12, 13)
            hovered = in_sidebar and in_rect(mpos, brect) or aediting[opt]
            crosshair |= bool(hovered) << 1
            v = max(0, min(1, (mpos2[0] - (screensize[0] + offs + 8)) / (sidebar_width - 16))) * (srange[1] - srange[0]) + srange[0]
            if len(srange) > 2:
                v = round_min(math.round(v / srange[2]) * srange[2])
            else:
                rv = round_min(math.round(v * 32) / 32)
                if type(rv) is int and rv not in srange:
                    v = rv
            if hovered and not hovertext:
                hovertext = cdict(
                    text=str(round(v * 100, 2)) + "%",
                    size=10,
                    colour=(255, 255, 127),
                )
                if aediting[opt]:
                    if not mheld[0]:
                        aediting[opt] = False
                elif mclick[0]:
                    aediting[opt] = True
                elif mclick[1]:
                    enter = easygui.get_string(
                        opt.capitalize(),
                        "Miza Player",
                        str(round_min(options.audio[opt] * 100)),
                    )
                    if enter:
                        v = round_min(float(eval(enter, {}, {})) / 100)
                        aediting[opt] = True
                if aediting[opt]:
                    orig, options.audio[opt] = options.audio[opt], v
                    if orig != v:
                        mixer.submit(f"~setting {opt} {v}", force=opt == "volume" or not queue)
            z = max(0, x - 4)
            rect = (offs2 + 8 + z, 17 + i * 32, sidebar_width - 16 - z, 9)
            col = (48 if hovered else 32,) * 3
            bevel_rectangle(
                DISP2,
                col,
                rect,
                3,
            )
            rainbow = quadratic_gradient((w, 9), pc() / 2 + i / 4)
            DISP2.blit(
                rainbow,
                (offs2 + 8 + x, 17 + i * 32),
                (x, 0, w - x, 9),
                special_flags=BLEND_RGB_MULT,
            )
            rect = (offs2 + 8, 17 + i * 32, x, 9)
            col = (223 if hovered else 191,) * 3
            bevel_rectangle(
                DISP2,
                col,
                rect,
                3,
            )
            rainbow = quadratic_gradient((w, 9), pc() + i / 4)
            DISP2.blit(
                rainbow,
                (offs2 + 8, 17 + i * 32),
                (0, 0, x, 9),
                special_flags=BLEND_RGB_MULT,
            )
            if hovered:
                bevel_rectangle(
                    DISP2,
                    progress.select_colour,
                    brect2,
                    2,
                    filled=False,
                )
        DISP.blit(
            DISP2,
            (screensize[0] - sidebar_width, 52),
        )

def render_dragging():
    base, maxitems = sidebar.base, sidebar.maxitems
    for i, entry in enumerate(queue[base:base + maxitems], base):
        if not entry.get("selected"):
            continue
        sat = 0.875
        val = 1
        entry.colour = col = [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
        x, y = mpos2 - sidebar.selection_offset
        y += 52 + 16
        x += screensize[0] - sidebar_width + 4
        if isfinite(lq2):
            y += (i - lq2) * 32
        rect = (x, y, sidebar_width - 32, 32)
        sat = 0.875
        val = 0.5
        flash = entry.get("flash", 16)
        if flash:
            sat = max(0, sat - flash / 16)
            val = min(1, val + flash / 16)
        bevel_rectangle(
            DISP,
            [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)],
            [rect[0] + 4, rect[1] + 4, rect[2] - 8, rect[3] - 8],
            0,
            alpha=191,
        )
        rounded_bev_rect(
            DISP,
            col,
            rect,
            4,
            alpha=255,
            filled=False,
        )
        if not entry.get("surf"):
            entry.surf = message_display(
                entry.name[:128],
                12,
                (0,) * 2,
                align=0,
                cache=True,
            )
        DISP.blit(
            entry.surf,
            (x + 6, y + 4),
            (0, 0, sidebar_width - 48, 24),
        )
        message_display(
            time_disp(entry.duration) if entry.duration else "N/A",
            10,
            [x + sidebar_width - 36, y + 28],
            (255,) * 3,
            surface=DISP,
            align=2,
            cache=True,
            font="Comic Sans MS",
        )
        anima_rectangle(
            DISP,
            [round(x * 255) for x in colorsys.hsv_to_rgb(i / 12 + 1 / 12, 0.9375, 1)],
            [rect[0] + 1, rect[1] + 1, rect[2] - 2, rect[3] - 2],
            frame=4,
            count=2,
            flash=1,
            ratio=pc() * 0.4,
            reduction=0.1,
        )