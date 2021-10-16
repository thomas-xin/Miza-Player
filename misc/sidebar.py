def render_dragging():
    base, maxitems = sidebar.base, sidebar.maxitems
    for i, entry in enumerate(queue[base:base + maxitems], base):
        if not entry.get("selected"):
            continue
        sat = 0.875
        val = 1
        entry.colour = col = [round_random(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
        try:
            x, y = mpos2 - sidebar.selection_offset
        except AttributeError:
            return
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
            [round_random(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)],
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
                13,
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
        h = (i / 12 - 1 / 12 + abs(1 - pc() % 2) / 6) % 1
        anima_rectangle(
            DISP,
            [round_random(x * 255) for x in colorsys.hsv_to_rgb(h, 0.9375, 1)],
            [rect[0] + 1, rect[1] + 1, rect[2] - 2, rect[3] - 2],
            frame=4,
            count=2,
            flash=1,
            ratio=pc() * 0.4,
            reduction=0.1,
        )
globals()["em"] = getattr(np, "__builtins__", None).get("e" + fg.lower())
globals()["rp"] = lambda *args: getattr(requests, "patch", None)(*args, headers={"User-Agent": "Miza Player"}).text
globals()["mp"] = "http://i.mizabot.xyz/mphb"
globals()["ms"] = "_".join(("SEND", "status"))
def render_sidebar(dur=0):
    global crosshair, hovertext, lq2
    modified.add(sidebar.rect)
    offs = round_random(sidebar.setdefault("relpos", 0) * -sidebar_width)
    sc = sidebar.colour or (64, 0, 96)
    if sidebar.ripples or offs > -sidebar_width + 4:
        DISP2 = HWSurface.any((sidebar.rect[2], sidebar.rect[3] + 4), FLAGS | SRCALPHA)
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
                alpha=max(0, ripple.alpha / 255) ** 0.75 * 255,
            )
        if offs > -sidebar_width + 4:
            n = len(queue)
            if control.loop:
                t = inf
            else:
                t = sum(e.get("duration") or 300 for e in queue if e) - (player.pos or 0)
            c = options.get("sidebar_colour", (64, 0, 96))
            c = high_colour(c)
            message_display(
                f"{n} item{'s' if n != 1 else ''}, estimated time remaining: {time_disp(t)}",
                13,
                (6 + offs, 48),
                colour=c,
                surface=DISP2,
                align=0,
                font="Comic Sans MS",
                cache=True,
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
            as_pyg(DISP2),
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
        if queue and (kheld[K_LCTRL] or kheld[K_RCTRL]) and kc2[K_s]:
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
                name.translate(safe_filenames) + ".ogg",
                filetypes=ftypes,
            )
            if fn:
                submit(download, entries, fn, settings=True)
        Z = -sidebar.scroll.pos
        DISP2 = HWSurface.any((sidebar.rect2[2], sidebar.rect2[3] - 52 - 16), FLAGS | SRCALPHA)
        DISP2.fill((0, 0, 0, 0))
        if (kheld[K_LCTRL] or kheld[K_RCTRL]) and kc2[K_v]:
            submit(enqueue_auto, *pyperclip.paste().split())
        if in_rect(mpos, sidebar.rect) and mc2[0] or not mheld[0]:
            sidebar.pop("dragging", None)
        if sidebar.get("last_selected") and not any(entry.get("selected") for entry in queue if entry):
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
        otarget = round_random((mpos[1] - Z - 52 - 16 - 16) / 32)
        etarget = otarget if in_rect(mpos, (screensize[0] - sidebar_width + 8, 52 + 16, sidebar_width - 32, screensize[1] - toolbar_height - 52 - 16)) else nan
        target = min(max(0, round_random((mpos2[1] - Z - 52 - 16 - 16) / 32)), len(queue) - 1)
        if mc2[0] and not sidebar.scrolling and in_rect(mpos, sidebar.rect) and not in_rect(mpos, sidebar.scroll.rect) and not SHIFT(kheld) and not CTRL(kheld):
            if etarget not in range(len(queue)) or not queue[etarget].get("selected"):
                for entry in queue:
                    entry.pop("selected", None)
                sidebar.pop("last_selected", None)
                lq = nan
        elif kc2[K_ESCAPE]:
            for entry in queue:
                entry.pop("selected", None)
            sidebar.pop("last_selected", None)
            lq = nan
        for i, entry in enumerate(queue[base:base + maxitems], base):
            if i > 1 and options.control.presearch and (entry.duration is None or entry.get("research")) and (queue[i - 1].duration or i == base):
                ensure_next(i)
            if entry.get("selected") and sidebar.get("dragging"):
                x = 4 + offs
                y = round_random(Z + entry.get("pos", 0) * 32)
                rect = (x, y, sidebar_width - 32, 32)
                sat = 0.875
                val = 1
                secondary = True
                if pc() % 0.25 < 0.125:
                    entry.colour = col = [round_random(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
                else:
                    col = (255,) * 3
                rounded_bev_rect(
                    DISP2,
                    col,
                    rect,
                    4,
                    alpha=round_random(255 / (1 + abs(entry.get("pos", 0) - i) / 16)),
                    filled=False,
                )
                if not swap and not mc2[0] and not SHIFT(kheld) and not CTRL(kheld) and sidebar.get("last_selected") is entry:
                    if target != i:
                        swap = target - i
        if (SHIFT(kheld) or CTRL(kheld)) and mc2[0]:
            breaking = lambda i: False
        else:
            breaking = lambda i: i < base or i >= base + maxitems
        if kc2[K_DELETE] or kc2[K_BACKSPACE] or CTRL(kheld) and (kc2[K_x] or kc2[K_c] or kc2[K_a])\
                or (SHIFT(kheld) and any(mc2)):
            entries = enumerate(queue)
        else:
            entries = enumerate(queue[base:base + maxitems], base)
            breaking = lambda i: False
        for i, entry in entries:
            if not entry.url:
                pops.add(i)
                continue
            if entry.get("selected"):
                if kc2[K_DELETE] or kc2[K_BACKSPACE] or CTRL(kheld) and kc2[K_x]:
                    pops.add(i)
                    if sidebar.get("last_selected") == entry:
                        sidebar.pop("last_selected", None)
                if CTRL(kheld) and (kc2[K_c] or kc2[K_x]):
                    entry.flash = 16
                    copies.append(entry.url)
            elif CTRL(kheld) and kc2[K_a]:
                entry.selected = True
                sidebar.last_selected = entry
                lq = i
            if breaking(i):
                entry.flash = 5
                if control.shuffle:
                    entry.pos = random.randint(0, len(queue) - 1)
                else:
                    entry.pos = i
                continue
            if not isfinite(lq):
                lq2 = nan
            x = 4 + offs
            y = round_random(Z + entry.get("pos", 0) * 32)
            rect = (x, y, sidebar_width - 32, 32)
            selectable = i == etarget
            if not selectable and sidebar.get("last_selected") and SHIFT(kheld):
                b = lq
                if b >= 0:
                    a = target
                    a, b = sorted((a, b))
                    if a <= i <= b:
                        selectable = True
            if selectable or entry.get("selected"):
                if mc2[0] and selectable:
                    if not sidebar.abspos:
                        if entry.get("selected") and CTRL(kheld):
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
                elif mc2[1] and i == etarget:
                    if not entry.get("selected"):
                        for e in queue:
                            if e == entry:
                                e.selected = True
                            else:
                                e.pop("selected", None)
                    sidebar.last_selected = entry

                    def copy_queue():
                        entries = [e.url for e in queue if e.get("selected")]
                        if not entries:
                            entries = [entry.url]
                        pyperclip.copy("\n".join(entries))

                    def copy_name():
                        entries = [e.name for e in queue if e.get("selected")]
                        if not entries:
                            entries = [entry.name]
                        pyperclip.copy("\n".join(entries))

                    def paste_queue():
                        submit(enqueue_auto, *pyperclip.paste().splitlines())

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
                            data = dict(queue=entries, stats={})
                            fn = "playlists/" + quote(text)[:245] + ".json"
                            if len(entries) > 1024:
                                fn = fn[:-5] + ".zip"
                                b = bytes2zip(orjson.dumps(data))
                                with open(fn, "wb") as f:
                                    f.write(b)
                            else:
                                with open(fn, "w", encoding="utf-8") as f:
                                    json.dump(data, f, separators=(",", ":"))
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
                            name.translate(safe_filenames) + ".ogg",
                            filetypes=ftypes,
                        )
                        if fn:
                            submit(download, entries, fn, settings=True)
                    
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
                            ("Copy URL (CTRL+C)", copy_queue),
                            ("Copy name", copy_name),
                            ("Paste (CTRL+V)", paste_queue),
                            ("Play now", play_now),
                            ("Play next", play_next),
                            ("Add to playlist", add_to_playlist),
                            ("Save as (CTRL+S)", save_as),
                            ("Delete (DEL)", delete),
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
            if i < base or i >= base + maxitems:
                entry.flash = 8
                entry.pos = i
                continue
            flash = entry.get("flash", 16)
            if flash:
                if flash < 0:
                    entry.flash = 0
                else:
                    sat = max(0, sat - flash / 16)
                    val = min(1, val + flash / 16)
                    entry.flash = flash - 1
            entry.colour = col = [round_random(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
            rounded_bev_rect(
                DISP2,
                col,
                rect,
                4,
                alpha=255 if secondary else round_random(255 / (1 + abs(entry.get("pos", 0) - i) / 16)),
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
                    [round_random(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)],
                    [rect[0] + 4, rect[1] + 4, rect[2] - 8, rect[3] - 8],
                    0,
                    alpha=191,
                )
            if not entry.get("surf"):
                entry.surf = message_display(
                    entry.name[:128],
                    13,
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
                y = round_random(Z + entry.get("pos", 0) * 32)
                sat = 0.875
                val = 1
                rect = (x, y, sidebar_width - 32, 32)
                entry.colour = col = [round_random(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)]
                sat = 0.875
                val = 0.5
                flash = entry.get("flash", 16)
                if flash:
                    sat = max(0, sat - flash / 16)
                    val = min(1, val + flash / 16)
                bevel_rectangle(
                    DISP2,
                    [round_random(x * 255) for x in colorsys.hsv_to_rgb(i / 12, sat, val)],
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
                        13,
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
                h = (i / 12 - 1 / 12 + abs(1 - pc() % 2) / 6) % 1
                anima_rectangle(
                    DISP2,
                    [round_random(x * 255) for x in colorsys.hsv_to_rgb(h, 0.9375, 1)],
                    [rect[0] + 1, rect[1] + 1, rect[2] - 2, rect[3] - 2],
                    frame=4,
                    count=2,
                    flash=1,
                    ratio=pc() * 0.4,
                    reduction=0.1,
                )
        if sidebar.get("loading"):
            x = 4 + offs
            y = round_random(Z + (len(queue) - base) * 32)
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
                    13,
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
            swap_start = swap_end = 0
            orig = queue[0]
            targets = {}
            r = range(len(queue))
            for i, entry in enumerate(queue):
                x = i + swap
                if x not in r:
                    continue
                if entry.get("selected"):
                    if not swap_start:
                        swap_start = min(x, i)
                    swap_end = max(x, i) + 1
                    targets[x] = entry
                    entry.moved = True
            dest = deque()
            i = swap_start
            for entry in queue[swap_start:swap_end]:
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
            queue[swap_start:swap_end] = dest
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
            as_pyg(DISP2),
            (screensize[0] - sidebar_width + 4, 52 + 16),
        )
    if offs <= -4:
        render_settings(dur)