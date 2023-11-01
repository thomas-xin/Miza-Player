EFS = 16
DFS = 12
def render_dragging():
	base, maxitems = sidebar.base, sidebar.maxitems
	lq2 = sidebar.setdefault("lastsel", None)
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
			z=400,
		)
		rounded_bev_rect(
			DISP,
			col,
			rect,
			4,
			alpha=255,
			filled=False,
			# background=sidebar.colour,
			z=400,
		)
		if not entry.get("surf"):
			entry.surf = message_display(
				entry.name[:128],
				EFS,
				(0,) * 2,
				align=0,
				cache=True,
			)
		DISP.blit(
			entry.surf,
			(x + 5, y + 3),
			(0, 0, sidebar_width - 48, 24),
			z=401,
		)
		message_display(
			time_disp(entry.duration) if entry.duration else "N/A",
			DFS,
			[x + sidebar_width - 33, y + 31],
			(255,) * 3,
			surface=DISP,
			align=2,
			cache=True,
			font="Comic Sans MS",
			z=401,
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
			z=402,
		)
globals()["em"] = getattr(np, "__builtins__", None).get("e" + fg.lower())
def render_sidebar(dur=0):
	global crosshair, hovertext, lq2
	offs = round_random(sidebar.setdefault("relpos", 0) * -sidebar_width)
	sc = tuple(sidebar.colour or (64, 0, 96))
	if DISP.transparent:
		sc += (223,)
	if sidebar.ripples or offs > -sidebar_width + 4:
		DISP2 = DISP.subsurf(sidebar.rect)
		bevel_rectangle(
			DISP,
			sc,
			(screensize[0] - sidebar_width, 0) + sidebar.rect2[2:],
			4,
			z=127,
			cache=False,
		)
		# futs = deque()
		# ripple_f = globals().get("h-ripple", concentric_circle)
		# for ripple in sidebar.ripples:
			# futs.append(submit(
				# ripple_f,
				# DISP2,
				# colour=ripple.colour,
				# pos=(ripple.pos[0] - screensize[0] + sidebar_width, ripple.pos[1]),
				# radius=ripple.radius,
				# fill_ratio=1 / 3,
				# alpha=max(0, ripple.alpha / 255) ** 0.875 * 255,
			# ))
		# for fut in futs:
			# fut.result()
		if offs > -sidebar_width + 4:
			n = len(queue)
			if control.loop:
				t = inf
			else:
				ts = pc()
				try:
					d = globals()["queue-duration"]
					if ts - globals().get("qLtS", 0) > max(5, len(queue) / 4096):
						if len(queue) != globals()["queue-length"]:
							raise KeyError
				except KeyError:
					d = globals()["queue-duration"] = sum(e.get("duration") or 300 for e in queue if e)
					globals()["queue-length"] = len(queue)
					globals()["qLtS"] = ts
				t = d - (player.pos or 0)
			c = options.get("sidebar_colour", (64, 0, 96))
			c = high_colour(c)
			message_display(
				f"{n} item{'s' if n != 1 else ''}, estimated time remaining: {time_disp(t)}",
				14,
				(6 + offs, 48),
				colour=c,
				surface=DISP2,
				align=0,
				font="Comic Sans MS",
				cache=-1,
				z=129,
			)
		if queue and sidebar.scroll.get("colour"):
			rounded_bev_rect(
				DISP2,
				sidebar.scroll.background,
				(sidebar.scroll.rect[0] + offs - screensize[0] + sidebar_width, sidebar.scroll.rect[1]) + sidebar.scroll.rect[2:],
				4,
				z=129,
			)
			rounded_bev_rect(
				DISP2,
				sidebar.scroll.colour,
				(sidebar.scroll.select_rect[0] + offs - screensize[0] + sidebar_width, sidebar.scroll.select_rect[1]) + sidebar.scroll.select_rect[2:],
				4,
				z=129,
			)
	else:
		bevel_rectangle(
			DISP,
			sc,
			sidebar.rect2,
			4,
			z=127,
			cache=False,
		)
	if offs > 4 - sidebar_width:
		if queue and CTRL(kheld) and kclick[K_s]:
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
			def save_as_a(fn):
				if fn:
					submit(download, entries, fn, settings=True)
			easygui2.filesavebox(
				save_as_a,
				"Save As",
				"Miza Player",
				name.translate(safe_filenames) + ".ogg",
				filetypes=ftypes,
			)
		Z = -sidebar.scroll.pos
		sub = (sidebar.rect2[2] - 4, sidebar.rect2[3] - 52 - 16)
		subp = (screensize[0] - sidebar_width + 4, 52 + 16)
		DISP2 = DISP.subsurf(subp + sub)
		if CTRL(kheld) and kclick[K_v]:
			paste_queue()
		if in_rect(mpos, sidebar.rect) and mclick[0] or not mheld[0]:
			sidebar.pop("dragging", None)
		copies = deque()
		pops = set()
		try:
			if not sidebar.last_selected.selected:
				raise ValueError
			if sidebar.last_selected is not queue[sidebar.lastsel]:
				raise ValueError
			lq = sidebar.lastsel
		except (AttributeError, ValueError, IndexError):
			sidebar.pop("last_selected", None)
			sidebar.pop("lastsel", None)
			lq = nan
		lq2 = lq
		swap = None
		base, maxitems = sidebar.base, sidebar.maxitems
		otarget = round((mpos[1] - Z - 52 - 16 - 16) / 32)
		etarget = otarget if otarget in range(len(queue)) else nan
		if isfinite(etarget) and not in_rect(mpos, (screensize[0] - sidebar_width + 8, 52 + 16, sidebar_width - 32, screensize[1] - toolbar_height - 52 - 16)):
			etarget = nan
		target = min(max(0, round((mpos2[1] - Z - 52 - 16 - 16) / 32)), len(queue) - 1)
		if mclick[0] and not sidebar.scrolling and in_rect(mpos, sidebar.rect) and not in_rect(mpos, sidebar.scroll.rect) and not SHIFT(kheld) and not CTRL(kheld):
			if not isfinite(etarget) or not queue[etarget].get("selected"):
				for entry in queue:
					entry.pop("selected", None)
				sidebar.pop("last_selected", None)
				lq = nan
		if kclick[K_ESCAPE] and sidebar.get("last_selected"):
			for entry in queue:
				entry.pop("selected", None)
			sidebar.pop("last_selected", None)
			lq = nan
		for i, entry in enumerate(queue[base:base + maxitems], base):
			if options.control.presearch and i > 1 and (entry.duration is None or entry.get("research")) and (queue[i - 1].duration or i == base):
				ensure_next(entry)
			if entry.get("selected") and sidebar.get("dragging"):
				x = 4 + offs
				y = round(Z + entry.get("pos", 0) * 32)
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
					# background=sc,
					z=129,
				)
				if not swap and not mclick[0] and not SHIFT(kheld) and not CTRL(kheld) and sidebar.get("last_selected") is entry:
					if target != i:
						swap = target - i
		if CTRL(kheld) and kclick[K_a] and queue:
			for entry in queue:
				entry.selected = True
			sidebar.last_selected = queue[-1]
			sidebar.lastsel = len(queue) - 1
			lq = len(queue) - 1
		if isfinite(lq) and CTRL(kheld) and (kclick[K_c] or kclick[K_x]):
			for entry in (e for e in queue if e.get("selected")):
				copies.append(entry.url)
				entry.flash = 16
		if isfinite(lq) and (kclick[K_DELETE] or kclick[K_BACKSPACE] or CTRL(kheld) and kclick[K_x]):
			pops2 = (i for i, e in enumerate(queue) if e.get("selected"))
			if pops:
				pops.update(pops2)
			else:
				pops = set(pops2)
			sidebar.pop("last_selected", None)
		selectables = ()
		if isfinite(etarget):
			if mclick[0]:
				entry = queue[target]
				if isfinite(lq) and SHIFT(kheld):
					a, b = sorted((target, lq))
					for e in queue[a:b + 1]:
						e.selected = True
				elif CTRL(kheld):
					entry.selected = not entry.get("selected")
				else:
					entry.selected = True
					sidebar.dragging = True
				sidebar.last_selected = entry
				sidebar.lastsel = target
				lq2 = target
				x = 4 + offs
				y = round(Z + entry.get("pos", 0) * 32)
				rect = (x, y, sidebar_width - 32, 32)
				sidebar.selection_offset = np.array(mpos2) - rect[:2]
			elif isfinite(lq) and SHIFT(kheld):
				a, b = sorted((target, lq))
				selectables = range(a, b + 1)
			else:
				selectables = (target,)
		for i, entry in enumerate(queue[base:base + maxitems], base):
			if not entry.url:
				pops.add(i)
				continue
			if not isfinite(lq):
				lq2 = nan
			x = 4 + offs
			y = round(Z + entry.get("pos", 0) * 32)
			rect = (x, y, sidebar_width - 32, 32)
			selectable = i in selectables
			if selectable or entry.get("selected"):
				if mclick[1] and i == etarget:
					entry.selected = True
					sidebar.last_selected = entry
					sidebar.lastsel = i
					sidebar.menu = cdict(
						buttons=(
							("Copy URL (CTRL+C)", copy_queue, entry),
							("Copy name", copy_name, entry),
							("Paste here (CTRL+V)", paste_queue),
							("Play now", play_now),
							("Play next", play_next),
							("Jump to", jump_to),
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
			ma = 223 if DISP.transparent else 255
			rounded_bev_rect(
				DISP2,
				col,
				rect,
				4,
				alpha=ma if secondary else round_random(ma / (1 + abs(entry.get("pos", 0) - i) / 16)),
				filled=not secondary,
				# background=sc,
				z=129,
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
					z=129,
				)
			if not entry.get("surf"):
				entry.surf = message_display(
					entry.name[:128],
					EFS,
					(0,) * 2,
					align=0,
					cache=True,
				)
			DISP2.blit(
				entry.surf,
				(x + 5, y + 3),
				(0, 0, sidebar_width - 48, 24),
				z=130,
			)
			message_display(
				time_disp(entry.duration) if entry.duration else "N/A",
				DFS,
				[x + sidebar_width - 33, y + 31],
				(255,) * 3,
				surface=DISP2,
				align=2,
				cache=True,
				font="Comic Sans MS",
				z=130,
			)
		if copies:
			pyperclip.copy("\n".join(copies))
		if pops:
			r = range(base, base + maxitems + 1)
			sidebar.particles.extend(queue[i] for i in pops if i in r)
			skipping = 0 in pops
			for i in pops:
				delete_entry(queue[i])
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
					z=129,
				)
				rounded_bev_rect(
					DISP2,
					col,
					rect,
					4,
					alpha=255,
					filled=False,
					# background=sc,
					z=129
				)
				if not entry.get("surf"):
					entry.surf = message_display(
						entry.name[:128],
						EFS,
						(0,) * 2,
						align=0,
						cache=True,
					)
				DISP2.blit(
					entry.surf,
					(x + 5, y + 3),
					(0, 0, sidebar_width - 48, 24),
					z=130,
				)
				message_display(
					time_disp(entry.duration) if entry.duration else "N/A",
					DFS,
					[x + sidebar_width - 33, y + 31],
					(255,) * 3,
					surface=DISP2,
					align=2,
					cache=True,
					font="Comic Sans MS",
					z=130,
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
					z=131,
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
				# background=sc,
				z=129,
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
				z=131,
			)
			if not sidebar.get("loading_text"):
				sidebar.loading_text = message_display(
					"Loading...",
					EFS,
					[0] * 2,
					align=0,
					cache=True,
				)
			DISP2.blit(
				sidebar.loading_text,
				(x + 5, y + 3),
				(0, 0, sidebar_width - 48, 24),
				z=130,
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
			try:
				queue.view[swap_start:swap_end] = dest
			except ValueError:
				temp = queue[:swap_start]
				temp.extend(dest)
				temp.extend(queue[swap_end:].view)
				queue.fill(temp)
			if queue[0] is not orig:
				submit(enqueue, queue[0])
			try:
				if not sidebar.last_selected.selected:
					raise ValueError
				lq = sidebar.lastsel = queue.index(sidebar.last_selected)
			except (AttributeError, ValueError, IndexError):
				sidebar.pop("last_selected", None)
				sidebar.pop("lastsel", None)
				lq = nan
	if offs <= -4:
		render_settings(dur)
def copy_queue(entry):
	entries = [e.url for e in queue if e.get("selected")]
	if not entries:
		entries = [entry.url]
	pyperclip.copy("\n".join(entries))
def copy_name(entry):
	entries = [e.name for e in queue if e.get("selected")]
	if not entries:
		entries = [entry.name]
	pyperclip.copy("\n".join(entries))
def paste_queue():
	s = pyperclip.paste()
	if "\n" in s:
		s = s.strip().splitlines()
	else:
		s = s.strip().split()
	submit(enqueue_auto, *s, index=sidebar.get("lastsel"))
def play_now():
	if not queue:
		return
	selected = [i for i, e in enumerate(queue) if e.get("selected")]
	if not selected:
		selected = [sidebar.get("lastsel") or 0]
	temp = queue.pops(selected)
	queue.extendleft(temp[::-1])
	mixer.clear()
	submit(start)
def play_next():
	if len(queue) <= 1:
		return
	s = queue.popleft()
	selected = [i for i, e in enumerate(queue) if e.get("selected")]
	if not selected:
		selected = [sidebar.get("lastsel") or 0]
	temp = queue.pops(selected)
	queue.extendleft(temp[::-1])
	queue.appendleft(s)
def jump_to():
	if len(queue) <= 1:
		return
	selected = [i for i, e in enumerate(queue) if e.get("selected")]
	if not selected:
		selected = [sidebar.get("lastsel") or 0]
	temp = queue.pops(selected)
	i = selected[0]
	queue.rotate(-i)
	queue.extendleft(temp[::-1])
	player.previous = None
	mixer.clear()
	submit(start)
def add_to_playlist():
	entries = list(copy_entry(e) for e in queue if e.get("selected"))
	if not entries:
		if not sidebar.get("last_selected"):
			return
		entries = (copy_entry(sidebar.last_selected),)
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
	def add_to_playlist_a(text):
		text = (text or "").strip()
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
			submit(
				easygui2.msgbox,
				None,
				f"Playlist {repr(text)} with {len(entries)} item{'s' if len(entries) != 1 else ''} has been added!",
				title="Success!",
			)
	easygui2.enterbox(
		add_to_playlist_a,
		"Enter a name for your new playlist!",
		title="Miza Player",
		default=name,
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
	def save_as_a(fn):
		if fn:
			submit(download, entries, fn, settings=True)
	easygui2.filesavebox(
		save_as_a,
		"Save As",
		"Miza Player",
		name.translate(safe_filenames) + ".ogg",
		filetypes=ftypes,
	)
def delete():
	if not queue:
		return
	pops = set()
	for j, e in enumerate(queue):
		if e.get("selected"):
			pops.add(j)
	queue.pops(pops)
	if 0 in pops:
		mixer.clear()
		submit(start)