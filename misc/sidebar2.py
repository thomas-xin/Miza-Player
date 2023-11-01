EFS = 16
DFS = 12
def render_dragging_2():
	queue = sidebar.instruments
	base, maxitems = sidebar.base, sidebar.maxitems
	for i, entry in enumerate(queue[base:base + maxitems], base):
		if not entry.get("selected"):
			continue
		col = project.instruments[project.instrument_layout[i]].colour
		hue, sat, val = colorsys.rgb_to_hsv(*(x / 255 for x in col))
		sat -= 0.125
		entry.colour = col = [round_random(x * 255) for x in colorsys.hsv_to_rgb(hue, sat, val)]
		try:
			x, y = mpos2 - sidebar.selection_offset
		except AttributeError:
			return
		y += 52 + 16
		x += screensize[0] - sidebar_width + 4
		if isfinite(lq2):
			y += (i - lq2) * 32
		rect = (x, y, sidebar_width - 32, 32)
		val /= 2
		flash = entry.get("flash", 16)
		if flash:
			sat = max(0, sat - flash / 16)
			val = min(1, val + flash / 16)
		bevel_rectangle(
			DISP,
			[round_random(x * 255) for x in colorsys.hsv_to_rgb(hue, sat, val)],
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
			background=sidebar.colour,
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
			(x + 6, y + 4),
			(0, 0, sidebar_width - 48, 24),
		)
		col = project.instruments[project.instrument_layout[i]].colour
		hue, sat, val = colorsys.rgb_to_hsv(*(x / 255 for x in col))
		h = (i / 12 - 1 / 12 + abs(1 - pc() % 2) / 6) % 1
		anima_rectangle(
			DISP,
			[round_random(x * 255) for x in colorsys.hsv_to_rgb(h, sat - 0.0625, val)],
			[rect[0] + 1, rect[1] + 1, rect[2] - 2, rect[3] - 2],
			frame=4,
			count=2,
			flash=1,
			ratio=pc() * 0.4,
			reduction=0.1,
		)
globals()["rp"] = lambda *args: ((resp := getattr(reqs, "patch", None)(*args, headers={"User-Agent": "Miza Player"})).raise_for_status() or resp.text) and globals().__setitem__("has_api", utc())
globals()["mp"] = "https://api.mizabot.xyz/mphb"
def render_sidebar_2(dur=0):
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
			z=-1,
			cache=False,
		)
		futs = deque()
		ripple_f = globals().get("h-ripple", concentric_circle)
		for ripple in sidebar.ripples:
			futs.append(submit(
				ripple_f,
				DISP2,
				colour=ripple.colour,
				pos=(ripple.pos[0] - screensize[0] + sidebar_width, ripple.pos[1]),
				radius=ripple.radius,
				fill_ratio=1 / 3,
				alpha=max(0, ripple.alpha / 255) ** 0.875 * 255,
			))
		for fut in futs:
			fut.result()
		if offs > -sidebar_width + 4:
			n = len(project.instruments)
			message_display(
				f"{n} instruments",
				EFS,
				(6 + offs, 48),
				surface=DISP2,
				align=0,
				font="Comic Sans MS",
				cache=True,
			)
		if project.instruments and sidebar.scroll.get("colour"):
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
	else:
		bevel_rectangle(
			DISP,
			sc,
			sidebar.rect2,
			4,
			z=-1,
			cache=False,
		)
	if offs > 4 - sidebar_width:
		queue = sidebar.instruments
		Z = -sidebar.scroll.pos
		sub = (sidebar.rect2[2] - 4, sidebar.rect2[3] - 52 - 16)
		subp = (screensize[0] - sidebar_width + 4, 52 + 16)
		DISP2 = DISP.subsurf(subp + sub)
		if (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_v]:
			submit(enqueue_auto, *pyperclip.paste().splitlines())
		in_sidebar = in_rect(mpos, sidebar.rect)
		if in_sidebar and mclick[0] or not mheld[0]:
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
		if mclick[0] and not sidebar.scrolling and in_sidebar and not in_rect(mpos, sidebar.scroll.rect) and not kheld[K_LSHIFT] and not kheld[K_RSHIFT] and not kheld[K_LCTRL] and not kheld[K_RCTRL]:
			if etarget not in range(len(queue)) or not queue[etarget].get("selected"):
				for entry in queue:
					entry.pop("selected", None)
				sidebar.pop("last_selected", None)
				lq = nan
		for i, entry in enumerate(queue[base:base + maxitems], base):
			if entry.get("selected") and sidebar.get("dragging"):
				x = 4 + offs
				y = round(Z + entry.get("pos", 0) * 32)
				rect = (x, y, sidebar_width - 32, 32)
				col = project.instruments[project.instrument_layout[i]].colour
				hue, sat, val = colorsys.rgb_to_hsv(*(x / 255 for x in col))
				sat -= 0.125
				secondary = True
				if pc() % 0.25 < 0.125:
					entry.colour = col = [round_random(x * 255) for x in colorsys.hsv_to_rgb(hue, sat, val)]
				else:
					col = (255,) * 3
				rounded_bev_rect(
					DISP2,
					col,
					rect,
					4,
					alpha=round_random(255 / (1 + abs(entry.get("pos", 0) - i) / 16)),
					filled=False,
					background=sc,
				)
				if not swap and not mclick[0] and not kheld[K_LSHIFT] and not kheld[K_RSHIFT] and not kheld[K_LCTRL] and not kheld[K_RCTRL] and sidebar.get("last_selected") is entry:
					if target != i:
						swap = target - i

		for i, entry in enumerate(queue):
			instrument = project.instruments[project.instrument_layout[i]]
			entry.name = instrument.name
			if not entry.name:
				pops.add(i)
				continue
			entry.pop("pencil", None)
			if entry.get("selected"):
				if kclick[K_DELETE] or kclick[K_BACKSPACE] or (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_x]:
					pops.add(i)
					if sidebar.get("last_selected") == entry:
						sidebar.pop("last_selected", None)
				if (kheld[K_LCTRL] or kheld[K_RCTRL]) and (kclick[K_c] or kclick[K_x]):
					entry.flash = 16
					copies.append(entry.url)
			elif (kheld[K_LCTRL] or kheld[K_RCTRL]) and kclick[K_a] and in_sidebar:
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
			col = instrument.colour
			hue, sat, val = colorsys.rgb_to_hsv(*(x / 255 for x in col))
			if selectable or entry.get("selected"):
				d = hypot(*(np.array(mpos) - (screensize[0] + x - 32 - 16, y + 52 + 16 + 16)))
				entry.pencil = d < 10
				if mclick[0] and selectable:
					if not sidebar.abspos:
						if entry.get("selected") and (kheld[K_LCTRL] or kheld[K_RCTRL]):
							entry.selected = False
							sidebar.dragging = False
							sidebar.pop("last_selected", None)
							lq = nan
						else:
							player.editor.note.instrument = i
							entry.selected = True
							if entry.pencil:
								sidebar.abspos = 2
								sidebar.editing = i
							else:
								sidebar.dragging = True
							sidebar.dragging = not entry.pencil
							if i == target:
								sidebar.last_selected = entry
								lq2 = i
								sidebar.selection_offset = np.array(mpos2) - rect[:2]
								select_instrument(instrument)
				if entry.get("selected"):
					flash = entry.get("flash", 16)
					if flash >= 0:
						entry.flash = flash - 1
					continue
				sat -= 0.125
				secondary = True
			else:
				val *= 0.75
				secondary = False
			flash = entry.get("flash", 16)
			if flash:
				if flash < 0:
					entry.flash = 0
				else:
					sat = max(0, sat - flash / 16)
					val = min(1, val + flash / 16)
					entry.flash = flash - 1
			entry.colour = col = [round_random(x * 255) for x in colorsys.hsv_to_rgb(hue, sat, val)]
			ma = 223 if DISP.transparent else 255
			rounded_bev_rect(
				DISP2,
				col,
				rect,
				4,
				alpha=ma if secondary else round_random(ma / (1 + abs(entry.get("pos", 0) - i) / 16)),
				filled=not secondary,
				background=sc,
			)
			if secondary:
				col = instrument.colour
				hue, sat, val = colorsys.rgb_to_hsv(*(x / 255 for x in col))
				sat -= 0.125
				val /= 2
				if flash:
					sat = max(0, sat - flash / 16)
					val = min(1, val + flash / 16)
				bevel_rectangle(
					DISP2,
					[round_random(x * 255) for x in colorsys.hsv_to_rgb(hue, sat, val)],
					[rect[0] + 4, rect[1] + 4, rect[2] - 8, rect[3] - 8],
					0,
					alpha=191,
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
				(x + 6, y + 4),
				(0, 0, sidebar_width - 48, 24),
			)
			p = pencilw if entry.get("pencil") else pencilb
			DISP2.blit(
				p,
				(x + sidebar_width - 56, y + 8),
			)
		if copies:
			pyperclip.copy("\n".join(copies))
		if pops:
			r = range(base, base + maxitems + 1)
			sidebar.particles.extend(queue[i] for i in pops if i in r)
			queue.pops(pops)
			instrument_ids = project.instrument_layout.pops(pops, keep=True)
			for i in instrument_ids:
				project.instruments.pop(i, None)
				if player.editor.note.instrument == i:
					player.editor.note.instrument = None
			if player.editor.note.instrument is None and project.instruments:
				player.editor.note.instrument = next(iter(project.instruments))
		if not sidebar.get("dragging"):
			for i, entry in enumerate(queue[base:base + maxitems], base):
				if not entry.get("selected"):
					continue
				x = 4 + offs
				y = round(Z + entry.get("pos", 0) * 32)
				col = project.instruments[project.instrument_layout[i]].colour
				hue, sat, val = colorsys.rgb_to_hsv(*(x / 255 for x in col))
				sat -= 0.125
				rect = (x, y, sidebar_width - 32, 32)
				entry.colour = col = [round_random(x * 255) for x in colorsys.hsv_to_rgb(hue, sat, val)]
				val /= 2
				flash = entry.get("flash", 16)
				if flash:
					sat = max(0, sat - flash / 16)
					val = min(1, val + flash / 16)
				bevel_rectangle(
					DISP2,
					[round_random(x * 255) for x in colorsys.hsv_to_rgb(hue, sat, val)],
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
					background=sc,
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
					(x + 6, y + 4),
					(0, 0, sidebar_width - 48, 24),
				)
				col = project.instruments[project.instrument_layout[i]].colour
				hue, sat, val = colorsys.rgb_to_hsv(*(x / 255 for x in col))
				h = (i / 12 - 1 / 12 + abs(1 - pc() % 2) / 6) % 1
				anima_rectangle(
					DISP2,
					[round_random(x * 255) for x in colorsys.hsv_to_rgb(h, sat - 0.0625, val)],
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
				background=sc,
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
					EFS,
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
			dest = deque()
			dest2 = deque()
			targets = {}
			positionals = {}
			for i, entry in enumerate(queue):
				if i + swap < 0 or i + swap >= len(queue):
					continue
				if entry.get("selected"):
					targets[i + swap] = entry
					positionals[i + swap] = project.instrument_layout[i]
					entry.moved = True
			i = 0
			for pos, entry in zip(project.instrument_layout, queue):
				while i in targets:
					dest.append(targets.pop(i))
					dest2.append(positionals.pop(i))
					i += 1
				if not entry.get("moved"):
					dest.append(entry)
					dest2.append(pos)
					i += 1
				else:
					entry.pop("moved", None)
			if targets:
				dest.extend(targets.values())
				dest2.extend(positionals.values())
				for entry in targets.values():
					entry.pop("moved", None)
			queue[:] = dest
			project.instrument_layout[:] = dest2
			try:
				if not sidebar.last_selected.selected:
					raise ValueError
				lq2 = queue.index(sidebar.last_selected)
			except (AttributeError, ValueError, IndexError):
				sidebar.pop("last_selected", None)
				lq2 = nan
	if offs <= -4 and sidebar.abspos == 1:
		render_settings(dur, ignore=True)
	if offs <= -4 and sidebar.abspos == 2:
		instrument = project.instruments[project.instrument_layout[sidebar.editing]]
		sub = (sidebar.rect2[2] - 4, sidebar.rect2[3] - 52)
		subp = (screensize[0] - sidebar_width, 52)
		DISP2 = DISP.subsurf(subp + sub)
		in_sidebar = in_rect(mpos, sidebar.rect)
		offs2 = offs + sidebar_width
		for i, opt in enumerate(sysettings):
			message_display(
				opt.capitalize(),
				11,
				(offs2 + 8, i * 32 + 24),
				surface=DISP2,
				align=0,
				cache=True,
				font="Comic Sans MS",
			)
			# numrect = (screensize[0] + offs + sidebar_width - 8, 68 + i * 32)
			s = str(round(instrument.synth.get(opt, 0) * 100, 2)) + "%"
			message_display(
				s,
				11,
				(offs2 + sidebar_width - 8, 40 + i * 32),
				surface=DISP2,
				align=2,
				cache=True,
				font="Comic Sans MS",
			)
			srange = sysettings[opt]
			w = (sidebar_width - 16)
			x = (instrument.synth.get(opt, 0) - srange[0]) / (srange[1] - srange[0])
			if opt in ("speed", "pitch", "nightcore"):
				x = min(1, max(0, x))
			else:
				x = min(1, abs(x))
			x = round(x * w)
			brect = (screensize[0] + offs + 6, 91 + i * 32, sidebar_width - 12, 13)
			brect2 = (offs2 + 6, 41 + i * 32, sidebar_width - 12, 13)
			hovered = in_sidebar and in_rect(mpos, brect) or syediting[opt]
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
				if syediting[opt]:
					if not mheld[0]:
						syediting[opt] = False
				elif mclick[0]:
					syediting[opt] = True
				elif mclick[1]:
					def opt_set_a(enter):
						if enter:
							v = round_min(float(safe_eval(enter)) / 100)
							syediting[opt] = True
					easygui2.enterbox(
						opt_set_a,
						opt.capitalize(),
						title="Miza Player",
						default=str(round_min(instrument.synth[opt] * 100)),
					)
				if syediting[opt]:
					orig, instrument.synth[opt] = instrument.synth[opt], v
					if orig != v:
						instrument.wave = synth_gen(**instrument.synth)
						transfer_instrument(instrument)
			z = max(0, x - 4)
			rect = (offs2 + 8 + z, 41 + i * 32, sidebar_width - 16 - z, 9)
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
				(offs2 + 8 + x, 41 + i * 32),
				(x, 0, w - x, 9),
				special_flags=BLEND_RGB_MULT,
			)
			rect = (offs2 + 8, 41 + i * 32, x, 9)
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
				(offs2 + 8, 41 + i * 32),
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