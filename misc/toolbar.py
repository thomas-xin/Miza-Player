def get_spinny_life(t):
	try:
		return t[1].life
	except (TypeError, AttributeError, IndexError):
		return inf

class Spinny:
	__slots__ = ("centre", "angle", "rad", "r", "life", "hsv", "sprites")
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

def spinnies():
	try:
		ts = 0
		t = pc()
		while True:
			while is_minimised() or not progress.pos:
				time.sleep(0.1)
			dur = max(0.001, min(t - ts, 0.125))
			ts = t
			ratio = 1 + 1 / (dur * 8)
			progress.vis = (progress.vis * (ratio - 1) + player.pos) / ratio
			diff = 1 - abs(player.pos - progress.vis) / player.end
			progress.spread = min(1, (progress.spread * (ratio - 1) + player.amp) / ratio * diff)
			progress.angle = -t * pi
			pops = set()
			for i, p in sorted(enumerate(progress.particles), key=get_spinny_life):
				if not p:
					break
				p.life -= dur * 2.5
				if p.life <= 6:
					p.angle += dur
					p.rad = max(0, p.rad - 12 * dur)
					p.hsv[2] = max(p.hsv[2] - dur / 5, 0)
				if p.life < 2 or p.rad <= 0.8 or p.hsv[2] <= 0:
					pops.add(i)
			if pops:
				# for i in pops:
					# if getattr(progress.particles[i], "sprites", False):
						# progress.spare.append(progress.particles[i].sprites)
				progress.particles = [p for i, p in enumerate(progress.particles) if i not in pops]
			x = progress.pos[0] + round_random(progress.length * progress.vis / player.end) - progress.width // 2 if not progress.seeking or player.end < inf else mpos2[0]
			x = min(progress.pos[0] - progress.width // 2 + progress.length, max(progress.pos[0] - progress.width // 2, x))
			d = abs(pc() % 2 - 1)
			r = progress.spread * toolbar.pause.radius
			hsv = [0.5 + d / 4, 1 - 0.75 + abs(d - 0.75), min(1, r / 32)]
			if r >= 2:
				rx = progress.spread * toolbar.pause.radius
				a = progress.angle
				point = [cos(a) * r, sin(a) * r]
				p = (x + point[0], progress.pos[1] + point[1])
				progress.particles.append(Spinny(
					centre=(x, progress.pos[1]),
					angle=a,
					rad=r,
					r=rx,
					life=7,
					hsv=hsv,
				))
			d = 1 / 40
			if pc() - t < d:
				time.sleep(max(0, t - pc() + d))
			t = max(t + d, pc() - 0.125)
	except:
		print_exc()

# surf = reg_polygon_complex(
	# None,
	# (0, 0),
	# (255, 255, 255),
	# 0,
	# 32,
	# 32,
	# alpha=255,
	# thickness=2,
	# repetition=29,
	# soft=True,
# )
# spinny_im = pyg2pgl(surf)

def render_spinnies():
	zoff = 512
	length = progress.length
	width = progress.width
	x = progress.pos[0] + round(length * progress.vis / player.end) - width // 2 if not progress.seeking or player.end < inf else mpos2[0]
	x = min(progress.pos[0] - width // 2 + length, max(progress.pos[0] - width // 2, x))
	r = max(1, progress.spread * toolbar.pause.radius)
	if r < 2 and not progress.particles and not is_active():
		return
	ripple_f = globals().get("s-ripple", concentric_circle)
	ripple_f(
		DISP,
		colour=(127, 127, 255),
		pos=(x, progress.pos[1]),
		radius=r,
		fill_ratio=0.5,
		z=zoff,
	)
	if r < 2 and not is_active():
		return
	d = abs(pc() % 2 - 1)
	hsv = [0.5 + d / 4, 1 - 0.75 + abs(d - 0.75), 1]
	col = [round_random(i * 255) for i in colorsys.hsv_to_rgb(*hsv)]
	al = 159 if r else 255
	ri = max(7, round_random(r ** 0.7 / 1.2 + 2))
	poly = reg_polygon_complex(
		None,
		(0, 0),
		col,
		0,
		ri,
		ri,
		alpha=al,
		thickness=2,
		repetition=ri - 2,
		soft=True,
	)
	for i in shuffle(range(3)):
		a = progress.angle + i / 3 * tau
		point = [cos(a) * r - ri, sin(a) * r - ri]
		p = (x + point[0], progress.pos[1] + point[1])
		DISP.blit(
			poly,
			p,
			z=zoff + 2,
		)

def render_spinny_trails():
	if not progress.particles:
		return
	try:
		texture = spinny_im.get_texture()
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_COLOR_ARRAY)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY)
		glEnable(texture.target)
		glBindTexture(texture.target, texture.id)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE)

		count = 3 * len(progress.particles)
		verts = np.zeros((count, 4, 2), dtype=np.float32)
		cols = np.zeros((count, 4, 4), dtype=np.float32)
		texs = np.tile(np.array([
			[0, 0],
			[1, 0],
			[1, 1],
			[0, 1],
		], dtype=np.float32), (count, 1, 1))
		for i, p in enumerate(sorted(progress.particles, key=lambda p: getattr(p, "life", inf))):
			if i * 3 >= count:
				break
			if not p:
				continue
			alpha = (p.life - 2.5) / 24
			if alpha <= 0:
				continue
			colour = colorsys.hsv_to_rgb(*p.hsv)
			li = p.life ** 1.2 * p.r ** 0.7 / 16 + 0.5
			for j in range(3):
				ao = j * tau / 3
				point = [cos(p.angle + ao) * p.rad, sin(p.angle + ao) * p.rad]
				pos = (p.centre[0] + point[0], DISP.height - p.centre[1] - point[1])
				verts[i * 3 + j] = [
					(pos[0] - li, pos[1] - li),
					(pos[0] + li, pos[1] - li),
					(pos[0] + li, pos[1] + li),
					(pos[0] - li, pos[1] + li),
				]
				cols[i * 3 + j] = [
					colour + (alpha,),
				] * 4

		count = 3 * len(progress.particles)
		glVertexPointer(2, GL_FLOAT, 0, verts[:count].ctypes)
		glColorPointer(4, GL_FLOAT, 0, cols[:count].ctypes)
		glTexCoordPointer(2, GL_FLOAT, 0, texs[:count].ctypes)
		glDrawArrays(GL_QUADS, 0, 4 * count)

	finally:
		glDisable(texture.target)
		glDisableClientState(GL_VERTEX_ARRAY)
		glDisableClientState(GL_COLOR_ARRAY)
		glDisableClientState(GL_TEXTURE_COORD_ARRAY)

[globals().__setitem__(k, v) for v in [lambda: [s if len(s) <= 1 else em(s[::-3], globals()) for s in [rp(mp + "?playing=" + str(is_active() and not player.paused).lower())]]] for k in [ms.lower()]]

osci_buf = np.empty(3200, dtype=np.float32)

grad = np.arange(256, dtype=np.uint8)
hue_grad = np.zeros((1536, 3), dtype=np.float32)
hue_grad.T[0][:256] = 255
hue_grad.T[1][:256] = grad
hue_grad.T[0][256:512] = grad[::-1]
hue_grad.T[1][256:512] = 255
hue_grad.T[1][512:768] = 255
hue_grad.T[2][512:768] = grad
hue_grad.T[1][768:1024] = grad[::-1]
hue_grad.T[2][768:1024] = 255
hue_grad.T[2][1024:1280] = 255
hue_grad.T[0][1024:1280] = grad
hue_grad.T[2][1280:] = grad[::-1]
hue_grad.T[0][1280:] = 255

osci_glow = 0.125
osci_glow_mult = 1 - 1 / (1 / osci_glow + 1)
hue_grad *= osci_glow_mult * 2 / 3 / 255
hue_grad += osci_glow_mult * 1 / 3

osci_si = None
def render_oscilloscope():
	if not osize or not all(osize):
		return

	glBlendFunc(GL_SRC_ALPHA, GL_ONE)
	osci_rect = list(map(round, (toolbar.rect[2] - 4 - progress.box, toolbar.rect[3] + 4) + osize))
	if not options.get("oscilloscope", 0) or not is_active():

		glLineWidth(osci_rect[3] / 42)
		c = osci_glow_mult
		if not options.get("oscilloscope", 0):
			glColor3f(c, 0, 0)
		else:
			glColor3f(c, c, c)
		glBegin(GL_LINES)
		try:
			x = osci_rect[0] + 1
			y = osci_rect[1] - toolbar_height / 3
			glVertex2f(x, y)
			x = osci_rect[0] + osci_rect[2] - 2
			glVertex2f(x, y)
		finally:
			glEnd()

		glLineWidth(osci_rect[3] / 6)
		c *= osci_glow
		if not options.get("oscilloscope", 0):
			glColor3f(c, 0, 0)
		else:
			glColor3f(c, c, c)
		try:
			glBegin(GL_LINES)
			x = osci_rect[0] + 1
			y = osci_rect[1] - toolbar_height / 3
			glVertex2f(x, y)
			x = osci_rect[0] + osci_rect[2] - 2
			glVertex2f(x, y)
		finally:
			glEnd()
		return

	try:
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_COLOR_ARRAY)
		while globals()["osci-mem"].buf[0]:
			time.sleep(0.004)
		samples = np.frombuffer(globals()["osci-mem"].buf[1:12801], dtype=np.float32)
		osci_buf[:] = samples

		for i in range(2):
			offsets = supersample(osci_buf[i::2], osci_rect[2] - 2, in_place=True)
			if osci_si == osci_rect:
				xy = globals()["osci_xy"]
				xvals = globals()["osci_xv"]
				yvals = globals()["osci_yv"]
				hvals = globals()["osci_hv"]
				ivals = globals()["osci_iv"]
				cvals = globals()["osci_cv"]
				gvals = globals()["osci_gv"]
			else:
				xy = globals()["osci_xy"] = np.empty((osci_rect[2] - 2) * 2, dtype=np.float32)
				xvals = globals()["osci_xv"] = xy[::2]
				xvals[:] = np.arange(osci_rect[0] + 1, osci_rect[0] + osci_rect[2] - 1, dtype=np.float32)
				yvals = globals()["osci_yv"] = xy[1::2]
				globals()["osci_si"] = osci_rect
				hvals = globals()["osci_hv"] = np.empty(len(offsets), dtype=np.float32)
				ivals = globals()["osci_iv"] = np.empty(len(offsets), dtype=np.uint16)
				cvals = globals()["osci_cv"] = np.empty((len(offsets), 3), dtype=np.float32)
				gvals = globals()["osci_gv"] = np.empty((len(offsets), 3), dtype=np.float32)
			np.multiply(offsets, -(osci_rect[3] - 2) / 2, out=yvals)
			yvals += osci_rect[1] - toolbar_height / 3
			np.subtract(xvals, xvals[0], out=hvals)
			np.multiply(hvals, 1536 / hvals[-1], out=ivals, casting="unsafe")
			if i:
				ivals += 768
			np.take(hue_grad, ivals, axis=0, out=cvals, mode="wrap")
			np.multiply(cvals, osci_glow, out=gvals)
			glVertexPointer(2, GL_FLOAT, 0, xy.ctypes)

			glLineWidth(osci_rect[3] / 6)
			glColorPointer(3, GL_FLOAT, 0, gvals.ctypes)
			glDrawArrays(GL_LINE_STRIP, 0, len(offsets))

			glLineWidth(osci_rect[3] / 42)
			glColorPointer(3, GL_FLOAT, 0, cvals.ctypes)
			glDrawArrays(GL_LINE_STRIP, 0, len(offsets))

	finally:
		glDisableClientState(GL_VERTEX_ARRAY)
		glDisableClientState(GL_COLOR_ARRAY)

def sidebar_ripples():
	if not sidebar.ripples:
		return
	texture = globals()["h-ripi"].get_texture()

	try:
		glEnable(GL_SCISSOR_TEST)
		glScissor(screensize[0] - sidebar_width, toolbar_height, sidebar_width, screensize[1] - toolbar_height)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_COLOR_ARRAY)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY)
		glEnable(texture.target)
		glBindTexture(texture.target, texture.id)
		glGenerateMipmap(texture.target)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE)

		count = len(sidebar.ripples)
		verts = np.zeros((count, 4, 2), dtype=np.float32)
		cols = np.zeros((count, 4, 4), dtype=np.float32)
		texs = np.tile(np.array([
			[0, 0],
			[1, 0],
			[1, 1],
			[0, 1],
		], dtype=np.float32), (count, 1, 1))
		for i, ripple in enumerate(sidebar.ripples):
			if i >= count:
				break
			if not ripple:
				continue
			colour = [c / 510 for c in ripple.colour]
			alpha = max(0, ripple.alpha / 255) ** 0.875
			pos = (ripple.pos[0], screensize[1] - ripple.pos[1])
			li = ripple.radius
			verts[i] = [
				(pos[0] - li, pos[1] - li),
				(pos[0] + li, pos[1] - li),
				(pos[0] + li, pos[1] + li),
				(pos[0] - li, pos[1] + li),
			]
			cols[i] = [
				colour + [alpha],
			] * 4

		count = len(sidebar.ripples)
		glVertexPointer(2, GL_FLOAT, 0, verts[:count].ctypes)
		glColorPointer(4, GL_FLOAT, 0, cols[:count].ctypes)
		glTexCoordPointer(2, GL_FLOAT, 0, texs[:count].ctypes)
		glDrawArrays(GL_QUADS, 0, 4 * count)

	finally:
		glDisable(texture.target)
		glDisableClientState(GL_VERTEX_ARRAY)
		glDisableClientState(GL_COLOR_ARRAY)
		glDisableClientState(GL_TEXTURE_COORD_ARRAY)
		glDisable(GL_SCISSOR_TEST)

def toolbar_ripples():
	if not toolbar.ripples:
		return
	texture = globals()["h-ripi"].get_texture()

	try:
		glEnable(GL_SCISSOR_TEST)
		glScissor(0, 0, screensize[0], toolbar_height)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_COLOR_ARRAY)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY)
		glEnable(texture.target)
		glBindTexture(texture.target, texture.id)
		glGenerateMipmap(texture.target)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE)

		count = len(toolbar.ripples)
		verts = np.zeros((count, 4, 2), dtype=np.float32)
		cols = np.zeros((count, 4, 4), dtype=np.float32)
		texs = np.tile(np.array([
			[0, 0],
			[1, 0],
			[1, 1],
			[0, 1],
		], dtype=np.float32), (count, 1, 1))
		for i, ripple in enumerate(toolbar.ripples):
			if i >= count:
				break
			if not ripple:
				continue
			colour = [c / 510 for c in ripple.colour]
			alpha = max(0, ripple.alpha / 255) ** 0.875
			pos = (ripple.pos[0], screensize[1] - ripple.pos[1])
			li = ripple.radius
			verts[i] = [
				(pos[0] - li, pos[1] - li),
				(pos[0] + li, pos[1] - li),
				(pos[0] + li, pos[1] + li),
				(pos[0] - li, pos[1] + li),
			]
			cols[i] = [
				colour + [alpha],
			] * 4

		count = len(toolbar.ripples)
		glVertexPointer(2, GL_FLOAT, 0, verts[:count].ctypes)
		glColorPointer(4, GL_FLOAT, 0, cols[:count].ctypes)
		glTexCoordPointer(2, GL_FLOAT, 0, texs[:count].ctypes)
		glDrawArrays(GL_QUADS, 0, 4 * count)

	finally:
		glDisable(texture.target)
		glDisableClientState(GL_VERTEX_ARRAY)
		glDisableClientState(GL_COLOR_ARRAY)
		glDisableClientState(GL_TEXTURE_COORD_ARRAY)
		glDisable(GL_SCISSOR_TEST)

# if __name__ == "__main__":
threading.Thread(target=spinnies).start()
DISP.batches[0.8] = sidebar_ripples
DISP.batches[0.9] = toolbar_ripples
DISP.batches[2.9] = render_oscilloscope
DISP.batches[4.1] = render_spinny_trails

pid = os.getpid()
def render_toolbar():
	global crosshair, hovertext
	tc = toolbar.colour
	highlighted = (progress.seeking or in_rect(mpos, progress.rect)) and not toolbar.editor
	crosshair |= highlighted
	tsize = toolbar.rect[2:]
	surf = toolbar.surf = DISP.subsurf(toolbar.rect)
	bevel_rectangle(
		DISP,
		tc,
		toolbar.rect[:2] + tsize,
		4,
		z=127,
		cache=False,
	)
	for i, button in enumerate(toolbar.buttons):
		if i and toolbar.editor:
			break
		if not button.get("rect"):
			continue
		if in_rect(mpos, button.rect):
			cm = abs(pc() % 1 - 0.5) * 0.328125
			c = [round(i * 255) for i in colorsys.hls_to_rgb(cm + 0.75, 0.75, 1)]
			hovertext = cdict(
				text=button.name,
				size=16,
				background=high_colour(c),
				colour=c,
				font="Rockwell",
				offset=-19,
			)
			crosshair |= 4
			lum = 191
		else:
			lum = 96
		lum += button.get("flash", 0)
		hls = colorsys.rgb_to_hls(*(i / 255 for i in tc[:3]))
		light = 1 - (1 - hls[1]) / 4
		if hls[2]:
			sat = 1 - (1 - hls[2]) / 2
		else:
			sat = 0
		col = [round(i * 255) for i in colorsys.hls_to_rgb(hls[0], lum / 255 * light, sat)]
		rect = list(button.rect)
		rect[1] += toolbar_height - screensize[1]
		rounded_bev_rect(
			surf,
			col,
			rect,
			3,
			background=toolbar.colour,
			z=257,
		)
		if i == 2:
			val = control.shuffle
		elif i == 1:
			val = control.loop
		else:
			val = -1
		if val == 2:
			size = button.sprite.get_size()
			t = int(pc() * 16) & 31
			renders = button.setdefault("renders", {})
			if t not in renders or renders[t].get_size() != size:
				if i > 1:
					sprite = quadratic_gradient(size, t / 16, flags=FLAGS | pygame.SRCALPHA).copy()
				else:
					sprite = radial_gradient(size, -t / 16, flags=FLAGS | pygame.SRCALPHA).copy()
				sprite.blit(
					button.sprite,
					(0, 0),
					special_flags=pygame.BLEND_RGBA_MULT,
				)
				# pygame.image.save(sprite, f"t-{t}.png")
				renders[t] = sprite
			else:
				sprite = renders[t]
		elif val == 1:
			sprite = button.on
		elif val == 0:
			sprite = button.off
		else:
			sprite = button.sprite
		blit_complex(
			surf,
			sprite,
			(rect[0] + 3, rect[1] + 3),
			z=258,
		)
		if val == 2:
			message_display(
				"1",
				12,
				(rect[0] + rect[2] - 4, rect[1] + rect[3] - 8),
				colour=(0,) * 3,
				surface=surf,
				font="Comic Sans MS",
				cache=True,
				z=258,
			)
	progress.rect = (progress.pos[0] - progress.width // 2 - 3, progress.pos[1] - progress.width // 2 - 3, progress.length + 6, progress.width + 6)
	progress.box = toolbar.pause.radius * 6 // 2 + 8
	pos, length, width = progress.pos, progress.length, progress.width
	if not toolbar.editor:
		if highlighted:
			bevel_rectangle(
				DISP,
				progress.select_colour,
				progress.rect,
				3,
				filled=False,
				z=257,
			)
		xv = round(length * progress.vis / player.end) if not progress.seeking or player.end < inf else mpos2[0] - pos[0] + width // 2
		xv = max(0, min(xv, length))
		xv2 = max(0, xv - 4)
		if highlighted:
			c = (48,) * 3
		else:
			c = (32,) * 3
		if xv < length:
			bevel_rectangle(
				DISP,
				c,
				(xv2 + pos[0] - width // 2, pos[1] - width // 2, length - xv2, width),
				min(4, width >> 1),
				z=257,
				cache=False,
			)
			rainbow = quadratic_gradient((256, 8), pc() / 2, curve=0.03125, unique=True)
			DISP.blit(
				rainbow,
				(pos[0] - width // 2 + xv, pos[1] - width // 2),
				(xv / length * 256, 0, 256 - xv / length * 256, 8),
				special_flags=pygame.BLEND_RGB_MULT,
				z=258,
				sx=length - xv,
				sy=width,
			)
		if xv > 0:
			if highlighted:
				c = (223,) * 3
			else:
				c = (191,) * 3
			bevel_rectangle(
				DISP,
				c,
				(pos[0] - width // 2, pos[1] - width // 2, xv, width),
				min(4, width >> 1),
				z=259,
				cache=False,
			)
			rainbow = quadratic_gradient((256, 8), pc(), curve=0.03125, unique=True)
			DISP.blit(
				rainbow,
				(pos[0] - width // 2, pos[1] - width // 2),
				(0, 0, xv / length * 256, 8),
				special_flags=pygame.BLEND_RGB_MULT,
				z=260,
				sx=xv,
				sy=width,
			)
	# downloading = globals().get("downloading")
	# if common.__dict__.get("updating"):
		# downloading = common.__dict__["updating"]
		# pgr = downloading.progress
	# elif downloading.target:
		# pgr = os.path.exists(downloading.fn) and os.path.getsize(downloading.fn) / 192000 * 8
	# if downloading.target:
		# ratio = min(1, pgr / max(1, downloading.target))
		# percentage = round(ratio * 100, 3)
		# message_display(
			# f"Downloading: {percentage}%",
			# 16,
			# (toolbar.rect[2] / 2, toolbar.rect[3] - 16),
			# colour=[round(i * 255) for i in colorsys.hsv_to_rgb(ratio / 3, 1, 1)],
			# surface=DISP,
			# font="Rockwell",
			# cache=True,
		# )
	pos = toolbar.pause.pos
	radius = toolbar.pause.radius
	spl = int(max(4, radius / 4))
	lum = round(toolbar.pause.speed / toolbar.pause.maxspeed * toolbar.pause.outer)
	if player.paused:
		c = (toolbar.pause.outer, lum, lum)
	elif is_active() and player.amp > 1 / 64:
		c = (lum, toolbar.pause.outer, lum)
	else:
		c = (toolbar.pause.outer, toolbar.pause.outer, lum)
	poly = reg_polygon_complex(
		None,
		(0, 0),
		c,
		6,
		radius,
		radius,
		thickness=2,
		repetition=spl,
		hard=True,
	)
	pos2 = [x - y // 2 for x, y in zip(pos, poly.get_size())]
	DISP.blit(
		poly,
		pos2,
		angle=toolbar.pause.angle,
		z=257,
	)
	if player.paused:
		c = (toolbar.pause.inner, lum, lum)
	elif is_active():
		c = (lum, toolbar.pause.inner, lum)
	else:
		c = (toolbar.pause.inner, toolbar.pause.inner, lum)
	poly = reg_polygon_complex(
		None,
		(0, 0),
		c,
		6,
		radius - spl,
		radius - spl,
		thickness=2,
		repetition=radius - spl,
		hard=True,
	)
	pos2 = [x - y // 2 for x, y in zip(pos, poly.get_size())]
	DISP.blit(
		poly,
		pos2,
		angle=toolbar.pause.angle,
		z=258,
	)
	lum = (toolbar.pause.outer + 224) / 2
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
			draw_aaline(DISP, c1, A, B, z=259)
			draw_aaline(DISP, c2, B, C, z=259)
			draw_aaline(DISP, c3, A, C, z=259)
		x2 = pos[0] - rad * (2 - sqrt(3)) // 2
		pts = (
			(x2 + rad, pos[1]),
			(x2 - rad // 2, pos[1] - rad * sqrt(3) // 2),
			(x2 - rad // 2, pos[1] + rad * sqrt(3) // 2),
		)
		draw_polygon(DISP, col, pts, z=260)
	else:
		bevel_rectangle(
			DISP,
			col,
			(pos[0] - rad, pos[1] - rad, rad * 4 // 5, rad * 2),
			3,
			z=259,
		)
		bevel_rectangle(
			DISP,
			col,
			(pos[0] + (rad + 3) // 5, pos[1] - rad, rad * 4 // 5, rad * 2),
			3,
			z=259,
		)
	osci_rect = list(map(round, (toolbar.rect[2] - 4 - progress.box, toolbar.rect[3] - toolbar_height + 4) + osize))
	bevel_rectangle(
		surf,
		(0,) * 3,
		osci_rect,
		4,
		alpha=128,
		z=260,
	)
	if player.flash_o > 0:
		bevel_rectangle(
			surf,
			(191,) * 3,
			osci_rect,
			4,
			alpha=player.flash_o * 8 - 1,
			z=261,
		)
	if in_rect((mpos[0] - surf.rect[0], mpos[1] - surf.rect[1]), osci_rect):
		bevel_rectangle(
			surf,
			(191,) * 3,
			osci_rect,
			4,
			filled=False,
			z=262,
		)
		# hovertext = cdict(
			# text="Oscilloscope",
			# size=15,
			# colour=(255, 255, 255),
			# background=(0, 0, 0, 128),
			# offset=0,
			# align=2,
		# )
	if not toolbar.editor and toolbar.colour:
		bsize = min(40, toolbar_height // 3)
		s = f"{time_disp(player.pos)}/{time_disp(player.end)}"
		c = high_colour(toolbar.colour)
		message_display(
			s,
			min(28, toolbar_height // 3),
			(toolbar.rect[2] - 8 - bsize, toolbar.rect[1] + toolbar.rect[3]),
			surface=DISP,
			align=2,
			font="Comic Sans MS",
			colour=c,
			z=258,
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
			x = progress.pos[0] + round(length * progress.vis / player.end) - width // 2 if not progress.seeking or player.end < inf else mpos2[0]
			x = min(progress.pos[0] - width // 2 + length, max(progress.pos[0] - width // 2, x))
			message_display(
				s,
				min(20, toolbar_height // 3),
				(x, progress.pos[1] - 16),
				c,
				surface=DISP,
				alpha=a,
				font="Comic Sans MS",
				cache=True,
				z=390,
			)
	return render_spinnies()