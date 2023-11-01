FONTS = {}


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
barcount2 = barcount // 4 + 1
barheight = 720

globals()["ms"] = "_".join(("SEND", "status"))

PARTICLES = set()
P_ORDER = 0
TICK = 0
TEXTS = {}

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

	def __init__(self, x, barcount):
		super().__init__()
		dark = False
		self.colour = [round(i * 255) for i in colorsys.hsv_to_rgb(x / barcount % 1, 1, 1)]
		note = highest_note - x + 9
		if note % 12 in (1, 3, 6, 8, 10):
			dark = True
		name = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")[note % 12]
		octave = note // 12
		self.line = name + str(octave + 1)
		self.x = x
		self.height = 0
		self.height2 = 0
		self.cache = {}

	def update(self, dur=1):
		if self.height:
			self.height = self.height * 0.16 ** dur
			if self.height < 1:
				self.height = 0
		if self.height2:
			self.height2 = self.height2 * 0.32 ** dur
			if self.height2 < 1:
				self.height2 = 0

	def ensure(self, value):
		if self.height < value:
			self.height = value
		value *= sqrt(2)
		if self.height2 < value:
			self.height2 = value

bars = [Bar(i - 1, barcount) for i in range(barcount)]
bars2 = [Bar(i - 1, barcount2) for i in range(barcount2)]

textsurf = pygame.Surface((1024, 1024), pygame.SRCALPHA)
W, H = textsurf.get_size()
x = y = z = 0
for bar in reversed(bars):
	surf = message_display(
		bar.line,
		28,
		colour=(255,) * 3,
		surface=None,
		font="Pacifico",
		cache=True,
	)
	w, h = surf.get_size()
	if x + w >= W:
		y = z
		x = 0
	textsurf.blit(
		surf,
		(x, y),
	)
	bar.tex_coords = (
		(x / W, 1 - (y + h) / H),
		((x + w) / W, 1 - (y + h) / H),
		((x + w) / W, 1 - y / H),
		(x / W, 1 - y / H),
	)
	bar.tex_size = (w / 28, h / 28)
	x += w
	z = max(z, y + h)
texttext = pyg2pgl(textsurf)

def animate_bars(changed=False):
	glClear(GL_COLOR_BUFFER_BIT)
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE)
	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_COLOR_ARRAY)
	verts = []
	cols = []
	for bar in bars:
		y = min(2 * barheight, bar.height)
		if not y:
			continue
		y += toolbar_height
		w = ssize[0] / barcount
		x = (barcount - 2 - bar.x) * w
		dark = False
		colour = [i for i in colorsys.hsv_to_rgb((pc_ / 3 + bar.x / barcount) % 1, 1, 1)] + [1]
		empty = (0,) * 4
		note = highest_note - bar.x + 9
		if note % 12 in (1, 3, 6, 8, 10):
			dark = True
			colour = colour[:3] + [0.5]
		verts.extend((
			(x, 0),
			(x + w, 0),
			(x + w, y / 3),
			(x, y / 3),
		))
		cols.extend((
			colour,
			colour,
			colour,
			colour,
		))
		if dark:
			verts.extend((
				(x, y / 3),
				(x + w, y / 3),
				(x + w, y * 2 / 3),
				(x, y * 2 / 3),
			))
			cols.extend((
				colour,
				colour,
				colour,
				colour,
			))
			verts.extend((
				(x, y * 2 / 3),
				(x + w, y * 2 / 3),
				(x + w, y),
				(x, y),
			))
			cols.extend((
				colour,
				colour,
				empty,
				empty,
			))
		else:
			verts.extend((
				(x, y / 3),
				(x + w, y / 3),
				(x + w, y),
				(x, y),
			))
			cols.extend((
				colour,
				colour,
				empty,
				empty,
			))
	verts = np.array(verts, dtype=np.float32)
	cols = np.array(cols, dtype=np.float32)
	glVertexPointer(2, GL_FLOAT, 0, verts.ctypes)
	glColorPointer(4, GL_FLOAT, 0, cols.ctypes)
	glDrawArrays(GL_QUADS, 0, len(verts))

	highbars = sorted(bars, key=lambda bar: bar.height, reverse=True)[:97]
	high = highbars[1]
	lowest = highbars.pop(-1).height
	dividend = globals().get("avgheight", high.height - lowest)
	high = max(1, high.height - lowest)
	globals()["avgheight"] = (dividend * 0.95) + high * 0.05

	texture = texttext.get_texture()
	glEnable(texture.target)
	glBindTexture(texture.target, texture.id)
	glEnableClientState(GL_TEXTURE_COORD_ARRAY)
	verts = []
	texs = []
	cols = []
	for bar in reversed(highbars):
		if bar.height > 1:
			scale = min(1, (bar.height - lowest) / dividend)
			if scale <= 1 / 127:
				continue
			w = ssize[0] / barcount
			x = (barcount - 2 - bar.x) * w + w / 2
			y = min(DISP.height - 84, toolbar_height + bar.height2 + 12 - w * (sqrt(5) + 1))
			fw, fh = bar.tex_size
			verts.extend((
				(x - w * fw, y - w * fh),
				(x + w * fw, y - w * fh),
				(x + w * fw, y + w * fh),
				(x - w * fw, y + w * fh),
			))
			texs.extend(bar.tex_coords)
			cols.extend((
				(1, 1, 1, scale),
			) * 4)
	verts = np.array(verts, dtype=np.float32)
	texs = np.array(texs, dtype=np.float32)
	cols = np.array(cols, dtype=np.float32)
	glVertexPointer(2, GL_FLOAT, 0, verts.ctypes)
	glTexCoordPointer(2, GL_FLOAT, 0, texs.ctypes)
	glColorPointer(4, GL_FLOAT, 0, cols.ctypes)
	glDrawArrays(GL_QUADS, 0, len(verts))
	glDisable(texture.target)

prism_setup = np.fromiter(map(int, """
0-2-3
1-3-4
2-4-5
3-5-0
4-0-1
5-1-2
6-8-9
7-9-10
8-10-11
9-11-6
10-6-7
11-7-8
0-1-7
0-7-6
1-2-8
1-8-7
2-3-9
2-9-8
3-4-10
3-10-9
4-5-11
4-11-10
5-0-6
5-6-11""".lstrip().replace("\n", "-").split("-")), dtype=np.uint8)

def animate_prism(changed=False):
	glClear(GL_COLOR_BUFFER_BIT)
	glMatrixMode(GL_PROJECTION)
	bars = bars2
	x = len(bars)
	try:
		if changed:
			raise KeyError
		spec = globals()["prism-s"]
	except KeyError:
		class Prism_Spec:
			pass
		spec = globals()["prism-s"] = Prism_Spec
	glLoadIdentity()
	ar = ssize[0] / ssize[1]
	glViewport(0, toolbar_height, *ssize)
	gluPerspective(30, ar, 1 / 16, 99999)
	glTranslatef(-x / 2 - 7.5, x / 2 - 28, -x * 2)
	glRotatef(-75, 1, 0.25, -0.125)
	glDisable(GL_DEPTH_TEST)
	glDisable(GL_CULL_FACE)
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE)
	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_COLOR_ARRAY)
	w, h = ssize
	glLineWidth(ssize[0] / 256)

	try:
		hsv = globals()["prism-hsv"]
	except KeyError:
		hsv = globals()["prism-hsv"] = np.empty((len(bars), 3), dtype=np.float32)
	try:
		rgba = globals()["prism-rgba"]
	except KeyError:
		rgba = globals()["prism-rgba"] = np.empty((len(bars), 4), dtype=np.float32)
	try:
		hue = globals()["prism-h"]
	except KeyError:
		hh = [bar.x / len(bars) for bar in bars]
		hue = globals()["prism-h"] = np.array(hh, dtype=np.float32)
	H = hue + (pc_ / 4 + sin(pc_ * tau / 8 / sqrt(2)) / 6) * 2 % 1
	hsv.T[0][:] = H % 1
	alpha = np.array([bar.height / barheight * 2 for bar in bars], dtype=np.float32)
	sat = np.clip(alpha - 1, 0, 1)
	np.subtract(1, sat, out=sat)
	hsv.T[1][:] = sat
	hsv.T[2] = alpha
	hsv.T[2] *= 0.5
	hsv *= 255
	hsv2 = hsv.astype(np.uint8)
	img = Image.frombuffer("HSV", (len(bars), 1), hsv2.tobytes()).convert("RGB")
	rgba.T[:3].T[:] = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((len(bars), 3))
	rgba.T[:3] *= 1 / 255
	rgba.T[:3] **= 4 / 3
	rgba.T[:3] *= 2
	# rgba.T[-1][:] = 2 / 3
	colours = rgba

	maxlen = len(bars)
	vertarray = None
	if "hexarray" not in globals() or hexarray.maxlen != maxlen + 1:
		globals()["hexarray"] = deque(maxlen=maxlen + 1)
	elif is_active() and len(hexarray) > maxlen:
		colarray, vertarray = hexarray.popleft()
	if is_active():
		if vertarray is None:
			vertarray = np.empty((len(bars), len(prism_setup), 3), dtype=np.float32)
			colarray = np.empty((len(bars), len(prism_setup), 4), dtype=np.float32)
		hexarray.append([colarray, vertarray])

		hexagon = np.array([[cos(z), sin(z), 0] for z in (i * tau / 6 for i in range(6))] * 2, dtype=np.float32)
		b = (0, 0, 0, 1 / 6)
		bc = np.array([b, b], dtype=np.float32)
		for i, h, c in zip(range(len(bars) - 1, -1, -1), alpha, colours):
			hexagon[6:].T[-1] = h * 3
			np.take(hexagon, prism_setup, axis=0, out=vertarray[i])
			vertarray[i].T[0] += i * 1.5
			if i & 1:
				vertarray[i].T[1] -= sqrt(3) / 2
			bc[0][:3] = c[:3]
			bc[1] = c
			np.take(bc, prism_setup < 6, axis=0, out=colarray[i])

	for c, v in hexarray:
		glColorPointer(4, GL_FLOAT, 0, c.ravel().ctypes)
		glVertexPointer(3, GL_FLOAT, 0, v.ravel().ctypes)
		glDrawArrays(GL_TRIANGLES, 0, len(bars) * len(prism_setup))
		if is_active():
			v.T[1] += sqrt(3)

def schlafli(symbol):
	args = (sys.executable, "misc/schlafli.py")
	proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	try:
		stdout, stderr = proc.communicate(symbol.encode("utf-8") + b"\n", timeout=5)
	except:
		proc.kill()
		print_exc()
		raise RuntimeError(symbol)
	resp = stdout.decode("utf-8", "replace").splitlines()
	verts = deque()
	edges = deque()
	vert = True
	for line in resp:
		if ":" in line:
			continue
		if not line:
			if vert:
				vert = False
				continue
			else:
				break
		if vert:
			v = verify_vert(map(float, line.split()))
			verts.append(v)
		else:
			e = list(map(int, line.split()))
			edges.append(e)
	verts, dims = normalise_verts(verts)
	verts = tuple(itertools.chain(*((verts[i], verts[j]) for i, j in edges)))
	verts = np.asanyarray(verts, dtype=np.float32)
	dist = verts[1::2] - verts[::2]
	dist *= dist
	dist = np.sum(dist, axis=1, out=dist.ravel()[:len(dist)])
	np.sqrt(dist, out=dist)
	complexity = np.sum(dist)
	return verts, dims, complexity

def verify_vert(v):
	v = astype(v, list)
	if len(v) < 3:
		v.append(0)
	return v

def normalise_verts(verts):
	verts = np.asanyarray(verts, dtype=np.float32)
	centre = np.mean(verts, axis=0)
	verts -= centre
	verts *= 1 / sqrt(sum(x ** 2 for x in centre))
	dims = verts.shape[-1]
	if dims <= 3:
		if np.all(verts.T[2] == 0):
			dims = 2
	return verts, dims

def parse_index(x):
	x = int(x.split("/", 1)[0])
	if x > 0:
		x -= 1
	return x

def load_model(fn):
	verts = deque()
	edges = deque()
	with open(fn, "r", encoding="utf-8") as f:
		lines = f.readlines()
	for line in lines:
		if not line:
			continue
		if line[0] == "v" and line[1] in " \t":
			v = verify_vert(map(float, line[2:].strip().split()))
			verts.append(v)
		elif line[0] == "f" and line[1] in " \t":
			e = [parse_index(x) for x in line[2:].strip().split()]
			if len(e) == 2:
				edges.append(e)
			elif len(e) > 2:
				edges.extend(zip(e[:-1], e[1:]))
				edges.append((e[0], e[-1]))
	verts, dims = normalise_verts(verts)
	verts = tuple(itertools.chain(*((verts[i], verts[j]) for i, j in edges)))
	verts = np.asanyarray(verts, dtype=np.float32)
	dist = verts[1::2] - verts[::2]
	dist *= dist
	dist = np.sum(dist, axis=1, out=dist.ravel()[:len(dist)])
	np.sqrt(dist, out=dist)
	complexity = np.sum(dist)
	return verts, dims, complexity

found_polytopes = {}
perms = {}
def get_dimension(dim):
	try:
		return perms[dim]
	except KeyError:
		pass
	perm = []
	for i in range(dim):
		for j in range(i + 1, dim):
			perm.append((i, j))
	if perm:
		perms[dim] = perm
	return perm
default_poly = schlafli("144")

def animate_polytope(changed=False):
	if not vertices:
		return
	glClear(GL_COLOR_BUFFER_BIT)
	if isinstance(vertices, (list, tuple)):
		s = " ".join(map(str, vertices))
	else:
		s = str(vertices)
	poly = None
	try:
		if changed:
			raise KeyError
		poly, dimcount, complexity = found_polytopes[s]
	except KeyError:
		pass
	if poly is None:
		try:
			if os.path.exists(s):
				poly, dimcount, complexity = load_model(s)
			else:
				poly, dimcount, complexity = schlafli(s)
		except:
			print_exc()
			poly, dimcount, complexity = default_poly
		found_polytopes[s] = (poly, dimcount, complexity)
	# print(poly)
	# return
	bars = globals()["bars"][::-1]
	dims = max(3, dimcount)
	try:
		if changed:
			raise KeyError
		spec = globals()["poly-s"]
		if spec.dims != dims or spec.dimcount != dimcount:
			raise KeyError
	except KeyError:
		class Poly_Spec:
			frame = 0
		spec = globals()["poly-s"] = Poly_Spec
		spec.dims = dims
		spec.dimcount = dimcount
		spec.rotmat = np.identity(dims)

	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	ar = ssize[0] / ssize[1]
	glViewport(0, toolbar_height, *ssize)
	gluPerspective(30, ar, 1 / 16, 99999)
	glTranslatef(0, 0, -4)
	glDisable(GL_DEPTH_TEST)
	glDisable(GL_CULL_FACE)
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE)
	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_COLOR_ARRAY)

	w, h = ssize
	thickness = ssize[0] / 64 / max(1, (complexity / 2 + 1) ** 0.5)
	glLineWidth(max(1, thickness))
	alpha_mult = min(1, thickness) / 2
	angle = tau / 512
	if is_active():
		i = 0
		perms = get_dimension(dimcount)
		for x, y in perms:
			i += 1
			if i <= 3:
				factor = i
			else:
				factor = (len(perms) - 0.5 - i) % dimcount + 0.5
			z = angle * sin(spec.frame / factor ** 0.5)
			rotation = np.identity(dims)
			a, b = cos(z), sin(z)
			rotation[x][x] = a
			rotation[x][y] = -b
			rotation[y][x] = b
			rotation[y][y] = a
			spec.rotmat = rotation @ spec.rotmat
		spec.frame += tau / 1920
	poly = poly @ spec.rotmat
	if poly.shape[-1] > 3:
		outs = poly.T[:3].T
		for dim in poly.T[3:]:
			dim /= 2
			dim += 1
			outs *= np.tile(dim, (3, 1)).T
		poly = outs
	try:
		radii = globals()["poly-r"]
	except KeyError:
		r = np.array([bar.x / len(bars) for bar in bars], dtype=np.float32)
		radii = globals()["poly-r"] = r

	try:
		hsv = globals()["poly-hsv"]
	except KeyError:
		hsv = globals()["poly-hsv"] = np.empty((len(bars), 4), dtype=np.float32)
	try:
		rgba = globals()["poly-rgba"]
	except KeyError:
		rgba = globals()["poly-rgba"] = np.empty((len(bars), 4), dtype=np.float32)
	try:
		hue = globals()["poly-h"]
	except KeyError:
		hh = [1 - bar.x / len(bars) for bar in bars]
		hue = globals()["poly-h"] = np.array(hh, dtype=np.float32)
	H = hue + (pc_ / 4 + sin(pc_ * tau / 12) / 6) % 1
	hsv.T[0][:] = H % 1
	alpha = np.array([bar.height / barheight * 2 for bar in bars], dtype=np.float32)
	sat = np.clip(alpha - 1, 0, 1)
	np.subtract(1, sat, out=sat)
	hsv.T[1][:] = sat
	hsv.T[:2] *= 255
	hsv2 = hsv.T[:3].T.astype(np.uint8)
	hsv2.T[2][:] = 255
	img = Image.frombuffer("HSV", (len(bars), 1), hsv2.tobytes()).convert("RGB")
	rgba.T[:3].T[:] = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((len(bars), 3))
	colours = rgba
	colours.T[:3] *= 1 / 255
	mult = np.linspace(alpha_mult, 0, len(alpha))
	mult *= alpha
	colours.T[-1][:] = mult

	maxb = sqrt(max(bar.height for bar in bars))
	ratio = min(48, max(8, 36864 / (len(poly) + 2 >> 1)))
	barm = sorted(((bar.height, i) for i, bar in enumerate(bars) if sqrt(bar.height) > maxb / ratio), reverse=True)
	bari = sorted((i for _, i in barm[:round_random(ratio * 2)]), reverse=True)
	if bari:
		radiii = radii[bari]
		colours = np.tile(colours[bari], (1, len(poly))).reshape((len(bari), len(poly), 4))
		verts = np.stack([np.asanyarray(poly, np.float32)] * len(bari))
		for vi, r in zip(verts, radiii):
			vi *= r
		c, v = colours.ravel(), verts.ravel()
		mults = v[2::3] + 1
		mults *= 0.5
		c[3::4] *= mults
		pointc = len(colours) * len(poly)
		# linec = pointc >> 1
		# depths = v[2::6] + v[5::6]
		# order = np.argsort(depths)
		# c = c.reshape((linec, 8))[order].ravel()
		# v = v.reshape((linec, 6))[order].ravel()
		glColorPointer(4, GL_FLOAT, 0, c.ravel().ctypes)
		glVertexPointer(3, GL_FLOAT, 0, v.ravel().ctypes)
		glDrawArrays(GL_LINES, 0, pointc)

def animate_ripple(changed=False):
	if not vertices or not vertices[0]:
		return
	V, R = vertices
	glClear(GL_COLOR_BUFFER_BIT)
	glMatrixMode(GL_PROJECTION)
	bars = globals()["bars"][::-1]
	try:
		if changed:
			raise KeyError
		spec = globals()["ripple-s"]
		if spec.R != R:
			raise KeyError
	except KeyError:
		class Ripple_Spec:
			angle = 0
			rx = 0
			ry = 0
			rz = 0
		spec = globals()["ripple-s"] = Ripple_Spec

	glLoadIdentity()
	ar = ssize[0] / ssize[1]
	glViewport(0, toolbar_height, *ssize)
	if R:
		w = 1 if ar <= 1 else ar
		h = 1 if ar >= 1 else 1 / ar
		glOrtho(-w, w, -h, h, -1, 1)
	else:
		gluPerspective(30, ar, 1 / 16, 99999)
		glTranslatef(0, 0, -4)
	glDisable(GL_DEPTH_TEST)
	glDisable(GL_CULL_FACE)
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE)
	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_COLOR_ARRAY)
	spec.R = R

	w, h = ssize
	depth = 3
	glLineWidth(ssize[0] / 144 / depth)
	if is_active():
		if R:
			rx = 0.5 * (0.8 - abs(spec.rx % 90 - 45) / 90)
			ry = 1 / sqrt(2) * (0.8 - abs(spec.ry % 90 - 45) / 90)
			rz = 0.8 - abs(spec.rz % 90 - 45) / 90
			spec.rx = (spec.rx + rx) % 360
			spec.ry = (spec.ry + ry) % 360
			spec.rz = (spec.rz + rz) % 360
		else:
			rz = 0.8 - abs(spec.rz % 90 - 45) / 90
			spec.rz = (spec.rz + rz) % 360
	glRotatef(spec.rx, 0, 1, 0)
	glRotatef(spec.ry, 1, 0, 0)
	glRotatef(spec.rz, 0, 0, 1)
	try:
		radii = globals()["ripple-r"]
	except KeyError:
		r = np.array([bar.x / len(bars) for bar in bars], dtype=np.float32)
		r.T[-1] = 1
		radii = globals()["ripple-r"] = np.repeat(r, 3).reshape((len(bars), 3))

	try:
		hsv = globals()["ripple-hsv"]
	except KeyError:
		hsv = globals()["ripple-hsv"] = np.empty((len(bars), 4), dtype=np.float32)
	try:
		rgba = globals()["ripple-rgba"]
	except KeyError:
		rgba = globals()["ripple-rgba"] = np.empty((len(bars), 4), dtype=np.float32)
	try:
		hue = globals()["ripple-h"]
	except KeyError:
		hh = [1 - bar.x / len(bars) for bar in bars]
		hue = globals()["ripple-h"] = np.array(hh, dtype=np.float32)
	if R:
		H = hue + (pc_ / 4 + sin(pc_ * tau / 8 / sqrt(2)) / 6) % 1
	else:
		H = hue - (pc_ / 4 + sin(pc_ * tau / 8 / sqrt(2)) / 6) % 1
	hsv.T[0][:] = H % 1
	alpha = np.array([bar.height / barheight * 2 for bar in bars], dtype=np.float32)
	sat = np.clip(alpha - 1, 0, 1)
	np.subtract(1, sat, out=sat)
	hsv.T[1][:] = sat
	hsv.T[:2] *= 255
	hsv2 = hsv.T[:3].T.astype(np.uint8)
	hsv2.T[2][:] = 255
	img = Image.frombuffer("HSV", (len(bars), 1), hsv2.tobytes()).convert("RGB")
	rgba.T[:3].T[:] = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((len(bars), 3))
	rgba.T[:3] *= 1 / 255
	mult = np.linspace(1, 0, len(alpha))
	if R:
		mult **= 2
	alpha *= mult
	rgba.T[-1][:] = alpha
	colours = np.repeat(rgba, 2, axis=0)[1:-1]
	colours = np.tile(colours, (V * depth, 1))

	hi = hue + pc_ / 3 % 1
	hi *= -tau
	np.sin(hi, out=hi)
	hi *= 0.25
	hi = np.repeat(np.asanyarray(hi, np.float32), 2, axis=0)[1:-1]
	hi = np.tile(hi, (V * depth, 1)).T
	maxlen = ceil(360 / V) - 1
	vertarray = None
	if "r_linearray" not in globals() or r_linearray.maxlen != maxlen + 1:
		globals()["r_linearray"] = deque(maxlen=maxlen + 1)
	elif is_active() and len(r_linearray) > maxlen:
		vertarray = r_linearray.popleft()[1]
	if is_active():
		if vertarray is None:
			vertarray = np.empty((depth * V, len(bars), 3), dtype=np.float32)
		angle = spec.angle + tau - tau / V
		zs = np.linspace(spec.angle, angle, V)
		i = 0
		for _ in range(depth):
			directions = ([cos(z), sin(z), 1] for z in zs)
			for d in directions:
				vertarray[i][:] = radii * d
				i += 1
			x = tau / 360 / depth
			spec.angle = (spec.angle + x) % tau
			zs += x
		rva = np.repeat(vertarray, 2, axis=1).swapaxes(0, 1)[1:-1].swapaxes(0, 1)
		r_linearray.append([colours, vertarray, rva])
	r = 1 if not r_linearray else 2 ** ((len(r_linearray) - 2) / len(r_linearray) - 1)

	for cols, _, verts in r_linearray:
		verts.T[-1][:] = hi
		ci = np.require(cols.ravel(), requirements="CA")
		vi = np.require(verts.ravel(), requirements="CA")
		glColorPointer(4, GL_FLOAT, 0, ci.ctypes)
		glVertexPointer(3, GL_FLOAT, 0, vi.ctypes)
		glDrawArrays(GL_LINES, 0, len(ci) // 4)
		if is_active():
			cols.T[-1] *= r

def animate_torus(changed=False):
	if not vertices or not vertices[0]:
		return
	V, R = vertices
	glClear(GL_COLOR_BUFFER_BIT)
	glMatrixMode(GL_PROJECTION)
	bars = globals()["bars"][::-1]
	try:
		if changed:
			raise KeyError
		spec = globals()["torus-s"]
		if spec.R != R:
			raise KeyError
	except KeyError:
		class Ripple_Spec:
			angle = 0
			rx = 0
			ry = 0
			rz = 0
		spec = globals()["torus-s"] = Torus_Spec

	glLoadIdentity()
	ar = ssize[0] / ssize[1]
	glViewport(0, toolbar_height, *ssize)
	if R:
		w = 1 if ar <= 1 else ar
		h = 1 if ar >= 1 else 1 / ar
		glOrtho(-w, w, -h, h, -1, 1)
	else:
		gluPerspective(30, ar, 1 / 16, 99999)
		glTranslatef(0, 0, -4)
	glDisable(GL_DEPTH_TEST)
	glDisable(GL_CULL_FACE)
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE)
	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_COLOR_ARRAY)
	spec.R = R

	w, h = ssize
	depth = 3
	glLineWidth(ssize[0] / 144 / depth)
	if is_active():
		if R:
			rx = 0.5 * (0.8 - abs(spec.rx % 90 - 45) / 90)
			ry = 1 / sqrt(2) * (0.8 - abs(spec.ry % 90 - 45) / 90)
			rz = 0.8 - abs(spec.rz % 90 - 45) / 90
			spec.rx = (spec.rx + rx) % 360
			spec.ry = (spec.ry + ry) % 360
			spec.rz = (spec.rz + rz) % 360
		else:
			rz = 0.8 - abs(spec.rz % 90 - 45) / 90
			spec.rz = (spec.rz + rz) % 360
	glRotatef(spec.rx, 0, 1, 0)
	glRotatef(spec.ry, 1, 0, 0)
	glRotatef(spec.rz, 0, 0, 1)
	try:
		radii = globals()["torus-r"]
	except KeyError:
		r = np.array([bar.x / len(bars) for bar in bars], dtype=np.float32)
		r.T[-1] = 1
		radii = globals()["torus-r"] = np.repeat(r, 3).reshape((len(bars), 3))

	try:
		hsv = globals()["torus-hsv"]
	except KeyError:
		hsv = globals()["torus-hsv"] = np.empty((len(bars), 4), dtype=np.float32)
	try:
		rgba = globals()["torus-rgba"]
	except KeyError:
		rgba = globals()["torus-rgba"] = np.empty((len(bars), 4), dtype=np.float32)
	try:
		hue = globals()["torus-h"]
	except KeyError:
		hh = [1 - bar.x / len(bars) for bar in bars]
		hue = globals()["torus-h"] = np.array(hh, dtype=np.float32)
	if R:
		H = hue + (pc_ / 4 + sin(pc_ * tau / 8 / sqrt(2)) / 6) % 1
	else:
		H = hue - (pc_ / 4 + sin(pc_ * tau / 8 / sqrt(2)) / 6) % 1
	hsv.T[0][:] = H % 1
	alpha = np.array([bar.height / barheight * 2 for bar in bars], dtype=np.float32)
	sat = np.clip(alpha - 1, 0, 1)
	np.subtract(1, sat, out=sat)
	hsv.T[1][:] = sat
	hsv.T[:2] *= 255
	hsv2 = hsv.T[:3].T.astype(np.uint8)
	hsv2.T[2][:] = 255
	img = Image.frombuffer("HSV", (len(bars), 1), hsv2.tobytes()).convert("RGB")
	rgba.T[:3].T[:] = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape((len(bars), 3))
	rgba.T[:3] *= 1 / 255
	mult = np.linspace(1, 0, len(alpha))
	if R:
		mult **= 2
	alpha *= mult
	rgba.T[-1][:] = alpha
	colours = np.repeat(rgba, 2, axis=0)[1:-1]
	colours = np.tile(colours, (V * depth, 1))

	hi = hue + pc_ / 3 % 1
	hi *= -tau
	np.sin(hi, out=hi)
	hi *= 0.25
	hi = np.repeat(np.asanyarray(hi, np.float32), 2, axis=0)[1:-1]
	hi = np.tile(hi, (V * depth, 1)).T
	maxlen = ceil(360 / V) - 1
	vertarray = None
	if "linearray" not in globals() or linearray.maxlen != maxlen + 1:
		globals()["linearray"] = deque(maxlen=maxlen + 1)
	elif is_active() and len(linearray) > maxlen:
		vertarray = linearray.popleft()[1]
	if is_active():
		if vertarray is None:
			vertarray = np.empty((depth * V, len(bars), 3), dtype=np.float32)
		angle = spec.angle + tau - tau / V
		zs = np.linspace(spec.angle, angle, V)
		i = 0
		for _ in range(depth):
			directions = ([cos(z), sin(z), 1] for z in zs)
			for d in directions:
				vertarray[i][:] = radii * d
				i += 1
			x = tau / 360 / depth
			spec.angle = (spec.angle + x) % tau
			zs += x
		rva = np.repeat(vertarray, 2, axis=1).swapaxes(0, 1)[1:-1].swapaxes(0, 1)
		linearray.append([colours, vertarray, rva])
	r = 1 if not linearray else 2 ** ((len(linearray) - 2) / len(linearray) - 1)

	for cols, _, verts in linearray:
		verts.T[-1][:] = hi
		ci = np.require(cols.ravel(), requirements="CA")
		vi = np.require(verts.ravel(), requirements="CA")
		glColorPointer(4, GL_FLOAT, 0, ci.ctypes)
		glVertexPointer(3, GL_FLOAT, 0, vi.ctypes)
		glDrawArrays(GL_LINES, 0, len(ci) // 4)
		if is_active():
			cols.T[-1] *= r


# r = 23 / 24
# s = 1 / 48
# colourmatrix = (
#	 r, s, 0, 0,
#	 0, r, s, 0,
#	 s, 0, r, 0,
# )

def ensure_bars():
	specs = options.get("spectrogram", 0)
	if specs == 3:
		bi = bars2
	else:
		bi = bars
	b = globals()["spec-mem"].buf[:len(bi) * 4]
	amp = np.frombuffer(b, dtype=np.float32)
	amp = amp[:len(bi)]
	for i, pwr in enumerate(amp):
		bi[i].ensure(pwr / 4)

def update_bars():
	specs = options.get("spectrogram", 0)
	if specs == 3:
		bi = bars2
	else:
		bi = bars
	dur = 1 / fps
	if specs == 3:
		dur *= 3
	elif specs == 4:
		dur *= 1.5
	elif specs == 5:
		dur *= 2
	for bar in bi:
		bar.update(dur=dur)

last = None
sp_changed = True
def render_spectrogram(rect):
	global sp_changed, last, pc_, vertices
	pc_ = player.extpos() if player.offpos > -inf else 0
	specs = options.get("spectrogram", 0)
	if specs == 4:
		vertices = options.control.get("gradient-vertices", (4, 3, 3))
	elif specs == 5:
		vertices = options.control.get("spiral-vertices", [24, 1])
	try:
		glEnable(GL_SCISSOR_TEST)
		glScissor(0, toolbar_height, screensize[0] - sidebar_width, screensize[1] - toolbar_height)
		# glClearColor(0, 0, 0, 0)
		if specs == 2:
			func = animate_bars
		elif specs == 3:
			func = animate_prism
		elif specs == 4:
			func = animate_polytope
		else:
			func = animate_ripple
		if last != func:
			last = func
			sp_changed = True
		ensure_bars()
		if func:
			func(sp_changed)
		sp_changed = False
		update_bars()
	except:
		print_exc()
	finally:
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		glViewport(0, 0, *DISP.get_size())
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glViewport(0, 0, *DISP.get_size())
		glOrtho(0, DISP.width, 0, DISP.height, -1, 1)
		glDisable(GL_DEPTH_TEST)
		glDisable(GL_CULL_FACE)
		glDisableClientState(GL_VERTEX_ARRAY)
		glDisableClientState(GL_TEXTURE_COORD_ARRAY)
		glDisableClientState(GL_COLOR_ARRAY)
		glDisable(GL_SCISSOR_TEST)