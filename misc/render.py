import time, numpy, math, random, orjson, collections, colorsys, traceback, subprocess, itertools, weakref, ctypes
from math import *
from traceback import print_exc
np = numpy
deque = collections.deque
from PIL import Image
import concurrent.futures
exc = concurrent.futures.ThreadPoolExecutor(max_workers=12)
submit = exc.submit

def pyg2pil(surf):
	mode = "RGBA" if surf.get_flags() & pygame.SRCALPHA else "RGB"
	b = pygame.image.tostring(surf, mode)
	return Image.frombuffer(mode, surf.get_size(), b)

def pil2pyg(im):
	mode = im.mode
	b = im.tobytes()
	return pygame.image.frombuffer(b, im.size, mode)

class HWSurface:

	cache = weakref.WeakKeyDictionary()
	anys = {}
	anyque = []
	maxlen = 128

	@classmethod
	def any(cls, size, flags=0, colour=None):
		size = astype(size, tuple)
		m = 4 if flags & pygame.SRCALPHA else 3
		t = (size, flags)
		try:
			self = cls.anys[t]
		except KeyError:
			if len(cls.anyque) >= cls.maxlen:
				cls.anys.pop(cls.anyque.pop(0))
			cls.anyque.append(t)
			self = cls.anys[t] = pygame.Surface(size, flags, m << 3)
		else:
			if t != cls.anyque[-1]:
				cls.anyque.remove(t)
				cls.anyque.append(t)
		if colour is not None:
			self.fill(colour)
		return self

import sys
write, sys.stdout.write = sys.stdout.write, lambda *args, **kwargs: None
import pygame
sys.stdout.write = write
print = lambda s: sys.stderr.write(str(s) + "\n")
pygame.font.init()
FONTS = {}

try:
	hwnd = int(sys.stdin.readline()[1:])
except ValueError:
	hwnd = 0
import socket
server = socket.create_server(("127.0.0.1", hwnd % 32768 + 16384))
server.listen(1)

is_minimised = lambda: ctypes.windll.user32.IsIconic(hwnd)

import multiprocessing.shared_memory
globals()["spec-mem"] = multiprocessing.shared_memory.SharedMemory(
	name=f"Miza-Player-{hwnd}-spec-mem",
)
globals()["spec-size"] = multiprocessing.shared_memory.SharedMemory(
	name=f"Miza-Player-{hwnd}-spec-size",
)
globals()["spec-locks"] = multiprocessing.shared_memory.SharedMemory(
	name=f"Miza-Player-{hwnd}-spec-locks",
)

def astype(obj, t, *args, **kwargs):
	try:
		if not isinstance(obj, t):
			if callable(t):
				return t(obj, *args, **kwargs)
			return t
	except TypeError:
		if callable(t):
			return t(obj, *args, **kwargs)
		return t
	return obj

def round_random(x):
	y = int(x)
	if y == x:
		return y
	x -= y
	if random.random() <= x:
		y += 1
	return y


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
barcount2 = barcount * 2 - 1
barcount3 = barcount // 4 + 1
barheight = 720

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

	__slots__ = ("x", "colour", "height", "height2", "surf", "line", "cache")

	fsize = 0

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

	def render(self, sfx, **void):
		size = min(2 * barheight, round_random(self.height))
		if not size:
			return
		dark = False
		self.colour = [round_random(i * 255) for i in colorsys.hsv_to_rgb((pc_ / 3 + self.x / barcount) % 1, 1, 1)]
		note = highest_note - self.x + 9
		if note % 12 in (1, 3, 6, 8, 10):
			dark = True
		if dark:
			self.colour = [i + 1 >> 1 for i in self.colour]
			surf = HWSurface.any((1, 3))
		else:
			surf = HWSurface.any((1, 2))
		surf.fill(self.colour)
		surf.set_at((0, 0), 0)
		self.surf = surf
		x = barcount - 2 - self.x
		y = barheight - size
		if x >= sfx.get_width():
			return
		if y >= 0:
			dest = sfx.subsurface((x, y, 1, size))
			return pygame.transform.smoothscale(self.surf, (1, size), dest)
		surf = HWSurface.any((1, size))
		surf = pygame.transform.smoothscale(self.surf, (1, size), surf)
		sfx.blit(surf, (x, y))

	def post_render(self, sfx, scale, **void):
		size = self.height2
		if size < 8:
			return
		ix = barcount - 2 - self.x
		sx = ix / (barcount - 2) * ssize2[0]
		w = (ix + 1) / (barcount - 2) * ssize2[0] - sx
		if scale < 7 / 255:
			alpha = round_random(scale * 255)
		else:
			alpha = round_random(85 * scale) * 3
		if not alpha:
			return
		t = (self.line, self.fsize)
		if t not in TEXTS:
			TEXTS[t] = self.font.render(self.line, True, (255,) * 3)
		surf = TEXTS[t]
		if alpha < 255:
			try:
				surf = self.cache[alpha]
			except KeyError:
				surf = surf.copy()
				surf.fill((255,) * 3 + (alpha,), special_flags=pygame.BLEND_RGBA_MULT)
				self.cache[alpha] = surf
		width, height = surf.get_size()
		x = round(sx + w / 2 - width / 2)
		y = round_random(max(84, ssize2[1] - size + 12 - width * (sqrt(5) + 1)))
		sfx.blit(surf, (x, y))

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
	bars = bars3
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
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		ar = ssize2[0] / ssize2[1]
		gluPerspective(45, ar, 1 / 16, 99999)
		glTranslatef(-x // 2 - 2.5, x // 2 - 9, -x * 1.5)
		glRotatef(75, 1, 0.25, -0.125)
		# glDisable(GL_DEPTH_TEST)
		# glEnable(GL_CULL_FACE)
		# glCullFace(GL_BACK)
		# glFrontFace(GL_CCW)
		# glDisable(GL_BLEND)
		glDisable(GL_DEPTH_TEST)
		glDisable(GL_CULL_FACE)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_COLOR_ARRAY)
	w, h = specsize
	glLineWidth(specsize[0] / 256)

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
	rgba.T[-1][:] = 2 / 3
	colours = rgba

	maxlen = len(bars)
	vertarray = None
	if "hexarray" not in globals() or hexarray.maxlen != maxlen + 1:
		globals()["hexarray"] = deque(maxlen=maxlen + 1)
	elif len(hexarray) > maxlen:
		colarray, vertarray = hexarray.popleft()
	if vertarray is None:
		vertarray = np.empty((len(bars), len(prism_setup), 3), dtype=np.float32)
		colarray = np.empty((len(bars), len(prism_setup), 4), dtype=np.float32)
	hexarray.append([colarray, vertarray])

	if skipping:
		return

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
		glColorPointer(4, GL_FLOAT, 0, c.ravel())
		glVertexPointer(3, GL_FLOAT, 0, v.ravel())
		glDrawArrays(GL_TRIANGLES, 0, len(bars) * len(prism_setup))
		v.T[1] -= sqrt(3)
	return "glReadPixels"

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
	if type(vertices) is list:
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
		glLoadIdentity()
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		ar = ssize2[0] / ssize2[1]
		gluPerspective(45, ar, 1 / 16, 99999)
		glTranslatef(0, 0, -2.5)
		glDisable(GL_DEPTH_TEST)
		glDisable(GL_CULL_FACE)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_COLOR_ARRAY)
	w, h = specsize
	thickness = specsize[0] / 64 / max(1, (complexity / 2 + 1) ** 0.5)
	glLineWidth(max(1, thickness))
	alpha_mult = min(1, thickness) / 2
	angle = tau / 512
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

	if skipping:
		return

	maxb = sqrt(max(bar.height for bar in bars))
	ratio = min(48, max(8, 36864 / (len(poly) + 2 >> 1)))
	barm = sorted(((bar.height, i) for i, bar in enumerate(bars) if sqrt(bar.height) > maxb / ratio), reverse=True)
	bari = sorted((i for _, i in barm[:round_random(ratio * 2)]), reverse=True)
	if bari:
		radiii = radii[bari]
		colours = np.tile(colours[bari], (1, len(poly))).reshape((len(bari), len(poly), 4))
		verts = np.stack([np.asanyarray(poly, np.float32)] * len(bari))
		for v, r in zip(verts, radiii):
			v *= r
		glColorPointer(4, GL_FLOAT, 0, colours.ravel())
		glVertexPointer(3, GL_FLOAT, 0, verts.ravel())
		glDrawArrays(GL_LINES, 0, len(colours) * len(poly))
	return "glReadPixels"

def animate_ripple(changed=False):
	if not vertices or not vertices[0]:
		return
	V, R = vertices
	glClear(GL_COLOR_BUFFER_BIT)
	bars = bars2[::-1]
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
		if R:
			glLoadIdentity()
			glMatrixMode(GL_MODELVIEW)
			glLoadIdentity()
		else:
			glLoadIdentity()
			glMatrixMode(GL_PROJECTION)
			glLoadIdentity()
			ar = ssize2[0] / ssize2[1]
			gluPerspective(45, ar, 1 / 16, 99999)
			glTranslatef(0, 0, -2.5)
		glDisable(GL_DEPTH_TEST)
		glDisable(GL_CULL_FACE)
		glEnable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_COLOR_ARRAY)
		spec.R = R
	w, h = specsize
	depth = 3
	glLineWidth(specsize[0] / 64 / depth)
	if R:
		rx = 0.5 * (0.8 - abs(spec.rx % 90 - 45) / 90)
		ry = 1 / sqrt(2) * (0.8 - abs(spec.ry % 90 - 45) / 90)
		rz = 0.8 - abs(spec.rz % 90 - 45) / 90
		glRotatef(-0.5, rx, ry, rz)
		spec.rx = (spec.rx + rx) % 360
		spec.ry = (spec.ry + ry) % 360
		spec.rz = (spec.rz + rz) % 360
	else:
		glRotatef(-0.5, 0, 0, 1)
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
	if "linearray" not in globals() or linearray.maxlen != maxlen + 1:
		globals()["linearray"] = deque(maxlen=maxlen + 1)
	elif len(linearray) > maxlen:
		vertarray = linearray.popleft()[-1].swapaxes(0, 1)[:len(bars)].swapaxes(0, 1)
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
	linearray.append([colours, rva])
	r = 2 ** ((len(linearray) - 2) / len(linearray) - 1)

	if skipping:
		return

	for c, verts in linearray:
		glColorPointer(4, GL_FLOAT, 0, c.ravel())
		verts.T[-1][:] = hi
		glVertexPointer(3, GL_FLOAT, 0, verts.ravel())
		glDrawArrays(GL_LINES, 0, len(c))
		c.T[-1] *= r
	return "glReadPixels"

bars = [Bar(i - 1, barcount) for i in range(barcount)]
bars2 = [Bar(i - 1, barcount2) for i in range(barcount2)]
bars3 = [Bar(i - 1, barcount3) for i in range(barcount3)]
last = None
# r = 23 / 24
# s = 1 / 48
# colourmatrix = (
#	 r, s, 0, 0,
#	 0, r, s, 0,
#	 s, 0, r, 0,
# )

clearing = 0
def spectrogram_render(bars):
	global ssize2, specs, dur, last, sp_changed
	try:
		if specs == 1:
			sfx = HWSurface.any((barcount - 2, barheight))
			if clearing >= 3:
				sfx.fill(0)
				globals()["clearing"] = 0
			else:
				globals()["clearing"] += 1
			for bar in bars:
				bar.render(sfx=sfx)
			func = None
		elif specs == 2:
			func = animate_prism
		elif specs == 3:
			func = animate_polytope
		else:
			func = animate_ripple
		if last != func:
			last = func
			sp_changed = True
		if func:
			sfx = func(sp_changed)
		sp_changed = False
		if not sfx:
			return
		if sfx == "glReadPixels":
			glFlush()

		length = ssize2[0] * ssize2[1] * 3
		ssize = np.frombuffer(globals()["spec-size"].buf[:8], dtype=np.uint32)
		locks = globals()["spec-locks"].buf
		if locks[0] == 255:
			locks[0] = 0
		else:
			while locks[0] > 0:
				time.sleep(0.005)
		locks[0] += 1
		try:
			if specs == 1:
				sfx2 = pygame.image.frombuffer(globals()["spec-mem"].buf[:length], ssize2, "RGB")
				sfx = pygame.transform.scale(sfx, ssize2, sfx2)
				fsize = max(12, round(ssize2[0] / barcount * (sqrt(5) + 1) / 2))
				if Bar.fsize != fsize:
					Bar.fsize = fsize
					Bar.font = pygame.font.Font("misc/Pacifico.ttf", bar.fsize)
					for bar in bars:
						bar.cache.clear()
				highbars = sorted(bars, key=lambda bar: bar.height, reverse=True)[:97]
				high = highbars[0]
				lowest = highbars.pop(-1).height
				dividend = globals().get("avgheight", high.height - lowest)
				for bar in reversed(highbars):
					if bar.height > 1:
						bar.post_render(sfx=sfx, scale=(bar.height - lowest) / dividend)
				high = max(1, high.height - lowest)
				globals()["avgheight"] = (dividend * 0.95) + high * 0.05
			if isinstance(sfx, bytes):
				globals()["spec-mem"].buf[:length] = sfx
			elif sfx == "glReadPixels":
				x = y = 0
				glReadPixels(x, y, *ssize2, GL_RGB, GL_UNSIGNED_BYTE, array=globals()["spec-mem"].buf[:length])
		except:
			raise
		finally:
			if locks[0] > 0:
				locks[0] -= 1

		if specs == 2:
			dur *= 3
		elif specs == 3:
			dur *= 1.5
		elif specs == 4:
			dur *= 2
		for bar in bars:
			bar.update(dur=dur)
	except:
		print_exc()

def ensure_bars(b):
	amp = np.frombuffer(b, dtype=np.float32)
	if specs == 2:
		bi = bars3
	elif specs == 4:
		bi = bars2
	else:
		bi = bars
	amp = amp[:len(bi)]
	for i, pwr in enumerate(amp):
		bi[i].ensure(pwr / 4)

def setup_window(size):
	glfw.window_hint(glfw.VISIBLE, False)
	window = glfw.create_window(*size, "render", None, None)
	glfw.make_context_current(window)
	glutInitDisplayMode(GL_RGB)
	glEnable(GL_LINE_SMOOTH)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
	glBlendEquation(GL_FUNC_ADD)
	glPixelStorei(GL_PACK_ALIGNMENT, 1)
	globals()["sp_changed"] = True
	return window

with open("misc/toolbar.py", "rb") as f:
	b = f.read()
exec(compile(b, "toolbar.py", "exec"))

tool_lock = concurrent.futures.Future()
tool_lock.set_result(None)
def _render_toolbar(x):
	global tool_lock
	locks = globals()["spec-locks"].buf
	tool_lock.result()
	tool_lock = concurrent.futures.Future()
	if locks[2] == 255:
		locks[2] = 0
	else:
		while locks[2] > 0:
			time.sleep(0.005)
	locks[2] += 1
	try:
		render_toolbar()
	except:
		print_exc()
	finally:
		locks[2] = 0
	tool_lock.set_result(None)
	sys.__stdout__.buffer.write(b"~r" + x)
	sys.__stdout__.flush()

def event():
	global tool_lock
	while True:
		try:
			line = sys.stdin.buffer.read(2)
			if line == b"~t":
				x = sys.stdin.buffer.readline()[1:]
				if globals().get("rtool"):
					rtool.result()
				globals()["rtool"] = submit(_render_toolbar, x)
			if line == b"~a":
				x = sys.stdin.buffer.readline().rstrip().split(b"~", 1)
				i = int(x[0])
				if i == 2:
					tool_lock.result()
					tool_lock = concurrent.futures.Future()
				while locks[i] > 0:
					time.sleep(0.004)
				locks[i] += 1
				sys.__stdout__.buffer.write(b"~r" + x[1] + b"\n")
				sys.__stdout__.flush()
			if line == b"~r":
				i = int(sys.stdin.buffer.read(1))
				locks[i] = 0
				if i == 2:
					tool_lock.set_result(None)
			elif not line:
				break
		except:
			print_exc()

ssize2 = (0, 0)
specs = 0
D = 9
specsize = (1, 1)#(barcount * D - D,) * 2

import glfw
from glfw.GLFW import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

glfw.init()
window = setup_window(specsize)
locks = globals()["spec-locks"].buf
last_changed = 0
skipping = False

submit(event)

import psutil
parent = psutil.Process(os.getppid())
receiver, _ = server.accept()

while True:
	try:
		comm = receiver.recv(2)
		i = receiver.recv(8)
		while len(i) < 8:
			i += receiver.recv(8 - i)
		nbytes = int(np.frombuffer(i, dtype=np.float64))
		line = receiver.recv(nbytes)
		while len(line) < nbytes:
			line += receiver.recv(nbytes - len(line))
		if comm == b"~r":
			ssize2, specs, vertices, dur, pc_ = map(orjson.loads, line.split(b"~"))
			if ssize2 != specsize:
				if last_changed <= 0:
					specsize = ssize2
					glfw.destroy_window(window)
					window = setup_window(specsize)
					last_changed = 8
				else:
					last_changed -= 1
					continue
			elif last_changed:
				last_changed -= 1
			if specs == 2:
				bi = bars3
			elif specs == 4:
				bi = bars2
			else:
				bi = bars
			spectrogram_render(bi)
			receiver.send(b"\x7f")
		elif comm == b"~e":
			ensure_bars(line)
		glfwPollEvents()
	except ConnectionResetError:
		if parent.is_running() and parent.status() != "zombie":
			time.sleep(0.5)
			receiver, _ = server.accept()
		else:
			psutil.Process().terminate()
	except:
		print_exc()

glfw.destroy_window(window)
glfw.terminate()