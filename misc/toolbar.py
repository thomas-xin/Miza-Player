import pygame.gfxdraw as gfxdraw
pc = time.perf_counter

FLAGS = 0

SURFS = {}
def load_surface(fn, greyscale=False, size=None, force=False):
	if type(fn) is str:
		tup = (fn, greyscale, size)
	else:
		tup = None
	if not force:
		try:
			return SURFS[tup]
		except KeyError:
			pass
	im = image = Image.open(fn)
	if im.mode == "P":
		im = im.convert("RGBA")
	if size:
		im = im.resize(size, Image.LANCZOS)
	if greyscale:
		if "A" in im.mode:
			A = im.getchannel("A")
		im2 = ImageOps.grayscale(im)
		if "A" in im.mode:
			im2.putalpha(A)
		if "RGB" not in im2.mode:
			im2 = im2.convert("RGB" + ("A" if "A" in im.mode else ""))
		im = im2
	surf = pil2pyg(im)
	image.close()
	out = surf.copy()
	if tup:
		SURFS[tup] = out
	return out

luma = lambda c: sqrt(0.299 * (c[0] / 255) ** 2 + 0.587 * (c[1] / 255) ** 2 + 0.114 * (c[2] / 255) ** 2) * (1 if len(c) < 4 else c[-1] / 255)
verify_colour = lambda c: [max(0, min(255, abs(i))) for i in c]
high_colour = lambda c, v=255: (255 - v if luma(c) > 0.5 else v,) * 3

def adj_colour(colour, brightness=0, intensity=1, hue=0):
	if hue != 0:
		h = colorsys.rgb_to_hsv(i / 255 for i in colour)
		c = adj_colour(colorsys.hsv_to_rgb((h[0] + hue) % 1, h[1], h[2]), intensity=255)
	else:
		c = astype(colour, list)
	for i in range(len(c)):
		c[i] = round(c[i] * intensity + brightness)
	return verify_colour(c)

gsize = (1920, 1)
gradient = ((np.arange(1, 0, -1 / gsize[0], dtype=np.float32)) ** 2 * 256).astype(np.uint8).reshape(tuple(reversed(gsize)))
qhue = Image.fromarray(gradient, "L")
qsat = qval = Image.new("L", gsize, 255)
QUADS = {}

def quadratic_gradient(size=gsize, t=None, curve=None, flags=None, copy=False):
	if flags is None:
		flags = FLAGS
	size = tuple(size)
	if t is None:
		t = pc()
	quadratics = QUADS.get(flags & pygame.SRCALPHA)
	if not quadratics:
		quadratics = QUADS[flags & pygame.SRCALPHA] = [None] * 256
	x = int(t * 128) & 255
	if not quadratics[x]:
		hue = qhue.point(lambda i: i + x & 255)
		img = Image.merge("HSV", (hue, qsat, qval)).convert("RGB")
		quadratics[x] = pil2pyg(img).copy()
	surf = quadratics[x]
	if surf.get_size() != size or surf.get_flags() != flags or curve:
		s2 = HWSurface.any(size, flags)
		surf = pygame.transform.scale(surf, size, s2)
		if curve:
			h = size[1]
			m = h + 1 >> 1
			for i in range(1, m):
				tx = t - curve * (i / (m - 1))
				g = quadratic_gradient((size[0], 1), tx)
				y = h // 2 - (not h & 1)
				try:
					surf.blit(g, (0, y - i))
				except pygame.error:
					continue
				y = h // 2
				try:
					surf.blit(g, (0, y + i))
				except pygame.error:
					continue
	elif copy:
		return HWSurface.copy(surf)
	return surf

rgw = 256
mid = (rgw - 1) / 2
row = np.arange(rgw, dtype=np.float32)
row -= mid
data = [None] * rgw
for i in range(rgw):
	data[i] = a = np.arctan2(i - mid, row)
	np.around(np.multiply(a, 256 / tau, out=a), 0, out=a)
data = np.uint8(data)
rhue = Image.fromarray(data, "L")
rsat = rval = Image.new("L", (rgw,) * 2, 255)
RADS = {}

def radial_gradient(size=(rgw,) * 2, t=None, flags=None, copy=False):
	if flags is None:
		flags = FLAGS
	size = tuple(size)
	if t is None:
		t = pc()
	radials = RADS.get(flags & pygame.SRCALPHA)
	if not radials:
		radials = RADS[flags & pygame.SRCALPHA] = [None] * 256
	x = int(t * 128) & 255
	if not radials[x]:
		hue = rhue.point(lambda i: i + x & 255)
		img = Image.merge("HSV", (hue, rsat, rval)).convert("RGB")
		radials[x] = pil2pyg(img)
	surf = radials[x]
	if surf.get_size() != size or surf.get_flags() != flags:
		s2 = HWSurface.any(size, flags)
		surf = pygame.transform.scale(surf, size, s2)
	elif copy:
		return HWSurface.copy(surf)
	return surf

draw_line = pygame.draw.line
draw_aaline = pygame.draw.aaline
draw_hline = gfxdraw.hline
draw_vline = gfxdraw.vline
draw_polygon = pygame.draw.polygon
draw_tpolygon = gfxdraw.textured_polygon

in_rect = lambda point, rect: point[0] >= rect[0] and point[0] < rect[0] + rect[2] and point[1] >= rect[1] and point[1] < rect[1] + rect[3]
in_circ = lambda point, dest, radius=1: hypot(dest[0] - point[0], dest[1] - point[1]) <= radius
def int_rect(r1, r2):
	x1, y1, x2, y2, = r1
	x2 += x1
	y2 += y1
	x3, y3, x4, y4 = r2
	x4 += x3
	y4 += y3
	return max(x1, x3) < min(x2, x4) and max(y1, y3) < min(y2, y4)

def draw_arc(surf, colour, pos, radius, start_angle=0, stop_angle=0):
	start_angle = int(start_angle % 360)
	stop_angle = int(stop_angle % 360)
	if radius <= 1:
		gfxdraw.filled_circle(surf, *pos, 1, colour)
	if start_angle == stop_angle:
		gfxdraw.circle(surf, *pos, radius, colour)
	else:
		gfxdraw.arc(surf, *pos, radius, start_angle, stop_angle, colour)

def garbage_collect(cache, lim=4096):
	while len(cache) >= lim:
		try:
			del cache[next(iter(cache))]
		except:
			return

import weakref
cb_cache = weakref.WeakKeyDictionary()
ALPHA = BASIC = 0
def blit_complex(dest, source, position=(0, 0), alpha=255, angle=0, scale=1, colour=(255,) * 3, area=None, copy=True, cache=True):
	pos = position
	if len(pos) > 2:
		pos = pos[:2]
	s1 = source.get_size()
	if dest:
		s2 = dest.get_size()
		if pos[0] >= s2[0] or pos[1] >= s2[1] or pos[0] <= -s1[0] or pos[1] <= -s1[1]:
			return
	alpha = round_random(min(alpha / 3, 85)) * 3
	if alpha <= 0:
		return
	s = source
	if alpha != 255 or any(i != 255 for i in colour) or dest is None:
		if copy:
			try:
				if not cache:
					raise KeyError
				for s, c, a in cb_cache[source]:
					if a == alpha and c == colour:
						break
				else:
					raise KeyError
			except KeyError:
				s = source.copy()
				try:
					cb_cache[source].append((s, colour, alpha))
				except KeyError:
					L = min(1024, max(12, 1048576 // s.get_width() // s.get_height()))
					cb_cache[source] = deque([(s, colour, alpha)], maxlen=L)
				# print(sum(len(c) for c in cb_cache.values()))
				if alpha != 255:
					s.fill(tuple(colour) + (alpha,), special_flags=pygame.BLEND_RGBA_MULT)
				elif any(i != 255 for i in colour):
					s.fill(tuple(colour), special_flags=pygame.BLEND_RGB_MULT)
		elif alpha != 255:
			s.fill(tuple(colour) + (alpha,), special_flags=pygame.BLEND_RGBA_MULT)
		elif any(i != 255 for i in colour):
			s.fill(tuple(colour), special_flags=pygame.BLEND_RGB_MULT)
	if angle:
		ckf = [s.get_colorkey(), s.get_flags()]
		s = pygame.transform.rotate(s, -angle / d2r)
		s.set_colorkey(*ckf)
		s3 = s.get_size()
		pos = [z - (y - x >> 1) for x, y, z in zip(s1, s3, pos)]
	if scale != 1:
		s = custom_scale(s, list(map(lambda i: round(i * scale), s.get_size())), hwany=True)
	if area is not None:
		area = list(map(lambda i: round(i * scale), area))
	if dest:
		if s.get_flags() & pygame.SRCALPHA or s.get_colorkey():
			globals()["ALPHA"] += 1
			return dest.blit(s, pos, area, special_flags=pygame.BLEND_ALPHA_SDL2)
		globals()["BASIC"] += 1
		return dest.blit(s, pos, area)
	return s

def draw_rect(dest, colour, rect, width=0, alpha=255, angle=0):
	alpha = max(0, min(255, round_random(alpha)))
	width = round(abs(width))
	if width > 0:
		if angle != 0 or alpha != 255:
			ssize = [i + width for i in rect[2:]]
			s = pygame.Surface(ssize, FLAGS)
			srec = [i + width // 2 for i in rect[2:]]
			pygame.draw.rect(s, colour, srec, width)
			blit_complex(dest, s, rect[:2], alpha, angle)
			#raise NotImplementedError("Alpha blending and rotation of rectangles with variable width is not implemented.")
		else:
			pygame.draw.rect(dest, colour, width)
	else:
		if angle != 0:
			bevel_rectangle(dest, colour, rect, 0, alpha, angle)
		else:
			rect = astype(rect, list)
			if rect[0] < 0:
				rect[2] += rect[0]
				rect[0] = 0
			if rect[1] < 0:
				rect[3] += rect[1]
				rect[1] = 0
			if alpha != 255:
				dest.fill((255 - alpha,) * 4, rect, special_flags=pygame.BLEND_RGBA_MULT)
				dest.fill([min(i + alpha / 255, 255) for i in colour] + [alpha], rect, special_flags=pygame.BLEND_RGBA_ADD)
			else:
				dest.fill(colour, rect)

def bevel_rectangle(dest, colour, rect, bevel=0, alpha=255, angle=0, grad_col=None, grad_angle=0, filled=True, cache=True, copy=True):
	rect = list(map(round, rect))
	if len(colour) > 3:
		colour, alpha = colour[:-1], colour[-1]
	if min(alpha, rect[2], rect[3]) <= 0:
		return
	s = dest.get_size()
	r = (0, 0) + s
	if not int_rect(r, rect):
		return
	br_surf = globals().setdefault("br_surf", {})
	colour = verify_colour(colour)
	if alpha == 255 and angle == 0 and (any(i > 160 for i in colour) or all(i in (0, 16, 32, 48, 64, 96, 127, 159, 191, 223, 255) for i in colour)):
		if cache:
			data = tuple(rect[2:]) + (grad_col, grad_angle, tuple(colour), filled)
		else:
			data = None
		try:
			surf = br_surf[data]
		except KeyError:
			surf = pygame.Surface(rect[2:], FLAGS)
			if not filled:
				surf.fill((1, 2, 3))
				surf.set_colorkey((1, 2, 3))
			r = rect
			rect = [0] * 2 + rect[2:]
			for c in range(bevel):
				p = [rect[0] + c, rect[1] + c]
				q = [a + b - c - 1 for a, b in zip(rect[:2], rect[2:])]
				v1 = 128 - c / bevel * 128
				v2 = c / bevel * 96 - 96
				col1 = col2 = colour
				if v1:
					col1 = [min(i + v1, 255) for i in col1]
				if v2:
					col2 = [max(i + v2, 0) for i in col1]
				try:
					draw_hline(surf, p[0], q[0], p[1], col1)
					draw_vline(surf, p[0], p[1], q[1], col1)
					draw_hline(surf, p[0], q[0], q[1], col2)
					draw_vline(surf, q[0], p[1], q[1], col2)
				except:
					print_exc()
			if filled:
				if grad_col is None:
					draw_rect(surf, colour, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
				else:
					gradient_rectangle(surf, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
			rect = r
			if data:
				garbage_collect(br_surf, 64)
				br_surf[data] = surf
		if dest:
			return blit_complex(dest, surf, rect[:2])
		return surf.copy() if copy else surf
	ctr = max(colour)
	contrast = min(round(ctr) + 2 >> 2 << 2, 255)
	data = tuple(rect[2:]) + (grad_col, grad_angle, contrast, filled)
	s = br_surf.get(data)
	if s is None:
		colour2 = (contrast,) * 3
		s = pygame.Surface(rect[2:], FLAGS)
		s.fill((1, 2, 3))
		s.set_colorkey((1, 2, 3))
		for c in range(bevel):
			p = [c, c]
			q = [i - c - 1 for i in rect[2:]]
			v1 = 128 - c / bevel * 128
			v2 = c / bevel * 96 - 96
			col1 = col2 = colour2
			if v1:
				col1 = [min(i + v1, 255) for i in col1]
			if v2:
				col2 = [max(i + v2, 0) for i in col1]
			draw_hline(s, p[0], q[0], p[1], col1)
			draw_vline(s, p[0], p[1], q[1], col1)
			draw_hline(s, p[0], q[0], q[1], col2)
			draw_vline(s, q[0], p[1], q[1], col2)
		if filled:
			if grad_col is None:
				draw_rect(s, colour2, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
			else:
				gradient_rectangle(s, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
		if cache:
			garbage_collect(br_surf, 64)
			br_surf[data] = s
	if ctr > 0:
		colour = tuple(round(i * 255 / ctr) for i in colour)
	else:
		colour = (0,) * 3
	return blit_complex(dest, s, rect[:2], angle=angle, alpha=alpha, colour=colour)

def rounded_bev_rect(dest, colour, rect, bevel=0, alpha=255, angle=0, grad_col=None, grad_angle=0, filled=True, background=None, cache=True, copy=True):
	rect = list(map(round, rect))
	if len(colour) > 3:
		colour, alpha = colour[:-1], colour[-1]
	if min(alpha, rect[2], rect[3]) <= 0:
		return
	s = dest.get_size()
	r = (0, 0) + s
	if not int_rect(r, rect):
		return
	rb_surf = globals().setdefault("rb_surf", {})
	colour = list(map(lambda i: min(i, 255), colour))
	if alpha == 255 and angle == 0:
		if cache:
			if background:
				background = astype(background, tuple)
			data = tuple(rect[2:]) + (grad_col, grad_angle, tuple(colour), filled, background)
		else:
			data = None
		try:
			surf = rb_surf[data]
		except KeyError:
			if background:
				surf = pygame.Surface(rect[2:], FLAGS)
				if not filled:
					surf.fill((1, 2, 3))
					surf.set_colorkey((1, 2, 3))
				elif any(background):
					surf.fill(background)
			else:
				surf = pygame.Surface(rect[2:], FLAGS | pygame.SRCALPHA)
			r = rect
			rect = [0] * 2 + rect[2:]
			s = surf
			for c in range(bevel):
				p = [rect[0] + c, rect[1] + c]
				q = [a + b - c - 1 for a, b in zip(rect[:2], rect[2:])]
				b = bevel - c
				v1 = 128 - c / bevel * 128
				v2 = c / bevel * 96 - 96
				col1 = col2 = colour
				if v1:
					col1 = [min(i + v1, 255) for i in col1]
				if v2:
					col2 = [max(i + v2, 0) for i in col1]
				n = b <= 1
				draw_hline(s, p[0] + b - n, q[0] - b, p[1], col1)
				draw_vline(s, p[0], p[1] + b, q[1] - b + n, col1)
				draw_hline(s, p[0] + b, q[0] - b + n, q[1], col2)
				draw_vline(s, q[0], p[1] + b - n, q[1] - b, col2)
				if b > 1:
					draw_arc(s, col1, [p[0] + b, p[1] + b], b, 180, 270)
					draw_arc(s, colour, [q[0] - b, p[1] + b], b, 270, 360)
					draw_arc(s, colour, [p[0] + b, q[1] - b], b, 90, 180)
					draw_arc(s, col2, [q[0] - b, q[1] - b], b, 0, 90)
			if filled:
				if grad_col is None:
					draw_rect(surf, colour, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
				else:
					gradient_rectangle(surf, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
			rect = r
			if data:
				garbage_collect(rb_surf, 256)
				rb_surf[data] = surf
		if dest:
			return blit_complex(dest, surf, rect[:2])
		return surf.copy() if copy else surf
	ctr = max(colour)
	contrast = min(round(ctr) + 2 >> 2 << 2, 255)
	data = tuple(rect[2:]) + (grad_col, grad_angle, contrast, filled)
	s = rb_surf.get(data)
	if s is None:
		colour2 = (contrast,) * 3
		s = pygame.Surface(rect[2:], FLAGS | pygame.SRCALPHA)
		for c in range(bevel):
			p = [c, c]
			q = [i - c - 1 for i in rect[2:]]
			b = bevel - c
			v1 = 128 - c / bevel * 128
			v2 = c / bevel * 96 - 96
			col1 = col2 = colour2
			if v1:
				col1 = [min(i + v1, 255) for i in col1]
			if v2:
				col2 = [max(i + v2, 0) for i in col1]
			n = b <= 1
			draw_hline(s, p[0] + b - n, q[0] - b, p[1], col1)
			draw_vline(s, p[0], p[1] + b, q[1] - b + n, col1)
			draw_hline(s, p[0] + b, q[0] - b + n, q[1], col2)
			draw_vline(s, q[0], p[1] + b - n, q[1] - b, col2)
			if b > 1:
				draw_arc(s, col1, [p[0] + b, p[1] + b], b, 180, 270)
				draw_arc(s, colour2, [q[0] - b, p[1] + b], b, 270, 360)
				draw_arc(s, colour2, [p[0] + b, q[1] - b], b, 90, 180)
				draw_arc(s, col2, [q[0] - b, q[1] - b], b, 0, 90)
		if filled:
			if grad_col is None:
				draw_rect(s, colour2, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
			else:
				gradient_rectangle(s, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle)
		if cache:
			garbage_collect(rb_surf, 256)
			rb_surf[data] = s
	if ctr > 0:
		colour = tuple(round_random(i * 255 / ctr) for i in colour)
	else:
		colour = (0,) * 3
	return blit_complex(dest, s, rect[:2], angle=angle, alpha=alpha, colour=colour)

reg_polygon_cache = {}

def reg_polygon_complex(dest, centre, colour, sides, width, height, angle=pi / 4, alpha=255, thickness=0, repetition=1, filled=False, rotation=0, soft=False, attempts=128, cache=True):
	width = max(round(width), 0)
	height = max(round(height), 0)
	repetition = int(repetition)
	if sides:
		angle %= tau / sides
	else:
		angle = 0
	cache |= angle % (pi / 4) == 0
	if cache:
		colour = tuple(min(255, round_random(i / 5) * 5) for i in colour)
		angle = round_random(angle / tau * 144) * tau / 144
		if soft and soft is not True:
			soft = astype(soft, tuple)
		h = (colour, sides, width, height, angle, thickness, repetition, filled, soft)
		try:
			newS = reg_polygon_cache[h]
		except KeyError:
			pass
		else:
			pos = [centre[0] - width, centre[1] - height]
			return blit_complex(dest, newS, pos, alpha, rotation, copy=True)
	construct = pygame.Surface if cache else HWSurface.any
	if not soft:
		newS = construct((width << 1, height << 1), FLAGS)
		newS.set_colorkey((1, 2, 3))
		newS.fill((1, 2, 3))
	elif soft is True:
		newS = construct((width << 1, height << 1), FLAGS | pygame.SRCALPHA)
	else:
		newS = construct((width << 1, height << 1), FLAGS)
		if any(soft):
			newS.fill(soft)
	draw_direction = 1 if repetition >= 0 else -1
	if draw_direction >= 0:
		a = draw_direction
		b = repetition + 1
	else:
		a = repetition + 1
		b = -draw_direction
	if sides > 32:
		sides = 0
	elif sides < 0:
		sides = 0
	draw_direction *= max(thickness, 3) - 2
	loop = a
	setted = filled
	att = 0
	while loop < b + draw_direction:
		if att >= attempts:
			break
		att += 1
		if loop - b > 0:
			loop = b
		move_direction = loop / repetition + 0.2
		points = []
		if soft is True:
			colourU = tuple(colour) + (min(round(255 * move_direction + 8), 255),)
		else:
			colourU = (colour[0] * move_direction + 8, colour[1] * move_direction + 8, colour[2] * move_direction + 8)
			colourU = list(map(lambda c: min(c, 255), colourU))
		try:
			size = (min(width, height) - loop)
			thickness = int(min(thickness, size))
			if setted:
				thickness = 0
				setted = False
			elif not filled:
				thickness = thickness + 4 >> 1
			if sides:
				for p in range(sides):
					points.append((
						width + (width - loop) * cos(angle + p * tau / sides),
						height + (height - loop) * sin(angle + p * tau / sides),
					))
				pygame.draw.polygon(newS, colourU, points, thickness)
			else:
				if thickness > loop:
					thickness = 0
				pygame.draw.ellipse(newS, colourU, (loop, loop, (width - loop) << 1, (height - loop) << 1), thickness)
		except:
			print_exc()
		loop += draw_direction
	pos = [centre[0] - width, centre[1] - height]
	if cache:
		garbage_collect(reg_polygon_cache, 16384)
		reg_polygon_cache[h] = newS
		# print(len(reg_polygon_cache), h)
	return blit_complex(dest, newS, pos, alpha, rotation, copy=cache, cache=False)

def concentric_circle(dest, colour, pos, radius, width=0, fill_ratio=1, alpha=255, gradient=False, filled=False, cache=True):
	reverse = fill_ratio < 0
	radius = max(0, round(radius * 2) / 2)
	if min(alpha, radius) > 0:
		cc_surf = globals().setdefault("cc_surf", {})
		width = max(0, min(width, radius))
		tw = width / radius
		fill_ratio = min(1, abs(fill_ratio))
		cr = bit_crush(round(fill_ratio * 64), 3)
		wr = bit_crush(round(tw * 64), 3)
		colour = verify_colour(colour)
		data = (radius, wr, cr, gradient, filled)
		s = cc_surf.get(data)
		if s == 0:
			cache = False
		if not s:
			radius2 = min(128, bit_crush(radius, 5, ceil))
			width2 = max(2, round(radius2 * tw))
			colour2 = (255,) * 3
			data2 = tuple(colour2) + (radius2 * 2, wr, cr, gradient, filled)
			s2 = cc_surf.get(data2)
			if not s2:
				width2 = round(width2)
				size = [radius2 * 2] * 2
				size2 = [round(radius2 * 4), round(radius2 * 4) + 1]
				s2 = pygame.Surface(size2, FLAGS | pygame.SRCALPHA)
				circles = round(radius2 * 2 * fill_ratio / width2)
				col = colour2
				r = radius2 * 2
				for i in range(circles):
					if reverse:
						it = (i + 1) / circles
					else:
						it = 1 - i / circles
					if filled and i == circles - 1:
						width2 = 0
					if gradient:
						col = adj_colour(colour2, 0, it)
					c = col + (round(255 * min(1, (it + gradient))),)
					pygame.draw.circle(s2, c, [i - 1 for i in size], r, min(r, width2 + (width2 > 0)))
					r -= width2
				cc_surf[data2] = s2
			size3 = [round(radius * 2) for i in range(2)]
			s = custom_scale(s2, size3, antialias=1)
			if cache:
				cc_surf[data] = s
		p = [i - radius for i in pos]
		return blit_complex(dest, s, p, alpha=alpha, colour=colour)

def load_spinner(spinner_path):
	try:
		globals()["s-img"] = Image.open(spinner_path)
		globals()["s-cache"] = {}

		def spinner_ripple(dest, colour, pos, radius, alpha=255, **kwargs):
			diameter = round_random(radius * 2)
			if not diameter > 0:
				return
			try:
				surf = globals()["s-cache"][diameter]
			except KeyError:
				im = globals()["s-img"].resize((diameter,) * 2, resample=Image.LANCZOS)
				if "RGB" not in im.mode:
					im = im.convert("RGBA")
				surf = pil2pyg(im)
				im.close()
				globals()["s-cache"][diameter] = surf
			blit_complex(
				dest,
				surf,
				[round_random(x - y / 2) for x, y in zip(pos, surf.get_size())],
				alpha=alpha,
				colour=colour,
			)
		globals()["s-ripple"] = spinner_ripple
	except:
		print_exc()

spinner_path = "misc/Default/bubble.png"
sfut = submit(load_spinner, spinner_path)

def text_objects(text, font, colour, background):
	text_surface = font.render(text, True, colour, background)
	return text_surface, text_surface.get_rect()

def get_font(font):
	try:
		fn = "misc/" + font + ".ttf"
		if "ct_font" in globals():
			ct_font.clear()
		if "ft_font" in globals():
			ft_font.clear()
		md_font.clear()
	except:
		print_exc()

loaded_fonts = set()

def sysfont(font, size, unicode=False):
	func = pygame.ftfont if unicode else pygame.font
	fn = "misc/" + font + ".ttf"
	if not os.path.exists(fn) and font not in loaded_fonts:
		if font in ("Rockwell", "OpenSansEmoji"):
			print("Downloading and applying required fonts...")
			for fnt in ("Rockwell", "OpenSansEmoji"):
				loaded_fonts.add(fnt)
				submit(get_font, fnt)
	if os.path.exists(fn):
		return func.Font(fn, size)
	return func.SysFont(font, size)

def surface_font(text, colour, background, size, font):
	size = round(size)
	unicode = any(ord(c) >= 65536 for c in text)
	if not unicode:
		ct_font = globals().setdefault("ct_font", {})
	else:
		ct_font = globals().setdefault("ft_font", {})
	data = (size, font)
	f = ct_font.get(data, None)
	if not f:
		f = ct_font[data] = sysfont(font, size, unicode=unicode)
	for i in range(4):
		try:
			return text_objects(text, f, colour, background)
		except:
			if i >= 3:
				raise
			f = ct_font[data] = sysfont(font, size, unicode=unicode)

def text_size(text, size, font="OpenSansEmoji"):
	size = round(size)
	asc = text.isascii()
	if asc:
		ct_font = globals().setdefault("ct_font", {})
	else:
		ct_font = globals().setdefault("ft_font", {})
	data = (size, font)
	f = ct_font.get(data, None)
	if not f:
		f = ct_font[data] = sysfont(font, size, unicode=not asc)
	for i in range(4):
		try:
			return f.size(text)
		except:
			if i >= 3:
				raise
			f = ct_font[data] = sysfont(font, size, unicode=not asc)

md_font = {}
def message_display(text, size, pos=(0, 0), colour=(255,) * 3, background=None, surface=None, font="OpenSansEmoji", alpha=255, align=1, cache=False):
	# text = "".join(c if ord(c) < 65536 else "\x7f" for c in text)
	text = str(text if type(text) is not float else round_min(text))
	colour = tuple(verify_colour(colour))
	data = (text, colour, background, size, font)
	try:
		resp = md_font[data]
	except KeyError:
		resp = surface_font(*data)
	TextSurf, TextRect = resp
	if cache:
		garbage_collect(md_font, 4096)
		md_font[data] = resp
	if surface:
		if align == 1:
			TextRect.center = pos
		elif align == 0:
			TextRect = astype(pos, list) + TextRect[2:]
		elif align == 2:
			TextRect = [y - x for x, y in zip(TextRect[2:], pos)] + TextRect[2:]
		blit_complex(surface, TextSurf, TextRect, alpha, copy=alpha != 255 and cache)
		return TextRect
	else:
		return TextSurf

def time_disp(s, rounded=True):
	if not isfinite(s):
		return str(s)
	if rounded:
		s = round(s)
	output = str(s % 60)
	if len(output) < 2:
		output = "0" + output
	if s >= 60:
		temp = str((s // 60) % 60)
		if len(temp) < 2 and s >= 3600:
			temp = "0" + temp
		output = temp + ":" + output
		if s >= 3600:
			temp = str((s // 3600) % 24)
			if len(temp) < 2 and s >= 86400:
				temp = "0" + temp
			output = temp + ":" + output
			if s >= 86400:
				output = str(s // 86400) + ":" + output
	else:
		output = "0:" + output
	return output

def shuffle(it):
	if not isinstance(it, list):
		it = list(it)
	random.shuffle(it)
	return it

def get_spinny_life(t):
	try:
		return t[1].life
	except (TypeError, AttributeError, IndexError):
		return inf

class Spinny:
	__slots__ = ("centre", "angle", "rad", "r", "life", "hsv")
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

def spinnies():
	ts = 0
	while "progress" not in globals():
		time.sleep(0.1)
	t = pc()
	while True:
		while is_minimised():
			time.sleep(0.1)
		dur = max(0.001, min(t - ts, 0.125))
		ts = t
		try:
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
				if p.life < 3:
					pops.add(i)
			if pops:
				progress.particles = [p for i, p in enumerate(progress.particles) if i not in pops]
			x = progress.pos[0] + round_random(progress.length * progress.vis / player.end) - progress.width // 2 if not progress.seeking or player.end < inf else mpos2[0]
			x = min(progress.pos[0] - progress.width // 2 + progress.length, max(progress.pos[0] - progress.width // 2, x))
			d = abs(pc() % 2 - 1)
			hsv = [0.5 + d / 4, 1 - 0.75 + abs(d - 0.75), 1]
			r = progress.spread * toolbar.pause.radius
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
			d = 1 / 60
			if pc() - t < d:
				time.sleep(max(0, t - pc() + d))
			t = max(t + d, pc() - 0.5)
		except:
			print_exc()
submit(spinnies)

def render_spinnies():
	sfut.result()
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
	)
	for i, p in sorted(enumerate(progress.particles), key=get_spinny_life):
		if not p:
			continue
		col = [round_random(i * 255) for i in colorsys.hsv_to_rgb(*p.hsv)]
		a = round(min(255, (p.life - 2.5) * 12))
		for j in shuffle(range(3)):
			point = [cos(p.angle + j * tau / 3) * p.rad, sin(p.angle + j * tau / 3) * p.rad]
			pos = [round_random(x) for x in (p.centre[0] + point[0], p.centre[1] + point[1])]
			ri = max(1, round_random(p.life ** 1.2 * p.r ** 0.7 / 16 + 0.5))
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
				gfxdraw.aacircle(
					DISP,
					*pos,
					1,
					col + [a],
				)
	if r < 2 and not is_active():
		return
	d = abs(pc() % 2 - 1)
	hsv = [0.5 + d / 4, 1 - 0.75 + abs(d - 0.75), 1]
	col = [round_random(i * 255) for i in colorsys.hsv_to_rgb(*hsv)]
	for i in shuffle(range(3)):
		a = progress.angle + i / 3 * tau
		point = [cos(a) * r, sin(a) * r]
		p = (x + point[0], progress.pos[1] + point[1])
		ri = max(7, round_random(r ** 0.7 / 1.2 + 2))
		reg_polygon_complex(
			DISP,
			p,
			col,
			0,
			ri,
			ri,
			alpha=159 if r else 255,
			thickness=2,
			repetition=ri - 2,
			soft=True,
		)

globals()["tool-vals"] = multiprocessing.shared_memory.SharedMemory(
	name=f"Miza-Player-{hwnd}-tool-vals",
)
globals()["tool-mem"] = multiprocessing.shared_memory.SharedMemory(
	name=f"Miza-Player-{hwnd}-tool-mem",
)
class toolbar:
	rect = (0,) * 4
	colour = (0,) * 3
	editor = False
	buttons = ()
	class pause:
		radius = 0
		pos = (0, 0)
		speed = 0
		angle = 0
		maxspeed = 5
class progress:
	vis = 0
	pos = (0, 0)
	length = 0
	width = 0
	seeking = False
	box = 0
	spread = 0
	particles = []
	alpha = 0
	angle = 0
	num = 0
class player:
	pos = 0
	end = inf
	paused = False
	flash_o = 0
	amp = 0
	active = False

is_active = lambda: player.active

def render_toolbar():
	args = np.frombuffer(globals()["tool-vals"].buf[:4096], dtype=np.float64)
	toolbar.rect = (0, 0, int(args[0]), int(args[1]))
	toolbar_height = int(args[1])
	screen_height = int(args[32])
	toolbar.colour = tuple(map(int, args[2:5]))
	length = round(toolbar.rect[2] * toolbar.rect[3] * 3)
	if "DISP" not in globals() or DISP.get_size() != toolbar.rect[2:]:
		globals()["DISP"] = pygame.image.frombuffer(globals()["tool-mem"].buf[:length], toolbar.rect[2:], "RGB")
	mpos = (args[5], args[6] - screen_height + toolbar_height)
	mpos2 = (args[7], args[8] - screen_height + toolbar_height)
	highlighted = args[9]
	progress.pos = (int(args[10]), int(args[11] - screen_height + toolbar_height))
	progress.length = int(args[12])
	progress.width = int(args[13])
	player.pos = args[14]
	player.end = args[15]
	oscilloscope = args[16]
	toolbar.pause.radius = args[17]
	toolbar.pause.pos = (args[18], args[19] - screen_height + toolbar_height)
	toolbar.pause.speed = args[20]
	toolbar.pause.angle = args[21]
	toolbar.pause.outer = args[22]
	toolbar.pause.inner = args[23]
	player.paused = args[24]
	player.flash_o = args[25]
	player.amp = args[26]
	player.active = args[27]
	progress.alpha = args[28]
	progress.select_colour = args[29], args[30], args[31]
	progress.num = args[33]
	toolbar.editor = args[34]
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
		rainbow = quadratic_gradient((length, width), pc() / 2, curve=0.03125)
		DISP.blit(
			rainbow,
			(pos[0] - width // 2 + xv, pos[1] - width // 2),
			(xv, 0, length - xv, width),
			special_flags=pygame.BLEND_RGB_MULT,
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
		rainbow = quadratic_gradient((length, width), curve=0.03125)
		DISP.blit(
			rainbow,
			(pos[0] - width // 2, pos[1] - width // 2),
			(0, 0, xv, width),
			special_flags=pygame.BLEND_RGB_MULT,
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
		c = (toolbar.pause.inner, lum, lum)
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
			(pos[0] - rad, pos[1] - rad, rad * 4 // 5, rad * 2),
			3,
		)
		bevel_rectangle(
			DISP,
			col,
			(pos[0] + (rad + 3) // 5, pos[1] - rad, rad * 4 // 5, rad * 2),
			3,
		)
	globals()["osize"] = tuple(np.frombuffer(globals()["spec-size"].buf[8:16], dtype=np.uint32))
	osci_rect = (toolbar.rect[2] - 4 - progress.box, toolbar.rect[3] - toolbar_height + 4) + osize
	locks = globals()["spec-locks"].buf
	if oscilloscope and locks[1] != 255:
		try:
			surf = player.osci
			if surf.get_size() != osize:
				raise AttributeError
		except AttributeError:
			globals()["osci-mem"] = multiprocessing.shared_memory.SharedMemory(
				name=f"Miza-Player-{hwnd}-osci-mem",
			)
			length = osize[0] * osize[1] * 3
			surf = player.osci = pygame.image.frombuffer(globals()["osci-mem"].buf[:length], osize, "BGR")
			surf.set_colorkey((0, 0, 0))
		for i in range(16):
			if locks[1] <= 0:
				break
			time.sleep(0.004)
		else:
			print(f"Oscilloscope lock expired ({i + 1}).")
		locks[1] += 1
		try:
			blit_complex(
				DISP,
				surf,
				osci_rect[:2],
			)
		except:
			raise
		finally:
			locks[1] = 0
	else:
		if oscilloscope:
			c = (255, 0, 0)
		else:
			c = (127, 0, 127)
		y = osci_rect[1] + osci_rect[3] // 2
		pygame.draw.line(
			DISP,
			c,
			(osci_rect[0], y),
			(osci_rect[0] + osci_rect[2], y)
		)
	if player.flash_o > 0:
		bevel_rectangle(
			DISP,
			(191,) * 3,
			osci_rect,
			4,
			alpha=player.flash_o * 8 - 1,
		)
	if in_rect(mpos, osci_rect):
		bevel_rectangle(
			DISP,
			(191,) * 3,
			osci_rect,
			4,
			filled=False,
		)
	if not toolbar.editor and toolbar.colour:
		bsize = min(40, toolbar_height // 3)
		s = f"{time_disp(player.pos)}/{time_disp(player.end)}"
		c = high_colour(toolbar.colour)
		message_display(
			s,
			min(24, toolbar_height // 3),
			(toolbar.rect[2] - 8 - bsize, toolbar.rect[3]),
			surface=DISP,
			align=2,
			font="Comic Sans MS",
			colour=c,
		)
		fut = render_spinnies()
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
			)
