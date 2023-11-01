import os, sys, subprocess, time, concurrent.futures

from install_update_p import *

print = lambda *args, sep=" ", end="\n": sys.stdout.write(str(sep).join(map(str, args)) + end)
from concurrent.futures import thread, _base

last_work = {}
last_used = {}

def _worker(executor_reference, work_queue, initializer, initargs):
	if initializer is not None:
		try:
			initializer(*initargs)
		except BaseException:
			_base.LOGGER.critical('Exception in initializer:', exc_info=True)
			executor = executor_reference()
			if executor is not None:
				executor._initializer_failed()
			return
	try:
		i = thread.threading.get_ident()
		last_used[i] = time.time()
		while True:
			t = time.time()
			work_item = work_queue.get(block=True)
			if work_item is not None:
				tup = (i, work_item.fn, work_item.args)
				last_work[i] = tup
				last_used[i] = t
				# print(tup)
				work_item.run()
				# Delete references to object. See issue16284
				del work_item

				# attempt to increment idle count
				executor = executor_reference()
				if executor is not None:
					executor._idle_semaphore.release()
				del executor
				continue

			executor = executor_reference()
			# Exit if:
			#   - The interpreter is shutting down OR
			#   - The executor that owns the worker has been collected OR
			#   - The executor that owns the worker has been shutdown.
			if getattr(_base, "_shutdown", None) or executor is None or executor._shutdown:
				# Flag the executor as shutting down as early as possible if it
				# is not gc-ed yet.
				if executor is not None:
					executor._shutdown = True
				# Notice other workers
				work_queue.put(None)
				return
			del executor
		print("Thread", i, "exited.")
	except BaseException:
		_base.LOGGER.critical('Exception in worker', exc_info=True)

def _adjust_thread_count(self):
	# if idle threads are available, don't spin new threads
	try:
		if self._idle_semaphore.acquire(timeout=0):
			return
	except AttributeError:
		pass

	# When the executor gets lost, the weakref callback will wake up
	# the worker threads.
	def weakref_cb(_, q=self._work_queue):
		q.put(None)

	num_threads = len(self._threads)
	if num_threads < self._max_workers:
		thread_name = '%s_%d' % (self._thread_name_prefix or self, num_threads)
		t = thread.threading.Thread(
			name=thread_name,
			target=_worker,
			args=(
				thread.weakref.ref(self, weakref_cb),
				self._work_queue,
				self._initializer,
				self._initargs,
			),
			daemon=True,
		)
		t.start()
		self._threads.add(t)
		thread._threads_queues[t] = self._work_queue

concurrent.futures.ThreadPoolExecutor._adjust_thread_count = _adjust_thread_count

exc = concurrent.futures.ThreadPoolExecutor(max_workers=96)
submit = exc.submit
def _settimeout(*args, timeout=0, **kwargs):
	if timeout > 0:
		time.sleep(timeout)
	args[0](*args[1:], **kwargs)
settimeout = lambda *args, **kwargs: submit(_settimeout, *args, **kwargs)

from rainbow_print import *

class MultiAutoImporter:

	class ImportedModule:

		def __init__(self, module, pool, _globals, start=True):
			object.__setattr__(self, "__module", module)
			if start:
				fut = pool.submit(__import__, module)
				object.__setattr__(self, "__fut", fut)
			object.__setattr__(self, "__globals", _globals)

		def __getattr__(self, k):
			m = self.force()
			return getattr(m, k)

		def __setattr__(self, k, v):
			m = self.force()
			return setattr(m, k, v)

		def force(self):
			module = object.__getattribute__(self, "__module")
			_globals = object.__getattribute__(self, "__globals")
			try:
				_globals[module] = m = object.__getattribute__(self, "__fut").result()
			except AttributeError:
				_globals[module] = m = __import__(module)
			return m

	def __init__(self, *args, pool=None, _globals=None):
		self.pool = pool
		if not _globals:
			_globals = globals()
		args = " ".join(args).replace(",", " ").split()
		if not pool:
			_globals.update((k, __import__(k)) for k in args)
		else:
			futs = []
			for arg in args:
				futs.append(self.ImportedModule(arg, pool, _globals, start=len(futs) < 3))
			self.futs = futs
			_globals.update(zip(args, futs))
			submit(self.scan)

	def scan(self):
		for i, sub in enumerate(self.futs):
			object.__getattribute__(sub, "__fut").result()
			for j in range(i + 1, len(self.futs)):
				try:
					object.__getattribute__(self.futs[j], "__fut")
				except AttributeError:
					module = object.__getattribute__(self.futs[j], "__module")
					fut = self.pool.submit(__import__, module)
					# sys.stderr.write(str(fut) + "\n")
					object.__setattr__(self.futs[j], "__fut", fut)
					break

importer = MultiAutoImporter(
	"PIL, numpy, traceback",
	"cffi, fractions, random, itertools, collections, re, colorsys, ast, contextlib, pyperclip, zipfile",
	"socket, io, pickle, hashlib, base64, urllib, weakref, orjson, copy, json",
	pool=exc,
	_globals=globals(),
)
import requests, psutil, ctypes, struct, io
import soundcard as sc

async_wait = lambda: time.sleep(0.005)
sys.setswitchinterval(0.008)
utc = time.time


hwaccel = "d3d11va" if os.name == "nt" else "auto"
ffmpeg = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
ffprobe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
sox = "sox.exe" if os.name == "nt" else "sox"
org2xm = "org2xm.exe" if os.name == "nt" else "org2xm"

collections2f = "misc/collections2.tmp"
try:
	if not os.path.exists("sndlib/ffmpeg.exe"):
		raise FileNotFoundError
	update_collections = utc() - os.path.getmtime(collections2f) >= 300
except FileNotFoundError:
	update_collections = True

hasmisc = os.path.exists("misc")
pyv = sys.version_info[1]
print_exc = traceback.print_exc


is_url = lambda url: "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")

downloader = concurrent.futures.Future()
lyrics_scraper = concurrent.futures.Future()
def import_audio_downloader():
	try:
		audio_downloader = __import__("audio_downloader")
		globals()["ytdl"] = ytdl = audio_downloader.AudioDownloader()
		downloader.set_result(ytdl)
		lyrics_scraper.set_result(audio_downloader.get_lyrics)
	except Exception as ex:
		print_exc()
		downloader.set_exception(ex)
		lyrics_scraper.set_exception(ex)


if os.name == "nt":
	user32 = ctypes.windll.user32
	shell32 = ctypes.windll.shell32
	user32.SetProcessDPIAware()


import pyglet
import pyglet.media.mediathreads
from pyglet.gl import *
from pyglet.math import *
pyglet.options["debug_gl"] = False
pyglet.options["debug_win32"] = False
pyglet.options["debug_lib"] = False
pyglet.options["debug_trace"] = False
pyglet.options["debug_font"] = False
pyglet.options["debug_graphics_batch"] = False
# pyglet.media.mediathreads.PlayerWorkerThread._nap_time = 1
import math
from math import *


def astype(obj, types, *args, **kwargs):
	if isinstance(types, tuple):
		tl = tuple(t for t in types if isinstance(t, type))
	else:
		tl = None
	tl = tl or types
	try:
		if not isinstance(obj, tl):
			raise TypeError
	except TypeError:
		t = types[0] if isinstance(types, tuple) else types
		if callable(t):
			return t(obj, *args, **kwargs)
		return t
	return obj

def as_str(s):
	if type(s) in (bytes, bytearray, memoryview):
		return bytes(s).decode("utf-8", "replace")
	return str(s)

def fuzzy_substring(sub, s, match_start=False, match_length=True):
	if not match_length and s in sub:
		return 1
	if s.startswith(sub):
		return len(sub) / len(s) * 2
	match = 0
	if not match_start or sub and s.startswith(sub[0]):
		found = [0] * len(s)
		x = 0
		for i, c in enumerate(sub):
			temp = s[x:]
			if temp.startswith(c):
				if found[x] < 1:
					match += 1
					found[x] = 1
				x += 1
			elif c in temp:
				y = temp.index(c)
				x += y
				if found[x] < 1:
					found[x] = 1
					match += 1 - y / len(s)
				x += 1
			else:
				temp = s[:x]
				if c in temp:
					y = temp.rindex(c)
					if found[y] < 1:
						match += 1 - (x - y) / len(s)
						found[y] = 1
					x = y + 1
		if len(sub) > len(s) and match_length:
			match *= len(s) / len(sub)
	# ratio = match / len(s)
	ratio = max(0, match / len(s))
	return ratio

def json_default(obj):
	if isinstance(obj, (deque, alist, np.ndarray)):
		return list(obj)
	if isinstance(obj, (bytes, bytearray)):
		return as_str(obj)
	if callable(obj):
		return
	raise TypeError(obj)

safe_filenames = {ord(c): "\x7f" for c in r'\/:*?"<>|'}


if update_collections:

	def add_to_path():
		p = os.path.abspath("sndlib")
		PATH = set(i.rstrip("/\\") for i in os.getenv("PATH", "").split(os.pathsep) if i)
		if p not in PATH:
			print(f"Adding {p} to PATH...")
			PATH.add(p)
			s = os.pathsep.join(PATH)
			subprocess.run(["setx", "path", s])
			os.environ["PATH"] = s

	print("Verifying FFmpeg, SoX, and Org2XM installations...")
	try:
		if not os.path.exists("sndlib/ffmpeg.exe") or os.path.getsize("sndlib/ffmpeg.exe") != 381440:
			raise FileNotFoundError
		subprocess.Popen(ffmpeg, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		subprocess.Popen(sox, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		subprocess.Popen(org2xm, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except FileNotFoundError:
		url = "https://mizabot.xyz/d/BfUX0q45bA"
		subprocess.run((sys.executable, "-O", "downloader.py", "-threads", "12", url, "sndlib.zip"), cwd="misc")
		with zipfile.ZipFile("misc/sndlib.zip", "r") as z:
			z.extractall()
		os.remove("misc/sndlib.zip")
		add_to_path()
		print("Sound library extraction complete.")


class seq(io.BufferedRandom, collections.abc.MutableSequence, contextlib.AbstractContextManager):

	BUF = 262144
	iter = None
	mode = "rb+"

	def __init__(self, obj, filename=None, buffer_size=None):
		if buffer_size:
			self.BUF = buffer_size
		self.closer = getattr(obj, "close", None)
		self.high = 0
		self.finished = False
		if isinstance(obj, io.IOBase):
			if isinstance(obj, io.BytesIO):
				self.data = obj
			elif hasattr(obj, "getbuffer"):
				self.data = io.BytesIO(obj.getbuffer())
			else:
				obj.seek(0)
				self.data = io.BytesIO(obj.read())
				obj.seek(0)
			self.finished = True
		elif isinstance(obj, bytes) or isinstance(obj, bytearray) or isinstance(obj, memoryview):
			self.data = io.BytesIO(obj)
			self.high = len(obj)
			self.finished = True
		elif isinstance(obj, collections.abc.Iterable):
			self.iter = iter(obj)
			self.data = io.BytesIO()
		elif getattr(obj, "iter_content", None):
			self.iter = obj.iter_content(self.BUF)
			self.data = io.BytesIO()
		else:
			raise TypeError(f"a bytes-like object is required, not '{type(obj)}'")
		self.filename = filename
		self.buffer = {}
		self.pos = 0
		self.limit = None

	seekable = lambda self: True
	readable = lambda self: True
	writable = lambda self: False
	isatty = lambda self: False
	flush = lambda self: None
	tell = lambda self: self.pos

	def seek(self, pos=0):
		self.pos = pos

	def read(self, size=None):
		out = self.peek(size)
		self.pos += len(out)
		return out

	def peek(self, size=None):
		if not size:
			if self.limit is not None:
				return self[self.pos:self.limit]
			return self[self.pos:]
		if self.limit is not None:
			return self[self.pos:min(self.pos + size, self.limit)]
		return self[self.pos:self.pos + size]

	def truncate(self, limit=None):
		self.limit = limit

	def fileno(self):
		raise OSError

	def __getitem__(self, k):
		if self.finished:
			return self.data.getbuffer()[k]
		if type(k) is slice:
			start = k.start or 0
			stop = k.stop or inf
			step = k.step or 1
			rev = step < 0
			if rev:
				start, stop, step = stop + 1, start + 1, -step
			curr = start // self.BUF * self.BUF
			out = deque()
			out.append(self.load(curr))
			curr += self.BUF
			while curr < stop:
				temp = self.load(curr)
				if not temp:
					break
				out.append(temp)
				curr += self.BUF
			b = memoryview(b"".join(out))
			b = b[start % self.BUF:]
			if isfinite(stop):
				b = b[:stop - start]
			if step != 1:
				b = b[::step]
			if rev:
				b = b[::-1]
			return b
		base = k // self.BUF
		with suppress(KeyError):
			return self.load(base)[k % self.BUF]
		raise IndexError("seq index out of range")

	def __str__(self):
		if self.filename is None:
			return str(self.data)
		if self.filename:
			return f"<seq name='{self.filename}'>"
		return f"<seq object at {hex(id(self))}"

	def __iter__(self):
		i = 0
		while True:
			x = self[i]
			if x:
				yield x
			else:
				break
			i += 1

	def __getattribute__(self, k):
		if k in ("name", "filename"):
			try:
				return object.__getattribute__(self, "filename")
			except AttributeError:
				k = "name"
		else:
			try:
				return object.__getattribute__(self, k)
			except AttributeError:
				pass
		return object.__getattribute__(self.data, k)

	close = lambda self: self.closer() if self.closer else None
	__exit__ = lambda self, *args: self.close()

	def load(self, k):
		if self.finished:
			return self.data.getbuffer()[k:k + self.BUF]
		with suppress(KeyError):
			return self.buffer[k]
		seek = getattr(self.data, "seek", None)
		if seek:
			if self.iter is not None and k + self.BUF >= self.high:
				out = deque()
				try:
					while k + self.BUF >= self.high:
						temp = next(self.iter)
						if not temp:
							raise StopIteration
						out.append(temp)
						self.high += len(temp)
				except StopIteration:
					out.appendleft(self.data.getbuffer())
					self.data = io.BytesIO(b"".join(out))
					self.finished = True
					return self.data.getbuffer()[k:k + self.BUF]
				out.appendleft(self.data.getbuffer())
				self.data = io.BytesIO(b"".join(out))
			self.buffer[k] = b = self.data.getbuffer()[k:k + self.BUF]
			return b
		try:
			while self.high < k:
				temp = next(self.data)
				if not temp:
					raise StopIteration
				if self.high in self.buffer:
					self.buffer[self.high] += temp
				else:
					self.buffer[self.high] = temp
				self.high += self.BUF
		except StopIteration:
			self.data = io.BytesIO(b"".join(self.buffer.values()))
			self.finished = True
			return self.data.getbuffer()[k:k + self.BUF]
		return self.buffer.get(k, b"")

def header():
	return {
		"User-Agent": f"Mozilla/5.{random.randint(1, 9)}",
		"DNT": "1",
		"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
	}


def mixer_verify(fut):
	try:
		fut.result(timeout=1)
		mixer.stdin.flush()
	except:
		print_exc()
		if mixer and mixer.is_running():
			mixer.kill()

def state(i):
	s = f"~state {int(i)}\n".encode("utf-8")
	fut = submit(mixer.stdin.write, s)
	submit(mixer_verify, fut)

def clear():
	s = f"~clear\n".encode("utf-8")
	fut = submit(mixer.stdin.write, s)
	submit(mixer_verify, fut)

def drop(i):
	s = f"~drop {i}\n".encode("utf-8")
	fut = submit(mixer.stdin.write, s)
	submit(mixer_verify, fut)

laststart = set()
def mixer_submit(s, force, debug):
	if not mixer.is_running():
		raise RuntimeError(mixer.stdout.read())
	if force < 2:
		while mixer.lock:
			mixer.lock.result()
	if not force:
		# A special rate limit system that will skip requests spammed too fast, but will allow the last one after a delay
		ts = pc()
		if laststart:
			diff = ts - min(laststart)
			if diff < 0.4:
				delay = 0.4 - diff
				laststart.add(ts)
				time.sleep(delay)
				if ts < max(laststart):
					return
			laststart.clear()
		laststart.add(ts)
	mixer.lock = concurrent.futures.Future()
	try:
		if not isinstance(s, (bytes, memoryview)):
			s = as_str(s)
			if not s.endswith("\n") and len(s) < 2048:
				s += "\n"
			if debug:
				sys.stdout.write(s)
			s = s.encode("utf-8")
		if not s.endswith(b"\n"):
			s += b"\n"
		fut = submit(mixer.stdin.write, s)
		fut.result(timeout=1)
		mixer.stdin.flush()
	except:
		temp, mixer.lock = mixer.lock, None
		temp.set_result(None)
		if mixer and mixer.is_running():
			mixer.kill()
		raise
	temp, mixer.lock = mixer.lock, None
	temp.set_result(None)

asettings = cdict(
	volume=(0, 5),
	speed=(-1, 3),
	pitch=(-12, 12, 0.5),
	pan=(0, 4),
	bassboost=(0, 7),
	reverb=(0, 3),
	compressor=(0, 6),
	chorus=(0, 5),
	nightcore=(-6, 18, 0.5),
)
audio_default = cdict(
	volume=1,
	speed=1,
	pitch=0,
	pan=1,
	bassboost=0,
	reverb=0,
	compressor=0,
	chorus=0,
	nightcore=0,
)
control_default = cdict(
	shuffle=1,
	loop=1,
	silenceremove=0,
	unfocus=0,
	subprocess=1,
	presearch=0,
	preserve=0,
	blur=0,
	transparency=0,
	ripples=1,
	autobackup=0,
	autoupdate=0,
	lyrics_size=16,
	playlist_sync="",
	playlist_files=0,
	playlist_size=0,
)
control_default["gradient-vertices"] = (4, 3, 3)
control_default["spiral-vertices"] = [24, 1]
editor_default = cdict(
	mode="I",
	freeform=False,
	bounded=False,
	instrument=False,
	duration=False,
	autoswap=False,
)
insettings = cdict(
	unison_count=(1, 8, 1),
	unison_depth=(0, 2),
	unison_phase=(0, 1),
	comb_delay=(0, 2),
	comb_amplitude=(0, 1),
)
default_instrument_opt = [
	1,  # unison-count
	0.5,# unison depth
	0,  # unison phase
	0,  # delay
	0,  # reverb
]
sysettings = cdict(
	shape=(0, 3),
	amplitude=(-1, 8),
	phase=(0, 1),
	pulse=(0, 1),
	shrink=(0, 1),
	exponent=(0, 3),
)
sasettings = cdict(
	shape=(0, 3),
	amplitude=(-1, 8),
	phase=(0, 1),
	pulse=(0, 1),
	shrink=(0, 1),
	exponent=(0, 3),
)
synth_default = cdict(
	type="synth",
	shape=0,
	amplitude=1,
	phase=0,
	pulse=0.5,
	shrink=0,
	exponent=1,
)
aediting = dict.fromkeys(asettings)
syediting = dict.fromkeys(sysettings)
config = "config.json"
options = None
if os.path.exists(config):
	try:
		with open(config, "r", encoding="utf-8") as f:
			options = json.load(f)
	except:
		print_exc()
if options:
	options = cdict(options)
else:
	options = cdict(
		screensize=[1280, 720],
		sidebar_width=256,
		toolbar_height=80,
		audio=cdict(audio_default),
		control=cdict(control_default),
		spectrogram=1,
		oscilloscope=1,
	)
screensize = options.screensize
if screensize[0] < 320:
	screensize[0] = 320
if screensize[1] < 240:
	screensize[1] = 240
if options.sidebar_width < 144:
	options.sidebar_width = 144
if options.toolbar_height < 64:
	options.toolbar_height = 64

options.audio = audio_default.union(options.get("audio") or ())
options.control = control_default.union(options.get("control") or ())
options.editor = editor_default.union(options.get("editor") or ())
orig_options = copy.deepcopy(options)

if not isinstance(options.control.get("spiral-vertices"), list):
	options.control["spiral-vertices"] = [options.control["spiral-vertices"], 1]


import soundcard as sc
CFFI = cffi.FFI()

DEVICE = None
OUTPUT_DEVICE = ""
reset_menu = lambda *args, **kwargs: None

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "TRUE"
import pygame

def start_mixer(devicename=None):
	global mixer, mixer_server
	pid = os.getpid()
	if "mixer" in globals() and mixer and mixer.is_running():
		try:
			mixer.kill()
		except psutil.NoSuchProcess:
			pass
	restarting = "DISP" in globals()
	if restarting and getattr(pygame, "closed", None):
		return
	if not restarting:
		pygame.display.init()
		start_display()
		mixer_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		mixer_server.bind(("127.0.0.1", pid & 32767 | 32768))
		mixer_server.listen(0)
	else:
		print("Restarting mixer subprocess...")
	# print(mixer_server)
	if hasmisc and not restarting:
		w, h = pygame.display.list_modes()[0]
		import multiprocessing.shared_memory
		globals()["multiprocessing"] = multiprocessing
		# Stores computed LDFT buckets to render as spectrogram
		globals()["spec-mem"] = multiprocessing.shared_memory.SharedMemory(
			name=f"Miza-Player-{pid}-spec-mem",
			create=True,
			size=8192,
		)
		# Stores computed PCM packets to render as oscilloscope
		globals()["osci-mem"] = multiprocessing.shared_memory.SharedMemory(
			name=f"Miza-Player-{pid}-osci-mem",
			create=True,
			size=12801,
		)
		# 0: minimised | unfocused
		# 6~8: barcount
		# 8~12, 12~16: osci width, osci height
		# 16~20, 20~24: spec width, spec height
		globals()["stat-mem"] = multiprocessing.shared_memory.SharedMemory(
			name=f"Miza-Player-{pid}-stat-mem",
			create=True,
			size=4096,
		)
	mixer = psutil.Popen(
		(sys.executable, "-O", "misc/mixer.py"),
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		bufsize=65536,
	)
	mixer.state = lambda i=0: state(i)
	mixer.clear = lambda: clear()
	mixer.drop = lambda i=0: drop(i)
	mixer.submit = lambda s, force=True, debug=False: submit(mixer_submit, s, force, debug)
	mixer.lock = None
	try:
		mixer.stdin.write(b"~init\n")
		mixer.stdin.flush()
		while True:
			fut = submit(mixer.stderr.readline)
			temp = fut.result(timeout=8).strip().decode("ascii")
			if temp.startswith("~"):
				if temp == "~I":
					break
				print(temp)
			else:
				raise RuntimeError(temp)
		if temp != "~I":
			print(temp)
			mixer.kill()
			raise RuntimeError(f"Unexpected response from mixer {mixer.stderr.read()}")
		mixer.client, mixer.addr = mixer_server.accept()
		# mixer.client = mixer_server
		if hasmisc:
			s = []
			d = options.audio.copy()
			d.update(options.control)
			j = orjson.dumps(d).decode("utf-8")
			s.append(f"~setting #{j}\n")
			s.append(f"~setting spectrogram {options.setdefault('spectrogram', 1) - 1}\n")
			s.append(f"~setting oscilloscope {options.setdefault('oscilloscope', 1)}\n")
			s.append(f"~setting insights {options.setdefault('insights', 1)}\n")
			if devicename:
				s.append(f"~output {devicename}\n")
			else:
				s.append(f"~output {OUTPUT_DEVICE}\n")
			mixer.stdin.write("".join(s).encode("utf-8"))
			try:
				mixer.stdin.flush()
			except OSError:
				print(mixer.stderr.read(), end="")
				raise
			while mixer.lock:
				mixer.lock.result()
			mixer.new = True
	except:
		print_exc()
		if mixer and mixer.is_running():
			mixer.kill()
	return mixer


def translate_pos(rect, pos):
	pos = astype(pos or [0, 0], list)
	pos[1] = DISP.height - rect[3] - pos[1]
	return pos

class KeyList(list):

	def __setitem__(self, k, v):
		if isinstance(k, int):
			k &= -1073741825
		return super().__setitem__(k, v)

	def __getitem__(self, k):
		if isinstance(k, int):
			k &= -1073741825
		return super().__getitem__(k)

	def reset(self):
		self[:] = [False] * len(self)


class AreaGroup(pyglet.graphics.OrderedGroup):

	def __init__(self, order, parent=None, area=None):
		super(self.__class__, self).__init__(parent)
		self.order = order
		self.area = area

	def __lt__(self, other):
		if isinstance(other, pyglet.graphics.OrderedGroup):
			return self.order < other.order
		return super(self.__class__, self).__lt__(other)

	def __eq__(self, other):
		return (
			self.__class__ is other.__class__ and
			self.order == other.order and
			self.parent == other.parent and
			self.area == other.area
		)

	def __hash__(self):
		return hash((self.order, self.parent, self.area))

	def __repr__(self):
		return '%s(%d)' % (self.__class__.__name__, self.order)

	def set_state(self):
		if self.area:
			area = [self.area[0], DISP.height - self.area[3] - self.area[1], self.area[2], self.area[3]]
			glEnable(GL_SCISSOR_TEST)
			try:
				glScissor(*area)
			except pyglet.gl.lib.GLException:
				glScissor(0, 0, 1, 1)
		else:
			glDisable(GL_SCISSOR_TEST)

	def unset_state(self):
		glDisable(GL_SCISSOR_TEST)

def set_state(self):
	glEnable(self.texture.target)
	glBindTexture(self.texture.target, self.texture.id)
	if self.blend_src in (GL_ONE, GL_SRC_ALPHA) and self.blend_dest == GL_ZERO:
		glDisable(GL_BLEND)
	else:
		glEnable(GL_BLEND)
		glBlendFunc(self.blend_src, self.blend_dest)

unset_state = lambda self: glDisable(self.texture.target)

pyglet.sprite.SpriteGroup.set_state = lambda self: set_state(self)
pyglet.sprite.SpriteGroup.unset_state = unset_state

# def _eq(self, other):
	# return (
		# other.__class__ is self.__class__ and
		# self.parent is other.parent and
		# self.texture.target == other.texture.target and
		# self.texture.id == other.texture.id and
		# self.blend_src == other.blend_src and
		# self.blend_dest == other.blend_dest and
		# getattr(self, "area", None) == getattr(other, "area", None)
	# )
# def _hash(self):
	# return hash((
		# id(self.parent),
		# self.texture.id, self.texture.target,
		# self.blend_src, self.blend_dest,
		# getattr(self, "area", None)
	# ))
# pyglet.sprite.SpriteGroup.__eq__ = lambda self, other: _eq(self, other)
# pyglet.sprite.SpriteGroup.__hash__ = lambda self: _hash(self)
pyglet.image.glFlush = lambda: None
pyglet.graphics.glFlush = lambda: None
pyglet.image.glFinish = lambda: None
pyglet.graphics.glFinish = lambda: None

def delete_sprite(self):
	if self._vertex_list is not None:
		self._vertex_list.delete()
		self._vertex_list = None

pyglet.shapes._ShapeBase.delete = lambda self: delete_sprite(self)

# def proj_set(self, window_width, window_height, viewport_width, viewport_height):
	# gl.glViewport(0, 0, max(1, viewport_width), max(1, viewport_height))
	# gl.glMatrixMode(gl.GL_PROJECTION)
	# gl.glLoadIdentity()
	# gl.glOrtho(0, max(1, window_width - 1), 0, max(1, window_height), -1, 1)
	# gl.glMatrixMode(gl.GL_MODELVIEW)
# pyglet.window.Projection2D.set = lambda *args: proj_set(*args)

class MultiSpriteGroup(pyglet.graphics.Group):

	def __init__(self, blend_src, blend_dest, texture=None, width=None, draw=None, parent=None):
		super().__init__(parent)
		self.texture = texture
		self.width = width
		self.blend_src = blend_src
		self.blend_dest = blend_dest
		self.draw = draw

	def set_state(self):
		if self.width is not None:
			glLineWidth(self.width)
		if self.texture is not None:
			glEnable(self.texture.target)
			glBindTexture(self.texture.target, self.texture.id)
		glEnable(GL_BLEND)
		glBlendFunc(self.blend_src, self.blend_dest)
		if self.draw is not None:
			self.draw()

	def unset_state(self):
		if self.texture is not None:
			glDisable(self.texture.target)

	def __eq__(self, other):
		return self is other

	def __hash__(self):
		return id(self)

class MultiSprite:

	def __init__(
		self, img,
		dests, sources, colours,
		blend_src=GL_SRC_ALPHA, blend_dest=GL_ONE_MINUS_SRC_ALPHA,
		batch=None, group=None,
	):
		if batch is not None:
			self._batch = batch

		self._dests = tuple(dests)
		self.dests = np.array(dests, dtype=np.float32)
		self._sources = tuple(sources)
		self.sources = np.array(sources, dtype=np.float32)

		self._texture = img.get_texture()
		self._group_1 = group
		self.init(blend_src, blend_dest)
		self._vertex_list = self._batch.add(0, GL_POINTS, self._group, "v2f")

	def init(self, blend_src=None, blend_dest=None):
		if blend_src is None:
			blend_src = self._group.blend_src
		if blend_dest is None:
			blend_dest = self._group.blend_dest
		self._group = MultiSpriteGroup(
			blend_src, blend_dest,
			texture=self._texture,
			draw=self.draw,
			parent=self._group_1,
		)

	def delete(self):
		if self._vertex_list is not None:
			self._vertex_list.delete()
		self._vertex_list = None
		self._texture = None
		self.draw = None
		if self._group is not None:
			self._group.draw = None
		self._group = None

	__del__ = delete

	def _set_texture(self, texture):
		if texture.id is not self._texture.id:
			self._texture = texture
			self.init()

	@property
	def batch(self):
		return self._batch

	@batch.setter
	def batch(self, batch):
		if self._batch == batch:
			return
		if batch is not None and self._batch is not None:
			self._batch.migrate(self._vertex_list, GL_POINTS, self._group, batch)
			self._batch = batch
		else:
			self._vertex_list.delete()
			self._batch = batch
			self._create_vertex_list()

	@property
	def group(self):
		return self._group.parent

	@group.setter
	def group(self, group):
		if self._group.parent == group:
			return
		self.init(self._group.blend_src, self._group.blend_dest)
		if self._batch is not None:
			self._batch.migrate(
				self._vertex_list,
				GL_QUADS,
				self._group,
				self._batch,
			)

	@property
	def image(self):
		return self._texture

	@image.setter
	def image(self, img):
		self._set_texture(img.get_texture())

class WidthShapeGroup(pyglet.graphics.Group):

	def __init__(self, blend_src, blend_dest, width=1, parent=None):
		super().__init__(parent)
		self.blend_src = blend_src
		self.blend_dest = blend_dest
		self.width = width

	def set_state(self):
		glEnable(GL_BLEND)
		glLineWidth(self.width)
		glDisable(GL_LINE_SMOOTH)
		glBlendFunc(self.blend_src, self.blend_dest)

	def unset_state(self):
		pass

	def __eq__(self, other):
		return (
			other.__class__ is self.__class__ and
			self.width == getattr(other, "width", None) and
			self.parent is other.parent and
			self.blend_src == other.blend_src and
			self.blend_dest == other.blend_dest
		)

	def __hash__(self):
		return hash((id(self.parent), self.blend_src, self.blend_dest, self.width))

class GL_Lines(pyglet.shapes.Line):

	_width = 1
	_rotation = 0
	_colours = ()

	def __init__(self, *coordinates, width=1, color=(255, 255, 255), mode=GL_LINES, batch=None, group=None, transparent=False):
		self._coordinates = coordinates
		self._width = width
		self._rgb = color

		self._mode = mode
		self._batch = batch or pyglet.graphics.Batch()
		self.group = group
		self.init(width=True, transparent=transparent)

	def init(self, width=True, transparent=False):
		if width:
			self._group = WidthShapeGroup(
				GL_SRC_ALPHA,
				GL_ONE_MINUS_SRC_ALPHA if transparent else GL_ZERO,
				self._width,
				self.group,
			)
		self._vertex_list = self._batch.add(
			len(self._coordinates),
			self._mode,
			self._group,
			"v2f",
			"c4B",
		)
		self._update_position()
		self.__colours = ()
		self._update_color()

	def _update_position(self):
		if not self._visible:
			self._vertex_list.vertices[:] = (0,) * sum(len(coords) for coords in self.coordinates)
		else:
			self._vertex_list.vertices[:] = tuple(itertools.chain(*self._coordinates))

	def _update_color(self):
		colours = self._colours
		if len(colours) != len(self._vertex_list.colors):
			colours = (*self._rgb, int(self._opacity)) * len(self._coordinates)
		self._vertex_list.colors[:] = colours

	@property
	def colours(self):
		try:
			return self.__colours
		except AttributeError:
			return self._colours

	@colours.setter
	def colours(self, colours):
		self.__colours = colours
		self._colours = tuple(map(round_random, itertools.chain(*colours)))
		self._update_color()

	@property
	def mode(self):
		return self._mode

	@mode.setter
	def mode(self, mode):
		self.delete()
		self._mode = mode
		self.init()

	@property
	def width(self):
		return self._width

	@width.setter
	def width(self, width):
		self.delete()
		self._width = width
		self.init(width=True)

	@property
	def coordinates(self):
		return self._coordinates

	@coordinates.setter
	def coordinates(self, coordinates):
		coordinates = tuple(coordinates)
		if len(coordinates) == len(self._coordinates):
			self._coordinates = coordinates
			self._update_position()
			return
		self.delete()
		self._coordinates = coordinates
		self.init()

class GL_Triangles(GL_Lines):

	def __init__(self, *coordinates, color=(255, 255, 255), mode=GL_TRIANGLES, batch=None, group=None):
		self._coordinates = coordinates
		self._rotation = 0
		self._rgb = color

		self._mode = mode
		self._batch = batch or pyglet.graphics.Batch()
		self.group = group
		self._group = pyglet.shapes._ShapeGroup(
			GL_SRC_ALPHA,
			GL_ONE_MINUS_SRC_ALPHA,
			self.group,
		)
		self.init()

class Rectangle(pyglet.shapes.Rectangle):
	def __init__(self, x, y, width, height, color=(255, 255, 255), batch=None, group=None):
		self._x = x
		self._y = y
		self._width = width
		self._height = height
		self._rotation = 0
		self._rgb = color

		self._batch = batch or pyglet.graphics.Batch()
		self._group = pyglet.shapes._ShapeGroup(GL_SRC_ALPHA, GL_ZERO, group)
		self._vertex_list = self._batch.add(6, GL_TRIANGLES, self._group, 'v2f', 'c4B')
		self._update_position()
		self._update_color()

RAWS = weakref.WeakKeyDictionary()
SUBSURFS = weakref.WeakKeyDictionary()

def start_display():
	config = pyglet.gl.Config(sample_buffers=0, samples=0)
	globals()["FLAGS"] = 0
	cls = pyglet.window.win32.Win32Window if os.name == "nt" else pyglet.window.Window
	transparent = options.control.get("transparency")
	globals()["DISP"] = cls(
		*screensize,
		config=config,
		resizable=True,
		visible=True,
		file_drops=True,
		vsync=False,
		style="transparent" if transparent else None,
	)
	DISP.groups = {}
	DISP.batches = {}
	DISP.surfaces = weakref.WeakKeyDictionary()
	DISP.sprites = weakref.WeakKeyDictionary()
	DISP.shapes = {}
	DISP.fonts = {}
	DISP.active = set()
	DISP.lastclear = True
	DISP.transparent = transparent
	print(DISP, DISP.event)

	def get_batch(i):
		try:
			return DISP.batches[i]
		except KeyError:
			batch = DISP.batches[i] = pyglet.graphics.Batch()
		return batch
	DISP.get_batch = get_batch

	def get_group(i, cls=pyglet.graphics.OrderedGroup, area=None):
		try:
			return DISP.groups[(i, cls, area)]
		except KeyError:
			if area and cls is pyglet.graphics.OrderedGroup:
				group = DISP.groups[(i, cls, area)] = AreaGroup(i, area=area)
			else:
				group = DISP.groups[(i, cls, area)] = cls(i)
		return group
	DISP.get_group = get_group

	globals()["NEW_SPRITES"] = []
	globals()["NEW_TEXTURES"] = []

	def ensure_texture(surf, texture=True):
		if surf not in DISP.surfaces:
			im = pyg2pgl(surf)
			if texture:
				im = im.get_texture(rectangle=True)
			DISP.surfaces[surf] = im
			globals()["NEW_TEXTURES"].append(surf)
		return DISP.surfaces[surf]
	DISP.ensure_texture = ensure_texture

	def display_blit(source, dest=None, area=None, special_flags=0, colour=None, angle=None, scale=None, z=0, redraw=False, permanent=False, sx=None, sy=None, dest_area=None):
		flags = astype(special_flags, int)
		if dest is not None:
			dest = astype(dest, tuple)
		if source.get_width() <= 0 or source.get_height() <= 0:
			return
		if area is not None:
			area = astype(area, list)
			different = False
			if area[0] <= 0:
				area[0] = 0
			else:
				different = True
			if area[1] <= 0:
				area[1] = 0
			else:
				different = True
			if area[2] + area[0] >= source.get_width():
				area[2] = source.get_width() - area[0]
			else:
				different = True
			if area[3] + area[1] >= source.get_height():
				area[3] = source.get_height() - area[1]
			else:
				different = True
			if area[2] <= 0 or area[3] <= 0:
				return
			area = tuple(area)
			# if different:
				# try:
					# source = SUBSURFS[source][area]
				# except KeyError:
					# SUBSURFS.setdefault(source, {})[area] = source = source.subsurf(area)
		if colour is not None:
			colour = astype(map(round, colour), tuple)
		batch = get_batch(z // 128)
		group = get_group(z, area=dest_area)
		try:
			if redraw or permanent:
				raise KeyError
			if area and different:
				raise KeyError
			sprites = DISP.sprites[source]
			for sp in sprites:
				if not sp.used and getattr(sp, "flags", 0) == flags and getattr(sp, "area", None) == area:
					break
			else:
				raise KeyError
		except KeyError:
			im = DISP.ensure_texture(source, texture=not redraw)
			if redraw:
				mode = "RGBA" if source.get_flags() & pygame.SRCALPHA else "RGB"
				b = RAWS.pop(source, False)
				if not b:
					b = pygame.image.tostring(source, mode, True)
				else:
					print(source)
				row = len(mode) * source.get_width()
				if isinstance(im, pyglet.image.Texture):
					i = im.get_image_data()
					i.set_data(mode, row, b)
					im.image = i
				else:
					im.set_data(mode, row, b)
			sp = None
			if area:
				r = list(area)
				r[1] = source.get_height() - r[3] + r[1]
				# print(r)
				try:
					im = im.get_region(*r)
				except ZeroDivisionError:
					return (0,) * 4
				for sp in DISP.sprites.get(source, ()):
					if not sp.used and getattr(sp, "flags", 0) == flags and getattr(sp, "area", None) == area:
						sp.image = im
						break
				else:
					sp = None
			if flags in (0, pygame.BLEND_PREMULTIPLIED, pygame.BLEND_ALPHA_SDL2):
				if source.get_flags() & pygame.SRCALPHA:
					blend_src = GL_SRC_ALPHA
					blend_dest = GL_ONE_MINUS_SRC_ALPHA
				else:
					blend_src = GL_ONE
					blend_dest = GL_ZERO
			elif flags in (pygame.BLEND_RGB_MULT, pygame.BLEND_RGBA_MULT):
				blend_src = GL_ZERO
				blend_dest = GL_SRC_COLOR
			elif flags == pygame.BLEND_RGB_ADD:
				blend_src = GL_ONE
				blend_dest = GL_ONE
			elif flags == pygame.BLEND_RGBA_ADD:
				if source.get_flags() & pygame.SRCALPHA:
					blend_src = GL_SRC_ALPHA
					blend_dest = GL_ONE
				else:
					blend_src = GL_ONE
					blend_dest = GL_ONE
			else:
				raise NotImplementedError(flags)
			if not sp:
				sp = pyglet.sprite.Sprite(im, blend_src=blend_src, blend_dest=blend_dest, batch=batch, group=group)
				globals()["NEW_SPRITES"].append(sp)
			sp.flags = flags
			if not redraw and not permanent and not (area and different):
				DISP.sprites.setdefault(source, weakref.WeakSet()).add(sp)
			# print(id(source), dest, area, colour, z)
		else:
			# if getattr(sp, "area", None) != area:
				# im = DISP.ensure_texture(source, texture=True)
				# if area:
					# r = list(area)
					# r[1] = source.get_height() - r[3] + r[1]
					# try:
						# im = im.get_region(*r)
					# except ZeroDivisionError:
						# return (0,) * 4
				# if sp.image != im:
					# sp.image = im
				# sp.area = area
			if sp.batch != batch:
				sp.batch = batch
			if sp.group != group:
				sp.group = group
		batch.used = True
		sp.used = True
		DISP.active.add(sp)
		kwargs = cdict()
		rect = [0, 0]
		if area and different:
			rect.extend(area[2:])
		else:
			rect.extend(source.get_size())
		if scale:
			kwargs.scale = scale
			rect[2] *= scale
			rect[3] *= scale
		if sx:
			sw = sx / rect[2]
			if sw != sp.scale_x:
				kwargs.scale_x = sw
			rect[2] *= sw
		if sy:
			sh = sy / rect[3]
			if sh != sp.scale_y:
				kwargs.scale_y = sh
			rect[3] *= sh
		pos = dest[:2]
		pos = translate_pos(rect, dest)
		if colour:
			if len(colour) > 3:
				if colour[3] <= 0:
					return sp
				if 0 < colour[3] < 255 and sp.opacity != colour[3]:
					sp.opacity = colour[3]
			if sp.color != colour[:3]:
				sp.color = colour[:3]
		if angle:
			half = (rect[2] / 2, rect[3] / 2)
			if sp.image.anchor_x != half[0]:
				sp.image.anchor_x = half[0]
			if sp.image.anchor_y != half[1]:
				sp.image.anchor_y = half[1]
			pos = tuple(x + y for x, y in zip(pos, half))
			kwargs.rotation = angle * 180 / pi
		if sp.x != pos[0] or sp.y != pos[1]:
			kwargs.x, kwargs.y = pos
		if kwargs:
			if len(kwargs) > 1:
				sp.update(**kwargs)
			else:
				k, v = next(iter(kwargs.items()))
				setattr(sp, k, v)
		if permanent:
			sp.permanent = permanent
		return sp
	DISP.blit = display_blit

	def display_fill(colour, rect=None, special_flags=0, z=0):
		w, h = DISP.get_size()
		if rect is not None:
			rect = astype(rect, list)
			rect[1] = h - rect[1] - rect[3]
			rect = astype(rect, tuple)
		else:
			rect = (0, 0, w, h)
		if colour is not None:
			colour = astype(colour, tuple)
		batch = get_batch(z // 128)
		group = get_group(z)
		t = (z, rect, colour)
		if t not in DISP.shapes:
			sp = Rectangle(*rect, [round(i) for i in colour[:3]], batch=batch, group=group)
			if len(colour) > 3:
				sp.opacity = colour[3]
			DISP.shapes[t] = sp
			globals()["NEW_TEXTURES"].append(sp)
		else:
			sp = DISP.shapes[t]
		batch.used = True
		sp.used = True
		if not sp.visible:
			sp.visible = True
		DISP.active.add(sp)
	DISP.fill = display_fill

	def display_shape(func, x, y, *args, z=0, **kwargs):
		batch = get_batch(z // 128)
		group = get_group(z)
		y = DISP.height - y
		if func is pyglet.shapes.Line:
			args = (args[0] - x, DISP.height - args[1] - y) + args[2:]
		if "color" in kwargs:
			kwargs["color"] = tuple(round(c) for c in kwargs["color"])
		t = (z, func) + args + tuple(kwargs.values())
		if t not in DISP.shapes:
			if func is pyglet.shapes.Line:
				args = (args[0] + x, args[1] + y) + args[2:]
			sp = func(x, y, *args, batch=batch, group=group, **kwargs)
			DISP.shapes[t] = sp
			globals()["NEW_TEXTURES"].append(sp)
		else:
			sp = DISP.shapes[t]
			if sp.x != x:
				sp.x = x
			if sp.y != y:
				sp.y = y
			if func is pyglet.shapes.Line:
				x2 = args[0] + x
				y2 = args[1] + y
				if sp.x2 != x2:
					sp.x2 = x2
				if sp.y2 != y2:
					sp.y2 = y2
		batch.used = True
		sp.used = True
		if not sp.visible:
			sp.visible = True
		DISP.active.add(sp)
	DISP.shape = display_shape

	def hardware_font(text, size, pos, colour, background, font, alpha, align, z):
		fn = "misc/" + font + ".ttf"
		if not os.path.exists(fn) and font not in loaded_fonts:
			if font in ("Rockwell", "OpenSansEmoji"):
				print("Downloading and applying required fonts...")
				for fnt in ("Rockwell", "OpenSansEmoji"):
					loaded_fonts.add(fnt)
					submit(get_font, fnt)
		if os.path.exists(fn):
			pyglet.font.add_file(fn)
		x, y = pos[0], DISP.height - pos[1] - 1
		if background is not None:
			background = astype(background, tuple)
		if colour is not None:
			colour = astype(colour, tuple)
			if len(colour) == 3 and alpha:
				colour += (alpha,)
		batch = get_batch(z // 128)
		group = get_group(z)
		if align == 0:
			anchor_y = "top"
			anchor_x = "left"
		elif align == 1:
			anchor_y = "center"
			anchor_x = "center"
		else:
			anchor_y = "baseline"
			anchor_x = "right"
			y += size / 2
		# size = size * 4
		for sp in DISP.fonts.get(z, ()):
			if not sp.used:
				if sp.text != text:
					sp.text = text
				if sp.x != x:
					sp.x = x
				if sp.y != y:
					sp.y = y
				if sp.font_name != font:
					sp.font_name = font
				if sp.font_size != size:
					sp.font_size = size
				if sp.color != colour:
					sp.color = colour
				if sp.anchor_x != anchor_x:
					sp.anchor_x = anchor_x
				if sp.anchor_y != anchor_y:
					sp.anchor_y = anchor_y
				if sp.batch != batch:
					sp.batch = batch
				break
		else:
			sp = pyglet.text.Label(
				text,
				x=x,
				y=y,
				font_name=font,
				font_size=size,
				color=colour,
				anchor_x=anchor_x,
				anchor_y=anchor_y,
				batch=batch,
				group=group,
				dpi=72,
			)
			DISP.fonts.setdefault(z, deque()).append(sp)
			globals()["NEW_TEXTURES"].append(sp)
		batch.used = True
		sp.used = True
		DISP.active.add(sp)
	DISP.hardware_font = hardware_font

	globals()["CURR_FLAGS"] = 0
	DISP.get_at_positions = {}
	def display_get(pos, force=True):
		x, y = pos
		cache = os.name != "nt"
		if not force and cache:
			try:
				return DISP.get_at_positions[(x, y)]
			except KeyError:
				pass
		if force:
			DISP.get_at_positions.clear()
		if not cache:
			try:
				hdc = DISP.hdc
			except AttributeError:
				hdc = DISP.hdc = ctypes.windll.user32.GetDC(DISP.canvas.hwnd)
			c = ctypes.windll.gdi32.GetPixel(hdc, x, y)
			col = (i & 255 for i in (c, c >> 8, c >> 16))
		else:
			col = (GLubyte * 3)(0, 0, 0)
			glReadPixels(x, DISP.height - y - 1, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, col)
		col = astype(col, tuple)
		DISP.get_at_positions[(x, y)] = col
		return col
	DISP.get_at = display_get

	def display_update():
		update_anima_parts()
		active = 0
		pops = deque()
		for sp in DISP.active:
			if getattr(sp, "permanent", False):
				active += 1
				continue
			if not getattr(sp, "used", False):
				if hasattr(sp, "batch"):
					if getattr(sp.batch, "used", None):
						sp.batch = None
						pops.append(sp)
						sp.used = False
				else:
					sp.visible = False
					pops.append(sp)
					sp.used = False
			else:
				active += 1
				sp.used = False
		if pops:
			DISP.active.difference_update(pops)
		batches = 0
		for i in sorted(DISP.batches):
			batch = DISP.batches[i]
			flags = getattr(batch, "flags", 0)
			if getattr(batch, "used", False):
				batch.draw()
				batch.used = False
				batches += 1
			elif callable(batch):
				try:
					batch()
				except:
					print_exc()
				batches += 1
			# glFlush()
		# print(f"{round(DISP.fps, 2)} FPS, {batches} batches, {len(DISP.groups)} groups, {active} sprites, {len(pops)} swapped, {len(NEW_TEXTURES)} new, {len(NEW_SPRITES)} copies, {len(DISP.sprites) + len(DISP.shapes)} cached")
		globals()["NEW_SPRITES"].clear()
		globals()["NEW_TEXTURES"].clear()
		async_wait()
		glFinish()
		if options.control.get("blur"):
			mbi = globals().get("-MBI-")
			mbs = globals().get("-MBS-")
			if mbi:
				glEnable(GL_BLEND)
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
				glEnable(mbi.target)
				glBindTexture(mbi.target, mbi.id)
				glCopyTexImage2D(
					mbi.target,
					0,
					GL_RGBA,
					0,
					0,
					DISP.width,
					DISP.height,
					0,
				)
				glDisable(mbi.target)
		DISP.flip()
		a = 0 if options.control.get("transparency") else 1
		glClearColor(0, 0, 0, 0)
		if not random.randint(0, 15):
			DISP.clear()
			DISP.lastclear = True
		else:
			DISP.lastclear = False
		if options.control.get("blur"):
			mbi = globals().get("-MBI-")
			mbs = globals().get("-MBS-")
			if not mbi or mbi.width != DISP.width or mbi.height != DISP.height:
				mbi = globals()["-MBI-"] = pyglet.image.Texture.create(
					DISP.width,
					DISP.height,
					rectangle=True,
				)
			if not mbs:
				mbs = globals()["-MBS-"] = pyglet.sprite.Sprite(mbi)
			if mbs.image != mbi:
				mbs.image = mbi
			ts = time.perf_counter()
			dur = max(1 / 60, ts - getattr(mbs, "ts", ts))
			mbs.opacity = round_random((1 / 768) ** dur * 255)
			# print(mbs.opacity)
			mbs.ts = ts
			if mbs.opacity > 0:
				mbs.draw()
	DISP.update = display_update

	class display_subsurface:

		@classmethod
		def init(cls, rect=None):
			if not rect:
				return DISP
			self = cls()
			self.rect = rect
			return self

		def blit(self, source, dest, area=None, *args, z=0, **kwargs):
			if dest:
				dest = astype(dest, list)
			else:
				dest = [0, 0]
			for i in range(2):
				dest[i] += self.rect[i]
			return DISP.blit(source, dest, area, *args, z=z, dest_area=self.rect, **kwargs)

		def fill(self, colour, rect=None, special_flags=0, z=0):
			if rect:
				rect = astype(rect, list)
				rect[0] += self.rect[0]
				rect[1] += self.rect[1]
				rect[2] = min(rect[0] + rect[2], self.rect[0] + self.rect[2]) - rect[0]
				rect[3] = min(rect[1] + rect[3], self.rect[1] + self.rect[3]) - rect[1]
			return DISP.fill(colour, rect, special_flags=special_flags, z=z)

		def __getattribute__(self, k):
			if k in ("blit", "fill", "rect"):
				return object.__getattribute__(self, k)
			return getattr(DISP, k)

	DISP.subsurf = display_subsurface.init
	DISP.subsurface = display_subsurface

	if hasmisc:
		appid = "Miza Player \x7f"
		shell32.SetCurrentProcessExplicitAppUserModelID(appid)
		icon = pyglet.image.load("misc/icon.png")
		DISP.set_icon(icon)
	DISP.set_caption("Miza Player")
	DISP.switch_to()
	if options.get("maximised"):
		DISP.maximize()
	elif options.get("screenpos"):
		x, y = options.screenpos
		w, h = DISP.get_size()
		if x + w > 0 and y + h > 0:
			DISP.set_location(x, y)
		else:
			options.pop("screenpos")
	globals()["screenpos2"] = DISP.get_location()
	globals()["screensize2"] = list(DISP.get_size())

	DISP.focused = True
	@DISP.event
	def on_activate():
		DISP.focused = True
		rect = get_window_rect()
		if screenpos2 != rect[:2] and not is_minimised():
			options.screenpos = rect[:2]
			globals()["screenpos2"] = None
	@DISP.event
	def on_deactivate():
		DISP.focused = False
	DISP.minimised = False
	@DISP.event
	def on_show():
		DISP.minimised = False
		rect = get_window_rect()
		if screenpos2 != rect[:2] and not is_minimised():
			options.screenpos = rect[:2]
			globals()["screenpos2"] = None
	@DISP.event
	def on_hide():
		DISP.minimised = True
	@DISP.event
	def on_resize(*size):
		flags = get_window_flags()
		if flags == 3:
			options.maximised = True
		else:
			options.pop("maximised", None)
			screensize2[:] = size
		screensize[:] = size
		if screensize[0] < 320:
			screensize[0] = 320
		if screensize[1] < 240:
			screensize[1] = 240
		globals().pop("rel-pos", None)
		DISP.reset = True

	DISP.mpos = [-1] * 2
	@DISP.event
	def on_mouse_motion(*mpos):
		DISP.mmoved = True
		mpos = (mpos[0], DISP.height - mpos[1] - 1)
		DISP.mpos[:] = mpos
		DISP.mouse_in = True
	DISP.mheld = [False] * 5
	DISP.mclick = list(DISP.mheld)
	DISP.mrelease = list(DISP.mheld)
	m_buttons = (pyglet.window.mouse.LEFT, pyglet.window.mouse.RIGHT, pyglet.window.mouse.MIDDLE)
	@DISP.event
	def on_mouse_press(x, y, button, modifiers):
		try:
			i = m_buttons.index(button)
		except ValueError:
			return
		DISP.mheld[i] = True
		DISP.mclick[i] = True
	@DISP.event
	def on_mouse_release(x, y, button, modifiers):
		try:
			i = m_buttons.index(button)
		except ValueError:
			return
		DISP.mheld[i] = False
		DISP.mrelease[i] = True

	DISP.kheld = KeyList([0] * len(pygame.key.get_pressed()))
	DISP.kclick = KeyList(DISP.kheld)
	DISP.krelease = KeyList(DISP.kheld)
	pygame.key.get_pressed = lambda: [bool(x) for x in DISP.kheld]
	from pyglet.window import key as K1
	import pygame.locals as K2
	for i in range(26):
		K2.__dict__[f"K_{chr(i + 97)}"] = i + 4
	for i in range(1, 11):
		K2.__dict__[f"K_{i % 10}"] = i + 29
	K2.K_ESCAPE = 41
	K2.K_BACKSPACE = 42
	K2.K_SPACE = 44
	K2.K_EQUALS = 46
	K2.K_LEFTBRACKET = 47
	K2.K_RIGHTBRACKET = 48
	K2.K_BACKSLASH = 49
	K2.K_SEMICOLON = 51
	K2.K_BACKQUOTE = 53
	K2.K_COMMA = 54
	K2.K_PERIOD = 55
	K2.K_SLASH = 56
	K2.K_DELETE = 76
	globals()["KEYMAP"] = {
		K1.EQUAL: K2.K_EQUALS,
		K1.LCOMMAND: K2.K_LCTRL,
		K1.RCOMMAND: K2.K_RCTRL,
		K1.LWINDOWS: K2.K_LSUPER,
		K1.RWINDOWS: K2.K_RSUPER,
		K1.APOSTROPHE: K2.K_QUOTE,
		K1.GRAVE: K2.K_BACKQUOTE,
		K1.BRACKETLEFT: K2.K_LEFTBRACKET,
		K1.BRACKETRIGHT: K2.K_RIGHTBRACKET,
		K1.LINEFEED: K2.K_RETURN,
	}
	for k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
		KEYMAP[getattr(K1, k)] = getattr(K2, "K_" + k.lower()) & -1073741825
	for k in "0123456789":
		KEYMAP[getattr(K1, "_" + k)] = getattr(K2, "K_" + k) & -1073741825
	for k in range(1, 16):
		KEYMAP[getattr(K1, f"F{k}")] = getattr(K2, f"K_F{k}") & -1073741825
	for k in (
		"RETURN", "SPACE", "TAB", "BACKSPACE", "INSERT", "DELETE", "MINUS",
		"SLASH", "BACKSLASH", "SEMICOLON", "COMMA", "PERIOD", "ESCAPE",
		"LEFT", "RIGHT", "UP", "DOWN", "HOME", "END", "PAGEUP", "PAGEDOWN",
		"LCTRL", "RCTRL", "LSHIFT", "RSHIFT", "LALT", "RALT", "LMETA", "RMETA",
	):
		KEYMAP[getattr(K1, k)] = getattr(K2, "K_" + k) & -1073741825

	if os.name == "nt":
		DISP.key_map = key_map = {k & -1073741825: v for k, v in {
			# mouse.LEFT: 0x1,
			# mouse.RIGHT: 0x2,
			# mouse.MIDDLE: 0x4,
			K2.K_BACKSPACE: 0x8,
			K2.K_TAB: 0x9,
			# K_CLEAR: 0xC,
			K2.K_RETURN: 0xD,
			K2.K_ESCAPE: 0x1B,
			K2.K_SPACE: 0x20,
			K2.K_LEFT: 0x25,
			K2.K_UP: 0x26,
			K2.K_RIGHT: 0x27,
			K2.K_DOWN: 0x28,
			K2.K_DELETE: 0x2E,
			K2.K_LSHIFT: 0xA0,
			K2.K_RSHIFT: 0xA1,
			K2.K_LCTRL: 0xA2,
			K2.K_RCTRL: 0xA3,
			K2.K_LALT: 0xA4,
			K2.K_RALT: 0xA5,
		}.items()}
		# print(key_map)
		key_map.update({K2.__dict__[f"K_{i}"] & -1073741825: i + 0x30 for i in range(10)})
		key_map.update({K2.__dict__[f"K_{chr(i + 32)}"] & -1073741825: i for i in range(65, 91)})
		key_map.update({K2.__dict__[f"K_F{i}"] & -1073741825: i + 0x70 - 1 for i in range(1, 16)})
		DISP.key_unmap = key_unmap = {v: k for k, v in key_map.items()}
		# print(key_map)
	KEYUNMAP = {v: k for k, v in KEYMAP.items()}

	def update_held():
		if os.name != "nt" or not any(DISP.kheld):
			return
		changed = False
		user32 = ctypes.windll.user32
		for v, s in enumerate(DISP.kheld):
			if not s:
				continue
			try:
				k = KEYUNMAP[v]
			except KeyError:
				continue
			try:
				s = user32.GetAsyncKeyState(key_map[v])
			except KeyError:
				s = True
			if not s:
				changed = True
				if v > 4:
					on_key_release(k)
				else:
					on_mouse_release(*DISP.mpos, k)
				print(k, v, DISP.kheld[v])
		return changed
	DISP.update_held = update_held

	@DISP.event
	def on_key_press(symbol, modifiers=0):
		t = pc()
		try:
			DISP.kheld[KEYMAP[symbol]] = DISP.kheld[KEYMAP[symbol]] or t
			DISP.kclick[KEYMAP[symbol]] = t
		except LookupError:
			pass
	@DISP.event
	def on_key_release(symbol, modifiers=0):
		try:
			DISP.kheld[KEYMAP[symbol]] = 0
			DISP.krelease[KEYMAP[symbol]] = pc()
		except LookupError:
			pass

	def display_update_keys():
		if os.name != "nt":
			return
		user32 = ctypes.windll.user32
		for k, s in enumerate(DISP.kheld):
			if not s:
				continue
			try:
				v = DISP.key_map[k]
			except KeyError:
				continue
			s = user32.GetAsyncKeyState(v)
			if not s:
				if v > 4:
					DISP.kheld[k] = 0
					DISP.krelease[k] = pc()
	DISP.update_keys = display_update_keys

	@DISP.event
	def on_context_lost():
		print("OpenGL context lost. Resetting menu...")
		DISP.groups.clear()
		DISP.batches.clear()
		DISP.surfaces.clear()
		DISP.sprites.clear()
		DISP.shapes.clear()
		DISP.fonts.clear()
		DISP.active.clear()
		DISP.reset = True
	@DISP.event
	def on_close():
		if DISP.kclick[K_ESCAPE]:
			return
		DISP.cLoSeD = True

	DISP.clear()

if os.name == "nt":
	psize = struct.calcsize("P")
	if psize == 8:
		win = "win_amd64"
	else:
		win = "win32"

	_pt = ctypes.wintypes.POINT()
	_ppt = ctypes.byref(_pt)
	def mouse_abs_pos():
		user32.GetCursorPos(_ppt)
		return (_pt.x, _pt.y)
	def mouse_rel_pos():
		if getattr(DISP, "mouse_in", False):
			DISP.mouse_in = False
			globals()["rel-pos"] = [x - y for x, y in zip(mouse_abs_pos(), DISP.mpos)]
		if "rel-pos" in globals():
			apos = mouse_abs_pos()
			DISP.mpos[:] = [x - y for x, y in zip(apos, globals()["rel-pos"])]
		return DISP.mpos

	class WR(ctypes.Structure):
		_fields_ = [
			("left", ctypes.c_long),
			("top", ctypes.c_long),
			("right", ctypes.c_long),
			("bottom", ctypes.c_long),
		]
	wr = WR()
	class WP(ctypes.Structure):
		_fields_ = [
			("length", ctypes.c_uint),
			("flags", ctypes.c_uint),
			("showCmd", ctypes.c_uint),
			("ptMinPosition", ctypes.c_void_p),
			("ptMaxPosition", ctypes.c_void_p),
			("rcNormalPosition", ctypes.c_void_p),
			("rcDevice", ctypes.c_void_p),
		]
	wp = WP()
	ptMinPosition = WR()
	ptMaxPosition = WR()
	rcNormalPosition = WR()
	rcDevice = WR()
	wp.length = 44
	wp.ptMinPosition = ctypes.cast(ctypes.byref(ptMinPosition), ctypes.c_void_p)
	wp.ptMaxPosition = ctypes.cast(ctypes.byref(ptMaxPosition), ctypes.c_void_p)
	wp.rcNormalPosition = ctypes.cast(ctypes.byref(rcNormalPosition), ctypes.c_void_p)
	wp.rcDevice = ctypes.cast(ctypes.byref(rcDevice), ctypes.c_void_p)

	def get_window_rect():
		user32.GetWindowRect(DISP.canvas.hwnd, ctypes.byref(wr))
		return wr.left, wr.top, wr.right - wr.left, wr.bottom - wr.top
	def get_window_flags():
		user32.GetWindowPlacement(DISP.canvas.hwnd, ctypes.byref(wp))
		return wp.showCmd
else:
	mouse_rel_pos = lambda: DISP.mpos
	get_window_rect = lambda: DISP.get_location() + DISP.get_size()
	get_window_flags = lambda: 0

if hasmisc:
	submit(import_audio_downloader)
	mixer = start_mixer()

	class MultiKey:

		__slot__ = ("keys",)

		def __init__(self, *keys):
			self.keys = keys

		def __call__(self, k):
			return any(k[i] for i in self.keys) 

		__getitem__ = __call__

	from pygame.locals import *
	CTRL = MultiKey(K_LCTRL, K_RCTRL)
	SHIFT = MultiKey(K_LSHIFT, K_RSHIFT)
	ALT = MultiKey(K_LALT, K_RALT)
else:
	mixer = cdict()

PROC = psutil.Process()


in_rect = lambda point, rect: point[0] >= rect[0] and point[0] < rect[0] + rect[2] and point[1] >= rect[1] and point[1] < rect[1] + rect[3]
in_circ = lambda point, dest, radius=1: hypot(dest[0] - point[0], dest[1] - point[1]) <= radius
def in_polygon(point, polygon):
	count = 0
	if polygon[0] != polygon[-1]:
		polygon = list(polygon)
		polygon.append(polygon[0])
	q = None
	for p in polygon:
		if q:
			if intervals_intersect((p, q), (point, (-2147483647, -2147483648))):
				count += 1
		q = p
	return count & 1

def int_rect(r1, r2):
	x1, y1, x2, y2, = r1
	x2 += x1
	y2 += y1
	x3, y3, x4, y4 = r2
	x4 += x3
	y4 += y3
	return max(x1, x3) < min(x2, x4) and max(y1, y3) < min(y2, y4)
def int_polygon(p1, p2):
	if p1[0] != p1[-1]:
		p1 = list(p1)
		p1.append(p1[0])
	if p2[0] != p2[-1]:
		p2 = list(p2)
		p2.append(p2[0])
	q = s = None
	for p in p1:
		if q:
			for r in p2:
				if s:
					if intervals_intersect((p, q), (r, s)):
						return True
				s = r
		q = p

def interval_interval_dist(line1, line2):
	if intervals_intersect(line1, line2):
		return 0
	distances = (
		point_interval_dist(line1[0], line2),
		point_interval_dist(line1[1], line2),
		point_interval_dist(line2[0], line1),
		point_interval_dist(line2[1], line1),
	)
	return min(distances)

def point_interval_dist(point, line):
	px, py = point
	x1, x2 = line[0][0], line[1][0]
	y1, y2 = line[0][1], line[1][1]
	dx = x2 - x1
	dy = y2 - y1
	if dx == dy == 0:
		return hypot(px - x1, py - y1)
	t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
	if t < 0:
		dx = px - x1
		dy = py - y1
	elif t > 1:
		dx = px - x2
		dy = py - y2
	else:
		dx = px - x1 - t * dx
		dy = py - y1 - t * dy
	return hypot(dx, dy)

def intervals_intersect(line1, line2):
	x11, y11 = line1[0]
	x12, y12 = line1[1]
	x21, y21 = line2[0]
	x22, y22 = line2[1]
	dx1 = x12 - x11
	dy1 = y12 - y11
	dx2 = x22 - x21
	dy2 = y22 - y21
	delta = dx2 * dy1 - dy2 * dx1
	if delta == 0:
		return False
	s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
	t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / -delta
	return (0 <= s <= 1) and (0 <= t <= 1)

rect_centre = lambda rect: (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
rect_points = lambda rect: (rect[:2], (rect[0] + rect[2], rect[1]), (rect[0], rect[1] + rect[3]), (rect[0] + rect[2], rect[1] + rect[3]))
point_dist = lambda p, q: hypot(p[0] - q[0], p[1] - q[1])

is_minimised = lambda: getattr(DISP, "minimised", None)
is_unfocused = lambda: not getattr(DISP, "focused", None)
get_focused = lambda: getattr(DISP, "focused", None)

proglast = (0, 0)
def taskbar_progress_bar(ratio=1, colour=0):
	if os.name != "nt":
		return
	if "shobjidl_core" not in globals():
		if win == "win32":
			spath = "misc/Shobjidl-32.dll"
		else:
			spath = "misc/Shobjidl.dll"
		try:
			globals()["shobjidl_core"] = ctypes.cdll.LoadLibrary(spath)
		except OSError:
			globals()["shobjidl_core"] = None
			print_exc()
	elif not shobjidl_core:
		return
	hwnd = DISP.canvas.hwnd
	global proglast
	if ratio <= 0 and not colour & 1 or not colour:
		ratio = colour = 0
	r = round(min(1, ratio) * 256)
	t = (r, colour)
	if t != proglast:
		proglast = t
		shobjidl_core.SetProgressState(hwnd, colour)
		if colour:
			shobjidl_core.SetProgressValue(hwnd, r, 256)


def submit_next(func, cb):
	fut = submit(func)
	fut.add_done_callback(lambda fut: cb())
	return fut

import easygui #, easygui_qt, PyQt5
# from PyQt5 import QtCore, QtWidgets
# if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
	# PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
# if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
	# PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
# easygui.__dict__.update(easygui_qt.easygui_qt.__dict__)

class cbfunc:

	exc = concurrent.futures.ThreadPoolExecutor(max_workers=1)

	def __init__(self, sub):
		self.sub = sub

	def __call__(self, *argv, args=(), **kwargs):
		DISP.kclick.reset()
		if easygui2.lock:
			return
		easygui2.lock = True
		self.cb = argv[0] if argv else None
		self.args = args
		fut = self.exc.submit(self.sub, *argv[1:], **kwargs)
		fut.add_done_callback(self.done)
		return fut

	def done(self, fut):
		easygui2.lock = False
		try:
			if callable(self.cb):
				self.cb(fut.result(), *self.args)
		except:
			print_exc()
		finally:
			DISP.kclick.reset()

easygui2 = cdict(lock=False)
for k, v in easygui.__dict__.items():
	if callable(v):
		easygui2[k] = cbfunc(v)

import PIL
from PIL import Image, ImageOps, ImageChops
Resampling = getattr(Image, "Resampling", Image)
Transpose = getattr(Image, "Transpose", Image)
Transform = getattr(Image, "Transform", Image)
np = numpy
deque = collections.deque
suppress = contextlib.suppress
d2r = pi / 180
ts_us = lambda: time.time_ns() // 1000
SR = 48000

commitf = ".git/refs/heads/main"
commitr = "misc/commit.tmp"
if not os.path.exists(commitf):
	commitf = commitr
elif os.path.exists(commitr):
	os.remove(commitr)

reqs = requests.Session()

def update_repo(force=False):
	# print("Checking for updates...")
	try:
		resp = reqs.get("https://github.com/thomas-xin/Miza-Player")
		s = resp.text
		try:
			search = '<include-fragment src="/thomas-xin/Miza-Player/tree-commit/'
			s = s[s.index(search) + len(search):]
		except ValueError:
			search = '<a data-pjax="true" data-test-selector="commit-tease-commit-message"'
			s = s[s.index(search) + len(search):]
			search = 'href="/thomas-xin/Miza-Player/commit/'
			s = s[s.index(search) + len(search):]
		commit = s.split('"', 1)[0]
		try:
			try:
				with open(commitf, "r") as f:
					s = f.read().strip()
			except FileNotFoundError:
				print("First run, treating as latest update...")
				raise EOFError
			if commit != s:
				print("Update found!")
				if not options.control.autoupdate:
					globals()["repo-update"] = fut = concurrent.futures.Future()
					if force:
						fut.set_result(True)
					else:
						return False
				try:
					if not os.path.exists(".git"):
						raise FileNotFoundError
					subprocess.run(["git"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
					b = None
				except FileNotFoundError:
					resp = reqs.get("https://codeload.github.com/thomas-xin/Miza-Player/zip/refs/heads/main")
					b = resp.content
				if not options.control.autoupdate:
					r = fut.result()
				else:
					r = True
				if r:
					globals()["updating"] = updating = cdict(target=1, progress=0)
					if b:
						with zipfile.ZipFile(io.BytesIO(b), allowZip64=True, strict_timestamps=False) as z:
							nl = z.namelist()
							updating.target = len(nl)
							for i, fn in enumerate(nl):
								updating.progress = i
								fn2 = fn[len("Miza-Player-main/"):]
								if fn2 and not fn2.endswith("/") and not fn2.endswith(".ttf"):
									try:
										folder = fn2.rsplit("/", 1)[0]
										if not os.path.exists(folder):
											os.mkdir(folder)
										with open(fn2, "wb") as f2:
											with z.open(fn, force_zip64=True) as f:
												f2.write(f.read())
									except PermissionError:
										pass
							updating.progress = len(nl)
					elif b is None:
						subprocess.run(["git", "reset", "--hard", "HEAD"])
						updating.progress = 0.5
						subprocess.run(["git", "pull"])
						updating.progress = 1
					else:
						raise ConnectionError(resp.status_code, resp.headers)
					update_collections2()
					globals().pop("updating", None)
					globals()["repo-update"] = True
				if r is not None:
					raise EOFError
			else:
				try:
					globals()["repo-update"].set_result(False)
				except KeyError:
					pass
				else:
					globals().pop("repo-update", None)
				# print("No updates found.")
				return True
		except EOFError:
			if commitf == commitr:
				with open(commitf, "w") as f:
					f.write(commit)
	except Exception as ex:
		print(repr(ex))

def update_collections2():
	try:
		resp = reqs.get("https://raw.githubusercontent.com/thomas-xin/Python-Extra-Classes/main/full.py")
		resp.raise_for_status()
	except:
		# print_exc()
		with open(collections2f, "rb+") as f:
			pass
		return
	b = resp.content
	with open(collections2f, "wb") as f:
		f.write(b)
	# print("collections2.tmp updated.")
	if "alist" in globals():
		return
	cd = cdict
	exec(compile(b, "collections2.tmp", "exec"), globals())
	globals()["cdict"] = cd

repo_fut = None
if not os.path.exists(collections2f):
	update_collections2()
	repo_fut = submit(update_repo)
try:
	with open(collections2f, "rb") as f:
		b = f.read()
	b = b.strip(b"\x00")
	if not b:
		raise FileNotFoundError
	exec(compile(b, "collections2.tmp", "exec"), globals())
except FileNotFoundError:
	try:
		update_collections2()
	except:
		if not os.path.getsize(collections2f):
			raise
		print_exc()
	repo_fut = submit(update_repo)
if utc() - os.path.getmtime(collections2f) > 3600:
	submit(update_collections2)
repo_fut = submit(update_repo)

options.history = astype(options.get("history", ()), alist)
globals().update(options)

def zip2bytes(data):
	if not hasattr(data, "read"):
		data = io.BytesIO(data)
	with zipfile.ZipFile(data, allowZip64=True, strict_timestamps=False) as z:
		b = z.read(z.namelist()[0])
	return b

def bytes2zip(data):
	b = io.BytesIO()
	with zipfile.ZipFile(b, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7, allowZip64=True) as z:
		z.writestr("D", data=data)
	return b.getbuffer()

shash = lambda s: base64.urlsafe_b64encode(hashlib.sha256(s if type(s) is bytes else as_str(s).encode("utf-8")).digest()).rstrip(b"=").decode("ascii")
unyt = lambda s: re.sub(r"https?:\/\/(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)|https?:\/\/(?:api\.)?mizabot\.xyz\/ytdl\?[vd]=(?:https:\/\/youtu\.be\/|https%3A%2F%2Fyoutu\.be%2F)", "https://youtu.be/", s)

def quote(s):
	if s.isascii():
		return urllib.parse.quote_plus(s)
	a = urllib.parse.quote_plus(s)
	b = base64.urlsafe_b64encode(s.encode("utf-8")).rstrip(b"=")
	if len(a) < len(b):
		return a
	return "\x7f" + as_str(b)

def unquote(s):
	if s.startswith("\x7f"):
		s = s[1:].encode("utf-8")
		s += b"=="
		if (len(s) - 1) & 3 == 0:
			s += b"="
		return as_str(base64.urlsafe_b64decode(s))
	return urllib.parse.unquote_plus(s.replace(" ", "+"))


pt = None
def pc():
	global pt
	t = time.time()
	if not pt:
		pt = t
		return 0
	return t - pt

math.round = round

def round(x, y=None):
	try:
		if isfinite(x):
			try:
				if x == int(x):
					return int(x)
				if y is None:
					return int(math.round(x))
			except:
				pass
			return round_min(math.round(x, y))
		else:
			return x
	except:
		pass
	if type(x) is complex:
		return round(x.real, y) + round(x.imag, y) * 1j
	try:
		return math.round(x, y)
	except:
		pass
	return x

def round_min(x):
	if isinstance(x, str):
		if "." in x:
			x = float(x.strip("0"))
		else:
			try:
				return int(x)
			except ValueError:
				return float(x)
	if isinstance(x, int):
		return x
	if isinstance(x, complex):
		if x.imag == 0:
			return round_min(x.real)
		else:
			return round_min(complex(x).real) + round_min(complex(x).imag) * (1j)
	if isfinite(x):
		y = math.round(x)
		if x == y:
			return int(y)
	return x

def round_random(x):
	try:
		y = int(x)
	except (ValueError, TypeError):
		return x
	if y == x:
		return y
	x -= y
	if random.random() <= x:
		y += 1
	return y
sleep = lambda secs: time.sleep(round_random(secs * 1000) / 1000)

def bit_crush(dest, b=0, f=round):
	if type(b) == int:
		a = 1 << b
	else:
		a = 2 ** b
	try:
		len(dest)
		dest = list(dest)
		for i in range(len(dest)):
			dest[i] = f(dest[i] / a) * a
	except TypeError:
		dest = f(dest / a) * a
	return dest

def shuffle(it):
	if not isinstance(it, list):
		it = list(it)
	random.shuffle(it)
	return it

def limit_size(w, h, wm, hm):
	r = h / w
	w2 = min(wm, hm / r)
	h2 = w2 * r
	return tuple(map(round, (w2, h2)))

import pygame.ftfont, pygame.gfxdraw
gfxdraw = pygame.gfxdraw
PRINT.start()

a = submit(pygame.ftfont.init)
b = submit(pygame.font.init)
a.result()
b.result()
globals()["fg"] = "xEC"

def pyg2pil(surf):
	mode = "RGBA" if surf.get_flags() & pygame.SRCALPHA else "RGB"
	# b = surf.getbuffer()
	b = pygame.image.tostring(surf, mode)
	return Image.frombuffer(mode, surf.get_size(), b)

def pil2pyg(im, convert=None):
	mode = im.mode
	b = im.tobytes()
	surf = pygame.image.frombuffer(b, im.size, mode)
	if convert:
		if "A" in mode:
			return surf
		return surf.convert(32)
	elif convert is not None:
		return HWSurface.copy(surf)
	return surf

def pyg2pgl(surf):
	mode = "RGBA" if surf.get_flags() & pygame.SRCALPHA else "RGB"
	b = pygame.image.tostring(surf, mode, True)
	return pyglet.image.ImageData(*surf.get_size(), mode, b)

def pil2pgl(im):
	mode = im.mode
	b = im.transpose(Transpose.FLIP_TOP_BOTTOM).tobytes()
	return pyglet.image.ImageData(*im.size, mode, b)

class HWSurface:

	cache = weakref.WeakKeyDictionary()
	anys = {}
	anyque = []
	maxlen = 24

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
			self = cls.anys[t] = pygame.Surface(size, flags)
		else:
			if t != cls.anyque[-1]:
				cls.anyque.remove(t)
				cls.anyque.append(t)
		if colour is not None:
			self.fill(colour)
		if self.get_colorkey():
			self.set_colorkey(None)
		return self

	@classmethod
	def copy(cls, surf):
		flags = pygame.SRCALPHA if surf.get_flags() & pygame.SRCALPHA else 0
		self = HWSurface.any(surf.get_size(), flags)
		if flags & pygame.SRCALPHA:
			self.fill((0, 0, 0, 0))
		self.blit(surf, (0, 0))
		return self

	# v1 = np.empty(8, dtype=np.float32)
	# v2 = np.empty(8, dtype=np.float32)

	def __init__(self, size, flags=0, colour=None, visible=False):
		self.c = 4 if flags & pygame.SRCALPHA else 3
		self.mode = "RGBA" if self.c > 3 else "RGB"
		self.width, self.height = size
		self.size = astype(size, tuple)
		self.rect = (0, 0) + self.size
		if visible:
			glfw.window_hint(glfw.VISIBLE, True)
		else:
			glfw.window_hint(glfw.VISIBLE, False)
		self.visible = visible
		self.wind = glfw.create_window(*size, "common", None, None)
		if colour:
			self.fill(colour)

	get_size = lambda self: self.size
	get_width = lambda self: self.width
	get_height = lambda self: self.height


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
	out = pil2pyg(im)
	image.close()
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
QUAD_SCALES = {}

def quadratic_gradient(size=gsize, t=None, repetition=1, curve=None, flags=None, copy=False, unique=False):
	if flags is None:
		flags = FLAGS
	size = tuple(size)
	if t is None:
		t = pc()
	quadratics = QUADS.get(flags & pygame.SRCALPHA)
	if not quadratics:
		quadratics = QUADS[flags & pygame.SRCALPHA] = {}
	x = int(t * 128) & 255
	xt = (x, repetition)
	if not quadratics.get(xt):
		hue = qhue.point(lambda i: round(i * repetition + x) & 255)
		img = Image.merge("HSV", (hue, qsat, qval))
		if flags & pygame.SRCALPHA:
			img = img.convert("RGBA")
		else:
			img = img.convert("RGB")
		quadratics[xt] = pil2pyg(img, convert=True)
	surf = quadratics[xt]
	if surf.get_size() != size or surf.get_flags() != flags or curve:
		if unique:
			h = (x, repetition, size, curve)
			try:
				return QUAD_SCALES[h]
			except KeyError:
				s2 = QUAD_SCALES[h] = pygame.Surface(size, flags)
		else:
			s2 = HWSurface.any(size, flags)
		surf = pygame.transform.scale(surf, size, s2)
		if curve:
			h = size[1]
			m = h + 1 >> 1
			for i in range(1, m):
				tx = t - curve * (i / (m - 1))
				g = quadratic_gradient((size[0], 1), tx, repetition)
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
		img = Image.merge("HSV", (hue, rsat, rval))
		if flags & pygame.SRCALPHA:
			img = img.convert("RGBA")
		else:
			img = img.convert("RGB")
		radials[x] = pil2pyg(img, convert=True)
	surf = radials[x]
	if surf.get_size() != size or surf.get_flags() != flags:
		s2 = HWSurface.any(size, flags)
		surf = pygame.transform.scale(surf, size, s2)
	elif copy:
		return HWSurface.copy(surf)
	return surf

def draw_line(dest, colour, start, end, width=1, z=0):
	if dest is DISP or isinstance(dest, DISP.subsurface):
		return DISP.shape(pyglet.shapes.Line, *start, *end, width=width, color=colour, z=z)
	else:
		return pygame.draw.line(dest, colour, start, end, width=width)
def draw_aaline(dest, colour, start, end, width=1, z=0):
	if dest is DISP or isinstance(dest, DISP.subsurface):
		return DISP.shape(pyglet.shapes.Line, *start, *end, width=width, color=colour, z=z)
	else:
		return pygame.draw.aaline(dest, colour, start, end)
def draw_hline(dest, x1, x2, y, colour, width=1, z=0):
	if dest is DISP or isinstance(dest, DISP.subsurface):
		return DISP.shape(pyglet.shapes.Line, x1, y, x2, y, width=width, color=colour, z=z)
	else:
		return gfxdraw.hline(dest, x1, x2, y, colour)
def draw_vline(dest, x, y1, y2, colour, width=1, z=0):
	if dest is DISP or isinstance(dest, DISP.subsurface):
		return DISP.shape(pyglet.shapes.Line, x, y1, x, y2, width=width, color=colour, z=z)
	else:
		return gfxdraw.vline(dest, x, y1, y2, colour)
def draw_polygon(dest, colour, points, z=0):
	if dest is DISP or isinstance(dest, DISP.subsurface):
		if len(colour) > 3:
			alpha = colour[3]
			colour = colour[:3]
		else:
			alpha = 255
		colour = tuple(round(c) for c in colour)
		points = tuple((p[0], screensize[1] - p[1]) for p in points)
		t = (z, pyglet.shapes.Polygon)
		batch = DISP.get_batch(z // 128)
		for sp in DISP.shapes.get(t, ()):
			if not sp.used:
				if tuple(sp._coordinates) != points:
					sp._coordinates = list(points)
					sp._update_position()
					sp._update_color()
				if sp.color != colour:
					sp.color = colour
				break
		else:
			sp = pyglet.shapes.Polygon(
				*points,
				color=colour,
				batch=batch,
				group=DISP.get_group(z),
			)
			DISP.shapes.setdefault(t, weakref.WeakSet()).add(sp)
		if sp.opacity != alpha:
			sp.opacity = alpha
		DISP.active.add(sp)
		batch.used = True
		sp.used = True
		if not sp.visible:
			sp.visible = True
		return sp
	else:
		return pygame.draw.polygon(dest, colour, points)
draw_tpolygon = gfxdraw.textured_polygon
def draw_arc(surf, colour, pos, radius, start_angle=0, stop_angle=0):
	start_angle = int(start_angle % 360)
	stop_angle = int(stop_angle % 360)
	if radius <= 1:
		gfxdraw.filled_circle(surf, *pos, 1, colour)
	if start_angle == stop_angle:
		gfxdraw.circle(surf, *pos, radius, colour)
	else:
		gfxdraw.arc(surf, *pos, radius, start_angle, stop_angle, colour)

poly_names = dict(
	septagram=7/3,
	star=2.5,
	pentagram=2.5,
	octagram=8/3,
	trigon=3,
	triangle=3,
	heptagram=3.5,
	quadrilateral=4,
	square=4,
	pentagon=5,
	hexagon=6,
	septagon=7,
	heptagon=7,
	octagon=8,
	nonagon=9,
	decagon=10,
	undecagon=11,
	hendecagon=11,
	dodecagon=12,
	tridecagon=13,
	tetradecagon=14,
	monogon=144,
	circle=144,
	tetrahedron=(3, 3),
	hexahedron=(4, 3),
	cube=(4, 3),
	octahedron=(3, 4),
	dodecahedron=(5, 3),
	icosahedron=(3, 5),
	stellated_dodecahedron=(2.5, 5),
	stellated_icosahedron=(2.5, 3),
	pentachoron=(3, 3, 3),
	octachoron=(4, 3, 3),
	tesseract=(4, 3, 3),
	hexadecachoron=(3, 3, 4),
	icositetrachoron=(3, 4, 3),
	dodecacontachoron=(5, 3, 3),
	hexacosichoron=(3, 3, 5),
	stellated_dodecacontachoron=(2.5, 5, 2.5),
	stellated_hexacosichoron=(2.5, 3, 3),
	hexateron=(3, 3, 3, 3),
	pentaract=(4, 3, 3, 3),
	decateron=(4, 3, 3, 3),
	triacontaditeron=(3, 3, 3, 4),
	heptapeton=(3, 3, 3, 3, 3),
	hexeract=(4, 3, 3, 3, 3),
	dodecapeton=(4, 3, 3, 3, 3),
	hexacontitetrapeton=(3, 3, 3, 3, 4),
	octaexon=(3, 3, 3, 3, 3, 3),
	hepteract=(4, 3, 3, 3, 3, 3),
	tetradecaexon=(4, 3, 3, 3, 3, 3),
	hecatonicosoctaexon=(3, 3, 3, 3, 3, 4),
	enneazetton=(3, 3, 3, 3, 3, 3, 3),
	octeract=(4, 3, 3, 3, 3, 3, 3),
	hexadecazetton=(4, 3, 3, 3, 3, 3, 3),
	diacosipentacontahexazetton=(3, 3, 3, 3, 3, 3, 4),
	decayotton=(3, 3, 3, 3, 3, 3, 3, 3),
	enneract=(4, 3, 3, 3, 3, 3, 3, 3),
	octadecayotton=(4, 3, 3, 3, 3, 3, 3, 3),
	pentacosidodecayotton=(3, 3, 3, 3, 3, 3, 3, 4),
	hendecaxennon=(3, 3, 3, 3, 3, 3, 3, 3, 3),
	dekeract=(4, 3, 3, 3, 3, 3, 3, 3, 3),
	icosaxennon=(4, 3, 3, 3, 3, 3, 3, 3, 3),
	chilliaicositetraxennon=(3, 3, 3, 3, 3, 3, 3, 3, 4),
)
poly_names.update((fn.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower(), os.path.abspath("misc/default/" + fn)) for fn in os.listdir("misc/default") if fn.endswith(".obj"))
poly_inv = {v: k for k, v in poly_names.items()}

def custom_scale(source, size, dest=None, antialias=False, hwany=False):
	if hwany and not dest:
		dest = HWSURFACE.any(size)
	dsize = tuple(map(round, size))
	ssize = source.get_size()
	if ssize == dsize:
		return source
	if antialias > 1 or (ssize[0] >= dsize[0] and ssize[1] >= dsize[1]):
		if dest:
			return pygame.transform.smoothscale(source, dsize, dest)
		return pygame.transform.smoothscale(source, dsize)
	else:
		while ssize[0] < dsize[0] or ssize[1] < dsize[1]:
			source = pygame.transform.scale2x(source)
			ssize = source.get_size()
	if antialias:
		scalef = pygame.transform.smoothscale
	else:
		scalef = pygame.transform.scale
	if dest:
		return scalef(source, dsize, dest)
	return scalef(source, dsize)

def garbage_collect(cache, lim=4096):
	if len(cache) >= lim * 2:
		items = tuple(cache.items())
		cache.clear()
		cache.update(dict(items[-lim:]))
		return len(items) - lim

QUEUED = deque()
def Enqueue(func, *args, **kwargs):
	fut = submit(func, *args, **kwargs)
	QUEUED.append(fut)
	return fut
def Finish():
	while QUEUED:
		try:
			QUEUED.popleft().result()
		except:
			print_exc()

cb_cache = weakref.WeakKeyDictionary()
ALPHA = BASIC = 0
def blit_complex(dest, source, position=(0, 0), alpha=255, angle=0, scale=1, colour=(255,) * 3, area=None, copy=True, cache=True, z=0):
	pos = position
	if len(pos) > 2:
		pos = pos[:2]
	s1 = source.get_size()
	if dest:
		s2 = dest.get_size()
		if pos[0] >= s2[0] or pos[1] >= s2[1] or pos[0] <= -s1[0] or pos[1] <= -s1[1]:
			return
	if alpha <= 0:
		return
	if dest is DISP or isinstance(dest, DISP.subsurface):
		return dest.blit(source, pos, area=area, colour=astype(colour, tuple) + (alpha,), angle=angle, z=z)
	alpha = round_random(min(alpha / 3, 85)) * 3
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
					s.fill(tuple(colour) + (alpha,), special_flags=BLEND_RGBA_MULT)
				elif any(i != 255 for i in colour):
					s.fill(tuple(colour), special_flags=BLEND_RGB_MULT)
		elif alpha != 255:
			s.fill(tuple(colour) + (alpha,), special_flags=BLEND_RGBA_MULT)
		elif any(i != 255 for i in colour):
			s.fill(tuple(colour), special_flags=BLEND_RGB_MULT)
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
			return dest.blit(s, pos, area, special_flags=BLEND_ALPHA_SDL2)
		globals()["BASIC"] += 1
		return dest.blit(s, pos, area)
	return s

def draw_rect(dest, colour, rect, width=0, alpha=255, angle=0, z=0):
	if len(colour) > 3:
		alpha = (alpha * colour[3] / 255)
		colour = colour[:3]
	alpha = max(0, min(255, round_random(alpha)))
	width = round(abs(width))
	if dest is DISP or isinstance(dest, DISP.subsurface):
		p = [rect[0], screensize[1] - rect[1] - rect[3]]
		q = [rect[0] + rect[2], screensize[1] - rect[1]]
		if dest is not DISP:
			p[0] += dest.rect[0]
			p[1] -= dest.rect[1]
			q[0] += dest.rect[0]
			q[1] -= dest.rect[1]
		width = q[0] - p[0]
		height = q[1] - p[1]
		x, y = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
		anchor_x = width / 2
		anchor_y = height / 2
		colour = tuple(map(round, colour))
		t = (z, pyglet.shapes.Rectangle)
		batch = DISP.get_batch(z // 128)
		for sp in DISP.shapes.get(t, ()):
			if not sp.used:
				if sp.position != (x, y):
					sp.position = (x, y)
				if sp.width != width:
					sp.width = width
				if sp.height != height:
					sp.height = height
				if sp.color != colour:
					sp.color = colour
				break
		else:
			sp = Rectangle(
				x, y,
				width, height,
				colour,
				batch=batch,
				group=DISP.get_group(z),
			)
			DISP.shapes.setdefault(t, weakref.WeakSet()).add(sp)
		if sp.anchor_x != anchor_x:
			sp.anchor_x = anchor_x
		if sp.anchor_y != anchor_y:
			sp.anchor_y = anchor_y
		if sp.opacity != alpha:
			sp.opacity = alpha
		r = angle / pi * 180
		if sp.rotation != r:
			sp.rotation = r
		DISP.active.add(sp)
		batch.used = True
		sp.used = True
		if not sp.visible:
			sp.visible = True
		return sp
	if width > 0:
		if angle != 0 or alpha != 255:
			ssize = [i + width for i in rect[2:]]
			s = pygame.Surface(ssize, FLAGS)
			srec = [i + width // 2 for i in rect[2:]]
			pygame.draw.rect(s, colour, srec, width)
			blit_complex(dest, s, rect[:2], alpha, angle, z=z)
			#raise NotImplementedError("Alpha blending and rotation of rectangles with variable width is not implemented.")
		else:
			pygame.draw.rect(dest, colour, width)
	else:
		rect = astype(rect, list)
		if rect[0] < 0:
			rect[2] += rect[0]
			rect[0] = 0
		if rect[1] < 0:
			rect[3] += rect[1]
			rect[1] = 0
		if alpha != 255:
			dest.fill((255 - alpha,) * 4, rect, special_flags=BLEND_RGBA_MULT)
			dest.fill([min(i + alpha / 255, 255) for i in colour] + [alpha], rect, special_flags=BLEND_RGBA_ADD)
		else:
			dest.fill(colour, rect)

BR_SPLIT = {}

def bevel_rectangle(dest, colour, rect, bevel=0, alpha=255, angle=0, filled=True, cache=True, copy=True, z=0):
	rect = list(map(round, rect))
	if len(colour) > 3:
		colour, alpha = colour[:-1], colour[-1]
	if min(alpha, rect[2], rect[3]) <= 0:
		return
	if dest:
		s = dest.get_size()
		r = (0, 0) + s
		if not int_rect(r, rect):
			return
	if not cache and (dest is DISP or isinstance(dest, DISP.subsurface)):
		if dest is not DISP:
			rect = (
				dest.rect[0] + rect[0],
				dest.rect[1] + rect[1],
				*rect[2:],
			)
		if bevel:
			lines = []
			colours = []
			for i in range(bevel):
				p = [rect[0] + i + 1, screensize[1] - rect[1] - i - 1]
				q = [rect[0] + rect[2] - i, screensize[1] - rect[1] - rect[3] + i + 1]
				v1 = 128 - i / bevel * 128
				v2 = i / bevel * 96 - 96
				col1 = col2 = colour
				if v1:
					col1 = tuple(min(x + v1, 255) for x in col1)
				if v2:
					col2 = tuple(max(x + v2, 0) for x in col1)
				col1 += (alpha,)
				col2 += (alpha,)
				lines.append((p[0] - 1, p[1]))
				colours.append(col1)
				lines.append((q[0], p[1]))
				colours.append(col1)
				lines.append((q[0], p[1]))
				colours.append(col2)
				lines.append((q[0], q[1] - 1))
				colours.append(col2)
				lines.append((q[0], q[1] - 1))
				colours.append(col2)
				lines.append((p[0] - 1, q[1] - 1))
				colours.append(col2)
				lines.append((p[0] - 1, q[1] - 1))
				# How to flex on people that you have a good monitor without telling people you have a good monitor: this bug fix actually means something to you
				# Also hi Lou was here on the 17th of Spooky Month 2023 *dabs and eats Miza's cables*
				colours.append(col1)
				lines.append((p[0], p[1]))
				colours.append(col1)
			lines = tuple(lines)
			colours = tuple(colours)
			t = (z, GL_Lines)
			batch = DISP.get_batch(z // 128)
			for sp in DISP.shapes.get(t, ()):
				if not sp.used:
					if sp.coordinates != lines:
						sp.coordinates = lines
					if sp.colours != colours:
						sp.colours = colours
					break
			else:
				sp = GL_Lines(
					*lines,
					batch=batch,
					group=DISP.get_group(z),
				)
				sp.colours = colours
				DISP.shapes.setdefault(t, weakref.WeakSet()).add(sp)
			DISP.active.add(sp)
			batch.used = True
			sp.used = True
			if not sp.visible:
				sp.visible = True
		if filled:
			lines = []
			draw_rect(
				DISP,
				colour,
				(rect[0] + bevel, rect[1] + bevel, rect[2] - bevel * 2, rect[3] - bevel * 2),
				alpha=alpha,
				z=z,
			)
		return rect
	br_surf = globals().setdefault("br_surf", {})
	colour = verify_colour(colour)
	if alpha == 255 and angle == 0 and (any(i > 160 for i in colour) or all(i in (0, 16, 32, 48, 64, 96, 127, 159, 191, 223, 255) for i in colour)):
		if cache:
			data = tuple(rect[2:]) + (tuple(colour), filled)
		else:
			data = None
		try:
			surf = br_surf[data]
		except KeyError:
			if filled:
				surf = pygame.Surface(rect[2:], FLAGS)
			else:
				surf = pygame.Surface(rect[2:], FLAGS | pygame.SRCALPHA)
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
				draw_rect(surf, colour, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
			rect = r
			if data:
				br_surf[data] = surf
		if dest:
			return blit_complex(dest, surf, rect[:2], z=z)
		return surf.copy() if copy else surf
	ctr = max(colour)
	contrast = min(round(ctr) + 2 >> 2 << 2, 255)
	data = tuple(rect[2:]) + (contrast, filled)
	s = br_surf.get(data)
	if s is None:
		colour2 = (contrast,) * 3
		s = pygame.Surface(rect[2:], FLAGS | pygame.SRCALPHA)
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
			draw_rect(s, colour2, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel])
		if cache:
			br_surf[data] = s
	if ctr > 0:
		colour = tuple(round(i * 255 / ctr) for i in colour)
	else:
		colour = (0,) * 3
	return blit_complex(dest, s, rect[:2], angle=angle, alpha=alpha, colour=colour, z=z)

def rounded_bev_rect(dest, colour, rect, bevel=0, alpha=255, angle=0, grad_col=None, grad_angle=0, filled=True, background=None, cache=True, copy=True, z=0):
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
				if filled:
					surf = pygame.Surface(rect[2:], FLAGS)
					if any(background):
						surf.fill(background)
				else:
					surf = pygame.Surface(rect[2:], FLAGS | pygame.SRCALPHA)
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
					draw_rect(surf, colour, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], alpha=alpha)
				else:
					gradient_rectangle(surf, [rect[0] + bevel, rect[1] + bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle, alpha=alpha)
			rect = r
			if data:
				rb_surf[data] = surf
		if dest:
			return blit_complex(dest, surf, rect[:2], z=z)
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
				draw_rect(s, colour2, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], alpha=alpha)
			else:
				gradient_rectangle(s, [bevel, bevel, rect[2] - 2 * bevel, rect[3] - 2 * bevel], grad_col, grad_angle, alpha=alpha)
		if cache:
			rb_surf[data] = s
	if ctr > 0:
		colour = tuple(round_random(i * 255 / ctr) for i in colour)
	else:
		colour = (0,) * 3
	return blit_complex(dest, s, rect[:2], angle=angle, alpha=alpha, colour=colour, z=z)

rp_surf = {}

def reg_polygon_complex(dest, centre, colour, sides, width, height, angle=pi / 4, alpha=255, thickness=0, repetition=1, filled=False, rotation=0, soft=False, hard=False, attempts=128, cache=True, z=0):
	width = max(round(width), 0)
	height = max(round(height), 0)
	repetition = int(repetition)
	acolour = verify_colour(colour)
	colour = (255, 255, 255)
	if dest and width == height and not rotation:
		rotation = angle
		angle = pi / 4
	elif sides:
		angle %= tau / sides
	else:
		angle = 0
	cache |= angle % (pi / 4) == 0
	if cache:
		# colour = tuple(min(255, round_random(i / 5) * 5) for i in colour)
		angle = round_random(angle / tau * 144) * tau / 144
		if soft and soft is not True:
			soft = astype(soft, tuple)
		h = (sides, width, height, angle, thickness, repetition, filled, soft)
		try:
			newS = rp_surf[h]
		except KeyError:
			pass
		else:
			pos = [centre[0] - width, centre[1] - height]
			return blit_complex(dest, newS, pos, alpha, rotation, colour=acolour, copy=True, z=z)
	construct = pygame.Surface if cache else HWSurface.any
	if soft is True or hard is True or not soft:
		newS = construct((width << 1, height << 1), FLAGS | pygame.SRCALPHA)
	# elif not soft:
		# newS = construct((width << 1, height << 1), FLAGS)
		# newS.set_colorkey((1, 2, 3))
		# newS.fill((1, 2, 3))
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
			colourU = astype(colour, tuple) + (min(round(255 * move_direction + 8), 255),)
		else:
			colourU = (colour[0] * move_direction + 8, colour[1] * move_direction + 8, colour[2] * move_direction + 8)
			colourU = [min(c, 255) for c in colourU]
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
		rp_surf[h] = newS
		# print(len(rp_surf), h)
	return blit_complex(dest, newS, pos, alpha, rotation, colour=acolour, copy=cache, cache=False, z=z)

def draw_circle(dest, colour, pos, radius, width=0, z=0):
	radius = round(radius)
	if radius <= 0:
		return
	width = round(width)
	dc_surf = globals().setdefault("dc_surf", {})
	colour = verify_colour(colour)
	data = (radius, width)
	s = dc_surf.get(data)
	if not s:
		size = (ceil(radius + width / 2) * 2 + 1,) * 2
		s = pygame.Surface(size, FLAGS | pygame.SRCALPHA)
		pygame.draw.circle(s, (255,) * 3, (size[0] >> 1, size[1] >> 1), radius, width)
		dc_surf[data] = s
	p = [x - ceil(y / 2) for x, y in zip(pos, s.get_size())]
	return blit_complex(dest, s, p, colour=colour, z=z)

def concentric_circle(dest, colour, pos, radius, width=0, fill_ratio=1, alpha=255, gradient=False, filled=False, z=0, cache=True):
	reverse = fill_ratio < 0
	radius = max(0, round(radius * 2) / 2)
	if min(alpha, radius) <= 0:
		return
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
	return blit_complex(dest, s, p, alpha=alpha, colour=colour, z=z)

anima_parts = {}

def anima_rectangle(surface, colour, rect, frame, count=2, speed=1, flash=1, ratio=0, reduction=0.2, z=1):
	s = surface.get_size()
	r = (-4, -4, s[0] + 8, s[1] + 8)
	if not int_rect(r, rect):
		return
	az = (z // 128) + 0.05
	if flash:
		n = 4
		a = (ratio * speed * n) % (flash * n)
		if a < speed:
			pos = round((a * 4 / flash - 1) * rect[3])
			bevel_rectangle(surface, (255,) * 3, (rect[0], rect[1] + max(pos, 0), rect[2], min(rect[3] + pos, rect[3]) - max(pos, 0)), 0, 159, z=z+0.5)
			bevel_rectangle(surface, (255,) * 3, (rect[0], rect[1] + max(pos + 8, 0), rect[2], min(rect[3] + pos, rect[3]) - max(pos + 16, 0)), 0, 159, z=z+0.5)
	perimeter = rect[2] * 2 + rect[3] * 2
	increment = 3
	orig = frame
	f = orig - reduction
	while frame > 1:
		c = list(colour)
		for i in range(count):
			pos = perimeter * ((i / count - increment / perimeter + ratio * speed) % 1)
			side = 0
			if pos >= rect[2]:
				pos -= rect[2]
				side = 1
				if pos >= rect[3]:
					pos -= rect[3]
					side = 2
					if pos >= rect[2]:
						pos -= rect[2]
						side = 3
			if side == 0:
				r = [round(rect[0] + pos), round(rect[1] + 0.5)]
			elif side == 1:
				r = [round(rect[0] + rect[2]), round(rect[1] + pos + 0.5)]
			elif side == 2:
				r = [round(rect[0] + rect[2] - pos), round(rect[1] + rect[3] + 0.5)]
			else:
				r = [round(rect[0]), round(rect[1] + rect[3] - pos + 0.5)]
			if surface is DISP or isinstance(surface, DISP.subsurface):
				if surface is not DISP:
					r[0] += surface.rect[0]
					r[1] = DISP.height - surface.rect[1] - r[1]
				else:
					r[1] = DISP.height - r[1]
				anima_parts.setdefault(az, []).append((r, frame * 2, adj_colour(c, (frame - f) * 16)))
			else:
				r_rect = [r[0] - round(frame), r[1] - round(frame), round(frame) << 1, round(frame) << 1]
				draw_rect(surface, adj_colour(c, (frame - f) * 16), r_rect, z=z)
		frame -= reduction
		increment += 3
	frame = orig
	for i in range(count):
		pos = perimeter * ((i / count + ratio * speed) % 1)
		side = 0
		if pos >= rect[2]:
			pos -= rect[2]
			side = 1
			if pos >= rect[3]:
				pos -= rect[3]
				side = 2
				if pos >= rect[2]:
					pos -= rect[2]
					side = 3
		if side == 0:
			r = [round(rect[0] + pos), round(rect[1] + 0.5)]
		elif side == 1:
			r = [round(rect[0] + rect[2]), round(rect[1] + pos + 0.5)]
		elif side == 2:
			r = [round(rect[0] + rect[2] - pos), round(rect[1] + rect[3] + 0.5)]
		else:
			r = [round(rect[0]), round(rect[1] + rect[3] - pos + 0.5)]
		if surface is DISP or isinstance(surface, DISP.subsurface):
			if surface is not DISP:
				r[0] += surface.rect[0]
				r[1] = DISP.height - surface.rect[1] - r[1]
			else:
				r[1] = DISP.height - r[1]
			anima_parts.setdefault(az, []).append((r, (frame - 1) * 2, adj_colour(c, (frame - f) * 16)))
		else:
			r_rect = [r[0] - round(frame) - 1, r[1] - round(frame) - 1, round(frame) + 1 << 1, round(frame) + 1 << 1]
			bevel_rectangle(surface, colour, r_rect, 3, z=z)

surf = reg_polygon_complex(
	None,
	(0, 0),
	(255, 255, 255),
	0,
	32,
	32,
	alpha=255,
	thickness=2,
	repetition=29,
	soft=True,
)
spinny_im = pyg2pgl(surf)
texture = spinny_im.get_texture()
glEnable(texture.target)
glBindTexture(texture.target, texture.id)
glGenerateMipmap(texture.target)
glDisable(texture.target)

def render_anima_parts(particles):
	texture = spinny_im.get_texture()
	glEnableClientState(GL_VERTEX_ARRAY)
	glEnableClientState(GL_COLOR_ARRAY)
	glEnableClientState(GL_TEXTURE_COORD_ARRAY)
	glEnable(texture.target)
	glBindTexture(texture.target, texture.id)
	glEnable(GL_BLEND)
	glBlendFunc(GL_SRC_ALPHA, GL_ONE)
	count = len(particles)
	verts = np.zeros((count, 4, 2), dtype=np.float32)
	cols = np.zeros((count, 4, 3), dtype=np.float32)
	texs = np.tile(np.array([
		[0, 0],
		[1, 0],
		[1, 1],
		[0, 1],
	], dtype=np.float32), (count, 1, 1))

	for i, p in enumerate(particles):
		verts[i] = [
			(p[0][0] - p[1], p[0][1] - p[1]),
			(p[0][0] + p[1], p[0][1] - p[1]),
			(p[0][0] + p[1], p[0][1] + p[1]),
			(p[0][0] - p[1], p[0][1] + p[1]),
		]
		cols[i] = [
			[c / 255 / 3 * 2 for c in p[2]],
		] * 4

	count = len(particles)
	glVertexPointer(2, GL_FLOAT, 0, verts[:count].ctypes)
	glColorPointer(3, GL_FLOAT, 0, cols[:count].ctypes)
	glTexCoordPointer(2, GL_FLOAT, 0, texs[:count].ctypes)
	glDrawArrays(GL_QUADS, 0, 4 * count)
	glDisable(texture.target)
	glDisableClientState(GL_VERTEX_ARRAY)
	glDisableClientState(GL_COLOR_ARRAY)
	glDisableClientState(GL_TEXTURE_COORD_ARRAY)
	particles.clear()

def update_anima_parts():

	def curry(a):
		return lambda: render_anima_parts(a)

	for k, v in anima_parts.items():
		DISP.batches[k] = curry(v)

def text_objects(text, font, colour, background):
	text_surface = font.render(text, True, colour, background)
	return text_surface, text_surface.get_rect()

def get_font(font):
	try:
		fn = "misc/" + font + ".ttf"
		if font == "OpenSansEmoji":
			resp = reqs.get("https://drive.google.com/u/0/uc?id=1OZs0gQ4J3vm9rEKzECatlgh5z3wfbNcZ&export=download", timeout=60)
			g = io.BytesIO(resp.content)
			with open(fn, "wb") as f:
				with zipfile.ZipFile(g) as z:
					content = z.read(font + ".ttf")
					f.write(content)
		elif font == "Rockwell":
			resp = reqs.get("https://drive.google.com/u/0/uc?id=1Lxr25oC003hfgjyzkVAjKUGuaEw9MCSf&export=download", timeout=60)
			g = io.BytesIO(resp.content)
			with open(fn, "wb") as f:
				with zipfile.ZipFile(g) as z:
					content = z.read(font + ".ttf")
					f.write(content)
		if "ct_font" in globals():
			ct_font.clear()
		if "ft_font" in globals():
			ft_font.clear()
		md_font.clear()
		globals()["font_reload"] = True
	except:
		print_exc()

loaded_fonts = set()
font_reload = False

def sysfont(font, size, unicode=False):
	func = pygame.ftfont if unicode else pygame.font
	fn = "misc/" + font + ".ttf"
	exists = os.path.exists(fn) and os.path.getsize(fn)
	if not exists and font not in loaded_fonts:
		if font in ("Rockwell", "OpenSansEmoji"):
			print("Downloading and applying required fonts...")
			for fnt in ("Rockwell", "OpenSansEmoji"):
				loaded_fonts.add(fnt)
				submit(get_font, fnt)
	if exists:
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

m_split = re.compile(r"[\s.\-/\\|:'%]")
md_font = {}
def message_display(text, size, pos=(0, 0), colour=(255,) * 3, background=None, surface=None, font="OpenSansEmoji", alpha=255, align=1, cache=False, z=0, clip=False):
	# text = "".join(c if ord(c) < 65536 else "\x7f" for c in text)
	text = str(text if type(text) is not float else round_min(text)).replace("\u200b", "").strip()
	if not text:
		return [0] * 4
	if not cache and (surface is DISP or isinstance(surface, DISP.subsurface)):
		if isinstance(surface, DISP.subsurface):
			pos = astype(pos, list)
			for i in range(2):
				pos[i] += surface.rect[i]
		return DISP.hardware_font(text, size, pos, colour, background, font, alpha, align, z)
	sep = cache < 0 and len(text) > 1 and re.search(m_split, text)
	if surface and sep and align == 0:
		pos = astype(pos, tuple)
		x, y = pos
		rect = pos + (0, 0)
		while text:
			offs = 0
			while text.startswith(" "):
				text = text[1:]
				offs += 1
			x += rect[2] + offs * size / 4
			m = re.search(m_split, text)
			if m:
				i = m.start()
				if i:
					s = text[:i]
					text = text[i:]
				else:
					s = text[0]
					text = text[1:]
			else:
				s = text
				text = ""
			if len(s) > 2 and s[:2].isnumeric():
				text = s[2:] + text
				s = s[:2]
			if s:
				rect = message_display(
					s,
					size,
					(x, y),
					colour=colour,
					background=background,
					surface=surface,
					font=font,
					alpha=alpha,
					align=align,
					cache=True,
					z=z,
				)
		return pos + (x, rect[3])
	colour = tuple(verify_colour(colour))
	data = (text, background, size, hash(font))
	args = (text, (255,) * 3, background, size, font)
	try:
		resp = md_font[data]
	except KeyError:
		resp = surface_font(*args)
	TextSurf, TextRect = resp
	if cache:
		md_font[data] = resp
	if surface:
		if align == 1:
			TextRect.center = pos
		elif align == 0:
			TextRect = astype(pos, list) + TextRect[2:]
		elif align == 2:
			TextRect = [y - x for x, y in zip(TextRect[2:], pos)] + TextRect[2:]
		if clip:
			rect = (
				max(0, min(surface.width - TextSurf.get_width(), TextRect[0])),
				max(0, min(surface.height - TextSurf.get_height(), TextRect[1])),
			)
			rect += tuple(TextRect[:2])
			TextRect = rect
		blit_complex(surface, TextSurf, TextRect, colour=colour, alpha=alpha, copy=cache, z=z)
		return TextRect
	else:
		return TextSurf

def char_display(char, size, font="OpenSansEmoji"):
	size = round(size)
	cs_font = globals().setdefault("cs_font", {})
	data = (char, size, font)
	f = cs_font.get(data, None)
	if not f:
		f = surface_font(char, (255,) * 3, size, font)[0]
	return f


class PipedProcess:

	procs = ()
	stdin = stdout = stderr = None

	def __init__(self, *args, stdin=None, stdout=None, stderr=None, cwd=".", bufsize=4096):
		if not args:
			return
		self.exc = concurrent.futures.ThreadPoolExecutor(max_workers=len(args) - 1) if len(args) > 1 else None
		self.procs = []
		for i, arg in enumerate(args):
			first = not i
			last = i >= len(args) - 1
			si = stdin if first else subprocess.PIPE
			so = stdout if last else subprocess.PIPE
			se = stderr if last else None
			proc = psutil.Popen(arg, stdin=si, stdout=so, stderr=se, cwd=cwd, bufsize=bufsize * 256)
			if first:
				self.stdin = proc.stdin
			if last:
				self.stdout = proc.stdout
				self.stderr = proc.stderr
			self.procs.append(proc)
		for i in range(len(args) - 1):
			self.exc.submit(self.pipe, i, bufsize=bufsize)
		self.pid = self.procs[0].pid

	def pipe(self, i, bufsize=4096):
		try:
			proc = self.procs[i]
			proc2 = self.procs[i + 1]
			si = 0
			while proc.is_running() and proc2.is_running():
				b = proc.stdout.read(si * (si + 1) * bufsize // 8 + bufsize)
				if not b:
					break
				proc2.stdin.write(b)
				proc2.stdin.flush()
				si += 1
			if proc2.is_running():
				proc2.stdin.close()
		except:
			import traceback
			traceback.print_exc()
			if not proc.is_running() or not proc2.is_running():
				self.terminate()
		if self.exc:
			self.exc.shutdown(wait=False)

	def is_running(self):
		for proc in self.procs:
			if proc.is_running():
				return True
		return False

	def terminate(self):
		for proc in self.procs:
			proc.terminate()

	def kill(self):
		for proc in self.procs:
			proc.kill()

	def wait(self):
		for proc in self.procs:
			proc.wait()

	def status(self):
		return self.procs[-1].status()


# Runs ffprobe on a file or url, returning the duration if possible.
def _get_duration_2(filename, _timeout=12):
	if filename.startswith("https://api.mizabot.xyz/ytdl"):
		url = filename.replace("?v=", "?q=").replace("?d=", "?q=")
		resp = reqs.get(url)
		return resp.json()[0].get("duration"), None, "webm"
	command = (
		ffprobe,
		"-v",
		"error",
		"-select_streams",
		"a:0",
		"-show_entries",
		"stream=codec_name,",
		"-show_entries",
		"format=duration,bit_rate",
		"-of",
		"default=nokey=1:noprint_wrappers=1",
		filename,
	)
	resp = None
	try:
		proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
		fut = submit(proc.wait, timeout=_timeout)
		res = fut.result(timeout=_timeout)
		resp = proc.stdout.read().splitlines()
	except:
		with suppress():
			proc.kill()
		print_exc()
	try:
		cdc = as_str(resp[0].rstrip())
	except (IndexError, ValueError, TypeError):
		cdc = "auto"
	try:
		dur = float(resp[1])
	except (IndexError, ValueError, TypeError):
		dur = None
	bps = None
	if resp and len(resp) > 2:
		with suppress(ValueError):
			bps = float(resp[2])
	return dur, bps, cdc

def get_duration_2(filename):
	if not is_url(filename):
		if filename.endswith(".pcm"):
			return os.path.getsize(filename) / (48000 * 2 * 2), "pcm"
		if os.path.exists(filename):
			with open(filename, "rb") as f:
				if f.read(4) == b"MThd":
					return None, "N/A"
	if filename:
		dur, bps, cdc = _get_duration_2(filename, 4)
		if not dur and is_url(filename):
			resp = reqs.head(filename)
			head = fcdict(resp.headers)
			if "content-length" not in head:
				dur, bps, cdc = _get_duration_2(filename, 20)
				return dur, cdc
			if bps:
				return (int(head["content-length"]) << 3) / bps, cdc
			ctype = [e.strip() for e in head.get("content-type", "").split(";") if "/" in e][0]
			if ctype.split("/", 1)[0] not in ("audio", "video"):
				return nan, cdc
			if ctype == "audio/midi":
				return nan, cdc
		return dur, cdc

def construct_options(full=True):
	stats = cdict(audio)
	pitchscale = 2 ** ((stats.pitch + stats.nightcore) / 12)
	reverb = stats.reverb
	volume = stats.volume
	if reverb:
		args = ["-i", "misc/SNB3,0all.wav"]
	else:
		args = []
	options = deque()
	if not isfinite(stats.compressor):
		options.extend(("anoisesrc=a=.001953125:c=brown", "amerge"))
	if pitchscale != 1 or stats.speed != 1:
		speed = abs(stats.speed) / pitchscale
		speed *= 2 ** (stats.nightcore / 12)
		if round(speed, 9) != 1:
			speed = max(0.005, speed)
			if speed >= 64:
				raise OverflowError
			opts = ""
			while speed > 3:
				opts += "atempo=3,"
				speed /= 3
			while speed < 0.5:
				opts += "atempo=0.5,"
				speed /= 0.5
			opts += "atempo=" + str(speed)
			options.append(opts)
	if pitchscale != 1:
		if abs(pitchscale) >= 64:
			raise OverflowError
		if full:
			options.append("aresample=48k")
		options.append("asetrate=" + str(48000 * pitchscale))
	if stats.chorus:
		chorus = abs(stats.chorus)
		ch = min(16, chorus)
		A = B = C = D = ""
		for i in range(ceil(ch)):
			neg = ((i & 1) << 1) - 1
			i = 1 + i >> 1
			i *= stats.chorus / ceil(chorus)
			if i:
				A += "|"
				B += "|"
				C += "|"
				D += "|"
			delay = (8 + 5 * i * tau * neg) % 39 + 19
			A += str(round(delay, 3))
			decay = (0.36 + i * 0.47 * neg) % 0.65 + 1.7
			B += str(round(decay, 3))
			speed = (0.27 + i * 0.573 * neg) % 0.3 + 0.02
			C += str(round(speed, 3))
			depth = (0.55 + i * 0.25 * neg) % max(1, stats.chorus) + 0.15
			D += str(round(depth, 3))
		b = 0.5 / sqrt(ceil(ch + 1))
		options.append(
			"chorus=0.5:" + str(round(b, 3)) + ":"
			+ A + ":"
			+ B + ":"
			+ C + ":"
			+ D
		)
	if stats.compressor:
		comp = min(8000, abs(stats.compressor * 10 + (1 if stats.compressor >= 0 else -1)))
		while abs(comp) > 1:
			c = min(20, comp)
			try:
				comp /= c
			except ZeroDivisionError:
				comp = 1
			mult = str(round((c * math.sqrt(2)) ** 0.5, 4))
			options.append(
				"acompressor=mode=" + ("upward" if stats.compressor < 0 else "downward")
				+ ":ratio=" + str(c) + ":level_in=" + mult + ":threshold=0.0625:makeup=" + mult
			)
	if stats.bassboost:
		opt = "firequalizer=gain_entry="
		entries = []
		high = 24000
		low = 13.75
		bars = 4
		small = 0
		for i in range(bars):
			freq = low * (high / low) ** (i / bars)
			bb = -(i / (bars - 1) - 0.5) * stats.bassboost * 64
			dB = log(abs(bb) + 1, 2)
			if bb < 0:
				dB = -dB
			if dB < small:
				small = dB
			entries.append(f"entry({round(freq, 5)},{round(dB, 5)})")
		entries.insert(0, f"entry(0,{round(small, 5)})")
		entries.append(f"entry(24000,{round(small, 5)})")
		opt += repr(";".join(entries))
		options.append(opt)
	if reverb:
		coeff = abs(reverb)
		wet = min(3, coeff) / 3
		if wet != 1:
			options.append("asplit[2]")
		volume *= 1.2
		if reverb < 0:
			volume = -volume
		options.append("afir=dry=10:wet=10")
		if wet != 1:
			dry = 1 - wet
			options.append("[2]amix=weights=" + str(round(dry, 6)) + " " + str(round(-wet, 6)))
		d = [round(1 - i ** 1.6 / (i ** 1.6 + coeff), 4) for i in range(2, 18, 2)]
		options.append(f"aecho=1:1:400|630:{d[0]}|{d[1]}")
		if d[2] >= 0.05:
			options.append(f"aecho=1:1:920|1450:{d[2]}|{d[3]}")
			if d[4] >= 0.06:
				options.append(f"aecho=1:1:1760|2190:{d[4]}|{d[5]}")
				if d[6] >= 0.07:
					options.append(f"aecho=1:1:2520|3000:{d[6]}|{d[7]}")
	if stats.pan != 1:
		pan = min(10000, max(-10000, stats.pan))
		while abs(abs(pan) - 1) > 0.001:
			p = max(-10, min(10, pan))
			try:
				pan /= p
			except ZeroDivisionError:
				pan = 1
			options.append("extrastereo=m=" + str(p) + ":c=0")
			volume *= 1 / max(1, round(math.sqrt(abs(p)), 4))
	if volume != 1:
		options.append("volume=" + str(round(volume, 7)))
	if options:
		if stats.compressor:
			options.append("alimiter")
		elif volume > 1 or abs(stats.bassboost):
			options.append("asoftclip=atan")
		args.append(("-af", "-filter_complex")[bool(reverb)])
		args.append(",".join(options))
	return args


# runs org2xm on a file, with an optional custom sample bank.
def org2xm(org, dat=None):
	if os.name != "nt":
		raise OSError("org2xm is only available on Windows.")
	r_org = None
	if not org or type(org) is not bytes:
		if is_url(org):
			r = reqs.get(org)
			data = r.content
		else:
			r_org = org
			with open(r_org, "rb") as f:
				data = f.read(6)
		if not data:
			raise FileNotFoundError("Error downloading file content.")
	else:
		if not org.startswith(b"Org-"):
			raise ValueError("Invalid file header.")
		data = org
	# Set compatibility option if file is not of org2 format.
	compat = not data.startswith(b"Org-02")
	ts = ts_us()
	# Write org data to file.
	if not r_org:
		r_org = "cache/" + str(ts) + ".org"
		with open(r_org, "wb") as f:
			f.write(data)
	r_dat = "cache/" + str(ts) + ".dat"
	orig = False
	# Load custom sample bank if specified
	if dat is not None and is_url(dat):
		with open(r_dat, "wb") as f:
			r = reqs.get(dat)
			f.write(r.content)
	else:
		if type(dat) is bytes and dat:
			with open(r_dat, "wb") as f:
				f.write(dat)
		else:
			r_dat = "sndlib/ORG210EN.DAT"
			orig = True
			if not os.path.exists(r_dat):
				resp = reqs.get("https://github.com/Clownacy/org2xm/blob/master/ORG210EN.DAT?raw=true")
				with open(r_dat, "wb") as f:
					f.write(resp.content)
	args = ["org2xm.exe", r_org, r_dat]
	if compat:
		args.append("c")
	print(args)
	subprocess.check_output(args)
	r_xm = f"cache/{ts}.xm"
	if not os.path.exists("cache/" + str(ts) + ".xm"):
		raise FileNotFoundError("Unable to locate converted file.")
	if not os.path.getsize(r_xm):
		raise RuntimeError("Converted file is empty.")
	for f in (r_org, r_dat)[:2 - orig]:
		with suppress():
			os.remove(f)
	return r_xm

def mid2mp3(mid):
	url = reqs.post(
		"https://hostfast.onlineconverter.com/file/send",
		files={
			"class": (None, "audio"),
			"from": (None, "midi"),
			"to": (None, "mp3"),
			"source": (None, "file"),
			"file": mid,
			"audio_quality": (None, "192"),
		},
	).text
	fn = url.rsplit("/", 1)[-1].strip("\x00")
	for i in range(360):
		t = utc()
		test = reqs.get(f"https://hostfast.onlineconverter.com/file/{fn}").content
		if test == b"d":
			break
		delay = utc() - t
		if delay < 1:
			time.sleep(1 - delay)
	ts = ts_us()
	r_mp3 = f"cache/{ts}.mp3"
	with open(r_mp3, "wb") as f:
		f.write(reqs.get(f"https://hostfast.onlineconverter.com/file/{fn}/download").content)
	return r_mp3

def png2wav(png):
	ts = ts_us()
	r_png = f"cache/{ts}"
	r_wav = f"cache/{ts}.wav"
	args = [sys.executable, "-O", "png2wav.py", "../" + r_png, "../" + r_wav]
	with open(r_png, "wb") as f:
		f.write(png)
	print(args)
	subprocess.run(args, cwd="misc", stderr=subprocess.PIPE)
	return r_wav

def ecdc_encode(ecdc, bitrate="24k", name=None, source=None):
	ts = ts_us()
	out = f"cache/{ts}.ecdc"
	if not isinstance(ecdc, str):
		fi = f"cache/{ts}"
		with open(fi, "wb") as f:
			f.write(ecdc)
		ecdc = fi
	args1 = [ffmpeg, "-v", "error", "-hide_banner", "-vn", "-nostdin", "-i", ecdc, "-f", "s16le", "-ac", "2", "-ar", "48k", "-"]
	args2 = [sys.executable, "misc/ecdc_stream.py", "-n", name or "", "-s", source or "", "-b", str(bitrate), "-e", out]
	print(args1)
	print(args2)
	PipedProcess(args1, args2).wait()
	return out

def ecdc_decode(ecdc, out=None):
	fmt = out.rsplit(".", 1)[-1] if out else "opus"
	ts = ts_us()
	out = out or f"cache/{ts}.{fmt}"
	if os.path.exists(out) and os.path.getsize(out):
		return out
	if not isinstance(ecdc, str):
		fi = f"cache/{ts}.ecdc"
		with open(fi, "wb") as f:
			f.write(ecdc)
		ecdc = fi
	args1 = [sys.executable, "misc/ecdc_stream.py", "-d", ecdc]
	args2 = [ffmpeg, "-v", "error", "-hide_banner", "-f", "s16le", "-ac", "2", "-ar", "48k", "-i", "-", "-b:a", "96k", out]
	print(args1)
	print(args2)
	PipedProcess(args1, args2).wait()
	return out


CONVERTERS = {
	b"MThd": mid2mp3,
	b"Org-": org2xm,
	b"ECDC": ecdc_decode,
}
CONVERTING = {}

def select_and_convert(stream):
	if stream in CONVERTING:
		resp = CONVERTING[stream].result()
		if resp:
			return resp
	CONVERTING[stream] = concurrent.futures.Future()
	try:
		if is_url(stream):
			with reqs.get(stream, timeout=8, stream=True) as resp:
				it = resp.iter_content(65536)
				b = bytes()
				while len(b) < 4:
					b += next(it)
				try:
					convert = CONVERTERS[b[:4]]
				except KeyError:
					convert = png2wav
				b += resp.content
		else:
			with open(stream, "rb") as file:
				b = file.read(65536)
				try:
					convert = CONVERTERS[b[:4]]
				except KeyError:
					convert = png2wav
				b += file.read()
		print(convert, stream)
		resp = convert(b)
		CONVERTING[stream].set_result(resp)
		return resp
	finally:
		if not CONVERTING[stream].done():
			CONVERTING[stream].set_result(None)


def supersample(a, size, hq=False, in_place=False):
	n = len(a)
	if n == size:
		return a
	if n < size:
		if hq:
			a = samplerate.resample(a, size / len(a), "sinc_fastest")
			return supersample(a, size, in_place=in_place)
		interp = np.linspace(0, n - 1, size)
		return np.interp(interp, range(n), a)
	try:
		dtype = a.dtype
	except AttributeError:
		dtype = object
	ftype = np.float64 if dtype is object or issubclass(dtype.type, np.integer) else dtype
	x = ceil(n / size)
	args = ("ss-interps", 0, n - 1, x * size)
	try:
		if not in_place:
			raise KeyError
		interp = globals()[args]
	except KeyError:
		interp = globals()[args] = np.linspace(*args[1:], dtype=ftype)
	a = np.interp(interp, range(n), a)
	if in_place:
		return np.mean(a.reshape(-1, x), 1, out=a[:size])
	return np.mean(a.reshape(-1, x), 1, dtype=dtype)


eval_const = {
	"none": None,
	"null": None,
	"NULL": None,
	"true": True,
	"false": False,
	"TRUE": True,
	"FALSE": False,
	"inf": inf,
	"nan": nan,
	"Infinity": inf,
}
safe_eval = lambda s: eval(s, {}, eval_const) if not s.isnumeric() else int(s)

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

def time_parse(ts):
	data = ts.split(":")
	if len(data) >= 5: 
		raise TypeError("Too many time arguments.")
	mults = (1, 60, 3600, 86400)
	return round_min(sum(float(count) * mult for count, mult in zip(data, reversed(mults[:len(data)]))))

def expired(stream):
	if is_youtube_url(stream):
		return True
	if stream.startswith("https://www.yt-download.org/download/"):
		if int(stream.split("/download/", 1)[1].split("/", 4)[3]) < utc() + 60:
			return True
	elif is_youtube_stream(stream):
		if int(stream.replace("/", "=").split("expire=", 1)[-1].split("=", 1)[0].split("&", 1)[0]) < utc() + 60:
			return True

is_youtube_stream = lambda url: url and re.findall(r"^https?:\/\/r+[0-9]+---.{2}-[A-Za-z0-9\-_]{4,}\.googlevideo\.com", url)
is_youtube_url = lambda url: url and re.findall("^https?:\\/\\/(?:www\\.)?youtu(?:\\.be|be\\.com)\\/[^\\s<>`|\"']+", url)
# Regex moment - Lou
