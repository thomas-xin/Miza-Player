# ONE SMALL STEP FOR MAN, ONE GIANT LEAP FOR SMUDGE KIND! Invaded once again on the 6th March >:D

import os, sys, traceback

# c = sys.stdin.readline()
# if c == "~init\n":
	# sys.stderr.write("~I\n")
# sys.stderr.write("~STARTED\n")

pid = os.getppid()

sys.stdout.write = lambda *args, **kwargs: None
import concurrent.futures

exc = concurrent.futures.ThreadPoolExecutor(max_workers=24)

class MultiAutoImporter:

	class ImportedModule:

		def __init__(self, module, pool, _globals):
			object.__setattr__(self, "__module", module)
			object.__setattr__(self, "__fut", pool.submit(__import__, module))
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
			_globals[module] = m = object.__getattribute__(self, "__fut").result()
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
				futs.append(self.ImportedModule(arg, pool, _globals))
			_globals.update(zip(args, futs))

# importer = MultiAutoImporter(
	# "numpy, pygame, pyglet, random, hashlib, orjson, traceback, base64",
	# "requests, ctypes, weakref, samplerate, itertools, io, zipfile",
	# "psutil", "subprocess, multiprocessing, re",
	# pool=exc,
	# _globals=globals(),
# )
# sys.stderr.write("~IMPORTING\n")
import numpy, pygame, pyglet, random, hashlib, orjson, traceback, base64
import requests, ctypes, weakref, samplerate, itertools, io, zipfile, socket
import psutil, subprocess, multiprocessing, re
import soundcard as sc
import math
from math import *
np = numpy
import cffi
CFFI = cffi.FFI()
import collections, contextlib, time
deque = collections.deque
suppress = contextlib.suppress

async_wait = lambda: time.sleep(0.005)
sys.setswitchinterval(0.005)

is_minimised = lambda: globals()["stat-mem"].buf[0] & 1

reqs = requests.Session()
sys.stdin.readline()
sys.stderr.write("~I\n")
mixer_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mixer_server.connect(("127.0.0.1", pid & 32767 | 32768))

pt = None
pt2 = None
def pc():
	global pt
	t = time.perf_counter()
	if not pt:
		pt = t
		return 0
	return t - pt

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

def as_str(s):
	if type(s) in (bytes, bytearray, memoryview):
		return bytes(s).decode("utf-8", "replace")
	return str(s)

is_url = lambda url: "://" in url and url.split("://", 1)[0].rstrip("s") in ("http", "hxxp", "ftp", "fxp")

def zip2bytes(data):
	if not hasattr(data, "read"):
		data = io.BytesIO(data)
	with zipfile.ZipFile(data, compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False) as z:
		b = z.read("D")
	return b

def bytes2zip(data):
	b = io.BytesIO()
	with zipfile.ZipFile(b, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
		z.writestr("D", data=data)
	return b.getbuffer()

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

SR = 48000
FR = 1600


from concurrent.futures import thread

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
			target=thread._worker,
			args=(
				thread.weakref.ref(self, weakref_cb),
				self._work_queue,
				self._initializer,
				self._initargs,
			),
			daemon=True
		)
		t.start()
		self._threads.add(t)
		thread._threads_queues[t] = self._work_queue

concurrent.futures.ThreadPoolExecutor._adjust_thread_count = lambda self: _adjust_thread_count(self)

# exc = concurrent.futures.ThreadPoolExecutor(max_workers=24)
submit = exc.submit

class cdict(dict):

	__slots__ = ()

	__init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
	__repr__ = lambda self: f"{self.__class__.__name__}({super().__repr__() if super().__len__() else ''})"
	__str__ = lambda self: super().__repr__()
	__iter__ = lambda self: iter(tuple(super().__iter__()))
	__call__ = lambda self, k: self.__getitem__(k)

	def __getattr__(self, k):
		try:
			return self.__getattribute__(k)
		except AttributeError:
			pass
		if not k.startswith("__") or not k.endswith("__"):
			try:
				return self.__getitem__(k)
			except KeyError as ex:
				raise AttributeError(*ex.args)
		raise AttributeError(k)

	def __setattr__(self, k, v):
		if k.startswith("__") and k.endswith("__"):
			return object.__setattr__(self, k, v)
		return self.__setitem__(k, v)

	def __dir__(self):
		data = set(object.__dir__(self))
		data.update(self)
		return data

	def union(self, other=None, **kwargs):
		temp = self.copy()
		if other:
			temp.update(other)
		if kwargs:
			temp.update(kwargs)
		return temp

	@property
	def __dict__(self):
		return self

	___repr__ = lambda self: super().__repr__()
	copy = __copy__ = lambda self: self.__class__(self)
	to_dict = lambda self: dict(**self)
	to_list = lambda self: list(super().values())

def point(s):
	b = str(s).encode("utf-8") if type(s) is not bytes else s
	if not b.endswith(b"\n"):
		b += b"\n"
	sys.__stdout__.buffer.write(b)
	sys.__stdout__.flush()

def bsend(*args):
	for b in args:
		# b = str(s).encode("utf-8") if type(s) is not bytes else s
		sys.__stderr__.buffer.write(b)
	sys.__stderr__.flush()

print = lambda *args, sep=" ", end="\n": point(repr(str(sep).join(map(str, args)) + end))
print_exc = lambda: point(repr(traceback.format_exc()))

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

def get_device(name=None):
	if not name:
		return
	global DEVICE
	try:
		DEVICE = sc.get_speaker(name)
	except (IndexError, RuntimeError):
		pass
	else:
		point("~W")
		return DEVICE
	point(f"~w {OUTPUT_DEVICE}")
	DEVICE = sc.default_speaker()
	return DEVICE

audio_format = pyglet.media.codecs.AudioFormat(
	channels=2,
	sample_size=16,
	sample_rate=48000,
)

class Source(pyglet.media.Source):

	emptybuff = b"\x00" * 8192
	audio_format = audio_format

	def __init__(self):
		self.buffer = deque()
		self.position = 0

	def get_audio_data(self, num_bytes, compensation_time=0):
		if not self.buffer:
			data = self.emptybuff[:num_bytes]
		elif len(self.buffer) == 1:
			data = bytes(self.buffer.popleft())
		else:
			data = b""
			while len(self.buffer) > 1 and len(data) < num_bytes:
				data += self.buffer.popleft()
		if len(data) > num_bytes:
			data, extra = data[:num_bytes], data[num_bytes:]
			self.buffer.appendleft(extra)
		pos = self.position
		ts = max(0.004, len(data) / (audio_format.sample_rate * audio_format.sample_size * audio_format.channels / 8))
		self.position += ts
		return pyglet.media.codecs.AudioData(data, len(data), pos, inf, [])

class Player(pyglet.media.Player):

	type = "pyglet"
	peak = 32767
	dtype = np.int16
	channels = 2
	re_paused = False

	def __init__(self):
		super().__init__()
		self.paused = False
		self.entry = Source()
		self.wait()
		# point(self.entry)

	def write(self, data):
		self.wait()
		data = data.data
		if len(self.entry.buffer) >= 3:
			ts = max(0.004, len(data) / (audio_format.sample_rate * audio_format.sample_size * audio_format.channels / 8)) - 0.004
			time.sleep(ts)
		self.re_paused = False
		if not self.paused:
			self.entry.buffer.append(data)

	def wait(self):
		while self.playing and not self.paused and self.source and len(self.entry.buffer) >= 4:
			async_wait()
		if not self.entry.buffer:
			if not self.re_paused:
				for i in range(3):
					self.entry.buffer.append(self.entry.emptybuff)
				self.re_paused = True
			else:
				super().pause()
		if len(self.entry.buffer) >= 1:
			if not self.source:
				self.queue(self.entry)
			if not self.paused and not self.playing:
				self.play()

	def pause(self):
		self.paused = True
		# self.entry.buffer.clear()
		# super().pause()

	def resume(self):
		self.paused = False
		self.play()

	stop = pause
	# def stop(self):
		# self.pause()

PG_USED = None
SC_EMPTY = np.zeros(3200, dtype=np.float32)
def sc_player(d=None):
	if not d:
		player = Player()
	else:
		cc = d.channels
		t = (d.name, cc)
		try:
			if not PG_USED or PG_USED == t:
				raise RuntimeError
			player = d.player(SR, cc, 2048)
		except RuntimeError:
			if not PG_USED:
				pygame.mixer.init(SR, -16, cc, 512, devicename=d.name)
			globals()["PG_USED"] = t
			player = pygame.mixer
			player.type = "pygame"
			player.dtype = np.int16
			player.peak = 32767
			player.resume = player.unpause
			def stop():
				player._data_ = ()
			player.pause = stop
			player.stop = stop
			try:
				player.resume()
			except:
				print_exc()
		else:
			player.__enter__()
			player.type = "soundcard"
			player.dtype = np.float32
			player.peak = 1
			player.resume = lambda: None
			player.pause = player.stop = lambda: setattr(player, "_data_", ())
		player.channels = cc
	if not getattr(player, "_data_", None):
		player._data_ = ()
	player.closed = False
	player.is_playing = None
	player.fut = None
	# a monkey-patched play function that has a better buffer
	# (soundcard's normal one is insufficient for continuous playback)
	def play(self):
		while True:
			if self.closed or paused and not paused.done() or not fut and not alphakeys or cleared:
				if len(self._data_) > 6400 * cc:
					self._data_ = self._data_[-6400 * cc:]
				return
			w2 = 1600 * cc
			towrite = self._render_available_frames()
			t2 = towrite << 1
			if towrite < w2:
				async_wait()
				continue
			if self.fut:
				self.fut.result()
			self.fut = concurrent.futures.Future()
			if not len(self._data_):
				self._data_ = SC_EMPTY[:w2]
			if t2 > len(self._data_) + w2:
				t2 = len(self._data_) + w2
			b = self._data_[:t2].data
			buffer = self._render_buffer(towrite)
			CFFI.memmove(buffer[0], b, b.nbytes)
			self._render_release(towrite)
			self._data_ = self._data_[t2:]
			if self.closed:
				return
			self.fut.set_result(None)
	def play2(self):
		channel = self.Channel(0)
		while True:
			if self._data_ and not channel.get_queue():
				channel.queue(self._data_.popleft())
			async_wait()
	def write(data):
		if player.closed:
			return
		cc = player.channels
		if cc < 2:
			if data.dtype == np.float32:
				data = np.add(data[::2], data[1::2], out=data[:len(data) >> 1])
				data *= 0.5
			else:
				data >>= 1
				data = np.add(data[::2], data[1::2], out=data[:len(data) >> 1])
		if player.type == "pygame":
			if cc >= 2:
				data = data.reshape((len(data) // cc, cc))
			sound = pygame.sndarray.make_sound(data)
			player.wait()
			channel = player.Channel(0)
			if channel.get_queue():
				try:
					player._data_.append(sound)
				except AttributeError:
					player._data_ = deque((sound,))
				return verify()
			channel.queue(sound)
			return verify()
		player.wait()
		if not len(player._data_):
			player._data_ = data
			return verify()
		player.fut = concurrent.futures.Future()
		player._data_ = np.concatenate((player._data_, data))
		player.fut.set_result(None)
		return verify()
	if player.type != "pyglet":
		player.write = write
	def close():
		if player.type == "pygame":
			player._data_ = ()
			return pygame.mixer.pause()
		if player.type == "pyglet":
			return player.delete()
		player.closed = True
		try:
			player.__exit__(None, None, None)
		except:
			print_exc()
	player.close = close
	if player.type != "pyglet":
		def wait():
			cc = player.channels
			if player.type == "pygame":
				verify()
				while len(player._data_) >= 4:
					async_wait()
				return
			if not len(player._data_):
				return
			verify()
			while len(player._data_) > 6400 * cc:
				async_wait()
			while player.fut and not player.fut.done():
				player.fut.result()
		player.wait = wait
	def verify():
		if not player.is_playing or player.is_playing.done():
			func = play2 if player.type == "pygame" else play
			player.is_playing = submit(func, player)
	if player.type == "pygame":
		verify()
	return player

get_channel = lambda: sc_player(get_device(OUTPUT_DEVICE))
DEVICE = None
OUTPUT_DEVICE = None
OUTPUT_FILE = OUTPUT_VIDEO = None
video_write = None
channel = get_channel()

ffmpeg = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
ffprobe = "ffprobe.exe" if os.name == "nt" else "ffprobe"

def _get_duration(filename, _timeout=12):
	command = (
		ffprobe,
		"-v",
		"error",
		"-select_streams",
		"a:0",
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
		resp = proc.stdout.read().split()
	except:
		with suppress():
			proc.kill()
		print_exc()
	try:
		dur = float(resp[0])
	except (IndexError, ValueError):
		dur = None
	bps = None
	if resp and len(resp) > 1:
		with suppress(ValueError):
			bps = float(resp[1])
	return dur, bps

def get_duration(filename):
	if not is_url(filename) and filename.endswith(".pcm"):
		return os.path.getsize(filename) / (48000 * 2 * 2)
	if filename:
		dur, bps = _get_duration(filename, 4)
		if not dur and is_url(filename):
			resp = reqs.head(filename)
			head = {k.casefold(): v for k, v in resp.headers.items()}
			if "content-length" not in head:
				return _get_duration(filename, 20)[0]
			if bps:
				return (int(head["content-length"]) << 3) / bps
			ctype = [e.strip() for e in head.get("content-type", "").split(";") if "/" in e][0]
			if ctype.split("/", 1)[0] not in ("audio", "video"):
				return nan
			if ctype == "audio/midi":
				return nan
		return dur

probe_cache = {}
def probe(stream):
	try:
		return probe_cache[stream]
	except KeyError:
		pass
	command = (
		ffprobe,
		"-v",
		"error",
		"-select_streams",
		"a:0",
		"-show_entries",
		"stream=codec_name,",
		"-show_entries",
		"format=format_name",
		"-of",
		"default=nokey=1:noprint_wrappers=1",
		stream,
	)
	print(command)
	resp = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out = list(s.strip() for s in reversed(as_str(resp.stdout).splitlines()))
	probe_cache[stream] = out
	return out

def kill(proc):
	try:
		return proc.kill2()
	except:
		pass
	proc.kill()

LT = 0
def duration_est():
	global duration
	last_fn = ""
	last_mt = 0
	last_fs = 0
	while True:
		if DEVICE:
			try:
				t = time.time()
				if floor(t / 8) * 8 > LT:
					globals()["LT"] = t
					if DEVICE:
						cc = sc.get_speaker(DEVICE.id).channels
						if cc != channel.channels:
							raise
				if hasattr(channel, "wait"):
					fut = submit(channel.wait)
					fut.result(timeout=3)
			except:
				print_exc()
				PROC.terminate()
		try:
			if not is_minimised() and stream and not is_url(stream) and (stream[0] != "<" or stream[-1] != ">") and os.path.exists(stream):
				stat = None
				if last_fn == stream:
					stat = os.stat(stream)
					if stat.st_mtime == last_mt and stat.st_size == last_fs:
						time.sleep(2)
						continue
				if not stat:
					stat = os.stat(stream)
				last_fn = stream
				last_mt = stat.st_mtime
				last_fs = stat.st_size
				duration = get_duration(stream)
		except:
			print_exc()
		time.sleep(0.5)

def header():
	return {
		"User-Agent": f"Mozilla/5.{random.randint(1, 9)}",
		"DNT": "1",
		"X-Forwarded-For": ".".join(str(random.randint(1, 254)) for _ in range(4)),
	}

def proxy_download(url, fn=None, proxy=True, timeout=24):
	if proxy:
		loc = random.choice(("eu", "us"))
		i = random.randint(1, 17)
		stream = f"https://{loc}{i}.proxysite.com/includes/process.php?action=update"
		print(url, stream, fn, sep="\n")
		req = reqs.post(
			stream,
			data=dict(d=url, allowCookies="on"),
			headers=header(),
			timeout=timeout,
			stream=True,
		)
	else:
		req = reqs.get(
			url,
			headers=header(),
			timeout=timeout,
			stream=True
		)
	with req as resp:
		if resp.status_code not in range(200, 400):
			raise ConnectionError(resp.status_code, resp)
		if not fn:
			return resp.content
		try:
			size = int(resp.headers["Content-Length"])
		except (KeyError, ValueError):
			size = None
		it = resp.iter_content(65536)
		with open(fn, "wb") as f:
			try:
				while True:
					b = next(it)
					if not b:
						break
					f.write(b)
			except StopIteration:
				pass
			except:
				from traceback import print_exc
				print_exc()
		if size:
			for i in range(3):
				pos = os.path.getsize(fn)
				if pos >= size:
					break
				print(f"Incomplete download ({pos} < {size}), resuming...")
				pos = max(0, pos - 65536)
				h = header()
				h["Range"] = f"bytes={pos}-"
				resp = reqs.get(url, headers=h, timeout=timeout, stream=True)
				resp.raise_for_status()
				it = resp.iter_content(65536)
				with open(fn, "ab") as f:
					f.truncate(pos)
					try:
						while True:
							b = next(it)
							if not b:
								raise StopIteration
							f.write(b)
					except StopIteration:
						continue
					except:
						from traceback import print_exc
						print_exc()
			pos = os.path.getsize(fn)
			if pos < size:
				raise EOFError(f"{url}: Incomplete download ({pos} < {size}), unable to resolve.")
		return fn

is_youtube_stream = lambda url: url and re.findall(r"^https?:\/\/r+[0-9]+---.{2}-[A-Za-z0-9\-_]{4,}\.googlevideo\.com", url)
downloading = set()
def download(url, fn):
	try:
		if fn in downloading:
			return
		downloading.add(fn)
		cmd = ffmpeg_start
		if is_youtube_stream(url) and (len(downloading) >= 3 or getattr(proc, "downloading", False)):
			fi = "cache/" + str(time.time_ns() + random.randint(1, 1000))
			try:
				proxy_download(url, fi)
			except ConnectionError as ex:
				print(f"[DEBUG] Pre-emptive download errored out with status {ex.errno}.")
			except:
				print_exc()
			else:
				if os.path.getsize(fi):
					with open(fi, "rb") as f:
						if f.read(15) != b"<!DOCTYPE html>":
							url = fi
						else:
							print("[DEBUG] Pre-emptive download returned invalid HTML.")
				else:
					print("[DEBUG] Pre-emptive download returned empty.")
				if url != fi:
					try:
						os.remove(fi)
					except:
						pass
		if (fn.endswith(".webm") or url.endswith(".pcm") and fn.endswith(".pcm")) and not is_url(url) and os.path.exists(url) and os.path.getsize(url):
			if url != fn:
				os.rename(url, fn)
			downloading.discard(fn)
			return
		if fn.endswith(".webm") and is_url(url) and url.rsplit(".", 1)[-1] not in ("mp4", "mov", "avi", "mkv"):
			resp = reqs.get(
				url,
				headers=header(),
			)
			b = resp.content
			with open(fn, "wb") as f:
				f.write(b)
			downloading.discard(fn)
			return
		if not is_url(url):
			fi = fn
		else:
			fi = "cache/" + str(time.time_ns() + random.randint(1, 1000))
		cmd += ("-nostdin", "-i", url)
		if fn.endswith(".pcm"):
			cmd += ("-f", "s16le")
		else:
			cmd += ("-b:a", "224k")
		cmd += ("-ar", "48k", "-ac", "2", fi)
		print(cmd)
		code = subprocess.Popen(cmd).wait()
		if code:
			raise RuntimeError(code)
		if not is_url(url) and os.path.exists(url):
			os.remove(url)
		if fi != fn and os.path.exists(fi):
			if os.path.exists(fn):
				os.remove(fn)
			os.rename(fi, fn)
	except StopIteration:
		pass
	except:
		print_exc()
	if is_url(url) and (not os.path.exists(fn) or os.path.getsize(fn) < 48000):
		try:
			os.remove(fn)
		except:
			pass
		point(f"~R {url}")
	downloading.discard(fn)


removing = set()
def remover():
	try:
		while True:
			for fn in tuple(removing):
				try:
					os.remove(fn)
				except PermissionError:
					continue
				except FileNotFoundError:
					pass
				removing.discard(file)
			time.sleep(0.5)
	except:
		print_exc()

def remove(file):
	if file:
		fn = file.name
		if "\x7f" in fn:
			removing.add(fn)

def stdclose(p):
	try:
		if not p.stdin.closed:
			fut = submit(p.stdin.close)
			fut.result(timeout=1)
		time.sleep(2)
	except concurrent.futures.TimeoutError:
		pass
	except:
		print_exc()
	print("Closing", p)
	p.kill()

shuffling = False
transfer = False
BSIZE = 1600
RSIZE = BSIZE << 2
TSIZE = BSIZE // 3
def reader(f, pos=None, reverse=False, shuffling=False, pcm=False):
	global proc, transfer
	try:
		if pcm:
			rsize = RSIZE << 3
		else:
			rsize = RSIZE
		if getattr(f, "closed", None):
			raise StopIteration
		if pos is None:
			pos = f.tell()
		if reverse:
			pos -= rsize
		if pos:
			f.seek(pos)
		opos = pos
		while True:
			# print(proc)
			lpos = pos
			while shuffling:
				for i in range(1024):
					pos = random.randint(0, fsize >> 2) << 2
					if abs(pos - lpos) > 65536:
						break
				f.seek(pos)
				pos += BSIZE
				b = f.read(BSIZE)
				if len(b) & 1:
					b = memoryview(b)[:-1]
				a = np.frombuffer(b, dtype=np.int16)
				u, c = np.unique(a, return_counts=True)
				s = np.sort(c)
				x = np.sum(s[-3:])
				if x >= TSIZE:
					while True:
						b = f.read(BSIZE)
						pos += BSIZE
						if not b:
							if type(proc) is cdict:
								pos = 0
								f.seek(0)
							break
						if len(b) & 1:
							b = memoryview(b)[:-1]
						a = np.frombuffer(b, dtype=np.int16)
						u, c = np.unique(a, return_counts=True)
						s = np.sort(c)
						x = np.sum(s[-3:])
						if not x >= TSIZE:
							pos = round(pos / 4) << 2
							if type(proc) is cdict and pos >= fsize - 65536:
								pos = 0
							f.seek(pos)
							break
					globals()["pos"] = pos / fsize * duration
					globals()["frame"] = globals()["pos"] * 30
					print(f"Autoshuffle {pos}/{fsize}")
					shuffling = False
					p = proc
					print(p.args)
					proc = psutil.Popen(p.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=65536)
					try:
						proc.kill2 = p.kill2
					except AttributeError:
						pass
					submit(stdclose, p)
					opos = pos
					transfer = True
			try:
				b = f.read(rsize)
			except ValueError:
				b = b""
			if settings.shuffle == 2 and abs(pos - opos) / fsize * duration >= 60:
				if b:
					if len(b) & 1:
						b = memoryview(b)[:-1]
					a = np.frombuffer(b, dtype=np.int16)
					u, c = np.unique(a, return_counts=True)
					s = np.sort(c)
					x = np.sum(s[-3:])
					if x >= rsize // 3:
						if pos != 0:
							print(x, rsize // 3)
						shuffling = True
						continue
			if not b:
				break
			if reverse:
				if len(b) & 1:
					b = memoryview(b)[:-1]
				b = np.flip(np.frombuffer(b, dtype=np.uint16)).data
				pos -= rsize
				if pos <= 0:
					proc.stdin.write(b)
					size = -pos
					if size:
						f.seek(0)
						b = f.read(size)
						if len(b) & 1:
							b = memoryview(b)[:-1]
						b = np.flip(np.frombuffer(b, dtype=np.uint16)).data
						proc.stdin.write(b)
					break
				f.seek(pos)
			else:
				pos += rsize
			p = proc
			try:
				if isinstance(b, memoryview) and not b.c_contiguous:
					b = bytes(b)
				fut = submit(proc.stdin.write, b)
				try:
					fut.result(timeout=12)
					proc.stdin.flush()
				except concurrent.futures.TimeoutError:
					if not paused:
						raise ValueError
					paused.result()
			except (OSError, BrokenPipeError, ValueError, AttributeError):
				if p and p.is_running():
					try:
						p.kill()
					except:
						pass
				if not proc or not proc.is_running():
					break
	except StopIteration:
		pass
	except:
		print_exc()
	try:
		if proc and proc.is_running():
			submit(stdclose, proc)
		f.close()
	except:
		pass

def construct_options(full=True):
	stats = cdict(settings)
	pitchscale = 2 ** ((stats.pitch + stats.nightcore) / 12)
	reverb = stats.reverb
	volume = 1
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
		# elif volume > 1 or abs(stats.bassboost):
		#	 options.append("asoftclip=atan")
		args.append(("-af", "-filter_complex")[bool(reverb)])
		args.append(",".join(options))
	return args

class HWSurface:

	cache = weakref.WeakKeyDictionary()
	anys = {}
	anyque = []
	maxlen = 12

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
		return self

import multiprocessing.shared_memory
globals()["stat-mem"] = multiprocessing.shared_memory.SharedMemory(
	name=f"Miza-Player-{pid}-stat-mem",
	create=False,
)
globals()["osize"] = np.frombuffer(globals()["stat-mem"].buf[8:16], dtype=np.uint32)
globals()["ssize"] = np.frombuffer(globals()["stat-mem"].buf[16:24], dtype=np.uint32)
sbuff2 = np.empty(3200, dtype=np.float32)

stderr_lock = None
def oscilloscope(buffer):
	global stderr_lock
	try:
		if not packet:
			return
		if "osci-mem" not in globals():
			globals()["osci-mem"] = multiprocessing.shared_memory.SharedMemory(
				name=f"Miza-Player-{pid}-osci-mem",
				create=False,
			)
		temp = np.clip(sbuffer[:3200], -1, 1, out=sbuff2, casting="unsafe")
		# temp = np.asanyarray(sbuffer[:3200], dtype=np.float32)
		globals()["osci-mem"].buf[0] = 1
		globals()["osci-mem"].buf[1:12801] = temp.view(np.uint8).data
		globals()["osci-mem"].buf[0] = 0
	except:
		print_exc()
osci_clear = b"\x00" * 12801
def clear_osci():
	globals()["osci-mem"].buf[:12801] = osci_clear

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

maxfreq = 27.5 * 2 ** ((highest_note + 1.5) / 12)
minfreq = 27.5 * 2 ** ((lowest_note - 1.5) / 12)
barcount = int(highest_note - lowest_note) + 1 + 2
freqmul = 1 / (1 - log(minfreq, maxfreq))

res_scale = 84000
dfts = res_scale // 2 + 1
fff = np.fft.fftfreq(res_scale, 1 / 48000)[:dfts]
fftrans = np.zeros(dfts, dtype=np.uint16)
freqscale = 4

bins = barcount * freqscale - 1
for i, x in enumerate(fff):
	if x <= 0:
		continue
	x = round((1 - log(x, maxfreq)) * freqmul * bins)
	if x > bins or x < 0:
		continue
	fftrans[i] = x

spec_update_fut = None
lastspec = 0
lastspec2 = 0

def spectrogram_update():
	global lastspec, spec_update_fut, spec2_fut, spec_buffer, packet_advanced2, packet_advanced3
	try:
		t = pc()
		dur = max(0.001, min(0.125, t - lastspec))
		lastspec = t
		dft1 = np.fft.rfft(spec_buffer)
		if dft1.dtype == np.complex64:
			dft = dft1
		else:
			try:
				dft = globals()["spec-dft-arr"]
			except KeyError:
				dft = globals()["spec-dft-arr"] = np.empty(len(dft1), dtype=np.complex64)
			dft[:] = dft1
		try:
			arr = globals()["spec-fft-arr"]
		except KeyError:
			arr = globals()["spec-fft-arr"] = np.zeros(barcount * freqscale, dtype=dft.dtype)
		else:
			arr.fill(0)
		np.add.at(arr, fftrans, dft)
		arr[0] = 0
		try:
			amp = globals()["spec-fft-amp"]
		except KeyError:
			amp = globals()["spec-fft-amp"] = np.empty(barcount * freqscale, dtype=np.float32)
		np.abs(arr, out=amp)
		x = barcount - np.argmax(amp) / freqscale - 0.5
		point(f"~n {x}")
		if settings.spectrogram > 0:
			if settings.spectrogram == 2:
				amp = supersample(amp, barcount // 4 + 1, in_place=True)
			else:
				amp = supersample(amp, barcount, in_place=True)
			amp = np.asanyarray(amp, dtype=np.float32)
			if "spec-mem" not in globals():
				globals()["spec-mem"] = multiprocessing.shared_memory.SharedMemory(
					name=f"Miza-Player-{pid}-spec-mem",
					create=False,
				)
			globals()["spec-mem"].buf[:len(amp) * 4] = amp.view(np.uint8).data
		spec_buffer *= (1 / 3) ** dur
	except:
		print_exc()

spec_empty = np.zeros(res_scale, dtype=np.float32)
spec_buffer = spec_empty
def spectrogram():
	global spec_buffer, spec2_fut, packet_advanced3
	try:
		if packet and sample is not None:
			try:
				buffer = globals()["spec-sample-32"]
			except KeyError:
				buffer = globals()["spec-sample-32"] = np.empty(len(sample), dtype=np.float32)
			if sample.dtype != np.float32:
				buffer[:] = sample
				buffer *= 1 / channel.peak
			else:
				buffer[:] = sample
			spec_buffer[:-len(buffer)] = spec_buffer[len(buffer):]
			spec_buffer[-len(buffer):] = buffer
			spectrogram_update()
	except:
		print_exc()

packet_advanced = None
packet_advanced2 = None
packet_advanced3 = None
emptyamp = np.zeros(barcount, dtype=np.float32)
osci_fut = None
spec_fut = None
spec2_fut = None
def render():
	global lastpacket, osci_fut, spec_fut, spec2_fut, packet_advanced, sbuffer, sblock
	try:
		while True:
			t = pc()
			sleep = 0.005
			if lastpacket != id(packet) and sbuffer is not None and not sblock:
				lastpacket = id(packet)
				sblock = True
				if len(sbuffer) > 3200:
					buffer = sbuffer[:3200]
					sbuffer = sbuffer[3200:]
					sleep = 1 / 30
				else:
					buffer = sbuffer.copy()

				amp = np.sum(np.abs(buffer)) / len(buffer)
				p_amp = sqrt(amp)

				if is_minimised():
					point(f"~y {p_amp}")
				else:
					if settings.oscilloscope and osize[0] and osize[1] and (not osci_fut or osci_fut.done()):
						osci_fut = submit(oscilloscope, buffer)
					out = []
					out.append(round(max(np.max(buffer), -np.min(buffer)) * 100, 3))
					out.append(round(amp * 100, 2))

					if packet:
						vel1 = buffer[::2][1:] - buffer[::2][:-1]
						vel2 = buffer[1::2][1:] - buffer[1::2][:-1]
						amp1 = np.mean(np.abs(vel1))
						amp2 = np.mean(np.abs(vel2))
						vel = (amp1 + amp2)
						out.append(round(vel / 4 * 100, 3))

						out = [min(i, 100) for i in out]
						out.append(p_amp)

						if packet:
							point("~x " + " ".join(map(str, out)))
				sblock = False

			packet_advanced = False
			dur = sleep + t - pc()
			if dur > 0:
				time.sleep(dur)
	except:
		print_exc()

emptybuff = b"\x00" * (FR * 2 * 2)
emptymem = memoryview(emptybuff)
emptysample = np.frombuffer(emptybuff, dtype=np.uint16)

def play(pos):
	global file, fn, proc, drop, quiet, frame, packet, lastpacket, sample, transfer, point_fut, spec_fut, sbuffer, sblock, packet_advanced, packet_advanced2, packet_advanced3, sfut, video_write
	skipzeros = False
	try:
		frame = pos * 30
		while True:
			if paused and drop <= 0:
				paused.result()
			if stopped:
				break
			p = proc
			b = b""
			if fn:
				if not file:
					try:
						# print(os.path.getsize(fn))
						if os.path.getsize(fn) < req:# or os.path.getsize(fn) < req * 150 and proc and proc.is_running():
							raise FileNotFoundError
						file = open(fn, "rb") # *cough* "FM", Miza is secretly a radio host and Txin's house is a radio station - Smudge
					except (OSError, FileNotFoundError, PermissionError):
						# print(proc)
						if proc and not proc.is_running():
							point("~r")
							proc = None
							raise
						async_wait()
						continue
				try:
					b = file.read(req)
				except ValueError:
					if getattr(file, "name") and os.path.exists(file.name):
						file.close()
						file = open(file.name, "rb")
						try:
							b = file.read(req)
						except ValueError:
							pass
					if not b:
						if proc_waiting:
							break
				except:
					print_exc()
				if not b and proc and proc.is_running():
					continue
				# print(len(b))
			else:
				while not getattr(proc, "readable", lambda: True)():
					async_wait()
				try:
					fut = submit(proc.stdout.read, req)
					b = fut.result(timeout=4)
				except (AttributeError, concurrent.futures.TimeoutError):
					pass
			if not b:
				# print(transfer, proc)
				if transfer and proc and proc.is_running():
					transfer = False
					fut = submit(proc.stdout.read, req)
					b = fut.result(timeout=4)
					drop = 0
				else:
					# print(f"{proc} {fn}, {file}")
					if proc:
						if proc.is_running():
							try:
								kill(proc)
							except:
								pass
						point("~s")
					if file:
						temp, file = file, None
						temp.close()
						remove(file)
					fn = file = proc = None
					packet = None
					drop = 0
					return
			if p and p != proc and p.is_running():
				try:
					p.kill()
				except:
					pass
			try:
				if drop > 0:
					drop -= 1
					raise StopIteration
				r = b
				if len(b) < req:
					b += bytes(emptymem[:req - len(b)])
				if len(b) & 1:
					b = memoryview(b)[:-1]
				sample = np.frombuffer(b, dtype=np.int16)
				if sfut:
					s = sfut.result()
				else:
					s = synthesize()
				if s is None:
					if settings.silenceremove and np.mean(np.abs(sample)) < 64:
						if quiet >= 15:
							raise StopIteration
						quiet += 1
					else:
						quiet = 0
				sfut = submit(synthesize)
				if channel.dtype == np.float32:
					try:
						globals()["s-temp32"][:] = sample
					except KeyError:
						globals()["s-temp32"] = sample.astype(np.float32)
					sample = globals()["s-temp32"]
					sample *= 1 / 32767
				if settings.volume != 1 or s is not None:
					if settings.volume != 1:
						if sample.dtype != np.float32:
							try:
								globals()["s-temp32"][:] = sample
							except KeyError:
								globals()["s-temp32"] = sample.astype(np.float32)
							sample = globals()["s-temp32"]
						try:
							np.multiply(sample, settings.volume, out=sample)
						except:
							sample = sample * settings.volume
						# sample = np.clip(sample, -channel.peak, channel.peak, out=sample).astype(np.int16)
					if s is not None:
						if sample.dtype != np.float32:
							try:
								globals()["s-temp32"][:] = sample
							except KeyError:
								globals()["s-temp32"] = sample.astype(np.float32)
							sample = globals()["s-temp32"]
							s *= 32767
						s += sample
						sample = s
				if s is not None:
					if settings.silenceremove and np.mean(np.abs(sample)) < channel.peak / 512:
						if quiet >= 15:
							raise StopIteration
						quiet += 1
					else:
						quiet = 0
				if sample.dtype != np.int16:
					sample = np.clip(sample, -channel.peak, channel.peak, out=sample)
				if sample.dtype != channel.dtype:
					# try:
						# if globals()["s-tempc"].dtype != channel.dtype:
							# raise KeyError
						# globals()["s-tempc"][:] = sample
					# except KeyError:
						# globals()["s-tempc"] = sample.astype(channel.dtype)
					# sample = globals()["s-tempc"]
					sample = sample.astype(channel.dtype)
				if channel.dtype != np.float32:
					while sblock:
						time.sleep(0.004)
					sblock = True
					try:
						globals()["s-buf32"][:] = sample
					except KeyError:
						globals()["s-buf32"] = sample.astype(np.float32)
					sbuffer = globals()["s-buf32"]
					sbuffer *= 1 / channel.peak
					sblock = False
				else:
					sbuffer = sample
				lastpacket = None
				packet = sample.data
				packet_advanced = True
				packet_advanced2 = True
				packet_advanced3 = True
				if OUTPUT_FILE:
					s = sample
					if s.dtype != np.float32:
						try:
							globals()["s-temp32"][:] = sample
						except KeyError:
							globals()["s-temp32"] = sample.astype(np.float32)
						s = globals()["s-temp32"]
						s *= 1 / 32767
					submit(OUTPUT_FILE.write, s.data)
				if not point_fut or point_fut.done():
					point_fut = submit(point, f"~{frame} {duration}")
				if settings.get("insights") != 0 or settings.spectrogram > 0:
					if ssize[0] and ssize[1] and not is_minimised():
						if spec_fut:
							spec_fut.result()
						spec_fut = submit(spectrogram)
				globals()["lastplay"] = pc()
				while waiting:
					waiting.result()
				for i in range(2147483648):
					try:
						# t = time.time()
						# if floor(t) > LT:
							# globals()["LT"] = t
							# if DEVICE:
								# cc = sc.get_speaker(DEVICE.id).channels
								# if cc != channel.channels:
									# raise
						if not settings.subprocess:
							b = None
							while not b:
								b = mixer_server.recv(1)
							submit(mixer_server.sendall, sample.data).result(timeout=3)
						else:
							fut = submit(channel.wait)
							fut.result(timeout=1.6)
							fut = submit(channel.write, sample)
							fut.result(timeout=1.2)
					except:
						if paused:
							break
						print_exc()
						print(f"{channel.type} timed out.")
						globals()["waiting"] = concurrent.futures.Future()
						if i > 1:
							PROC.terminate()
						else:
							channel.close()
						globals()["channel"] = get_channel()
						globals()["waiting"], w = None, waiting
						w.set_result(None)
						PROC.terminate()
						# globals()["waiting"] = concurrent.futures.Future()
						# if i > 1:
							# PROC.terminate()
						# else:
							# channel.close()
						# globals()["channel"] = get_channel()
						# globals()["waiting"], w = None, waiting
						# w.set_result(None)
					else:
						break
				if OUTPUT_VIDEO and settings.spectrogram > 0:
					if ssize[0] and ssize[1]:
						t = pc()
						while not video_write and pc() - t < 1:
							async_wait()
						try:
							video_write.result()
						except (AttributeError, ValueError):
							pass
						video_write = None
			except StopIteration:
				pass
			frame += settings.speed * 2 ** (settings.nightcore / 12)
	except:
		try:
			kill(proc)
		except:
			pass
		print_exc()


baserange = np.arange(FR, dtype=np.float64)
basewave = baserange / FR
c = SR // 48
cm = np.linspace(-2, 2, c, endpoint=True)
import scipy.special
cm = scipy.special.erf(cm)
cm += 1
cm /= 2
den = 2
da = 0
db = 0
dc = 8

wavecache = {}
sel_instrument = 0

class Sample(collections.abc.Hashable):

	def __init__(self, data, opt):
		self.data = np.frombuffer(zip2bytes(base64.b85decode(data)), dtype=np.float32)
		self.cache = {}
		self.opt = opt

	def __hash__(self):
		data = self.data
		x = ceil(log2(len(data)))
		y = [round(i) for i in np.linspace(0, len(data) - 1, num=x, dtype=np.float32)]
		return hash(np.sum(data[y]))

	def __eq__(self, other):
		if not isinstance(other, self.__class__):
			return False
		return np.all(self.data == other.data)

	__len__ = lambda self: len(self.data)

	def get(self, period=None):
		if period is None:
			return self.data
		period = round(period / 8) << 3
		if period not in self.cache:
			self.cache[period] = supersample(self.data, period)
			print(period)
			print(len(self.cache))
		return self.cache[period]


lastplay = 0
sfut = None

def synthesize():
	global prevkeys, den, da, db, dc, synth_buff
	if not wavecache:
		return
	totalkeys = alphakeys.union(prevkeys) if alphakeys or prevkeys else ()
	dt = max(1, len(totalkeys)) + 1
	if db != dt:
		if da != 0:
			dc = 0
		da = den
		db = dt
	instrument = wavecache[sel_instrument]
	u = instrument.opt
	s = None
	for i in totalkeys:
		partials = [(x / (u[0] - 1) * u[1] * 2) - u[1] for x in range(u[0])] if u[0] > 1 and u[1] else [0] * u[0]
		for p in partials:
			freq = 110 * 2 ** ((i + p + 3 + settings.get("pitch", 0) + settings.get("nightcore", 0)) / 12)
			period = SR / freq
			synth_period = period * ceil(4096 / period)
			offs = buffoffs % synth_period
			wave = instrument.get(synth_period)
			if len(partials) > 1:
				wave = wave * (1 / len(partials))
			c = SR // 48
			if i in alphakeys:
				xa = np.linspace(0, period, len(wave), endpoint=False, dtype=np.float32)
				space = np.linspace(offs, FR + offs, FR, endpoint=False, dtype=np.float32)
				space %= period
				wave = np.interp(space, xa, wave)
				if i not in prevkeys:
					x = min(int((period - offs) % period), FR - c)
					wave[:x] = 0
					wave[x:x + c] *= cm
				if s is not None:
					s += wave
				else:
					s = wave
			elif i in prevkeys:
				x = min(int((period - offs - c) % period), FR - c)
				if s is None:
					s = np.zeros(FR, dtype=np.float32)
				xa = np.linspace(0, period, len(wave), endpoint=False, dtype=np.float32)
				space = np.linspace(offs, x + c + offs, x + c, endpoint=False, dtype=np.float32)
				space %= period
				wave = np.interp(space, xa, wave)
				samplespace = np.linspace(-2, 2, x + c, dtype=np.float32)
				samplespace = np.asanyarray(scipy.special.erf(samplespace), dtype=np.float32)
				samplespace -= 1
				wave *= samplespace
				wave *= -0.5
				s[:x + c] += wave
	if s is not None:
		globals()["buffoffs"] += FR
		m = 1
		if dc < 5:
			lin = basewave + dc
			lin -= 2
			lin = np.asanyarray(scipy.special.erf(lin), dtype=np.float32)
			lin += 1
			lin *= (db - da) / 2
			lin += da
			den = lin[-1]
			s /= lin
			dc += 1
		else:
			den = db
			m /= den
			da = db
		m *= settings.get("volume", 1)
		if m != 1:
			s *= m
		s = np.repeat(s, 2)
	else:
		globals()["buffoffs"] = 0
		den = db
		dc = 8
		da = 0
	prevkeys = alphakeys
	return s


def piano_player():
	global sample, sbuffer, spec_fut, point_fut, ssize, lastpacket, packet, packet_advanced, packet_advanced2, packet_advanced3, sfut, video_write
	try:
		while not hasattr(channel, "wait"):
			time.sleep(0.1)
		while True:
			while is_minimised():
				time.sleep(0.1)
			channel.wait()
			if pc() - lastplay < 0.5:
				async_wait()
				continue
			if sfut:
				s = sfut.result()
			else:
				s = synthesize()
			sfut = submit(synthesize)
			if s is not None and len(s):
				sbuffer = s
				if channel.dtype != np.float32:
					sample = s * 32768
					sample = np.asanyarray(np.clip(sample, -32768, 32767, out=sample), dtype=np.int16)
				else:
					sample = np.clip(s, -1, 1, out=s)
				lastpacket = None
				packet = sbuffer.data
				packet_advanced = True
				packet_advanced2 = True
				packet_advanced3 = True
				if OUTPUT_FILE:
					submit(OUTPUT_FILE.write, packet)
				if not point_fut or point_fut.done():
					point_fut = submit(point, f"~{frame} {duration}")
				if settings.get("insights") != 0 or settings.spectrogram > 0:
					if ssize[0] and ssize[1] and not is_minimised():
						if spec_fut:
							spec_fut.result()
						spec_fut = submit(spectrogram)
				while waiting:
					waiting.result()
				for i in range(2147483648):
					try:
						# t = time.time()
						# if floor(t) > LT:
							# globals()["LT"] = t
							# if DEVICE:
								# cc = sc.get_speaker(DEVICE.id).channels
								# if cc != channel.channels:
									# raise
						if not settings.subprocess:
							b = None
							while not b:
								b = mixer_server.recv(1)
							submit(mixer_server.sendall, sample.data).result(timeout=3)
						else:
							fut = submit(channel.wait)
							fut.result(timeout=1.6)
							fut = submit(channel.write, sample)
							fut.result(timeout=1.2)
					except:
						print_exc()
						print(f"{channel.type} timed out.")
						globals()["waiting"] = concurrent.futures.Future()
						if i > 1:
							PROC.terminate()
						else:
							channel.close()
						globals()["channel"] = get_channel()
						globals()["waiting"], w = None, waiting
						w.set_result(None)
						PROC.terminate()
						# globals()["waiting"] = concurrent.futures.Future()
						# if i > 1:
							# PROC.terminate()
						# else:
							# channel.close()
						# globals()["channel"] = get_channel()
						# globals()["waiting"], w = None, waiting
						# w.set_result(None)
					else:
						break
				if OUTPUT_VIDEO and settings.spectrogram > 0:
					t = pc()
					while not video_write and pc() - t < 1:
						async_wait()
					try:
						video_write.result()
					except (AttributeError, ValueError):
						pass
					video_write = None
			async_wait()
	except:
		print_exc()


PROC = psutil.Process()
def ensure_parent():
	par = psutil.Process(os.getppid())
	while True:
		if par.is_running():
			time.sleep(0.5)
		else:
			PROC.kill()


n_measure = lambda n: n[0]
n_pos = lambda n: n[1]
n_end = lambda n: n[1] + n[4]
n_instrument = lambda n: n[2]
n_pitch = lambda n: n[3]
n_length = lambda n: n[4]
n_volume = lambda n: n[5] if len(n) > 5 else 0.25
n_pan = lambda n: n[6] if len(n) > 6 else 0
n_effects = lambda n: n[7] if len(n) > 7 else ()


synth_samples = np.zeros(0, dtype=np.float32)
def render_notes(i, notes):
	global synth_samples
	timesig = editor.timesig
	note_length = editor.note_length
	samplecount = round_random(timesig[0] * note_length)
	if len(synth_samples) < samplecount << 1:
		if not i:
			synth_samples = np.zeros(samplecount << 1, dtype=np.float32)
		else:
			synth_samples = np.append(synth_samples, np.zeros(samplecount * 2 - len(synth_samples), dtype=np.float32))
	left, right = synth_samples[::2], synth_samples[1::2]
	bufofs = bar * editor.timesig[0] * note_length
	for n in notes:
		v = n_volume(n)
		if v <= 0:
			continue
		instrument = wavecache[n_instrument(n)]
		u = instrument.opt
		wave = instrument.data
		offs = n_pos(n) * note_length
		length = n_length(n) * note_length
		pitch = n_pitch(n)
		partials = [(x / (u[0] - 1) * u[1] * 2) - u[1] for x in range(u[0])] if u[0] > 1 and u[1] else [0] * u[0]
		if len(partials) > 1:
			v /= len(partials)
		for p in partials:
			freq = 440 * 2 ** ((p + pitch - 57) / 12)
			slength = 48000 / freq
			pos = round_random(ceil(offs / slength) * slength + (bufofs + offs) % slength)
			if n_measure(n) < bar:
				pos -= slength
			if length > 2 * slength:
				over = length % slength
				if over:
					length += slength - over
			if v != 1:
				wave = wave * v
			pos = round_random(pos)
			length = round_random(length)
			if pos + length > len(left):
				synth_samples = np.append(synth_samples, np.zeros(pos + length - len(left) << 1, dtype=np.float32))
				left, right = synth_samples[::2], synth_samples[1::2]
				# length = len(right) - pos
				# length -= length % slength
			if length <= 0:
				continue
			positions = np.linspace(0, length / slength * len(wave), length, endpoint=False, dtype=np.float32)
			positions %= len(wave)
			sample = np.interp(positions, np.arange(len(wave)), wave)
			if pos < 0:
				sample = sample[-pos:]
				pos = 0
			sl = slice(pos, pos + len(sample))
			pan = n_pan(n)
			if pan <= -1:
				left[sl] += sample
			elif pan >= 1:
				right[sl] += sample
			elif pan == 0:
				left[sl] += sample
				right[sl] += sample
			else:
				p = 0.5 + (pan / 2)
				lsamp = sample * (1 - p)
				sample *= p
				rsamp = sample
				left[sl] += lsamp
				right[sl] += rsamp
	samples, synth_samples = synth_samples[:samplecount << 1], synth_samples[samplecount << 1:]
	np.clip(samples, -1, 1, out=samples)
	return samples


# sys.stderr.write("~LOADED\n")
seen_urls = set()
hwaccel = "d3d11va" if os.name == "nt" else "auto"
ffmpeg_start = (ffmpeg, "-y", "-hide_banner", "-v", "error", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-err_detect", "ignore_err", "-hwaccel", hwaccel, "-vn")
ffmpeg_stream = ("-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "60")
settings = cdict()
alphakeys = prevkeys = ()
buffoffs = 0
lastpacket = None
packet = None
sample = None
sbuffer = None
sblock = False
cdc = "auto"
duration = inf
stream = ""
sh = ""
fut = None
sf = None
reading = None
stopped = False
cleared = False
proc_waiting = False
point_fut = None
proc = None
fn = None
file = None
paused = None
req = FR * 2 * 2
frame = 0
drop = 0
quiet = 0
failed = 0
editor = None
submit(render)
submit(remover)
submit(duration_est)
submit(piano_player)
submit(ensure_parent)
pc()
waiting = None
# with open("test.txt", "wb"):
	# pass
# sys.stderr.write("~INITIATED\n")
while not sys.stdin.closed and failed < 8:
	try:
		command = sys.stdin.readline()
		# print("\x7f" + command)
		if not command:
			failed += 1
			continue
		failed = 0
		command = command.rstrip().rstrip("\x00")
		pos = frame / 30
		# if command.startswith("~l1"):
			# if stream_locked:
				# stream_locked.set_result(None)
			# stream_locked = concurrent.futures.Future()
			# continue
		# if command.startswith("~u1"):
			# if stream_locked:
				# stream_locked.set_result(None)
			# stream_locked = None
			# continue
		if command.startswith("~render"):
			s = command[8:]
			p, b, s = s.split(" ", 2)
			bar = int(b)
			notes = orjson.loads(s)
			samples = render_notes(b, notes)
			out = np.asanyarray(samples, dtype=np.float32).data
			with open(f"cache/&p{p}b{b}.pcm", "wb") as f:
				f.write(out)
			continue
		if command.startswith("~wave"):
			b = command[6:]
			if b == "clear":
				wavecache.clear()
				continue
			sid, opt, data = b.split(None, 2)
			sid = int(sid)
			s = Sample(data.encode("ascii"), orjson.loads(opt))
			wavecache[sid] = s
			sel_instrument = sid
			continue
		if command.startswith("~select"):
			s = int(command[8:])
			if s in wavecache:
				sel_instrument = s
			else:
				point(f"~t {s}")
			continue
		if command.startswith("~keys"):
			s = command[6:]
			alphakeys = set(map(int, s.split(","))) if s else set()
			cleared = False
			continue
		if command.startswith("~editor"):
			s = command[8:]
			editor = cdict(orjson.loads(s))
			editor.note_length = 60 / editor.tempo * SR
			continue
		if command == "~clear":
			if proc:
				temp, proc = proc, None
				try:
					temp.kill()
				except:
					pass
			if fn and file:
				temp, file = file, None
				temp.close()
				submit(remove, file)
			fn = file = proc = None
			synth_samples = np.zeros(0, dtype=np.float32)
			cleared = True
			channel.stop()
			channel.resume()
			continue
		if command.startswith("~state"):
			i = int(command[6:])
			if i and not paused:
				paused = concurrent.futures.Future()
				channel.pause()
			elif not i and paused:
				paused.set_result(None)
				paused = None
				channel.resume()
			packet_advanced = paused
			if paused:
				pt2 = time.perf_counter() - pt
			else:
				if pt2:
					pt = time.perf_counter() - pt2
				pt2 = None
			continue
		if command.startswith("~download"):
			st, fn2 = command[10:].rsplit(" ", 1)
			if not os.path.exists(fn2) or time.time() - os.path.getmtime(fn2) > 86400 * 7:
				st2 = base64.b85decode(st.encode("ascii")).decode("utf-8", "replace")
				submit(download, st2, fn2)
			continue
		if command.startswith("~output"):
			OUTPUT_DEVICE = command[8:]
			waiting = concurrent.futures.Future()
			submit(channel.close)
			if OUTPUT_DEVICE or channel.type != "pyglet":
				channel = get_channel()
			waiting, w = None, waiting
			w.set_result(None)
			continue
		if command.startswith("~record"):
			s = command[8:]
			if s:
				if " " in s:
					s, v = s.split(None, 1)
					args = (
						ffmpeg, "-y", "-hide_banner", "-v", "error",
						"-hwaccel", hwaccel, "-an",
						"-f", "rawvideo", "-pix_fmt", "rgb24",
						"-video_size", "x".join(map(str, ssize)),
						"-i", "-", "-r", "30", "-b:v", "8M", "-c:v", "h264", v
					)
					print(args)
					OUTPUT_VIDEO = psutil.Popen(args, stdin=subprocess.PIPE, bufsize=int(np.prod(ssize) * 3))
				OUTPUT_FILE = open(s, "ab")
			else:
				if OUTPUT_VIDEO:
					OUTPUT_VIDEO.stdin.close()
				if OUTPUT_FILE:
					OUTPUT_FILE.close()
				if OUTPUT_VIDEO:
					OUTPUT_VIDEO.wait()
					point("~V")
				OUTPUT_FILE = OUTPUT_VIDEO = None
			continue
		if command.startswith("~eval"):
			s = command[6:]
			try:
				code = compile(s, "<terminal.py>", "eval")
			except:
				code = compile(s, "<terminal.py>", "exec")
			resp = eval(code, globals())
			point(resp)
			continue
		if command.startswith("~setting"):
			s = command[9:]
			if s.startswith("#"):
				s = s[1:]
				nostart = True
			else:
				nostart = False
			if s.startswith("{"):
				sets = orjson.loads(s)
				settings.update(sets)
			else:
				setting, value = s.split(None, 1)
				if not os.path.exists(value):
					value = eval(value, {}, {})
				settings[setting] = value
				if setting in ("volume", "shuffle", "spectrogram", "oscilloscope", "unfocus", "insights"):
					continue
			if nostart or not stream:
				continue
		elif command.startswith("~drop"):
			drop += float(command[5:]) * 30
			if drop <= 60 * 30:
				continue
			pos = (frame + drop) / 30
			drop = 0
		elif command == "~quit":
			if proc:
				temp, proc = proc, None
				try:
					temp.kill()
				except:
					pass
			if fn and file:
				temp, file = file, None
				temp.close()
				submit(remove, file)
			fn = file = proc = None
			pygame.mixer.quit()
			break
		elif command == "~init":
			bsend(b"~I\n")
			continue
		elif command != "~replay":
			s = sys.stdin.readline().rstrip().split(" ", 3)
			pos, duration, cdc, sh = s
			pos, duration = map(float, (pos, duration))
			stream = base64.b85decode(command.encode("ascii")).decode("utf-8", "replace")
		# print(stream)
		shuffling = False
		if proc:
			proc_waiting = True
			temp, proc = proc, None
			try:
				temp.kill()
			except:
				pass
			if fn and file:
				temp, file = file, None
				temp.close()
				remove(file)
		if fut:
			stopped = True
			if paused:
				channel._data_ = ()
				paused.set_result(None)
			try:
				fut.result(timeout=2)
			except:
				print_exc()
				print("Previously playing song timed out, killing relevant subprocesses and skipping...")
				drop = 0
				fut = None
			if paused:
				paused = concurrent.futures.Future()
		stopped = False
		if reading:
			reading.result()
			reading = None
		ext = construct_options()
		if is_url(stream):
			for fmt in ("webm", "pcm"):
				fn = "cache/~" + sh + "." + fmt
				if os.path.exists(fn) and abs(os.path.getsize(fn) / 48000 / 2 / 2 - duration) < 1:
					stream = fn
					fn = None
					file = None
					break
			else:
				if pos or not duration < inf or ext:
					ts = time.time_ns() // 1000
					fn = "cache/\x7f" + str(ts) + ".pcm"
		else:
			fn = None
			file = None
		if not stream:
			continue
		cleared = False
		if not is_url(stream) and stream.endswith(".pcm") and not ext and settings.speed >= 0:
			f = open(stream, "rb")
			proc = cdict(
				stdout=cdict(
					read=f.read,
				),
				stderr=cdict(
					read=lambda: b"",
				),
				is_running=lambda: True,
				kill=f.close,
			)
			if pos:
				i = round(pos * SR * 2) * 2
				f.seek(i)
		elif stream[0] == "<" and stream[-1] == ">":
			pya = afut.result()
			i = int(stream.strip("<>"))
			d = pya.get_device_info_by_index(i)
			ai = pya.open(
				SR,
				2,
				pyaudio.paInt16,
				input=True,
				frames_per_buffer=800,
				input_device_index=i,
			)
			proc = cdict(
				stdout=cdict(
					read=lambda n: ai.read(n >> 2, exception_on_overflow=False),
				),
				stderr=cdict(
					read=lambda: b"",
				),
				is_running=lambda: True,
				kill=ai.close,
			)
			proc.readable = lambda: ai.get_read_available() >= req >> 2
			if ext:
				fsize = inf
				f = proc.stdout
				f.seek = f.tell = lambda *args: 0
				f.close = ai.close
				cmd = list(ffmpeg_start + ("-f", "s16le", "-ar", "48k", "-ac", "2", "-i", "-"))
				cmd.extend(ext)
				cmd.extend(("-f", "s16le", "-ar", "48k", "-ac", "2", "-"))
				print(cmd)
				proc = psutil.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=65536)
				reading = submit(reader, f, pos=0, pcm=True)
		else:
			f = None
			if not fn and (cdc == "mp3" and pos >= 960 or settings.shuffle == 2 and duration >= 960) or settings.speed < 0:
				if (fn or not stream.endswith(".pcm")) and (settings.speed < 0 or cdc != "mp3"):
					ostream = stream
					stream = "cache/~" + sh + ".pcm"
					if not os.path.exists(stream) or os.path.getsize(stream) / 48000 / 2 / 2 < duration - 1:
						cmd = ffmpeg_start + ("-nostdin", "-i", ostream, "-f", "s16le", "-ar", "48k", "-ac", "2", stream)
						print(cmd)
						resp = subprocess.run(cmd)
				fn = None
				f = open(stream, "rb")
			cmd = ffmpeg_start
			if is_url(stream):
				cmd += ffmpeg_stream
				fn2 = "cache/~" + sh + ".webm"
				if sh in seen_urls and not os.path.exists(fn2):
					submit(download, stream, fn2)
				else:
					seen_urls.add(sh)
			cmd = list(cmd)
			pcm = False
			if not fn:
				if stream.endswith(".pcm"):
					cmd.extend(("-f", "s16le", "-ar", "48k", "-ac", "2"))
					pcm = True
				elif cdc == "mp3":
					cmd.extend(("-c:a", "mp3"))
				elif cdc == "webm":
					cmd.extend(("-c:a", "copy"))
			cmd.extend(("-nostdin", "-i", "-" if f else stream))
			cmd.extend(ext)
			if fn and stream.rsplit(".", 1)[-1] in ("mp4", "mov", "avi", "mkv"):
				fn = fn.rsplit(".", 1) + ".pcm"
			if fn and fn.endswith(".webm"):
				cmd.extend(("-f", "webm", "-c:a", "copy", fn or "-"))
			else:
				cmd.extend(("-f", "s16le", "-ar", "48k", "-ac", "2", fn or "-"))
			if pos and not f:
				i = cmd.index("-i")
				ss = "-ss"
				cmd = cmd[:i] + [ss, str(pos)] + cmd[i:]
			if not fn and f and pos == 0 and settings.shuffle == 2 and duration >= 960:
				proc = cdict(
					args=cmd,
					is_running=lambda: True,
					stdin=cdict(
						write=lambda b: None,
						close=lambda: None,
						closed=None,
					),
					stdout=cdict(
						read=lambda n: emptymem[:n],
					),
					stderr=cdict(
						read=lambda n: emptymem[:n],
					),
					kill=lambda: None,
					kill2=f.close,
				)
			else:
				print(cmd)
				proc = psutil.Popen(cmd, stdin=subprocess.PIPE if f else subprocess.DEVNULL, stdout=subprocess.DEVNULL if fn else subprocess.PIPE, bufsize=65536)
			if fn and not pos:
				proc.kill = lambda: None
			elif f:
				fsize = os.path.getsize(stream)
				if pos:
					fp = round(pos / duration * fsize / 4) << 2
					if fp > fsize - 2:
						fp = fsize - 2
				else:
					fp = 0 if settings.speed >= 0 else fsize - 2
				reading = submit(reader, f, pos=fp, reverse=settings.speed < 0, shuffling=pos == 0 and settings.shuffle == 2, pcm=pcm)
			if is_url(stream):
				proc.downloading = True
		if point_fut and not point_fut.done():
			point_fut.result()
		point(f"~{pos * 30} {duration}")
		fut = submit(play, pos)
		proc_waiting = False
	except:
		print_exc()
	async_wait()
channel.close()
