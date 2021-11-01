import os, sys, io, time, itertools, traceback, threading

pc = time.perf_counter


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


def as_str(s):
	if type(s) in (bytes, bytearray, memoryview):
		return bytes(s).decode("utf-8", "replace")
	return str(s)


C = COLOURS = cdict(
	red="\x1b[38;5;196m",
	orange="\x1b[38;5;208m",
	yellow="\x1b[38;5;226m",
	chartreuse="\x1b[38;5;118m",
	green="\x1b[38;5;46m",
	spring_green="\x1b[38;5;48m",
	cyan="\x1b[38;5;51m",
	azure="\x1b[38;5;33m",
	blue="\x1b[38;5;21m",
	violet="\x1b[38;5;93m",
	magenta="\x1b[38;5;201m",
	rose="\x1b[38;5;198m",
	black="\u001b[30m",
	white="\u001b[37m",
	reset="\u001b[0m",
)
os.system("color")


class RainbowPrint:

	rainbow = (
		C.red, C.orange, C.yellow, C.chartreuse, C.green, C.spring_green,
		C.cyan, C.azure, C.blue, C.violet, C.magenta, C.rose,
	)
	history = ""
	count = 0
	closed = True
	empty = True

	def __init__(self, stack=False):
		self.buffer = self
		self.stack = stack
		self.data = io.StringIO()
		i = time.time_ns() // 1000 % len(self.rainbow)
		self.it = itertools.cycle(itertools.chain(self.rainbow[i:], self.rainbow[:i]))

	def start(self):
		if self.stack:
			self.thread = threading.Thread(target=self.update_print, daemon=True)
			self.thread.start()
			self.closed = False
		__builtins__["print"] = sys.stdout = sys.stderr = self

	def flush(self):
		m = 120
		try:
			if self.empty:
				return
			self.empty = True
			try:
				self.data.seek(0)
				spl = self.data.readlines()
			except ValueError:
				self.data = io.StringIO()
				return
			if not spl:
				return
			self.data.close()
			self.data = io.StringIO()
			for i, line in enumerate(spl):
				s = ""
				while len(line) > m:
					s += next(self.it) + line[:m]
					line = line[m:]
				if line:
					s += next(self.it) + line + C.reset
				spl[i] = s
			sys.__stdout__.write("".join(spl))
		except:
			sys.__stdout__.write(traceback.format_exc())

	def update_print(self):
		while True:
			try:
				t = pc()
				self.flush()
				time.sleep(max(0.01, t + 0.04 - pc()))
				while self.closed:
					time.sleep(0.5)
			except:
				sys.__stdout__.write(traceback.format_exc())

	def __call__(self, *args, sep=" ", end="\n", prefix="", **void):
		out = str(sep).join(i if type(i) is str else str(i) for i in args) + str(end) + str(prefix)
		if not out:
			return
		temp = out.strip()
		if temp and self.stack:
			if self.history == temp:
				self.count += 1
				return
			elif self.count:
				count = self.count
				self.count = 0
				times = "s" if count != 1 else ""
				out, self.history = f"<Last message repeated {count} time{times}>\n{out}", out
			else:
				self.history = temp
				self.count = 0
		self.empty = False
		self.data.write(out)
		if self.closed:
			self.flush()

	def write(self, *args, end="", **kwargs):
		args2 = [as_str(arg) for arg in args]
		return self.__call__(*args2, end=end, **kwargs)

	read = lambda self, *args, **kwargs: bytes()
	close = lambda self, force=False: self.__setattr__("closed", force)
	isatty = lambda self: False

PRINT = RainbowPrint()