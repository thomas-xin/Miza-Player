print("loading...")
import sys, os, subprocess, traceback

python = ("python3", "python")[os.name == "nt"]


if sys.version_info[0] < 3:
    raise ImportError("Python 3 required.")

print("Loading and checking modules...")

with open("requirements.txt", "rb") as f:
    modlist = f.read().decode("utf-8", "replace").replace("\r", "\n").split("\n")

import pkg_resources

installing = []
install = lambda m: installing.append(subprocess.Popen(["python", "-m", "pip", "install", "--upgrade", m, "--user"]))

for mod in modlist:
    if mod:
        try:
            name = mod
            version = None
            for op in (">=", "==", "<="):
                if op in mod:
                    name, version = mod.split(op)
                    break
            v = pkg_resources.get_distribution(name).version
            if version is not None:
                assert eval(repr(v) + op + repr(version), {}, {})
        except:
            traceback.print_exc()
            inst = name
            if op in ("==", "<="):
                inst += "==" + version
            install(inst)

if installing:
    print("Installing missing or outdated modules, please wait...")
    subprocess.Popen([python, "-m", "pip", "install", "--upgrade", "pip", "--user"]).wait()
    for i in installing:
        i.wait()
    print("Installer terminated.")

if os.name == "nt":
    os.system("color")

print_prefix = ""

class Printer:

    buffer = sys.stdout.buffer

    def flush(self):
        sys.stdout.flush()

    def write(self, s):
        global print_prefix
        if type(s) is not str:
            s = s.decode("utf-8", "replace")
        if print_prefix:
            s, print_prefix = print_prefix + s, ""
        sys.__stdout__.write(s)

sys.stdout = Printer()

def print(*args, sep=" ", end="\n", prefix="\033[1;0;40m", file=None, **void):
    sys.stdout.write("\033[1;37;40m" + str(sep).join(str(i) for i in args) + str(end) + str(prefix))


import contextlib, concurrent.futures

class MultiThreadedImporter(contextlib.AbstractContextManager, contextlib.ContextDecorator):

    def __init__(self, glob=None):
        self.glob = glob
        self.exc = concurrent.futures.ThreadPoolExecutor(max_workers=12)
        self.out = {}

    def __enter__(self):
        return self

    def __import__(self, *modules):
        for module in modules:
            self.out[module] = self.exc.submit(__import__, module)
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type and exc_value:
            raise exc_value
    
    def close(self):
        for k in tuple(self.out):
            self.out[k] = self.out[k].result()
        glob = self.glob if self.glob is not None else globals()
        glob.update(self.out)
        self.exc.shutdown(True)

with MultiThreadedImporter() as importer:
    importer.__import__(
        "collections",
        "time",
        "datetime",
        "ast",
        "copy",
        "pickle",
        "io",
        "random",
        "math",
        "shlex",
        "numpy",
        "re",
        "hashlib",
        "base64",
        "dateutil",
        "itertools",
        "nacl",
        "zipfile",
        "cmath",
        "json",
        "psutil",
        "youtube_dlc",
        "bs4",
        "requests",
    )
# import collections, time, datetime, ast, copy, pickle, io, random, math, shlex, numpy, re, hashlib, base64, dateutil, itertools, nacl, zipfile, cmath, json, psutil, youtube_dlc, bs4, requests


AUTH = {}
if os.path.exists("auth.json"):
    with open("auth.json") as f:
        AUTH = eval(f.read())


suppress = lambda *args, **kwargs: contextlib.suppress(BaseException) if not args and not kwargs else contextlib.suppress(*args + tuple(kwargs.values()))

print_exc = lambda: sys.stdout.write(traceback.format_exc())

class EmptyContext(contextlib.AbstractContextManager):
    __enter__ = lambda self, *args: self
    __exit__ = lambda *args: None

emptyctx = EmptyContext()


class Semaphore(contextlib.AbstractContextManager, contextlib.ContextDecorator, collections.abc.Callable):

    __slots__ = ("limit", "buffer", "delay", "active", "passive", "last", "ratio")

    def __init__(self, limit=256, buffer=32, delay=0.05, rate_limit=None, randomize_ratio=2):
        self.limit = limit
        self.buffer = buffer
        self.delay = delay
        self.active = 0
        self.passive = 0
        self.rate_limit = rate_limit
        self.rate_bin = alist()
        self.last = utc()
        self.ratio = randomize_ratio

    def _update_bin(self):
        while self.rate_bin and utc() - self.rate_bin[0] >= self.rate_limit:
            self.rate_bin.popleft()
        return self.rate_bin
    
    def __enter__(self):
        self.last = utc()
        if self.is_busy():
            if self.passive >= self.buffer:
                raise SemaphoreOverflowError(f"Semaphore object of limit {self.limit} overloaded by {self.passive}")
            self.passive += 1
            while self.is_busy():
                time.sleep(self.delay if not self.ratio else (random.random() * self.ratio + 1) * self.delay)
                self._update_bin()
            self.passive -= 1
        if self.rate_limit:
            self.rate_bin.append(utc())
        self.active += 1
        return self

    def __exit__(self, *args):
        self.active -= 1
        self.last = utc()

    def __call__(self):
        while self.value >= self.limit:
            time.sleep(self.delay)

    def is_active(self):
        return self.active or self.passive

    def is_busy(self):
        return self.active >= self.limit or len(self._update_bin()) >= self.limit

    @property
    def busy(self):
        return self.is_busy()

class SemaphoreOverflowError(RuntimeError):
    __slots__ = ()


class TracebackSuppressor(contextlib.AbstractContextManager, contextlib.ContextDecorator, collections.abc.Callable):

    def __init__(self, *args, **kwargs):
        self.exceptions = args + tuple(kwargs.values())
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type and exc_value:
            for exception in self.exceptions:
                if issubclass(type(exc_value), exception):
                    return True
            try:
                raise exc_value
            except:
                print_exc()
        return True

    __call__ = lambda self, *args, **kwargs: self.__class__(*args, **kwargs)

tracebacksuppressor = TracebackSuppressor()


class delay(contextlib.AbstractContextManager, contextlib.ContextDecorator, collections.abc.Callable):

    def __init__(self, duration=0):
        self.duration = duration
        self.start = utc()

    def __call__(self):
        return self.exit()
    
    def __exit__(self, *args):
        remaining = self.duration - utc() + self.start
        if remaining > 0:
            time.sleep(remaining)


def retry(func, *args, attempts=5, delay=1, exc=(), **kwargs):
    for i in range(attempts):
        t = utc()
        try:
            return func(*args, **kwargs)
        except BaseException as ex:
            if i >= attempts - 1 or ex in exc:
                raise
        remaining = delay - utc() + t
        if remaining > 0:
            time.sleep(delay)


closing = contextlib.closing
repeat = itertools.repeat
loop = lambda x: repeat(None, x)
from zipfile import ZipFile
import urllib.request, urllib.parse
import nacl.secret
from dateutil import parser as tparser
from math import *

np = numpy
array = numpy.array
deque = collections.deque

random.seed(random.random() + time.time() % 1)

math.round = round

ts_us = lambda: time.time_ns() // 1000
utc = lambda: time.time_ns() / 1e9
utc_dt = datetime.datetime.utcnow
utc_ft = datetime.datetime.utcfromtimestamp
ep = datetime.datetime(1970, 1, 1)

def zerot():
    today = utc_dt()
    return datetime.datetime(today.year, today.month, today.day, tzinfo=datetime.timezone.utc).timestamp()

to_utc = lambda dt: dt.replace(tzinfo=datetime.timezone.utc)
to_naive = lambda dt: dt.replace(tzinfo=None)

def utc_ts(dt):
    with suppress(TypeError):
        return (dt - ep).total_seconds()
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()

inf = Infinity = math.inf
nan = math.nan
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
null = None

TRUE, FALSE = True, False
true, false = True, False


def exclusive_range(range, *excluded):
    ex = frozenset(excluded)
    return tuple(i for i in range if i not in ex)

def exclusive_set(range, *excluded):
    ex = frozenset(excluded)
    return frozenset(i for i in range if i not in ex)


class UniversalSet(collections.abc.Set):

    __slots__ = ()

    __str__ = lambda self: "ξ"
    __repr__ = lambda self: f"{self.__class__.__name__}()"
    __contains__ = lambda self, key: True
    __bool__ = lambda self: True
    __iter__ = lambda self: repeat(None)
    __len__ = lambda self: inf
    __le__ = lambda self, other: type(self) is type(other)
    __lt__ = lambda self, other: False
    __eq__ = lambda self, other: type(self) is type(other)
    __ne__ = lambda self, other: type(self) is not type(other)
    __gt__ = lambda self, other: type(self) is not type(other)
    __ge__ = lambda self, other: True
    __and__ = lambda self, other: other
    __or__ = lambda self, other: self
    __sub__ = lambda self, other: self
    __xor__ = lambda self, other: self
    index = lambda self, obj: 0
    isdisjoint = lambda self, other: False

universal_set = UniversalSet()


class alist(collections.abc.MutableSequence, collections.abc.Callable):

    """
custom list-like data structure that incorporates the functionality of numpy arrays but allocates more space on the ends in order to have faster insertion."""

    maxoff = (1 << 24) - 1
    minsize = 256
    __slots__ = ("hash", "block", "offs", "size", "data", "frozenset", "queries")

    def waiting(self):
        func = self
        def call(self, *args, force=False, **kwargs):
            if not force:
                t = utc()
                while self.block:
                    time.sleep(0.001)
                    if utc() - t > 1:
                        raise TimeoutError("Request timed out.")
            return func(self, *args, **kwargs)
        return call

    def blocking(self):
        func = self
        def call(self, *args, force=False, **kwargs):
            if not force:
                t = utc()
                while self.block:
                    time.sleep(0.001)
                    if utc() - t > 1:
                        raise TimeoutError("Request timed out.")
            self.block = True
            self.hash = None
            self.frozenset = None
            self.queries = 0
            try:
                output = func(self, *args, **kwargs)
                self.block = False
            except:
                self.block = False
                raise
            return output
        return call

    def __init__(self, *args, fromarray=False, **void):
        self.block = True if not getattr(self, "block", None) else 2
        self.hash = None
        self.frozenset = None
        self.queries = 0
        if not args:
            self.offs = 0
            self.size = 0
            self.data = None
            self.block = False
            return
        elif len(args) == 1:
            iterable = args[0]
        else:
            iterable = args
        if issubclass(type(iterable), self.__class__) and iterable:
            self.offs = iterable.offs
            self.size = iterable.size
            if fromarray:
                self.data = iterable.data
            else:
                self.data = iterable.data.copy()
        elif fromarray:
            self.offs = 0
            self.size = len(iterable)
            self.data = iterable
        else:
            if not issubclass(type(iterable), collections.abc.Sequence) or issubclass(type(iterable), collections.abc.Mapping) or type(iterable) in (str, bytes):
                try:
                    iterable = deque(iterable)
                except TypeError:
                    iterable = [iterable]
            self.size = len(iterable)
            size = max(self.minsize, self.size * 3)
            self.offs = size // 3
            self.data = np.empty(size, dtype=object)
            self.view[:] = iterable
        self.block = True if self.block >= 2 else False

    def __getattr__(self, k):
        with suppress(AttributeError):
            return self.__getattribute__(k)
        return getattr(self.__getattribute__("view"), k)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(dir(self.view))
        return data

    @property
    def view(self):
        data = self.__getattribute__("data")
        if data is None:
            return []
        offs, size = [self.__getattribute__(i) for i in ("offs", "size")]
        return data[offs:offs + size]

    @waiting
    def __call__(self, arg=1, *void1, **void2):
        if arg == 1:
            return self.copy()
        return self * arg

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.view.tobytes())
        return self.hash

    def to_frozenset(self):
        if self.frozenset is None:
            self.frozenset = frozenset(self)
        return self.frozenset

    __str__ = lambda self: "[" + ", ".join(repr(i) for i in iter(self)) + "]"
    __repr__ = lambda self: f"{self.__class__.__name__}({tuple(self) if self.__bool__() else ''})"
    __bool__ = lambda self: bool(self.size)


    @blocking
    def __iadd__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.add(arr, iterable, out=arr)
        return self

    @blocking
    def __isub__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.subtract(arr, iterable, out=arr)
        return self

    @blocking
    def __imul__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.multiply(arr, iterable, out=arr)
        return self

    @blocking
    def __imatmul__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        temp = np.matmul(arr, iterable)
        self.size = len(temp)
        arr[:self.size] = temp
        return self

    @blocking
    def __itruediv__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.true_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __ifloordiv__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.floor_divide(arr, iterable, out=arr)
        return self

    @blocking
    def __imod__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.mod(arr, iterable, out=arr)
        return self

    @blocking
    def __ipow__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.power(arr, iterable, out=arr)
        return self

    @blocking
    def __ilshift__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        try:
            np.left_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.multiply(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __irshift__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        try:
            np.right_shift(arr, iterable, out=arr)
        except (TypeError, ValueError):
            np.divide(arr, np.power(2, iterable), out=arr)
        return self

    @blocking
    def __iand__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_and(arr, iterable, out=arr)
        return self

    @blocking
    def __ixor__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_xor(arr, iterable, out=arr)
        return self

    @blocking
    def __ior__(self, other):
        iterable = self.to_iterable(other)
        arr = self.view
        np.logical_or(arr, iterable, out=arr)
        return self

    @waiting
    def __neg__(self):
        return self.__class__(-self.view)

    @waiting
    def __pos__(self):
        return self

    @waiting
    def __abs__(self):
        d = self.data
        return self.__class__(np.abs(self.view))

    @waiting
    def __invert__(self):
        return self.__class__(np.invert(self.view))

    @waiting
    def __add__(self, other):
        temp = self.copy()
        temp += other
        return temp

    @waiting
    def __sub__(self, other):
        temp = self.copy()
        temp -= other
        return temp

    @waiting
    def __mul__(self, other):
        temp = self.copy()
        temp *= other
        return temp

    @waiting
    def __matmul__(self, other):
        temp1 = self.view
        temp2 = self.to_iterable(other)
        result = temp1 @ temp2
        return self.__class__(result)

    @waiting
    def __truediv__(self, other):
        temp = self.copy()
        temp /= other
        return temp

    @waiting
    def __floordiv__(self, other):
        temp = self.copy()
        temp //= other
        return temp

    @waiting
    def __mod__(self, other):
        temp = self.copy()
        temp %= other
        return temp

    @waiting
    def __pow__(self, other):
        temp = self.copy()
        temp **= other
        return temp

    @waiting
    def __lshift__(self, other):
        temp = self.copy()
        temp <<= other
        return temp

    @waiting
    def __rshift__(self, other):
        temp = self.copy()
        temp >>= other
        return temp

    @waiting
    def __and__(self, other):
        temp = self.copy()
        temp &= other
        return temp

    @waiting
    def __xor__(self, other):
        temp = self.copy()
        temp ^= other
        return temp

    @waiting
    def __or__(self, other):
        temp = self.copy()
        temp |= other
        return temp

    @waiting
    def __round__(self, prec=0):
        temp = np.round(self.view, prec)
        return self.__class__(temp)

    @waiting
    def __trunc__(self):
        temp = np.trunc(self.view)
        return self.__class__(temp)

    @waiting
    def __floor__(self):
        temp = np.floor(self.view)
        return self.__class__(temp)

    @waiting
    def __ceil__(self):
        temp = np.ceil(self.view)
        return self.__class__(temp)

    __index__ = lambda self: self.view
    __radd__ = __add__
    __rsub__ = lambda self, other: -self + other
    __rmul__ = __mul__
    __rmatmul__ = __matmul__

    @waiting
    def __rtruediv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.true_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rfloordiv__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.floor_divide(iterable, arr, out=arr)
        return temp

    @waiting
    def __rmod__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.mod(iterable, arr, out=arr)
        return temp

    @waiting
    def __rpow__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        np.power(iterable, arr, out=arr)
        return temp

    @waiting
    def __rlshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        try:
            np.left_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.multiply(iterable, np.power(2, arr), out=arr)
        return temp

    @waiting
    def __rrshift__(self, other):
        temp = self.__class__(self.data)
        iterable = self.to_iterable(other)
        arr = temp.view
        try:
            np.right_shift(iterable, arr, out=arr)
        except (TypeError, ValueError):
            np.divide(iterable, np.power(2, arr), out=arr)
        return temp
    
    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__


    @waiting
    def __lt__(self, other):
        it = self.to_iterable(other)
        return self.view < other

    @waiting
    def __le__(self, other):
        it = self.to_iterable(other)
        return self.view <= other

    @waiting
    def __eq__(self, other):
        try:
            it = self.to_iterable(other)
            return self.view == other
        except (TypeError, IndexError):
            return

    @waiting
    def __ne__(self, other):
        try:
            it = self.to_iterable(other)
            return self.view != other
        except (TypeError, IndexError):
            return

    @waiting
    def __gt__(self, other):
        it = self.to_iterable(other)
        return self.view > other

    @waiting
    def __ge__(self, other):
        it = self.to_iterable(other)
        return self.view >= other

    @waiting
    def __getitem__(self, *args):
        if len(args) == 1:
            key = args[0]
            if type(key) in (float, complex):
                return get(self.view, key, 1)
            if type(key) is int:
                key = key % self.size
                return self.view.__getitem__(key)
            if type(key) is slice:
                if key.step in (None, 1):
                    start = key.start
                    if start is None:
                        start = 0
                    stop = key.stop
                    if stop is None:
                        stop = self.size
                    if start >= self.size or stop <= start and (stop >= 0 or stop + self.size <= start):
                        return self.__class__()
                    temp = self.__class__(self, fromarray=True)
                    if start < 0:
                        if start < -self.size:
                            start = 0
                        else:
                            start %= self.size
                    if stop < 0:
                        stop %= self.size
                    elif stop > self.size:
                        stop = self.size
                    temp.offs += start
                    temp.size = stop - start
                    if not temp.size:
                        return self.__class__()
                    return temp
            return self.__class__(self.view.__getitem__(key), fromarray=True)
        return self.__class__(self.view.__getitem__(*args), fromarray=True)

    @blocking
    def __setitem__(self, *args):
        if len(args) == 2:
            key = args[0]
            if type(key) is int:
                key = key % self.size
            return self.view.__setitem__(key, args[1])
        return self.view.__setitem__(*args)

    @blocking
    def __delitem__(self, key):
        if type(key) is slice:
            s = key.indices(self.size)
            return self.pops(xrange(*s))
        try:
            len(key)
        except TypeError:
            return self.pop(key, force=True)
        return self.pops(key)

    __len__ = lambda self: self.size
    __length_hint__ = __len__
    __iter__ = lambda self: iter(self.view)
    __reversed__ = lambda self: iter(np.flip(self.view))

    @waiting
    def __bytes__(self):
        return bytes(round(i) & 255 for i in self.view)

    def __contains__(self, item):
        if self.queries >= 8:
            return item in self.to_frozenset()
        if self.frozenset is not None:
            return item in self.frozenset
        self.queries += 1
        return item in self.view

    __copy__ = lambda self: self.copy()

    def to_iterable(self, other, force=False):
        if not issubclass(type(other), collections.abc.Sequence) or issubclass(type(other), collections.abc.Mapping):
            try:
                other = list(other)
            except TypeError:
                other = [other]
        if len(other) not in (1, self.size) and not force:
            raise IndexError(f"Unable to perform operation on objects with size {self.size} and {len(other)}.")
        return other

    @blocking
    def clear(self):
        self.size = 0
        self.offs = self.size >> 1
        return self

    @waiting
    def copy(self):
        return self.__class__(self.view)

    @waiting
    def sort(self, *args, **kwargs):
        return self.__class__(sorted(self.view, *args, **kwargs))

    @waiting
    def shuffle(self, *args, **kwargs):
        return self.__class__(shuffle(self.view, *args, **kwargs))

    @waiting
    def reverse(self):
        return self.__class__(np.flip(self.view))

    @blocking
    def rotate(self, steps):
        s = self.size
        if not s:
            return self
        steps %= s
        if steps > s >> 1:
            steps -= s
        if abs(steps) < self.minsize:
            while steps > 0:
                self.appendleft(self.popright(force=True), force=True)
                steps -= 1
            while steps < 0:
                self.appendright(self.popleft(force=True), force=True)
                steps += 1
            return self
        self.view[:] = np.roll(self.view, steps)
        return self

    @blocking
    def rotateleft(self, steps):
        return self.rotate(-steps, force=True)

    rotateright = rotate

    @blocking
    def isempty(self):
        if self.size:
            if abs(len(self.data) // 3 - self.offs) > self.maxoff:
                self.reconstitute(force=True)
            return False
        if len(self.data) > 4096:
            self.data = None
            self.offs = 0
        else:
            self.offs = len(self.data) // 3
        return True

    @waiting
    def get(self, key, default=None):
        try:
            return self[key]
        except LookupError:
            return default

    @blocking
    def popleft(self):
        temp = self.data[self.offs]
        self.offs += 1
        self.size -= 1
        self.isempty(force=True)
        return temp

    @blocking
    def popright(self):
        temp = self.data[self.offs + self.size - 1]
        self.size -= 1
        self.isempty(force=True)
        return temp

    @blocking
    def pop(self, index=None, *args):
        try:
            if index is None:
                return self.popright(force=True)
            if index >= len(self.data):
                return self.popright(force=True)
            elif index == 0:
                return self.popleft(force=True)
            index %= self.size
            temp = self.data[index + self.offs]
            if index > self.size >> 1:
                self.view[index:-1] = self.data[self.offs + index + 1:self.offs + self.size]
            else:
                self.view[1:index + 1] = self.data[self.offs:self.offs + index]
                self.offs += 1
            self.size -= 1
            return temp
        except LookupError:
            if not args:
                raise
            return args[0]

    @blocking
    def insert(self, index, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if index >= self.size:
            return self.append(value, force=True)
        elif index == 0:
            return self.appendleft(value, force=True)
        index %= self.size
        if index > self.size >> 1:
            if self.size + self.offs + 1 >= len(self.data):
                self.reconstitute(force=True)
            self.size += 1
            self.view[index + 1:] = self.view[index:-1]
        else:
            if self.offs < 1:
                self.reconstitute(force=True)
            self.size += 1
            self.offs -= 1
            self.view[:index] = self.view[1:index + 1]
        self.view[index] = value
        return self

    @blocking
    def insort(self, value, key=None, sorted=True):
        if self.data is None:
            self.__init__((value,))
            return self
        if not sorted:
            self.__init__(sorted(self, key=key))
        if key is None:
            return self.insert(np.searchsorted(self.view, value), value, force=True)
        v = key(value)
        x = self.size
        index = (x >> 1) + self.offs
        gap = 3 + x >> 2
        seen = {}
        d = self.data
        while index not in seen and index >= self.offs and index < self.offs + self.size:
            check = key(d[index])
            if check < v:
                seen[index] = True
                index += gap
            else:
                seen[index] = False
                index -= gap
            gap = 1 + gap >> 1
        index -= self.offs - seen.get(index, 0)
        if index <= 0:
            return self.appendleft(value, force=True)
        return self.insert(index, value, force=True)

    @blocking
    def remove(self, value, key=None, sorted=False):
        pops = self.search(value, key, sorted, force=True)
        if pops:
            self.pops(pops, force=True)
        return self

    discard = remove

    @blocking
    def removedups(self, sorted=True):
        if sorted:
            try:
                temp = np.unique(self.view)
            except:
                temp = sorted(set(self.view))
        elif sorted is None:
            temp = tuple(set(self.view))
        else:
            temp = {}
            for x in self.view:
                if x not in temp:
                    temp[x] = None
            temp = tuple(temp.keys())
        self.size = len(temp)
        self.view[:] = temp
        return self

    uniq = unique = removedups

    @waiting
    def index(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[0]

    @waiting
    def rindex(self, value, key=None, sorted=False):
        return self.search(value, key, sorted, force=True)[-1]
    
    @waiting
    def search(self, value, key=None, sorted=False):
        if key is None:
            if sorted and self.size > self.minsize:
                i = np.searchsorted(self.view, value)
                if self.view[i] != value:
                    raise IndexError(f"{value} not found.")
                pops = self.__class__()
                pops.append(i)
                for x in range(i + self.offs - 1, -1, -1):
                    if self.data[x] == value:
                        pops.appendleft(x - self.offs)
                    else:
                        break
                for x in range(i + self.offs + 1, self.offs + self.size):
                    if self.data[x] == value:
                        pops.append(x - self.offs)
                    else:
                        break
                return pops
            else:
                return self.__class__(np.arange(self.size, dtype=np.uint32)[self.view == value])
        if sorted:
            v = value
            d = self.data
            pops = self.__class__()
            x = len(d)
            index = (x >> 1) + self.offs
            gap = x >> 2
            seen = {}
            while index not in seen and index >= self.offs and index < self.offs + self.size:
                check = key(d[index])
                if check < v:
                    seen[index] = True
                    index += gap
                elif check == v:
                    break
                else:
                    seen[index] = False
                    index -= gap
                gap = 1 + gap >> 1
            i = index + seen.get(index, 0)
            while i in d and key(d[i]) == v:
                pops.append(i - self.offs)
                i += 1
            i = index + seen.get(index, 0) - 1
            while i in d and key(d[i]) == v:
                pops.append(i - self.offs)
                i -= 1
        else:
            pops = self.__class__(i for i, x in enumerate(self.view) if key(x) == value)
        if not pops:
            raise IndexError(f"{value} not found.")
        return pops
    
    find = findall = search

    @waiting
    def count(self, value, key=None):
        if key is None:
            return sum(self.view == value)
        return sum(1 for i in self if key(i) == value)

    concat = lambda self, value: self.__class__(np.concatenate([self.view, value]))

    @blocking
    def appendleft(self, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if self.offs <= 0:
            self.reconstitute(force=True)
        self.offs -= 1
        self.size += 1
        self.data[self.offs] = value
        return self

    @blocking
    def append(self, value):
        if self.data is None:
            self.__init__((value,))
            return self
        if self.offs + self.size >= len(self.data):
            self.reconstitute(force=True)
        self.data[self.offs + self.size] = value
        self.size += 1
        return self

    appendright = add = append

    @blocking
    def extendleft(self, value):
        if self.data is None:
            self.__init__(reversed(value))
            return self
        value = self.to_iterable(reversed(value), force=True)
        if self.offs >= len(value):
            self.data[self.offs - len(value):self.offs] = value
            self.offs -= len(value)
            self.size += len(value)
            return self
        self.__init__(np.concatenate([value, self.view]))
        return self

    @blocking
    def extend(self, value):
        if self.data is None:
            self.__init__(value)
            return self
        value = self.to_iterable(value, force=True)
        if len(self.data) - self.offs - self.size >= len(value):
            self.data[self.offs + self.size:self.offs + self.size + len(value)] = value
            self.size += len(value)
            return self
        self.__init__(np.concatenate([self.view, value]))
        return self

    extendright = extend

    @waiting
    def join(self, iterable):
        iterable = self.to_iterable(iterable)
        temp = deque()
        for i, v in enumerate(iterable):
            try:
                temp.extend(v)
            except TypeError:
                temp.append(v)
            if i != len(iterable) - 1:
                temp.extend(self.view)
        return self.__class__(temp)

    @blocking
    def replace(self, original, new):
        view = self.view
        for i, v in enumerate(view):
            if v == original:
                view[i] = new
        return self

    @blocking
    def fill(self, value):
        self.view[:] = value
        return self

    keys = lambda self: range(len(self))
    values = lambda self: iter(self)
    items = lambda self: enumerate(self)

    @waiting
    def isdisjoint(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().isdisjoint(other)
    
    @waiting
    def issubset(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().issubset(other)

    @waiting
    def issuperset(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().issuperset(other)
    
    @waiting
    def union(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().union(*args)
    
    @waiting
    def intersection(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().intersection(*args)

    @waiting
    def difference(self, *others):
        args = deque()
        for other in others:
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            args.append(other)
        return self.to_frozenset().difference(*args)
    
    @waiting
    def symmetric_difference(self, other):
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        return self.to_frozenset().symmetric_difference(other)

    @blocking
    def update(self, *others, uniq=True):
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            self.extend(other, force=True)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def intersection_update(self, *others, uniq=True):
        pops = set()
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            for i, v in enumerate(self):
                if v not in other:
                    pops.add(i)
        self.pops(pops)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def difference_update(self, *others, uniq=True):
        pops = set()
        for other in others:
            if issubclass(other, collections.abc.Mapping):
                other = other.values()
            if type(other) not in (set, frozenset):
                other = frozenset(other)
            for i, v in enumerate(self):
                if v in other:
                    pops.add(i)
        self.pops(pops)
        if uniq:
            self.uniq(False, force=True)
        return self

    @blocking
    def symmetric_difference_update(self, other):
        data = set(self)
        if issubclass(other, collections.abc.Mapping):
            other = other.values()
        if type(other) not in (set, frozenset):
            other = frozenset(other)
        data.symmetric_difference_update(other)
        self.__init__(data)
        return self

    @blocking
    def clip(self, a, b=None):
        if b is None:
            b = -a
        if a > b:
            a, b = b, a
        arr = self.view
        np.clip(arr, a, b, out=arr)
        return self


    @waiting
    def real(self):
        return self.__class__(np.real(self.view))

    @waiting
    def imag(self):
        return self.__class__(np.imag(self.view))
    
    @waiting
    def float(self):
        return self.__class__(float(i.real) for i in self.view)

    @waiting
    def complex(self):
        return self.__class__(complex(i) for i in self.view)

    @waiting
    def mpf(self):
        return self.__class__(mpf(i.real) for i in self.view)

    @waiting
    def sum(self):
        return np.sum(self.view)

    @waiting
    def mean(self):
        return np.mean(self.view)
    
    @waiting
    def product(self):
        return np.prod(self.view)
    
    prod = product
        
    @blocking
    def reconstitute(self, data=None):
        self.__init__(data if data is not None else self.view)
        return self

    @blocking
    def delitems(self, iterable):
        iterable = self.to_iterable(iterable, force=True)
        if len(iterable) == 1:
            return self.pop(iterable[0], force=True)
        temp = np.delete(self.view, iterable)
        self.size = len(temp)
        self.view[:] = temp
        return self

    pops = delitems

hlist = alist

arange = lambda a, b=None, c=None: alist(xrange(a, b, c))

azero = lambda size: alist(repeat(0, size))


class cdict(dict):

    __slots__ = ()

    __init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
    __repr__ = lambda self: f"{self.__class__.__name__}({super().__repr__() if super().__len__() else ''})"
    __str__ = lambda self: super().__repr__()
    __iter__ = lambda self: iter(tuple(super().__iter__()))
    __call__ = lambda self, k: self.__getitem__(k)

    def __getattr__(self, k):
        with suppress(AttributeError):
            return self.__getattribute__(k)
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

    @property
    def __dict__(self):
        return self

    ___repr__ = lambda self: super().__repr__()
    to_dict = lambda self: dict(**self)
    to_list = lambda self: list(super().values())


class mdict(cdict):

    __slots__ = ()

    count = lambda self: sum(len(v) for v in super().values())

    def extend(self, k, v):
        try:
            values = super().__getitem__(k)
        except KeyError:
            return super().__setitem__(k, alist(v).uniq(sorted=False))
        return values.extend(v).uniq(sorted=False)

    def append(self, k, v):
        values = set_dict(super(), k, alist())
        if v not in values:
            values.append(v)

    add = append

    def popleft(self, k):
        values = super().__getitem__(k)
        if len(values):
            v = values.popleft()
        else:
            v = None
        if not values:
            super().pop(k)
        return v

    def popright(self, k):
        values = super().__getitem__(k)
        if len(values):
            v = values.popright()
        else:
            v = None
        if not values:
            super().pop(k)
        return v

    def __init__(self, *args, **kwargs):
        super().__init__()
        for it in args:
            for k, v in it.items():
                self.extend(k, v)
        for k, v in kwargs:
            self.extend(k, v)


sgn = lambda x: (((x > 0) << 1) - 1) * (x != 0)

def is_finite(x):
    if type(x) is int:
        return True
    if type(x) is complex:
        return not (cmath.isinf(x) or cmath.isnan(x))
    with suppress():
        return x.is_finite()
    return math.isfinite(x)

def round_min(x):
    if type(x) is int:
        return x
    if type(x) is not complex:
        if is_finite(x):
            y = math.round(x)
            if x == y:
                return int(y)
        return x
    else:
        if x.imag == 0:
            return round_min(x.real)
        else:
            return round_min(complex(x).real) + round_min(complex(x).imag) * (1j)

def round(x, y=None):
    with suppress():
        if is_finite(x):
            with suppress():
                if x == int(x):
                    return int(x)
                if y is None:
                    return int(math.round(x))
            return round_min(math.round(x, y))
        else:
            return x
    if type(x) is complex:
        return round(x.real, y) + round(x.imag, y) * 1j
    with suppress():
        return math.round(x)
    return x

def xrand(x, y=None, z=0):
    if y == None:
        y = 0
    if x == y:
        return x
    return random.randint(floor(min(x, y)), ceil(max(x, y)) - 1) + z

def xrange(a, b=None, c=None):
    if b == None:
        b = ceil(a.real)
        a = 0
    if c == None:
        if a > b:
            c = -1
        else:
            c = 1
    return range(floor(a.real), ceil(b.real), c)

def set_dict(d, k, v, ignore=False):
    try:
        v = d.__getitem__(k)
        if v is None and ignore:
            raise LookupError
    except LookupError:
        d.__setitem__(k, v)
    return v

def add_dict(a, b, replace=True, insert=None):
    if not issubclass(type(a), collections.abc.MutableMapping):
        if replace:
            r = b
        else:
            r = copy.copy(b)
        try:
            r[insert] += a
        except KeyError:
            r[insert] = a
        return r
    elif not issubclass(type(b), collections.abc.MutableMapping):
        if replace:
            r = a
        else:
            r = copy.copy(a)
        try:
            r[insert] += b
        except KeyError:
            r[insert] = b
        return r
    else:
        if replace:
            r = a
        else:
            r = copy.copy(a)
        for k in b:
            try:
                temp = a[k]
            except KeyError:
                r[k] = b[k]
                continue
            if issubclass(type(temp), collections.abc.MutableMapping) or issubclass(type(b[k]), collections.abc.MutableMapping):
                r[k] = add_dict(b[k], temp, replace)
                continue
            r[k] = b[k] + temp
    return r

def sub_dict(d, key):
    output = dict(d)
    try:
        key[0]
    except TypeError:
        key = [key]
    for k in key:
        output.pop(k, None)
    return output


def shuffle(it):
    if type(it) is list:
        random.shuffle(it)
        return it
    elif type(it) is tuple:
        it = list(it)
        random.shuffle(it)
        return it
    elif type(it) is dict:
        ir = shuffle(list(it))
        new = {}
        for i in ir:
            new[i] = it[i]
        it.clear()
        it.update(new)
        return it
    elif type(it) is deque:
        it = list(it)
        random.shuffle(it)
        return deque(it)
    elif isinstance(it, alist):
        temp = it.shuffle()
        it.data = temp.data
        it.offs = temp.offs
        it.size = temp.size
        del temp
        return it
    else:
        try:
            it = list(it)
            random.shuffle(it)
            return it
        except TypeError:
            raise TypeError(f"Shuffling {type(it)} is not supported.")

def reverse(it):
    if type(it) is list:
        return list(reversed(it))
    elif type(it) is tuple:
        return list(reversed(it))
    elif type(it) is dict:
        ir = tuple(reversed(it))
        new = {}
        for i in ir:
            new[i] = it[i]
        it.clear()
        it.update(new)
        return it
    elif type(it) is deque:
        return deque(reversed(it))
    elif isinstance(it, alist):
        temp = it.reverse()
        it.data = temp.data
        it.offs = temp.offs
        it.hash = None
        del temp
        return it
    else:
        try:
            return list(reversed(it))
        except TypeError:
            raise TypeError(f"Reversing {type(it)} is not supported.")

def sort(it, key=None, reverse=False):
    if type(it) is list:
        it.sort(key=key, reverse=reverse)
        return it
    elif type(it) is tuple:
        it = sorted(it, key=key, reverse=reverse)
        return it
    elif issubclass(type(it), collections.abc.Mapping):
        keys = sorted(it, key=it.get if key is None else lambda x: key(it.get(x)))
        if reverse:
            keys = reversed(keys)
        items = tuple((i, it[i]) for i in keys)
        it.clear()
        it.__init__(items)
        return it
    elif type(it) is deque:
        it = sorted(it, key=key, reverse=reverse)
        return deque(it)
    elif isinstance(it, alist):
        it.__init__(sorted(it, key=key, reverse=reverse))
        it.hash = None
        return it
    else:
        try:
            it = list(it)
            it.sort(key=key, reverse=reverse)
            return it
        except TypeError:
            raise TypeError(f"Sorting {type(it)} is not supported.")


def get(v, i, mode=1):
    size = len(v)
    i = i.real + i.imag * size
    if i == int(i) or mode == 0:
        return v[round(i) % size]
    elif mode > 0 and mode < 1:
        return get(v, i, 0) * (1 - mode) + mode * get(v, i, 1)
    elif mode == 1:
        return v[floor(i) % size] * (1 - i % 1) + v[ceil(i) % size] * (i % 1)
    return get(v, i, floor(mode)) * (1 - mode % 1) + (mode % 1) * get(v, i, ceil(mode))


TIMEUNITS = {
    "galactic year": 7157540528801820.28133333333333,
    "millenium": [31556925216., "millenia"],
    "century": [3155692521.6, "centuries"],
    "decade": 315569252.16,
    "year": 31556925.216,
    "month": 2629743.768,
    "week": 604800.,
    "day": 86400.,
    "hour": 3600.,
    "minute": 60.,
    "second": 1,
}

def time_convert(s):
    if not is_finite(s):
        high = "galactic years"
        return [str(s) + " " + high]
    r = s < 0
    s = abs(s)
    taken = []
    for i in TIMEUNITS:
        a = None
        t = m = TIMEUNITS[i]
        if type(t) is list:
            t = t[0]
        if type(t) is int:
            a = round(s, 3)
        elif s >= t:
            a = int(s // t)
            s = s % t
        if a:
            if a != 1:
                if type(m) is list:
                    i = m[1]
                else:
                    i += "s"
            taken.append("-" * r + str(round_min(a)) + " " + str(i))
    if not len(taken):
        return [str(round_min(s)) + " seconds"]
    return taken

sec2time = lambda s: " ".join(time_convert(s))

def time_disp(s):
    if not is_finite(s):
        return str(s)
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
    t = 0
    mult = 1
    while len(data):
        t += float(data[-1]) * mult
        data = data[:-1]
        if mult <= 60:
            mult *= 60
        elif mult <= 3600:
            mult *= 24
        elif len(data):
            raise TypeError("Too many time arguments.")
    return t


OP = {
    "=": None,
    ":=": None,
    "+=": "__add__",
    "-=": "__sub__",
    "*=": "__mul__",
    "/=": "__truediv__",
    "//=": "__floordiv__",
    "**=": "__pow__",
    "^=": "__pow__",
    "%=": "__mod__",
}

def eval_math(expr, default=0, op=True):
    if op:
        _op = None
        for op, at in OP.items():
            if expr.startswith(op):
                expr = expr[len(op):].strip()
                _op = at
        num = eval_math(expr, op=False)
        if _op is not None:
            num = getattr(float(default), _op)(num)
        return num
    f = expr.strip()
    try:
        if not f:
            r = [0]
        else:
            s = f.casefold()
            if s in ("t", "true", "y", "yes", "on"):
                r = [True]
            elif s in ("f", "false", "n", "no", "off"):
                r = [False]
            else:
                try:
                    r = [float(f)]
                except:
                    r = [ast.literal_eval(f)]
    except (ValueError, TypeError, SyntaxError):
        r = eval(f, {}, globals())
    x = r[0]
    with suppress(TypeError):
        while True:
            if type(x) is str:
                raise TypeError
            x = tuple(x)[0]
    if type(x) is str and x.isnumeric():
        return int(x)
    return round_min(float(x))


timeChecks = {
    "galactic year": ("gy", "galactic year", "galactic years"),
    "millenium": ("ml", "millenium", "millenia"),
    "century": ("c", "century", "centuries"),
    "decade": ("dc", "decade", "decades"),
    "year": ("y", "year", "years"),
    "month": ("mo", "mth", "month", "mos", "mths", "months"),
    "week": ("w", "wk", "week", "wks", "weeks"),
    "day": ("d", "day", "days"),
    "hour": ("h", "hr", "hour", "hrs", "hours"),
    "minute": ("m", "min", "minute", "mins", "minutes"),
    "second": ("s", "sec", "second", "secs", "seconds"),
}
num_words = "(?:(?:(?:[0-9]+|[a-z]{1,}illion)|thousand|hundred|ten|eleven|twelve|(?:thir|four|fif|six|seven|eigh|nine)teen|(?:twen|thir|for|fif|six|seven|eigh|nine)ty|zero|one|two|three|four|five|six|seven|eight|nine)\\s*)"
numericals = re.compile("^(?:" + num_words + "|(?:a|an)\\s*)(?:" + num_words + ")*", re.I)
connectors = re.compile("\\s(?:and|at)\\s", re.I)
alphabet = frozenset("abcdefghijklmnopqrstuvwxyz")

def eval_time(expr, default=0, op=True):
    if op:
        _op = None
        for op, at in OP.items():
            if expr.startswith(op):
                expr = expr[len(op):].strip(" ")
                _op = at
        num = eval_time(expr, op=False)
        if _op is not None:
            num = getattr(float(default), _op)(num)
        return num
    t = 0
    if expr:
        f = None
        try:
            if ":" in expr:
                data = expr.split(":")
                mult = 1
                while len(data):
                    t += eval_math(data[-1]) * mult
                    data = data[:-1]
                    if mult <= 60:
                        mult *= 60
                    elif mult <= 3600:
                        mult *= 24
                    elif len(data):
                        raise TypeError("Too many time arguments.")
            else:
                f = single_space(connectors.sub(" ", expr.replace(",", " "))).casefold()
                for tc in timeChecks:
                    for check in reversed(timeChecks[tc]):
                        if check in f:
                            i = f.index(check)
                            isnt = i + len(check) < len(f) and f[i + len(check)] in alphabet
                            if not i or f[i - 1] in alphabet or isnt:
                                continue
                            n = eval_math(f[:i])
                            s = TIMEUNITS[tc]
                            if type(s) is list:
                                s = s[0]
                            t += s * n
                            f = f[i + len(check):]
                if f.strip():
                    t += eval_math(f)
        except:
            t = utc_ts(tparser.parse(f if f is not None else expr)) - utc_ts(tparser.parse("0s"))
    if type(t) is not float:
        t = float(t)
    return t


def iter2str(it, key=None, limit=None, offset=0, left="[", right="]"):
    try:
        try:
            len(it)
        except TypeError:
            it = alist(i for i in it)
    except:
        it = alist(it)
    if issubclass(type(it), collections.abc.Mapping):
        keys = it.keys()
        values = iter(it.values())
    else:
        keys = range(offset, offset + len(it))
        values = iter(it)
    spacing = int(math.log10(max(1, len(it) + offset - 1)))
    s = ""
    with suppress(StopIteration):
        for k in keys:
            index = k if type(k) is str else " " * (spacing - int(math.log10(max(1, k)))) + str(k)
            s += f"\n{left}{index}{right} "
            if key is None:
                s += str(next(values))
            else:
                s += str(key(next(values)))
    return lim_str(s, limit)


def lim_str(s, maxlen=10):
    if maxlen is None:
        return s
    if type(s) is not str:
        s = str(s)
    over = (len(s) - maxlen) / 2
    if over > 0:
        half = len(s) / 2
        s = s[:ceil(half - over - 1)] + ".." + s[ceil(half + over + 1):]
    return s


def parse_fs(fs):
    if type(fs) is not bytes:
        fs = str(fs).encode("utf-8")
    if fs.endswith(b"TB"):
        scale = 1099511627776
    if fs.endswith(b"GB"):
        scale = 1073741824
    elif fs.endswith(b"MB"):
        scale = 1048576
    elif fs.endswith(b"KB"):
        scale = 1024
    else:
        scale = 1
    return float(fs.split(None, 1)[0]) * scale


RE = cdict()

def regexp(s, flags=0):
    global RE
    if issubclass(type(s), re.Pattern):
        return s
    elif type(s) is not str:
        s = s.decode("utf-8", "replace")
    t = (s, flags)
    try:
        return RE[t]
    except KeyError:
        RE[t] = re.compile(s, flags)
        return RE[t]


single_space = lambda s: regexp("  +").sub(" ", s)


single_space = lambda s: regexp("  +").sub(" ", s)

ZeroEnc = "\xad\u061c\u180e\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\u2065\u2066\u2067\u2068\u2069\u206a\u206b\u206c\u206d\u206e\u206f\ufe0f\ufeff"

UNIFMTS = [
    "𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝐚𝐛𝐜𝐝𝐞𝐟𝐠𝐡𝐢𝐣𝐤𝐥𝐦𝐧𝐨𝐩𝐪𝐫𝐬𝐭𝐮𝐯𝐰𝐱𝐲𝐳𝐀𝐁𝐂𝐃𝐄𝐅𝐆𝐇𝐈𝐉𝐊𝐋𝐌𝐍𝐎𝐏𝐐𝐑𝐒𝐓𝐔𝐕𝐖𝐗𝐘𝐙",
    "𝟢𝟣𝟤𝟥𝟦𝟧𝟨𝟩𝟪𝟫𝓪𝓫𝓬𝓭𝓮𝓯𝓰𝓱𝓲𝓳𝓴𝓵𝓶𝓷𝓸𝓹𝓺𝓻𝓼𝓽𝓾𝓿𝔀𝔁𝔂𝔃𝓐𝓑𝓒𝓓𝓔𝓕𝓖𝓗𝓘𝓙𝓚𝓛𝓜𝓝𝓞𝓟𝓠𝓡𝓢𝓣𝓤𝓥𝓦𝓧𝓨𝓩",
    "𝟢𝟣𝟤𝟥𝟦𝟧𝟨𝟩𝟪𝟫𝒶𝒷𝒸𝒹𝑒𝒻𝑔𝒽𝒾𝒿𝓀𝓁𝓂𝓃𝑜𝓅𝓆𝓇𝓈𝓉𝓊𝓋𝓌𝓍𝓎𝓏𝒜𝐵𝒞𝒟𝐸𝐹𝒢𝐻𝐼𝒥𝒦𝐿𝑀𝒩𝒪𝒫𝒬𝑅𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵",
    "𝟘𝟙𝟚𝟛𝟜𝟝𝟞𝟟𝟠𝟡𝕒𝕓𝕔𝕕𝕖𝕗𝕘𝕙𝕚𝕛𝕜𝕝𝕞𝕟𝕠𝕡𝕢𝕣𝕤𝕥𝕦𝕧𝕨𝕩𝕪𝕫𝔸𝔹ℂ𝔻𝔼𝔽𝔾ℍ𝕀𝕁𝕂𝕃𝕄ℕ𝕆ℙℚℝ𝕊𝕋𝕌𝕍𝕎𝕏𝕐ℤ",
    "0123456789𝔞𝔟𝔠𝔡𝔢𝔣𝔤𝔥𝔦𝔧𝔨𝔩𝔪𝔫𝔬𝔭𝔮𝔯𝔰𝔱𝔲𝔳𝔴𝔵𝔶𝔷𝔄𝔅ℭ𝔇𝔈𝔉𝔊ℌℑ𝔍𝔎𝔏𝔐𝔑𝔒𝔓𝔔ℜ𝔖𝔗𝔘𝔙𝔚𝔛𝔜ℨ",
    "0123456789𝖆𝖇𝖈𝖉𝖊𝖋𝖌𝖍𝖎𝖏𝖐𝖑𝖒𝖓𝖔𝖕𝖖𝖗𝖘𝖙𝖚𝖛𝖜𝖝𝖞𝖟𝕬𝕭𝕮𝕯𝕰𝕱𝕲𝕳𝕴𝕵𝕶𝕷𝕸𝕹𝕺𝕻𝕼𝕽𝕾𝕿𝖀𝖁𝖂𝖃𝖄𝖅",
    "０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ",
    "⓪①②③④⑤⑥⑦⑧⑨🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉🄰🄱🄲🄳🄴🄵🄶🄷🄸🄹🄺🄻🄼🄽🄾🄿🅀🅁🅂🅃🅄🅅🅆🅇🅈🅉",
    "⓿➊➋➌➍➎➏➐➑➒🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉🅰🅱🅲🅳🅴🅵🅶🅷🅸🅹🅺🅻🅼🅽🅾🅿🆀🆁🆂🆃🆄🆅🆆🆇🆈🆉",
    "⓪①②③④⑤⑥⑦⑧⑨ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ",
    "⓿➊➋➌➍➎➏➐➑➒🅐🅑🅒🅓🅔🅕🅖🅗🅘🅙🅚🅛🅜🅝🅞🅟🅠🅡🅢🅣🅤🅥🅦🅧🅨🅩🅐🅑🅒🅓🅔🅕🅖🅗🅘🅙🅚🅛🅜🅝🅞🅟🅠🅡🅢🅣🅤🅥🅦🅧🅨🅩",
    "0123456789𝘢𝘣𝘤𝘥𝘦𝘧𝘨𝘩𝘪𝘫𝘬𝘭𝘮𝘯𝘰𝘱𝘲𝘳𝘴𝘵𝘶𝘷𝘸𝘹𝘺𝘻𝘈𝘉𝘊𝘋𝘌𝘍𝘎𝘏𝘐𝘑𝘒𝘓𝘔𝘕𝘖𝘗𝘘𝘙𝘚𝘛𝘜𝘝𝘞𝘟𝘠𝘡",
    "𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗𝙖𝙗𝙘𝙙𝙚𝙛𝙜𝙝𝙞𝙟𝙠𝙡𝙢𝙣𝙤𝙥𝙦𝙧𝙨𝙩𝙪𝙫𝙬𝙭𝙮𝙯𝘼𝘽𝘾𝘿𝙀𝙁𝙂𝙃𝙄𝙅𝙆𝙇𝙈𝙉𝙊𝙋𝙌𝙍𝙎𝙏𝙐𝙑𝙒𝙓𝙔𝙕",
    "𝟶𝟷𝟸𝟹𝟺𝟻𝟼𝟽𝟾𝟿𝚊𝚋𝚌𝚍𝚎𝚏𝚐𝚑𝚒𝚓𝚔𝚕𝚖𝚗𝚘𝚙𝚚𝚛𝚜𝚝𝚞𝚟𝚠𝚡𝚢𝚣𝙰𝙱𝙲𝙳𝙴𝙵𝙶𝙷𝙸𝙹𝙺𝙻𝙼𝙽𝙾𝙿𝚀𝚁𝚂𝚃𝚄𝚅𝚆𝚇𝚈𝚉",
    "₀₁₂₃₄₅₆₇₈₉ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖqʳˢᵗᵘᵛʷˣʸᶻ🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿",
    "0123456789ᗩᗷᑢᕲᘿᖴᘜᕼᓰᒚҠᒪᘻᘉᓍᕵᕴᖇSᖶᑘᐺᘺ᙭ᖻᗱᗩᗷᑕᗪᗴᖴǤᕼIᒍKᒪᗰᑎOᑭᑫᖇᔕTᑌᐯᗯ᙭Yᘔ",
    "0ƖᘔƐᔭ59Ɫ86ɐqɔpǝɟɓɥᴉſʞןɯuodbɹsʇnʌʍxʎzꓯᗺƆᗡƎℲ⅁HIſꓘ⅂WNOԀΌᴚS⊥∩ΛMX⅄Z",
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
]
__umap = {UNIFMTS[k][i]: UNIFMTS[-1][i] for k in range(len(UNIFMTS) - 1) for i in range(len(UNIFMTS[k]))}

__unfont = "".maketrans(__umap)
unfont = lambda s: str(s).translate(__unfont)

DIACRITICS = {
    "ÀÁÂÃÄÅĀĂĄ": "A",
    "Æ": "AE",
    "ÇĆĈĊČ": "C",
    "ĎĐ": "D",
    "ÈÉÊËĒĔĖĘĚ": "E",
    "ĜĞĠĢ": "G",
    "ĤĦ": "H",
    "ÌÍÎÏĨĪĬĮİ": "I",
    "Ĳ": "IJ",
    "Ĵ": "J",
    "Ķ": "K",
    "ĹĻĽĿŁ": "L",
    "ÑŃŅŇŊ": "N",
    "ÒÓÔÕÖØŌŎŐ": "O",
    "Œ": "OE",
    "ŔŖŘ": "R",
    "ŚŜŞŠ": "S",
    "ŢŤŦ": "T",
    "ÙÚÛÜŨŪŬŮŰŲ": "U",
    "Ŵ": "W",
    "ÝŶŸ": "Y",
    "ŹŻŽ": "Z",
    "àáâãäåāăą": "a",
    "æ": "ae",
    "çćĉċč": "c",
    "ďđ": "d",
    "èéêëðēĕėęě": "e",
    "ĝğġģ": "g",
    "ĥħ": "h",
    "ìíîïĩīĭįı": "i",
    "ĳ": "ij",
    "ĵ": "j",
    "ķĸ": "k",
    "ĺļľŀł": "l",
    "ñńņňŉŋ": "n",
    "òóôõöøōŏő": "o",
    "œ": "oe",
    "þ": "p",
    "ŕŗř": "r",
    "śŝşšſ": "s",
    "ß": "ss",
    "ţťŧ": "t",
    "ùúûüũūŭůűų": "u",
    "ŵ": "w",
    "ýÿŷ": "y",
    "źżž": "z",
}
for i, k in DIACRITICS.items():
    __umap.update({c: k for c in i})
__umap.update({c: "" for c in ZeroEnc})
__umap["\u200a"] = ""
for c in tuple(__umap):
    if c in UNIFMTS[-1]:
        __umap.pop(c)
__trans = "".maketrans(__umap)
extra_zalgos = (
    range(768, 880),
    range(1155, 1162),
    exclusive_range(range(1425, 1478), 1470, 1472, 1475),
    range(1552, 1560),
    range(1619, 1632),
    exclusive_range(range(1750, 1774), 1757, 1758, 1765, 1766, 1769),
    exclusive_range(range(2260, 2304), 2274),
    range(7616, 7627),
    (8432,),
    range(11744, 11776),
    (42607,), range(42612, 42622), (42654, 42655),
    range(65056, 65060),
)
zalgo_array = np.concatenate(extra_zalgos)
zalgo_map = {n: "" for n in zalgo_array}
__trans.update(zalgo_map)
__unitrans = ["".maketrans({UNIFMTS[-1][x]: UNIFMTS[i][x] for x in range(len(UNIFMTS[-1]))}) for i in range(len(UNIFMTS) - 1)]

def uni_str(s, fmt=0):
    if type(s) is not str:
        s = str(s)
    return s.translate(__unitrans[fmt])

def unicode_prune(s):
    if type(s) is not str:
        s = str(s)
    if s.isascii():
        return s
    return s.translate(__trans)

__qmap = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "〝": '"',
    "〞": '"',
    "⸌": "'",
    "⸍": "'",
    "⸢": "'",
    "⸣": "'",
    "⸤": "'",
    "⸥": "'",
}
__qtrans = "".maketrans(__qmap)

full_prune = lambda s: unicode_prune(s).translate(__qtrans).casefold()


def fuzzy_substring(sub, s, match_start=False, match_length=True):
    if not match_length and s in sub:
        return 1
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
    ratio = max(0, min(1, match / len(s)))
    return ratio


def bytes2hex(b, space=True):
    if type(b) is str:
        b = b.encode("utf-8")
    if space:
        return b.hex(" ").upper()
    return b.hex().upper()

hex2bytes = lambda b: bytes.fromhex(b if type(b) is str else b.decode("utf-8", "replace"))

def bytes2b64(b, alt_char_set=False):
    if type(b) is str:
        b = b.encode("utf-8")
    b = base64.b64encode(b)
    if alt_char_set:
        b = b.replace(b"=", b"-").replace(b"/", b".")
    return b

def b642bytes(b, alt_char_set=False):
    if type(b) is str:
        b = b.encode("utf-8")
    if alt_char_set:
        b = b.replace(b"-", b"=").replace(b".", b"/")
    b = base64.b64decode(b)
    return b

python = ("python3", "python")[os.name == "nt"]
PROC = Process = psutil.Process()
quit = lambda *args, **kwargs: PROC.kill()
url_parse = urllib.parse.quote_plus

DISCORD_EPOCH = 1420070400000

def time_snowflake(datetime_obj, high=False):
    unix_seconds = (datetime_obj - type(datetime_obj)(1970, 1, 1)).total_seconds()
    discord_millis = int(unix_seconds * 1000 - DISCORD_EPOCH)
    return (discord_millis << 22) + ((1 << 22) - 1 if high else 0)

snowflake_time = lambda i: datetime.datetime.utcfromtimestamp(((i >> 22) + DISCORD_EPOCH) / 1000)


class ArgumentError(LookupError):
    __slots__ = ()

class TooManyRequests(PermissionError):
    __slots__ = ()

class CommandCancelledError(RuntimeError):
    __slots__ = ()


def html_decode(s):
    while len(s) > 7:
        try:
            i = s.index("&#")
        except ValueError:
            break
        try:
            if s[i + 2] == "x":
                h = "0x"
                p = i + 3
            else:
                h = ""
                p = i + 2
            for a in range(4):
                if s[p + a] == ";":
                    v = int(h + s[p:p + a])
                    break
            c = chr(v)
            s = s[:i] + c + s[p + a + 1:]
        except ValueError:
            continue
        except IndexError:
            continue
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return s.replace("&quot;", '"').replace("&apos;", "'")


C = COLOURS = cdict(
    black="\u001b[30m",
    red="\u001b[31m",
    green="\u001b[32m",
    yellow="\u001b[33m",
    blue="\u001b[34m",
    magenta="\u001b[35m",
    cyan="\u001b[36m",
    white="\u001b[37m",
    reset="\u001b[0m",
)

bar = "∙░▒▓█"
col = [C.red, C.yellow, C.green, C.cyan, C.blue, C.magenta]
def create_progress_bar(ratio, length=32, offset=None):
    high = length * 4
    position = min(high, round(ratio * high))
    items = deque()
    if offset is not None:
        offset = round(offset * len(col))
    for i in range(length):
        new = min(4, position)
        if offset is not None:
            items.append(col[offset % len(col)])
            offset += 1
        items.append(bar[new])
        position -= new
    return "".join(items)


ESCAPE_T = {
    "[": "⦍",
    "]": "⦎",
    "@": "＠",
    "`": "",
    "#": "♯",
    ";": ";",
}
__emap = "".maketrans(ESCAPE_T)

ESCAPE_T2 = {
    "@": "＠",
    "`": "",
    "#": "♯",
    ";": ";",
}
__emap2 = "".maketrans(ESCAPE_T2)

no_md = lambda s: str(s).translate(__emap)
clr_md = lambda s: str(s).translate(__emap2)
sqr_md = lambda s, colour=C.blue, reset=C.white: f"{colour}[{no_md(s)}]{reset}"

is_alphanumeric = lambda string: string.replace(" ", "").isalnum()
to_alphanumeric = lambda string: single_space(regexp("[^a-z 0-9]", re.I).sub(" ", unicode_prune(string)))
is_numeric = lambda string: regexp("[0-9]", re.I).search(string)


def to_png(url):
    if type(url) is not str:
        url = str(url)
    if url.endswith("?size=1024"):
        url = url[:-10] + "?size=4096"
    return url.replace(".webp", ".png")


if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
    randbytes = random.randbytes
else:
    randbytes = lambda size: (np.random.random_sample(size) * 256).astype(np.uint8).tobytes()

shash = lambda s: base64.b64encode(hashlib.sha256(s.encode("utf-8")).digest()).replace(b"/", b"-").decode("utf-8", "replace")
hhash = lambda s: bytes2hex(hashlib.sha256(s.encode("utf-8")).digest(), space=False)
ihash = lambda s: int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest(), "little") % 4294967296 - 2147483648

def bxor(b1, b2):
    x = np.frombuffer(b1, dtype=np.uint8)
    y = np.frombuffer(b2, dtype=np.uint8)
    return (x ^ y).tobytes()


__imap = {
    "#": "",
    "<": "",
    ">": "",
    "@": "",
    "!": "",
    "&": "",
}
__itrans = "".maketrans(__imap)

def verify_id(obj):
    if type(obj) is int:
        return obj
    if type(obj) is str:
        with suppress(ValueError):
            return int(obj.translate(__itrans))
        return obj
    with suppress(AttributeError):
        return obj.recipient.id
    with suppress(AttributeError):
        return obj.id
    return int(obj)


def strip_acc(url):
    if url.startswith("<") and url[-1] == ">":
        s = url[1:-1]
        if is_url(s):
            return s
    return url

__smap = {"|": "", "*": ""}
__strans = "".maketrans(__smap)
verify_search = lambda f: strip_acc(single_space(f.strip().translate(__strans)))
find_urls = lambda url: regexp("(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+").findall(url)
is_url = lambda url: regexp("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+$").findall(url)
is_discord_url = lambda url: regexp("^https?:\\/\\/(?:[a-z]+\\.)?discord(?:app)?\\.com\\/").findall(url)
is_tenor_url = lambda url: regexp("^https?:\\/\\/tenor.com(?:\\/view)?/[a-zA-Z0-9\\-_]+-[0-9]+").findall(url)
is_imgur_url = lambda url: regexp("^https?:\\/\\/(?:[a-z]\\.)?imgur.com/[a-zA-Z0-9\\-_]+").findall(url)
is_giphy_url = lambda url: regexp("^https?:\\/\\/giphy.com/gifs/[a-zA-Z0-9\\-_]+").findall(url)
is_youtube_url = lambda url: regexp("^https?:\\/\\/(?:www\\.)?youtu(?:\\.be|be\\.com)\\/[^\\s<>`|\"']+").findall(url)

verify_url = lambda url: url if is_url(url) else url_parse(url)


IMAGE_FORMS = {
    ".gif": True,
    ".png": True,
    ".bmp": False,
    ".jpg": True,
    ".jpeg": True,
    ".tiff": False,
    ".webp": True,
}
def is_image(url):
    url = url.split("?", 1)[0]
    if "." in url:
        url = url[url.rindex("."):]
        url = url.casefold()
        return IMAGE_FORMS.get(url)


class RequestManager(contextlib.AbstractContextManager, collections.abc.Callable):

    session = None
    semaphore = emptyctx

    def _init_(self):
        self.semaphore = Semaphore(512, 256, delay=0.25)

    def __call__(self, url, headers={}, files=None, data=None, raw=False, timeout=8, method="get", decode=False, bypass=True):
        if bypass:
            if "user-agent" not in headers:
                headers["user-agent"] = f"Mozilla/5.{xrand(1, 10)}"
            headers["DNT"] = "1"
        method = method.casefold()
        with self.semaphore:
            with getattr(requests, method)(url, headers=headers, files=files, data=data, stream=True, timeout=timeout) as resp:
                if resp.status_code >= 400:
                    raise ConnectionError(f"Error {resp.status_code}: {resp.text}")
                if raw:
                    data = resp.raw.read()
                else:
                    data = resp.content
                if decode:
                    return data.decode("utf-8", "replace")
                return data

    def __exit__(self, *args):
        self.session.close()

Request = RequestManager()
Request._init_()


athreads = concurrent.futures.ThreadPoolExecutor(max_workers=32)

def create_future_ex(func, *args, timeout=None, **kwargs):
    fut = athreads.submit(func, *args, **kwargs)
    if timeout is not None:
        fut = athreads.submit(fut.result, timeout=timeout)
    return fut

def evalEX(exc):
    try:
        ex = eval(exc)
    except NameError:
        if type(exc) is bytes:
            exc = exc.decode("utf-8", "replace")
        s = exc[exc.index("(") + 1:exc.index(")")]
        with suppress(TypeError, SyntaxError, ValueError):
            s = ast.literal_eval(s)
        ex = RuntimeError(s)
    except:
        print(exc)
        raise
    if issubclass(type(ex), BaseException):
        raise ex
    return ex


enc_key = None
if AUTH:
    with tracebacksuppressor:
        enc_key = AUTH["encryption_key"]

if not enc_key:
    AUTH["encryption_key"] = base64.b64encode(randbytes(32)).decode("utf-8", "replace")
    # try:
        # s = json.dumps(AUTH).encode("utf-8")
    # except:
        # print_exc()
        # s = repr(AUTH).encode("utf-8")
    # with open("auth.json", "wb") as f:
        # f.write(s)

enc_key = AUTH.pop("encryption_key")

enc_box = nacl.secret.SecretBox(base64.b64decode(enc_key)[:32])


def zip2bytes(data):
    if not hasattr(data, "read"):
        data = io.BytesIO(data)
    z = ZipFile(data, compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False)
    b = z.open("DATA").read()
    z.close()
    return b

def bytes2zip(data):
    b = io.BytesIO()
    z = ZipFile(b, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    z.writestr("DATA", data=data)
    z.close()
    b.seek(0)
    return b.read()


# Safer than raw eval, more powerful than json.decode
def eval_json(s):
    if type(s) is memoryview:
        s = bytes(s)
    with suppress(json.JSONDecodeError):
        return json.loads(s)
    return safe_eval(s)

encrypt = lambda s: b">~MIZA~>" + enc_box.encrypt(s if type(s) is bytes else str(s).encode("utf-8"))
def decrypt(s):
    if type(s) is not bytes:
        s = str(s).encode("utf-8")
    if s[:8] == b">~MIZA~>":
        return enc_box.decrypt(s[8:])
    raise ValueError("Data header not found.")

def select_and_loads(s, mode="safe", size=None):
    if not s:
        raise ValueError("Data must not be empty.")
    if size and size < len(s):
        raise OverflowError("Data input size too large.")
    if type(s) is str:
        s = s.encode("utf-8")
    if mode != "unsafe":
        try:
            s = decrypt(s)
        except ValueError:
            pass
        except:
            raise
        else:
            time.sleep(0.1)
    b = io.BytesIO(s)
    if zipfile.is_zipfile(b):
        print(f"Loading zip file of size {len(s)}...")
        b.seek(0)
        z = ZipFile(b, compression=zipfile.ZIP_DEFLATED, allowZip64=True, strict_timestamps=False)
        if size:
            x = z.getinfo("DATA").file_size
            if size < x:
                raise OverflowError(f"Data input size too large ({x} > {size}).")
        s = z.open("DATA").read()
        z.close()
    data = None
    with tracebacksuppressor:
        if s[0] == 128:
            data = pickle.loads(s)
    if data and type(data) in (str, bytes):
        s, data = data, None
    if data is None:
        if mode == "unsafe":
            data = eval(compile(s, "<loader>", "eval", optimize=2, dont_inherit=False))
        else:
            if b"{" in s:
                s = s[s.index(b"{"):s.rindex(b"}") + 1]
            data = eval_json(s)
    return data

def select_and_dumps(data, mode="safe"):
    if mode == "unsafe":
        s = pickle.dumps(data)
        if len(s) > 65536:
            s = bytes2zip(s)
        return s
    try:
        s = json.dumps(data)
    except:
        s = None
    if not s or len(s) > 262144:
        s = pickle.dumps(data)
        if len(s) > 1048576:
            s = bytes2zip(s)
        return encrypt(s)
    return s.encode("utf-8")


commands = cdict()


class Command(collections.abc.Hashable, collections.abc.Callable):
    min_level = 0
    rate_limit = 0
    description = ""
    usage = ""

    def perm_error(self, perm, req=None, reason=None):
        if req is None:
            req = self.min_level
        if reason is None:
            reason = f"for command {self.name[-1]}"
        return PermissionError(f"Insufficient priviliges {reason}. Required level: {req}, Current level: {perm}.")

    def __init__(self, bot=None, catg=None):
        self.used = {}
        if not hasattr(self, "data"):
            self.data = cdict()
        if not hasattr(self, "min_display"):
            self.min_display = self.min_level
        if not hasattr(self, "name"):
            self.name = []
        self.__name__ = self.__class__.__name__
        if not hasattr(self, "alias"):
            self.alias = self.name
        else:
            self.alias.append(self.__name__)
        self.name.append(self.__name__)
        self.aliases = {full_prune(alias).replace("*", "").replace("_", "").replace("||", ""): alias for alias in self.alias}
        self.aliases.pop("", None)
        for a in self.aliases:
            if a in commands:
                commands[a].append(self)
            else:
                commands[a] = alist((self,))
        self.catg = catg
        self.bot = bot
        self._globals = globals()
        f = getattr(self, "__load__", None)
        if callable(f):
            try:
                f()
            except:
                print_exc()
                self.data.clear()
                f()

    __hash__ = lambda self: hash(self.__name__) ^ hash(self.catg)
    __str__ = lambda self: f"Command <{self.__name__}>"
    __call__ = lambda self, **void: None

    def unload(self):
        bot = self.bot
        for alias in self.alias:
            alias = alias.replace("*", "").replace("_", "").replace("||", "")
            coms = commands.get(alias)
            if coms:
                coms.remove(self)
                print("unloaded", alias, "from", self)
            if not coms:
                commands.pop(alias, None)


youtube_dl = youtube_dlc
from bs4 import BeautifulSoup


SAMPLE_RATE = 48000


genius_key = None
google_api_key = None
if AUTH:
    try:
        genius_key = AUTH["genius_key"]
    except:
        print("WARNING: genius_key not found. Unable to use API to search song lyrics.")
    try:
        google_api_key = AUTH["google_api_key"]
    except:
        print("WARNING: google_api_key not found. Unable to use API to search youtube playlists.")


e_dur = lambda d: float(d) if type(d) is str else (d if d is not None else 300)


def get_duration(filename):
    command = ["ffprobe", "-hide_banner", filename]
    resp = None
    for _ in loop(3):
        try:
            proc = psutil.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            fut = create_future_ex(proc.communicate, timeout=2)
            res = fut.result(timeout=2)
            resp = bytes().join(res)
            break
        except:
            with suppress():
                proc.kill()
            print_exc()
    if not resp:
        return None
    s = resp.decode("utf-8", "replace")
    with tracebacksuppressor(ValueError):
        i = s.index("Duration: ")
        d = s[i + 10:]
        i = 2147483647
        for c in ", \n\r":
            with suppress(ValueError):
                x = d.index(c)
                if x < i:
                    i = x
        dur = time_parse(d[:i])
        return dur


def get_best_icon(entry):
    with suppress(KeyError):
        return entry["icon"]
    with suppress(KeyError):
        return entry["thumbnail"]
    try:
        thumbnails = entry["thumbnails"]
    except KeyError:
        try:
            url = entry["webpage_url"]
        except KeyError:
            url = entry["url"]
        if is_discord_url(url):
            if not is_image(url):
                return "https://cdn.discordapp.com/embed/avatars/0.png"
        return url
    return sorted(thumbnails, key=lambda x: float(x.get("width", x.get("preference", 0) * 4096)), reverse=True)[0]["url"]


def get_best_audio(entry):
    with suppress(KeyError):
        return entry["stream"]
    best = -1
    try:
        fmts = entry["formats"]
    except KeyError:
        fmts = ()
    try:
        url = entry["webpage_url"]
    except KeyError:
        url = entry["url"]
    replace = True
    for fmt in fmts:
        q = fmt.get("abr", 0)
        if type(q) is not int:
            q = 0
        vcodec = fmt.get("vcodec", "none")
        if vcodec not in (None, "none"):
            q -= 1
        if not fmt["url"].startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
            replace = False
        if q > best or replace:
            best = q
            url = fmt["url"]
    if "dropbox.com" in url:
        if "?dl=0" in url:
            url = url.replace("?dl=0", "?dl=1")
    if url.startswith("https://manifest.googlevideo.com/api/manifest/dash/"):
        resp = Request(url)
        fmts = alist()
        with suppress(ValueError, KeyError):
            while True:
                search = b'<Representation id="'
                resp = resp[resp.index(search) + len(search):]
                f_id = resp[:resp.index(b'"')].decode("utf-8")
                search = b"><BaseURL>"
                resp = resp[resp.index(search) + len(search):]
                stream = resp[:resp.index(b'</BaseURL>')].decode("utf-8")
                fmt = cdict(youtube_dl.extractor.youtube.YoutubeIE._formats[f_id])
                fmt.url = stream
                fmts.append(fmt)
        entry["formats"] = fmts
        return get_best_audio(entry)
    return url


def ensure_url(url):
    if url.startswith("ytsearch:"):
        url = f"https://www.youtube.com/results?search_query={verify_url(url[9:])}"
    return url


copy_entry = lambda item: {"name": item.name, "url": item.url, "duration": item.duration}


class CustomAudio(collections.abc.Hashable):

    emptyopus = b"\xfc\xff\xfe"
    defaults = {
        "volume": 1,
        "reverb": 0,
        "pitch": 0,
        "speed": 1,
        "pan": 1,
        "bassboost": 0,
        "compressor": 0,
        "chorus": 0,
        "resample": 0,
        "bitrate": 1966.08,
        "loop": False,
        "repeat": False,
        "shuffle": False,
        "quiet": False,
        "stay": False,
        "position": 0,
    }

    def __init__(self, bot=None, vc=None, channel=None):
        with tracebacksuppressor:
            self.paused = False
            self.stats = cdict(self.defaults)
            self.source = None
            self.channel = channel
            self.vc = vc
            self.reverse = False
            self.reading = 0
            self.has_read = False
            self.searching = False
            self.preparing = True
            self.player = None
            self.timeout = utc()
            self.ts = None
            self.lastsent = 0
            self.last_end = 0
            self.pausec = False
            self.curr_timeout = 0
            self.bot = bot
            self.args = []
            self.queue = AudioQueue()
            self.new(update=False)
            self.queue._init_(auds=self)
            self.semaphore = Semaphore(1, 4, rate_limit=1 / 8)
            self.announcer = Semaphore(1, 2, rate_limit=1 / 3)
            self.started = 0
            if os.path.exists("audiosettings.json"):
                with open("audiosettings.json", "r") as f:
                    self.stats.update(eval(f.read()))

    def __str__(self):
        classname = str(self.__class__).replace("'>", "")
        classname = classname[classname.index("'") + 1:]
        return f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>: " + "{" + f'"vc": {self.vc}, "queue": {len(self.queue)}, "stats": {self.stats}, "source": {self.source}' + "}"

    __hash__ = lambda self: self.channel.id ^ self.guild.id

    def __getattr__(self, key):
        with suppress(AttributeError):
            return self.__getattribute__(key)
        with suppress(AttributeError, KeyError):
            return getattr(self.__getattribute__("source"), key)
        return getattr(self.__getattribute__("queue"), key)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(dir(self.source))
        data.update(dir(self.vc))
        data.update(dir(self.queue))
        data.update(dir(self.channel))
        return data

    def has_options(self):
        stats = self.stats
        return stats.volume != 1 or stats.reverb != 0 or stats.pitch != 0 or stats.speed != 1 or stats.pan != 1 or stats.bassboost != 0 or stats.compressor != 0 or stats.chorus != 0 or stats.resample != 0

    def get_dump(self, position, js=False):
        with self.semaphore:
            lim = 1024
            q = [copy_entry(item) for item in self.queue.verify()]
            s = dict(self.stats)
            d = {
                "stats": s,
                "queue": q,
            }
            if not position:
                d["stats"].pop("position")
            if js:
                if len(q) > lim:
                    s = pickle.dumps(d)
                    if len(s) > 262144:
                        return encrypt(bytes2zip(s)), "dump.bin"
                    return encrypt(s), "dump.bin"
                return json.dumps(d).encode("utf-8"), "dump.json"
            return d, None

    def ensure_play(self):
        with tracebacksuppressor(RuntimeError):
            if self.started == 0 and not self.paused:
                self.play()

    def play(self):
        self.started = utc()

    def pause(self):
        if getattr(self, "source", None) is None:
            return
        if not self.source.closed:
            create_future_ex(self.source.close, timeout=60)

    def stop(self):
        if self.queue:
            self.queue[0].pop("played", None)
        self.started = 0
        self.pause()
        self.source = None

    def new(self, source=None, pos=0, update=True):
        self.speed = abs(self.stats.speed)
        self.is_playing = source is not None
        if source is not None:
            new_source = None
            try:
                self.stats.position = 0
                t = utc()
                while source.stream is None and utc() - t < 10:
                    time.sleep(0.1)
                new_source = source.create_reader(pos, auds=self)
            except OverflowError:
                source = None
            else:
                self.preparing = False
                self.is_playing = True
                self.has_read = False
            self.stop()
            self.source = new_source
            self.file = source
        else:
            self.stop()
            self.file = None
        self.stats.position = pos
        if pos == 0:
            if self.reverse and len(self.queue):
                self.stats.position = e_dur(self.queue[0].duration)
        if self.source is not None and self.player:
            self.player.time = 1 + utc()
        if self.speed < 0.005:
            self.speed = 1
            self.paused |= 2
        else:
            self.paused &= -3
        if update:
            self.update()
            self.queue.update_play()

    def seek(self, pos):
        duration = e_dur(self.queue[0].duration)
        pos = max(0, pos)
        if (pos >= duration and not self.reverse) or (pos <= 0 and self.reverse):
            create_future_ex(self.new, update=True, timeout=60)
            return duration
        create_future_ex(self.new, self.file, pos, update=False, timeout=60)
        self.stats.position = pos
        return self.stats.position

    def kill(self, reason=None):
        self.dead = None
        self.stop()

    def update(self, *void1, **void2):
        vc = self.vc
        if hasattr(self, "dead"):
            if self.dead is not None:
                self.kill()
            return
        self.queue.update_load()

    def construct_options(self, full=True):
        stats = self.stats
        pitchscale = 2 ** ((stats.pitch + stats.resample) / 12)
        chorus = min(16, abs(stats.chorus))
        reverb = stats.reverb
        volume = stats.volume
        if reverb:
            args = ["-i", "misc/SNB3,0all.wav"]
        else:
            args = []
        options = deque()
        if not is_finite(stats.compressor):
            options.extend(("anoisesrc=a=.001953125:c=brown", "amerge"))
        if self.reverse:
            options.append("areverse")
        if pitchscale != 1 or stats.speed != 1:
            speed = abs(stats.speed) / pitchscale
            speed *= 2 ** (stats.resample / 12)
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
                options.append("aresample=" + str(SAMPLE_RATE))
            options.append("asetrate=" + str(SAMPLE_RATE * pitchscale))
        if chorus:
            A = B = C = D = ""
            for i in range(ceil(chorus)):
                neg = ((i & 1) << 1) - 1
                i = 1 + i >> 1
                i *= stats.chorus / ceil(chorus)
                if i:
                    A += "|"
                    B += "|"
                    C += "|"
                    D += "|"
                delay = (25 + i * tau * neg) % 39 + 18
                A += str(round(delay, 3))
                decay = (0.125 + i * 0.03 * neg) % 0.25 + 0.25
                B += str(round(decay, 3))
                speed = (2 + i * 0.61 * neg) % 4.5 + 0.5
                C += str(round(speed, 3))
                depth = (i * 0.43 * neg) % max(4, stats.chorus) + 0.5
                D += str(round(depth, 3))
            b = 0.5 / sqrt(ceil(chorus + 1))
            options.append(
                "chorus=0.5:" + str(round(b, 3)) + ":"
                + A + ":"
                + B + ":"
                + C + ":"
                + D
            )
            volume *= 2
        if stats.compressor:
            comp = min(8000, abs(stats.compressor * 10 + sgn(stats.compressor)))
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
            opt = "anequalizer="
            width = 4096
            x = round(sqrt(1 + abs(stats.bassboost)), 5)
            coeff = width * max(0.03125, (0.25 / x))
            ch = " f=" + str(coeff if stats.bassboost > 0 else width - coeff) + " w=" + str(coeff / 2) + " g=" + str(max(0.5, min(48, 4 * math.log2(x * 5))))
            opt += "c0" + ch + "|c1" + ch
            options.append(opt)
        if reverb:
            coeff = abs(reverb)
            wet = min(3, coeff) / 3
            if wet != 1:
                options.append("asplit[2]")
            volume *= 1.2
            options.append("afir=dry=10:wet=10")
            if wet != 1:
                dry = 1 - wet
                options.append("[2]amix=weights=" + str(round(dry, 6)) + " " + str(round(wet, 6)))
            if coeff > 1:
                decay = str(round(1 - 4 / (3 + coeff), 4))
                options.append("aecho=1:1:479|613:" + decay + "|" + decay)
                if not is_finite(coeff):
                    options.append("aecho=1:1:757|937:1|1")
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
            elif volume > 1:
                options.append("asoftclip=atan")
            args.append(("-af", "-filter_complex")[bool(reverb)])
            args.append(",".join(options))
        return args

    def read(self):
        try:
            found = empty = False
            if self.queue.loading or self.paused:
                self.is_playing = True
                raise EOFError
            try:
                source = self.source
                if source is None:
                    raise StopIteration
                temp = source.read()
                if not temp:
                    raise StopIteration
                found = True
            except (StopIteration, ValueError):
                empty = True
                raise EOFError
            except:
                empty = True
                print_exc()
                raise EOFError
            if not empty:
                self.stats.position = round(
                    self.stats.position + self.speed / 50 * (self.reverse * -2 + 1),
                    4,
                )
                self.has_read = True
                self.curr_timeout = 0
            self.is_playing = True
        except EOFError:
            if self.source is not None and self.source.closed:
                self.source = None
            if (empty or not self.paused) and not self.queue.loading:
                queueable = self.queue
                if self.queue and not self.queue[0].get("played", False):
                    if not found and not self.queue.loading:
                        if self.source is not None:
                            self.source.advanced = True
                        create_future_ex(self.queue.advance, timeout=120)
                elif empty and queueable and self.source is not None:
                    if utc() - self.last_end > 0.5:
                        if self.reverse:
                            ended = self.stats.position <= 0.5
                        else:
                            ended = ceil(self.stats.position) >= e_dur(self.queue[0].duration) - 0.5
                        if self.curr_timeout and utc() - self.curr_timeout > 0.5 or ended:
                            if not found:
                                self.last_end = utc()
                                if not self.has_read or not self.queue:
                                    if self.queue:
                                        self.queue[0].url = ""
                                    self.source.advanced = True
                                    create_future_ex(self.queue.update_play, timeout=120)
                                    self.preparing = False
                                else:
                                    self.source.advanced = True
                                    create_future_ex(self.queue.update_play, timeout=120)
                                    self.preparing = False
                        elif self.curr_timeout == 0:
                            self.curr_timeout = utc()
                elif (empty and not queueable) or self.pausec:
                    self.curr_timeout = 0
                    self.vc.stop()
            temp = self.emptyopus
            self.pausec = self.paused & 1
        else:
            self.pausec = False
        return temp

    is_opus = lambda self: True
    cleanup = lambda self: None


class AudioQueue(alist):

    maxitems = 262144
        
    def _init_(self, auds):
        self.auds = auds
        self.bot = auds.bot
        self.vc = auds.vc
        self.lastsent = 0
        self.loading = False
        self.playlist = None

    def update_load(self):
        q = self
        if q:
            dels = deque()
            for i, e in enumerate(q):
                if i >= len(q) or i > 8191:
                    break
                if i < 2:
                    if not e.get("stream", None):
                        if not i:
                            callback = self.update_play
                        else:
                            callback = None
                        create_future_ex(ytdl.get_stream, e, callback=callback, timeout=90)
                        break
                if "file" in e:
                    e["file"].ensure_time()
                if not e.url:
                    if not self.auds.stats.quiet:
                        print(f"A problem occured while loading {sqr_md(e.name)}, and it has been automatically removed from the queue.")
                    dels.append(i)
                    continue
            q.pops(dels)
            self.advance(process=False)

    def advance(self, looped=True, repeated=True, shuffled=True, process=True):
        q = self
        s = self.auds.stats
        if q and process:
            if q[0].get("played"):
                q[0].pop("played")
                if not (s.repeat and repeated):
                    if s.loop:
                        temp = q[0]
                    q.popleft()
                    if s.shuffle and shuffled:
                        if len(q) > 1:
                            temp = q.popleft()
                            shuffle(q)
                            q.appendleft(temp)
                    if s.loop and looped:
                        q.append(temp)
                if self.auds.player:
                    self.auds.player.time = 1 + utc()
        self.update_play()

    def update_play(self):
        auds = self.auds
        q = self
        if q:
            entry = q[0]
            if (auds.source is None or auds.source.closed or auds.source.advanced) and not entry.get("played", False):
                if not auds.paused and (entry.get("file", None) or entry.get("stream", None) not in (None, "none")):
                    entry.played = True
                    if not auds.stats.quiet:
                        print(f"Now playing {sqr_md(q[0].name)}!")
                    self.loading = True
                    try:
                        source = ytdl.get_stream(entry, force=True)
                        # print(source)
                        auds.new(source)
                        # print(auds.source)
                        self.loading = False
                    except:
                        print(entry)
                        print(source)
                        self.loading = False
                        print_exc()
                        raise
                    auds.preparing = False
            elif auds.source is None and not self.loading and not auds.preparing:
                self.advance()
            elif auds.source is not None and auds.source.advanced:
                auds.source.advanced = False
                auds.source.closed = True
                self.advance()
            auds.ensure_play()
        else:
            if auds.source is None or auds.source.closed or auds.source.advanced:
                auds.stop()
                auds.source = None

    def verify(self):
        if len(self) > self.maxitems + 2048:
            self.__init__(self[1 - self.maxitems:].appendleft(self[0]), fromarray=True)
        elif len(self) > self.maxitems:
            self.rotate(-1)
            while len(self) > self.maxitems:
                self.pop()
            self.rotate(1)
        return self

    def enqueue(self, items, position):
        with self.auds.semaphore:
            if len(items) > self.maxitems:
                items = items[:self.maxitems]
            if not self:
                self.__init__(items)
                self.auds.source = None
                create_future_ex(self.update_load, timeout=120)
                return self
            if position == -1:
                self.extend(items)
            else:
                self.rotate(-position)
                self.extend(items)
                self.rotate(len(items) + position)
            return self.verify()


auds = CustomAudio()


def org2xm(org, dat=None):
    if os.name != "nt":
        raise OSError("org2xm is only available on Windows.")
    if not org or type(org) is not bytes:
        if not is_url(org):
            raise TypeError("Invalid input URL.")
        org = verify_url(org)
        data = Request(org)
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
    r_org = "cache/" + str(ts) + ".org"
    with open(r_org, "wb") as f:
        f.write(data)
    r_dat = "cache/" + str(ts) + ".dat"
    orig = False
    # Load custom sample bank if specified
    if dat is not None and is_url(dat):
        dat = verify_url(dat)
        with open(r_dat, "wb") as f:
            dat = Request(dat)
            f.write(dat)
    else:
        if type(dat) is bytes and dat:
            with open(r_dat, "wb") as f:
                f.write(dat)
        else:
            r_dat = "misc/ORG210EN.DAT"
            orig = True
    args = ["misc/org2xm.exe", r_org, r_dat]
    if compat:
        args.append("c")
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
    url = Request(
        "https://hostfast.onlineconverter.com/file/send",
        files={
            "class": (None, "audio"),
            "from": (None, "midi"),
            "to": (None, "mp3"),
            "source": (None, "file"),
            "file": mid,
            "audio_quality": (None, "192"),
        },
        method="post",
        decode=True,
    )
    # print(url)
    fn = url.rsplit("/", 1)[-1].strip("\x00")
    for i in range(360):
        with delay(1):
            test = Request(f"https://hostfast.onlineconverter.com/file/{fn}")
            if test == b"d":
                break
    ts = ts_us()
    r_mp3 = f"cache/{ts}.mp3"
    with open(r_mp3, "wb") as f:
        f.write(Request(f"https://hostfast.onlineconverter.com/file/{fn}/download"))
    return r_mp3


CONVERTERS = {
    b"MThd": mid2mp3,
    b"Org-": org2xm,
}

def select_and_convert(stream):
    with requests.get(stream, timeout=8, stream=True) as resp:
        it = resp.iter_content(4096)
        b = bytes()
        while len(b) < 4:
            b += next(it)
        try:
            convert = CONVERTERS[b[:4]]
        except KeyError:
            raise ValueError("Invalid file header.")
        b += resp.content
    return convert(b)


LOADED = set()
if os.path.exists("loaded.json"):
    with suppress():
        with open("loaded.json", "r") as f:
            LOADED = eval(f.read())


def subread(proc, parse=False):
    global print_prefix
    output = parse and not auds.stats.quiet
    with tracebacksuppressor:
        b = io.BytesIO()
        data = bytearray()
        last = None
        while proc.is_running():
            try:
                new = proc.stderr.read(1)
            except (BrokenPipeError, os.InvalidArgument):
                pass
            except:
                print_exc()
            if not new:
                break
            if parse:
                data.append(new[0])
                if new == b"\r":
                    try:
                        num = data[:data.index(b"M") - 1]
                        pos = float(num)
                        dur = e_dur(auds.queue[0].duration)
                        auds.stats.position = pos
                        if output:
                            out = f"\r{C.white}|{create_progress_bar(pos / dur, 64, ((-pos) % (1 / 3)) * 3)}{C.white}| ({time_disp(pos)}/{time_disp(dur)})"
                            if out != last:
                                print_prefix = "\n"
                                sys.__stdout__.write(out)
                                last = out
                    except (ValueError, TypeError, IndexError, ZeroDivisionError):
                        pass
                    data.clear()
            b.write(new)
        if parse:
            print_prefix = ""
            print()
        if getattr(proc, "killed", None) is not True and proc.returncode:
            b.seek(0)
            s = b.read().decode("utf-8", "replace").strip()
            if s:
                print(s)


class AudioFile:

    def __init__(self, fn, stream=None):
        self.file = fn
        self.proc = None
        self.stream = stream
        self.wasfile = False
        self.loading = False
        self.expired = False
        self.buffered = False
        self.loaded = False
        self.readers = cdict()
        self.assign = deque()
        self.semaphore = Semaphore(1, 1, delay=5)
        self.ensure_time()
        self.webpage_url = None

    def __str__(self):
        classname = str(self.__class__).replace("'>", "")
        classname = classname[classname.index("'") + 1:]
        return f"<{classname} object at {hex(id(self)).upper().replace('X', 'x')}>"
    
    def load(self, stream=None, check_fmt=False, force=False, webpage_url=None):
        if self.loading and not force:
            return
        if stream is not None:
            self.stream = stream
        stream = self.stream
        if webpage_url is not None:
            self.webpage_url = webpage_url
        self.loading = True
        # Collects data from source, converts to 48khz 192kbps mp3 format, outputting to target file
        # else:
        #     cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-vn", "-i", stream, "-map_metadata", "-1", "-f", "mp3", "-c:a", "mp3", "-ar", str(SAMPLE_RATE), "-ac", "2", "-b:a", "196608", "cache/" + self.file]
        cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-vn", "-i", stream, "-map_metadata", "-1", "-c:a", "copy", "cache/" + self.file]
        if not stream.startswith("https://www.yt-download.org/download/"):
            with suppress():
                fmt = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=format_name", "-of", "default=nokey=1:noprint_wrappers=1", stream]).decode("utf-8", "replace").rsplit(",", 1)[-1].strip()
                if fmt not in ("wav", "mp3", "webm", "opus", "ogg", "flac", "aac", "wma", "m4a", "mp4", "weba"):
                    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-vn", "-i", stream, "-map_metadata", "-1", "-f", "mp3", "-c:a", "mp3", "cache/" + self.file]
                    # print("Converting", fmt, "to mp3")
                else:
                    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-vn", "-i", stream, "-map_metadata", "-1", "-f", fmt, "-c:a", "copy", "cache/" + self.file]
        self.proc = None
        try:
            try:
                self.proc = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                create_future_ex(subread, self.proc)
            except:
                print(cmd)
                raise
            fl = 0
            # Attempt to monitor status of output file
            while fl < 4096:
                with delay(0.1):
                    if not self.proc.is_running():
                        err = self.proc.stderr.read().decode("utf-8", "replace")
                        if self.webpage_url and ("Server returned 5XX Server Error reply" in err or "Server returned 404 Not Found" in err or "Server returned 403 Forbidden" in err):
                            with tracebacksuppressor:
                                entry = ytdl.extract_backup(self.webpage_url)
                                print(err)
                                return self.load(get_best_audio(entry), check_fmt=False, force=True)
                        if check_fmt:
                            new = None
                            with suppress(ValueError):
                                new = select_and_convert(stream)
                            if new is not None:
                                return self.load(new, check_fmt=False, force=True)
                        print(self.proc.args)
                        if err:
                            ex = RuntimeError(err)
                        else:
                            ex = RuntimeError("FFmpeg did not start correctly, or file was too small.")
                        raise ex
                try:
                    fl = os.path.getsize("cache/" + self.file)
                except FileNotFoundError:
                    fl = 0
            self.buffered = True
            self.ensure_time()
            # print(self.file, "buffered", fl)
        except:
            # File errored, remove from cache and kill corresponding FFmpeg process if possible
            ytdl.cache.pop(self.file, None)
            if self.proc is not None:
                with suppress():
                    self.proc.killed = True
                    self.proc.kill()
            with suppress():
                os.remove("cache/" + self.file)
            raise
        return self

    # Touch the file to update its cache time.
    ensure_time = lambda self: setattr(self, "time", utc())

    # Update event run on all cached files
    def update(self):
        # Newly loaded files have their duration estimates copied to all queue entries containing them
        if self.loaded:
            if not self.wasfile:
                dur = self.duration()
                if dur is not None:
                    for e in self.assign:
                        e["duration"] = dur
                    self.assign.clear()
        # Check when file has been fully loaded
        elif self.buffered and not self.proc.is_running():
            if not self.loaded:
                self.loaded = True
                if not self.proc.returncode and self.file not in LOADED:
                    if len(LOADED) >= 256:
                        LOADED.pop()
                    LOADED.add(self.file)
                    with open("loaded.json", "w") as f:
                        f.write(repr(LOADED))
                if not is_url(self.stream):
                    retry(os.remove, self.stream, attempts=3, delay=0.5)
                try:
                    fl = os.path.getsize("cache/" + self.file)
                except FileNotFoundError:
                    fl = 0
                # print(self.file, "loaded", fl)
        # Touch file if file is currently in use
        if self.readers:
            self.ensure_time()
            return
        # Remove any unused file that has been left for a long time
        if utc() - self.time > 24000:
            try:
                fl = os.path.getsize("cache/" + self.file)
            except FileNotFoundError:
                fl = 0
                if self.buffered:
                    self.time = -inf
            ft = 24000 / (math.log2(fl / 16777216 + 1) + 1)
            if ft > 86400:
                ft = 86400
            if utc() - self.time > ft:
                self.destroy()

    def open(self):
        self.ensure_time()
        return "cache/" + self.file

    def destroy(self):
        self.expired = True
        if self.proc.is_running():
            with suppress():
                self.proc.killed = True
                self.proc.kill()
        with suppress():
            with self.semaphore:
                retry(os.remove, "cache/" + self.file, attempts=8, delay=5, exc=(FileNotFoundError,))
                # File is removed from cache data
                ytdl.cache.pop(self.file, None)
                # print(self.file, "deleted.")

    def create_reader(self, pos=0, auds=None):
        with suppress():
            auds.proc.killed = True
            auds.proc.kill()
        if not os.path.exists("cache/" + self.file):
            t = utc()
            while utc() - t < 10 and not self.stream:
                time.sleep(0.1)
            self.load(force=True)
        stats = auds.stats
        auds.reverse = stats.speed < 0
        if auds.speed < 0.005:
            auds.speed = 1
            auds.paused |= 2
        else:
            auds.paused &= -3
        stats.position = pos
        if not is_finite(stats.pitch * stats.speed):
            raise OverflowError("Speed setting out of range.")
        # Construct FFmpeg options
        options = auds.construct_options(full=False)
        if options or auds.reverse or pos or auds.stats.bitrate != 1966.08:
            args = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            if pos:
                arg = "-to" if auds.reverse else "-ss"
                args += [arg, str(pos)]
            args.append("-i")
            if self.loaded:
                buff = False
                args.insert(1, "-nostdin")
                args.append("cache/" + self.file)
            else:
                buff = True
                args.append("pipe:0")
            if options or auds.stats.bitrate != 1966.08:
                br = 100 * auds.stats.bitrate
                sr = SAMPLE_RATE
                while br < 4096:
                    br *= 2
                    sr >>= 1
                if sr < 8000:
                    sr = 8000
                options.extend(("-f", "s16le", "-ar", str(sr), "-ac", "2", "-b:a", str(round_min(br)), "-bufsize", "8192"))
                if options:
                    args.extend(options)
            else:
                options.extend(("-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-bufsize", "8192"))
            args.append("pipe:1")
            key = 0
            self.readers[key] = True
            callback = lambda: self.readers.pop(key, None)
            if buff:
                while not self.buffered and not self.expired:
                    time.sleep(0.1)
                # Select buffered reader for files not yet fully loaded, convert while downloading
                player = BufferedAudioReader(self, args, callback=callback)
            else:
                # Select loaded reader for loaded files
                player = LoadedAudioReader(self, args, callback=callback)
            auds.args = args
            return player.start()
        # Select raw file stream for direct audio playback
        auds.args.clear()
        return BasicAudioReader(self, (), None)

    # Audio duration estimation: Get values from file if possible, otherwise URL
    duration = lambda self: self.dur if getattr(self, "dur", None) is not None else set_dict(self.__dict__, "dur", get_duration("cache/" + self.file) if self.loaded else get_duration(self.stream), ignore=True)


def advance_after(proc):
    with suppress():
        proc.wait()
    if not getattr(proc, "killed", None):
        if auds.queue:
            if auds.stats.repeat:
                auds.queue[0].pop("played", None)
            elif auds.stats.loop:
                e = auds.queue.popleft()
                e.pop("played", None)
                auds.queue.append(e)
            else:
                auds.queue.popleft()
        auds.new()


class BasicAudioReader:

    def __init__(self, file, args, callback=None):
        self.closed = False
        self.advanced = False
        fn = "cache/" + file.file
        while not os.path.exists(fn) or not os.path.getsize(fn):
            time.sleep(0.1)
        args = ["ffplay", "-autoexit", "-hide_banner", "-nodisp", "-i", fn]
        self.proc = psutil.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        create_future_ex(advance_after, self.proc)
        create_future_ex(subread, self.proc, True)
        self.read = lambda: b""
        self.file = file
        self.buffer = None
        self.callback = callback

    def start(self):
        self.buffer = None
        self.buffer = self.read()
        return self

    def close(self, *void1, **void2):
        self.closed = True
        with suppress():
            self.proc.killed = True
            self.proc.kill()
        if callable(self.callback):
            self.callback()

    is_opus = lambda self: True
    cleanup = close


# Audio reader for fully loaded files. FFmpeg with single pipe for output.
class LoadedAudioReader:

    def __init__(self, file, args, callback=None):
        self.closed = False
        self.advanced = False
        args = " ".join(args) + f" | ffplay -autoexit -hide_banner -nodisp -f s16le -ar {str(SAMPLE_RATE)} -ac 2 -i pipe:0"
        self.proc = psutil.Popen(args, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        create_future_ex(advance_after, self.proc)
        create_future_ex(subread, self.proc, True)
        self.read = self.proc.stdout.read
        self.file = file
        self.buffer = None
        self.callback = callback
    
    def start(self):
        self.buffer = None
        self.buffer = self.read()
        return self

    def close(self, *void1, **void2):
        self.closed = True
        with suppress():
            self.proc.killed = True
            self.proc.kill()
        if callable(self.callback):
            self.callback()

    is_opus = lambda self: True
    cleanup = close


# Audio player for audio files still being written to. Continuously reads and sends data to FFmpeg process, only terminating when file download is confirmed to be finished.
class BufferedAudioReader:

    def __init__(self, file, args, callback=None):
        self.closed = False
        self.advanced = False
        args = " ".join(args) + f" | ffplay -autoexit -hide_banner -nodisp -f s16le -ar {str(SAMPLE_RATE)} -ac 2 -i pipe:0"
        self.proc = psutil.Popen(args, shell=True, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        create_future_ex(advance_after, self.proc)
        create_future_ex(subread, self.proc, True)
        self.read = self.proc.stdout.read
        self.file = file
        self.stream = file.open()
        self.buffer = None
        self.callback = callback
        self.full = False

    # Required loop running in background to feed data to FFmpeg
    def run(self):
        while not self.file.buffered and not self.closed:
            time.sleep(0.1)
        while True:
            b = bytes()
            try:
                b = self.stream._read(65536)
                if not b:
                    raise EOFError
                self.proc.stdin.write(b)
                self.proc.stdin.flush()
            except (ValueError, EOFError):
                # Only stop when file is confirmed to be finished
                if self.file.loaded or self.closed:
                    break
                time.sleep(0.1)
        self.full = True
        self.proc.stdin.close()
    
    def start(self):
        # Run loading loop in parallel thread obviously
        create_future_ex(self.run, timeout=86400)
        self.buffer = None
        self.buffer = self.read()
        return self

    def close(self):
        self.closed = True
        with suppress():
            self.stream.close()
        with suppress():
            self.proc.killed = True
            self.proc.kill()
        if callable(self.callback):
            self.callback()

    is_opus = lambda self: True
    cleanup = close

    
class AudioDownloader:
    
    _globals = globals()
    ydl_opts = {
        # "verbose": 1,
        "quiet": 1,
        "format": "bestaudio/best",
        "nocheckcertificate": 1,
        "no_call_home": 1,
        "nooverwrites": 1,
        "noplaylist": 1,
        "logtostderr": 0,
        "ignoreerrors": 0,
        "default_search": "auto",
        "source_address": "0.0.0.0",
    }
    youtube_x = 0
    youtube_dl_x = 0
    spotify_x = 0
    other_x = 0

    def __init__(self):
        self.bot = None
        self.lastclear = 0
        self.downloading = cdict()
        self.cache = cdict()
        self.searched = cdict()
        self.semaphore = Semaphore(4, 128, delay=0.25)
        self.update_dl()
        self.setup_pages()

    # Fetches youtube playlist page codes, split into pages of 50 items
    def setup_pages(self):
        with open("misc/page_tokens.txt", "r", encoding="utf-8") as f:
            s = f.read()
        page10 = s.splitlines()
        self.yt_pages = {i * 10: page10[i] for i in range(len(page10))}
        # self.yt_pages = [page10[i] for i in range(0, len(page10), 5)]

    # Initializes youtube_dl object as well as spotify tokens, every 720 seconds.
    def update_dl(self):
        if utc() - self.lastclear > 720:
            self.lastclear = utc()
            with tracebacksuppressor:
                self.youtube_dl_x += 1
                self.downloader = youtube_dl.YoutubeDL(self.ydl_opts)
                self.spotify_x += 1
                token = retry(Request, "https://open.spotify.com/get_access_token", attempts=8, delay=0.5)
                self.spotify_header = {"authorization": f"Bearer {json.loads(token[:512])['accessToken']}"}
                self.other_x += 1
                resp = Request("https://keepv.id/")
                search = b"<script>apikey='"
                resp = resp[resp.rindex(search) + len(search):]
                search = b";sid='"
                resp = resp[resp.index(search) + len(search):]
                self.keepvid_token = resp[:resp.index(b"';</script>")].decode("utf-8", "replace")

    # Gets data from yt-download.org, keepv.id, or y2mate.guru, adjusts the format to ensure compatibility with results from youtube-dl. Used as backup.
    def extract_backup(self, url):
        url = verify_url(url)
        if is_url(url) and not is_youtube_url(url):
            raise TypeError("Not a youtube link.")
        excs = alist()
        if ":" in url:
            url = url.rsplit("/", 1)[-1].split("v=", 1)[-1].split("&", 1)[0]
        webpage_url = f"https://www.youtube.com/watch?v={url}"
        resp = None
        try:
            yt_url = f"https://www.yt-download.org/file/mp3/{url}"
            self.other_x += 1
            resp = Request(yt_url)
            search = b'<img class="h-20 w-20 md:h-48 md:w-48 mt-0 md:mt-12 lg:mt-0 rounded-full mx-auto md:mx-0 md:mr-6" src="'
            resp = resp[resp.index(search) + len(search):]
            thumbnail = resp[:resp.index(b'"')].decode("utf-8", "replace")
            search = b'<h2 class="text-lg text-teal-600 font-bold m-2 text-center">'
            resp = resp[resp.index(search) + len(search):]
            title = html_decode(resp[:resp.index(b"</h2>")].decode("utf-8", "replace"))
            resp = resp[resp.index(f'<a href="https://www.yt-download.org/download/{url}/mp3/192'.encode("utf-8")) + 9:]
            stream = resp[:resp.index(b'"')].decode("utf-8", "replace")
            resp = resp[:resp.index(b"</a>")]
            search = b'<div class="text-shadow-1">'
            fs = parse_fs(resp[resp.rindex(search) + len(search):resp.rindex(b"</div>")])
            dur = fs / 192000 * 8
            entry = {
                "formats": [
                    {
                        "abr": 192,
                        "url": stream,
                    },
                ],
                "duration": dur,
                "thumbnail": thumbnail,
                "title": title,
                "webpage_url": webpage_url,
            }
            print("Successfully resolved with yt-download.")
            return entry
        except Exception as ex:
            if resp:
                excs.append(resp)
            excs.append(ex)
        try:
            self.other_x += 1
            resp = Request(
                "https://keepv.id/",
                headers={"Accept": "*/*", "Cookie": "PHPSESSID=" + self.keepvid_token, "X-Requested-With": "XMLHttpRequest"},
                data=(("url", webpage_url), ("sid", self.keepvid_token)),
                method="POST",
            )
            search = b'<h2 class="mb-3">'
            resp = resp[resp.index(search) + len(search):]
            title = html_decode(resp[:resp.index(b"</h3>")].decode("utf-8", "replace"))
            search = b'<img src="'
            resp = resp[resp.index(search) + len(search):]
            thumbnail = resp[:resp.index(b'"')].decode("utf-8", "replace")
            entry = {
                "formats": [],
                "thumbnail": thumbnail,
                "title": title,
                "webpage_url": webpage_url,
            }
            with suppress(ValueError):
                search = b"Download Video</a><br>"
                resp = resp[resp.index(search) + len(search):]
                search = b"Duration: "
                resp = resp[resp.index(search) + len(search):]
                entry["duration"] = time_parse(resp[:resp.index("<br><br>")])
            search = b"</a></td></tr></tbody></table><h3>Audio</h3>"
            resp = resp[resp.index(search) + len(search):]
            with suppress(ValueError):
                while resp:
                    search = b"""</td><td class='text-center'><span class="btn btn-sm btn-outline-"""
                    resp = resp[resp.index(search) + len(search):]
                    search = b"</span></td><td class='text-center'>"
                    resp = resp[resp.index(search) + len(search):]
                    fs = parse_fs(resp[:resp.index(b"<")])
                    abr = fs / dur * 8
                    search = b'class="btn btn-sm btn-outline-primary shadow vdlbtn" href='
                    resp = resp[resp.index(search) + len(search):]
                    stream = resp[resp.index(b'"') + 1:resp.index(b'" download="')]
                    entry["formats"].append(dict(abr=abr, url=stream))
            if not entry["formats"]:
                raise FileNotFoundError
            print("Successfully resolved with keepv.id.")
            return entry
        except Exception as ex:
            if resp:
                excs.append(resp)
            excs.append(ex)
        try:
            self.other_x += 1
            resp = Request("https://y2mate.guru/api/convert", decode=True, data={"url": webpage_url}, method="POST")
            data = eval_json(resp)
            meta = data["meta"]
            entry = {
                "formats": [
                    {
                        "abr": stream.get("quality", 0),
                        "url": stream["url"],
                    } for stream in data["url"] if "url" in stream and stream.get("audio")
                ],
                "thumbnail": data.get("thumb"),
                "title": meta["title"],
                "webpage_url": meta["source"],
            }
            if meta.get("duration"):
                entry["duration"] = time_parse(meta["duration"])
            if not entry["formats"]:
                raise FileNotFoundError
            print("Successfully resolved with y2mate.")
            return entry
        except Exception as ex:
            if resp:
                excs.append(resp)
            excs.append(ex)
            print(excs)
            raise
        
    # def from_pytube(self, url):
    #     # pytube only accepts direct youtube links
    #     url = verify_url(url)
    #     if not url.startswith("https://www.youtube.com/"):
    #         if not url.startswith("http://youtu.be/"):
    #             if is_url(url):
    #                 raise TypeError("Not a youtube link.")
    #             url = f"https://www.youtube.com/watch?v={url}"
    #     try:
    #         resp = retry(pytube.YouTube, url, attempts=3, exc=(pytube.exceptions.RegexMatchError,))
    #     except pytube.exceptions.RegexMatchError:
    #         raise RuntimeError("Invalid single youtube link.")
    #     entry = {
    #         "formats": [
    #             {
    #                 "abr": 0,
    #                 "vcodec": stream.video_codec,
    #                 "url": stream.url,
    #             } for stream in resp.streams.fmt_streams
    #         ],
    #         "duration": resp.length,
    #         "thumbnail": getattr(resp, "thumbnail_url", None),
    #     }
    #     # Format bitrates
    #     for i in range(len(entry["formats"])):
    #         stream = resp.streams.fmt_streams[i]
    #         try:
    #             abr = stream.abr.casefold()
    #         except AttributeError:
    #             abr = "0"
    #         if type(abr) is not str:
    #             abr = str(abr)
    #         if abr.endswith("kbps"):
    #             abr = float(abr[:-4])
    #         elif abr.endswith("mbps"):
    #             abr = float(abr[:-4]) * 1024
    #         elif abr.endswith("bps"):
    #             abr = float(abr[:-3]) / 1024
    #         else:
    #             try:
    #                 abr = float(abr)
    #             except (ValueError, TypeError):
    #                 continue
    #         entry["formats"][i]["abr"] = abr
    #     return entry

    # Returns part of a spotify playlist.
    def get_spotify_part(self, url):
        out = deque()
        self.spotify_x += 1
        resp = Request(url, headers=self.spotify_header)
        d = eval_json(resp)
        with suppress(KeyError):
            d = d["tracks"]
        try:
            items = d["items"]
            total = d.get("total", 0)
        except KeyError:
            if "type" in d:
                items = (d,)
                total = 1
            else:
                items = []
        for item in items:
            try:
                track = item["track"]
            except KeyError:
                try:
                    track = item["episode"]
                except KeyError:
                    if "id" in item:
                        track = item
                    else:
                        continue
            name = track.get("name", track["id"])
            artists = ", ".join(a["name"] for a in track.get("artists", []))
            dur = track.get("duration_ms")
            if dur:
                dur /= 1000
            temp = cdict(
                name=name,
                url="ytsearch:" + f"{name} ~ {artists}".replace(":", "-"),
                id=track["id"],
                duration=dur,
                research=True,
            )
            out.append(temp)
        return out, total

    # Returns part of a youtube playlist.
    def get_youtube_part(self, url):
        out = deque()
        self.youtube_x += 1
        resp = Request(url)
        d = eval_json(resp)
        items = d["items"]
        total = d.get("pageInfo", {}).get("totalResults", 0)
        for item in items:
            try:
                snip = item["snippet"]
                v_id = snip["resourceId"]["videoId"]
            except KeyError:
                continue
            name = snip.get("title", v_id)
            url = f"https://www.youtube.com/watch?v={v_id}"
            temp = cdict(
                name=name,
                url=url,
                duration=None,
                research=True,
            )
            out.append(temp)
        return out, total

    # Returns a full youtube playlist.
    def get_youtube_playlist(self, p_id):
        out = deque()
        self.youtube_x += 1
        resp = Request(f"https://www.youtube.com/playlist?list={p_id}")
        try:
            search = b'window["ytInitialData"] = '
            try:
                resp = resp[resp.index(search) + len(search):]
            except ValueError:
                search = b"var ytInitialData = "
                resp = resp[resp.index(search) + len(search):]
            try:
                resp = resp[:resp.index(b'window["ytInitialPlayerResponse"] = null;')]
                resp = resp[:resp.rindex(b";")]
            except ValueError:
                resp = resp[:resp.index(b";</script><title>")]
            data = eval_json(resp)
        except:
            print(resp)
            raise
        count = int(data["sidebar"]["playlistSidebarRenderer"]["items"][0]["playlistSidebarPrimaryInfoRenderer"]["stats"][0]["runs"][0]["text"].replace(",", ""))
        for part in data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"][0]["playlistVideoListRenderer"]["contents"]:
            try:
                video = part["playlistVideoRenderer"]
            except KeyError:
                # print(part)
                continue
            v_id = video['videoId']
            try:
                dur = round_min(float(video["lengthSeconds"]))
            except (KeyError, ValueError):
                try:
                    dur = time_parse(video["lengthText"]["simpleText"])
                except KeyError:
                    dur = None
            temp = cdict(
                name=video["title"]["runs"][0]["text"],
                url=f"https://www.youtube.com/watch?v={v_id}",
                duration=dur,
                thumbnail=f"https://i.ytimg.com/vi/{v_id}/maxresdefault.jpg",
            )
            out.append(temp)
        if count > 100:
            url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&key={google_api_key}&playlistId={p_id}"
            page = 50
            futs = deque()
            for curr in range(100, page * ceil(count / page), page):
                search = f"{url}&pageToken={self.yt_pages[curr]}"
                futs.append(create_future_ex(self.get_youtube_part, search))
            for fut in futs:
                out.extend(fut.result()[0])
        return out

    # Repeatedly makes calls to youtube-dl until there is no more data to be collected.
    def extract_true(self, url):
        while not is_url(url):
            with suppress(NotImplementedError):
                return self.search_yt(regexp("ytsearch[0-9]*:").sub("", url, 1))[0]
            resp = self.extract_from(url)
            if "entries" in resp:
                resp = resp["entries"][0]
            if "duration" in resp and "formats" in resp:
                return resp
            try:
                url = resp["webpage_url"]
            except KeyError:
                try:
                    url = resp["url"]
                except KeyError:
                    url = resp["id"]
        if is_discord_url(url):
            title = url.split("?", 1)[0].rsplit("/", 1)[-1]
            if "." in title:
                title = title[:title.rindex(".")]
            return dict(url=url, name=title, direct=True)
        try:
            self.youtube_dl_x += 1
            entries = self.downloader.extract_info(url, download=False, process=True)
        except Exception as ex:
            s = str(ex).casefold()
            if type(ex) is not youtube_dl.DownloadError or ("403" in s or "429" in s or "no video formats found" in s or "unable to extract video data" in s or "unable to extract js player" in s or "geo restriction" in s):
                try:
                    entries = self.extract_backup(url)
                except youtube_dl.DownloadError:
                    raise FileNotFoundError("Unable to fetch audio data.")
            else:
                raise
        if "entries" in entries:
            entries = entries["entries"]
        else:
            entries = [entries]
        out = deque()
        for entry in entries:
            temp = cdict(
                name=entry["title"],
                url=entry["webpage_url"],
                duration=entry.get("duration"),
                stream=get_best_audio(entry),
                icon=get_best_icon(entry),
            )
            if not temp.duration:
                temp.research = True
            out.append(temp)
        return out

    # Extracts audio information from a single URL.
    def extract_from(self, url):
        if is_discord_url(url):
            title = url.split("?", 1)[0].rsplit("/", 1)[-1]
            if "." in title:
                title = title[:title.rindex(".")]
            return dict(url=url, webpage_url=url, title=title, direct=True)
        try:
            self.youtube_dl_x += 1
            return self.downloader.extract_info(url, download=False, process=False)
        except Exception as ex:
            s = str(ex).casefold()
            if type(ex) is not youtube_dl.DownloadError or ("403" in s or "429" in s or "no video formats found" in s or "unable to extract video data" in s or "unable to extract js player" in s or "geo restriction" in s):
                if is_url(url):
                    try:
                        return self.extract_backup(url)
                    except youtube_dl.DownloadError:
                        raise FileNotFoundError("Unable to fetch audio data.")
            raise

    # Extracts info from a URL or search, adjusting accordingly.
    def extract_info(self, item, count=1, search=False, mode=None):
        if mode or search and not item.startswith("ytsearch:") and not is_url(item):
            if count == 1:
                c = ""
            else:
                c = count
            item = item.replace(":", "-")
            if mode:
                self.youtube_dl_x += 1
                return self.downloader.extract_info(f"{mode}search{c}:{item}", download=False, process=False)
            exc = ""
            try:
                self.youtube_dl_x += 1
                return self.downloader.extract_info(f"ytsearch{c}:{item}", download=False, process=False)
            except Exception as ex:
                exc = repr(ex)
            try:
                self.youtube_dl_x += 1
                return self.downloader.extract_info(f"scsearch{c}:{item}", download=False, process=False)
            except Exception as ex:
                raise ConnectionError(exc + repr(ex))
        if is_url(item) or not search:
            return self.extract_from(item)
        self.youtube_dl_x += 1
        return self.downloader.extract_info(item, download=False, process=False)

    # Main extract function, able to extract from youtube playlists much faster than youtube-dl using youtube API, as well as ability to follow soundcloud links.
    def extract(self, item, force=False, count=1, mode=None, search=True):
        try:
            page = None
            output = deque()
            if google_api_key and ("youtube.com" in item or "youtu.be/" in item):
                p_id = None
                for x in ("?list=", "&list="):
                    if x in item:
                        p_id = item[item.index(x) + len(x):]
                        p_id = p_id.split("&", 1)[0]
                        break
                if p_id:
                    with tracebacksuppressor:
                        output.extend(self.get_youtube_playlist(p_id))
                        # Scroll to highlighted entry if possible
                        v_id = None
                        for x in ("?v=", "&v="):
                            if x in item:
                                v_id = item[item.index(x) + len(x):]
                                v_id = v_id.split("&", 1)[0]
                                break
                        if v_id:
                            for i, e in enumerate(output):
                                if v_id in e.url:
                                    output.rotate(-i)
                                    break
                        return output
                # # Pages may contain up to 50 items each
                # if p_id:
                #     url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&key={google_api_key}&playlistId={p_id}"
                #     page = 50
                # if page:
                #     futs = deque()
                #     maxitems = 5000
                #     # Optimized searching with lookaheads
                #     for i, curr in enumerate(range(0, maxitems, page)):
                #         with delay(0.03125):
                #             if curr >= maxitems:
                #                 break
                #             search = f"{url}&pageToken={self.yt_pages[curr]}"
                #             fut = create_future_ex(self.get_youtube_part, search, timeout=90)
                #             print("Sent 1 youtube playlist snippet.")
                #             futs.append(fut)
                #             if not (i < 1 or math.log2(i + 1) % 1) or not 1 + i & 15:
                #                 while futs:
                #                     fut = futs.popleft()
                #                     res = fut.result()
                #                     if not i:
                #                         maxitems = res[1] + page
                #                     if not res[0]:
                #                         maxitems = 0
                #                         futs.clear()
                #                         break
                #                     output += res[0]
                #     while futs:
                #         output.extend(futs.popleft().result()[0])
                #     # Scroll to highlighted entry if possible
                #     v_id = None
                #     for x in ("?v=", "&v="):
                #         if x in item:
                #             v_id = item[item.index(x) + len(x):]
                #             v_id = v_id.split("&", 1)[0]
                #             break
                #     if v_id:
                #         for i, e in enumerate(output):
                #             if v_id in e.url:
                #                 output.rotate(-i)
                #                 break
            elif regexp("(play|open|api)\\.spotify\\.com").search(item):
                # Spotify playlist searches contain up to 100 items each
                if "playlist" in item:
                    url = item[item.index("playlist"):]
                    url = url[url.index("/") + 1:]
                    key = url.split("/", 1)[0]
                    url = f"https://api.spotify.com/v1/playlists/{key}/tracks?type=track,episode"
                    page = 100
                # Spotify album searches contain up to 50 items each
                elif "album" in item:
                    url = item[item.index("album"):]
                    url = url[url.index("/") + 1:]
                    key = url.split("/", 1)[0]
                    url = f"https://api.spotify.com/v1/albums/{key}/tracks?type=track,episode"
                    page = 50
                # Single track links also supported
                elif "track" in item:
                    url = item[item.index("track"):]
                    url = url[url.index("/") + 1:]
                    key = url.split("/", 1)[0]
                    url = f"https://api.spotify.com/v1/tracks/{key}"
                    page = 1
                else:
                    raise TypeError("Unsupported Spotify URL.")
                if page == 1:
                    output.extend(self.get_spotify_part(url)[0])
                else:
                    futs = deque()
                    maxitems = 10000
                    # Optimized searching with lookaheads
                    for i, curr in enumerate(range(0, maxitems, page)):
                        with delay(0.03125):
                            if curr >= maxitems:
                                break
                            search = f"{url}&offset={curr}&limit={page}"
                            fut = create_future_ex(self.get_spotify_part, search, timeout=90)
                            print("Sent 1 spotify search.")
                            futs.append(fut)
                            if not (i < 1 or math.log2(i + 1) % 1) or not i & 7:
                                while futs:
                                    fut = futs.popleft()
                                    res = fut.result()
                                    if not i:
                                        maxitems = res[1] + page
                                    if not res[0]:
                                        maxitems = 0
                                        futs.clear()
                                        break
                                    output += res[0]
                    while futs:
                        output.extend(futs.popleft().result()[0])
                    # Scroll to highlighted entry if possible
                    v_id = None
                    for x in ("?highlight=spotify:track:", "&highlight=spotify:track:"):
                        if x in item:
                            v_id = item[item.index(x) + len(x):]
                            v_id = v_id.split("&", 1)[0]
                            break
                    if v_id:
                        for i, e in enumerate(output):
                            if v_id == e.get("id"):
                                output.rotate(-i)
                                break
            # Only proceed if no items have already been found (from playlists in this case)
            if not len(output):
                # Allow loading of files output by ~dump
                if is_url(item):
                    url = verify_url(item)
                    if url[-5:] == ".json" or url[-4:] in (".txt", ".bin", ".zip"):
                        s = Request(url)
                        d = select_and_loads(s, size=268435456)
                        q = d["queue"][:262144]
                        return [cdict(name=e["name"], url=e["url"], duration=e.get("duration")) for e in q]
                elif mode in (None, "yt"):
                    with suppress(NotImplementedError):
                        return self.search_yt(item)[:count]
                # Otherwise call automatic extract_info function
                resp = self.extract_info(item, count, search=search, mode=mode)
                if resp.get("_type", None) == "url":
                    resp = self.extract_from(resp["url"])
                if resp is None or not len(resp):
                    raise LookupError(f"No results for {item}")
                # Check if result is a playlist
                if resp.get("_type", None) == "playlist":
                    entries = list(resp["entries"])
                    if force or len(entries) <= 1:
                        for entry in entries:
                            # Extract full data if playlist only contains 1 item
                            data = self.extract_from(entry["url"])
                            temp = {
                                "name": data["title"],
                                "url": data["webpage_url"],
                                "duration": float(data["duration"]),
                                "stream": get_best_audio(resp),
                                "icon": get_best_icon(resp),
                            }
                            output.append(cdict(temp))
                    else:
                        for i, entry in enumerate(entries):
                            if not i:
                                # Extract full data from first item only
                                temp = self.extract(entry["url"], search=False)[0]
                            else:
                                # Get as much data as possible from all other items, set "research" flag to have bot lazily extract more info in background
                                with tracebacksuppressor:
                                    found = True
                                    if "title" in entry:
                                        title = entry["title"]
                                    else:
                                        title = entry["url"].rsplit("/", 1)[-1]
                                        if "." in title:
                                            title = title[:title.rindex(".")]
                                        found = False
                                    if "duration" in entry:
                                        dur = float(entry["duration"])
                                    else:
                                        dur = None
                                    url = entry.get("webpage_url", entry.get("url", entry.get("id")))
                                    if not url:
                                        continue
                                    temp = {
                                        "name": title,
                                        "url": url,
                                        "duration": dur,
                                    }
                                    if not is_url(url):
                                        if entry.get("ie_key", "").casefold() == "youtube":
                                            temp["url"] = f"https://www.youtube.com/watch?v={url}"
                                    temp["research"] = True
                            output.append(cdict(temp))
                else:
                    # Single item results must contain full data, we take advantage of that here
                    found = "duration" in resp
                    if found:
                        dur = resp["duration"]
                    else:
                        dur = None
                    temp = {
                        "name": resp["title"],
                        "url": resp["webpage_url"],
                        "duration": dur,
                        "stream": get_best_audio(resp),
                        "icon": get_best_icon(resp),
                    }
                    output.append(cdict(temp))
            return output
        except:
            if force != "spotify":
                raise
            print_exc()
            return 0

    def item_yt(self, item):
        video = next(iter(item.values()))
        if "videoId" not in video:
            return
        try:
            dur = time_parse(video["lengthText"]["simpleText"])
        except KeyError:
            dur = None
        try:
            title = video["title"]["runs"][0]["text"]
        except KeyError:
            title = video["title"]["simpleText"]
        try:
            tn = video["thumbnail"]
        except KeyError:
            thumbnail = None
        else:
            if type(tn) is dict:
                thumbnail = sorted(tn["thumbnails"], key=lambda t: t.get("width", 0) * t.get("height", 0))[-1]["url"]
            else:
                thumbnail = tn
        try:
            views = int(video["viewCountText"]["simpleText"].replace(",", "").replace("views", "").replace(" ", ""))
        except (KeyError, ValueError):
            views = 0
        return cdict(
            name=video["title"]["runs"][0]["text"],
            url=f"https://www.youtube.com/watch?v={video['videoId']}",
            duration=dur,
            icon=thumbnail,
            views=views,
        )

    def parse_yt(self, s):
        data = eval_json(s)
        results = alist()
        try:
            pages = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
        except KeyError:
            pages = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"]["contents"]
        for page in pages:
            try:
                items = next(iter(page.values()))["contents"]
            except KeyError:
                continue
            for item in items:
                if "promoted" not in next(iter(item)).casefold():
                    entry = self.item_yt(item)
                    if entry is not None:
                        results.append(entry)
        return sorted(results, key=lambda entry: entry.views, reverse=True)

    def search_yt(self, query):
        url = f"https://www.youtube.com/results?search_query={verify_url(query)}"
        self.youtube_x += 1
        resp = Request(url, timeout=12)
        result = None
        with suppress(ValueError):
            s = resp[resp.index(b"// scraper_data_begin") + 21:resp.rindex(b"// scraper_data_end")]
            s = s[s.index(b"var ytInitialData = ") + 20:s.rindex(b";")]
            result = self.parse_yt(s)
        with suppress(ValueError):
            s = resp[resp.index(b'window["ytInitialData"] = ') + 26:]
            s = s[:s.index(b'window["ytInitialPlayerResponse"] = null;')]
            s = s[:s.rindex(b";")]
            result = self.parse_yt(s)
        if result is None:
            raise NotImplementedError("Unable to read json response.")
        q = to_alphanumeric(full_prune(query))
        high = alist()
        low = alist()
        for entry in result:
            if entry.duration:
                name = full_prune(entry.name)
                aname = to_alphanumeric(name)
                spl = aname.split()
                if entry.duration < 960 or "extended" in q or "hour" in q or "extended" not in spl and "hour" not in spl and "hours" not in spl:
                    if fuzzy_substring(aname, q, match_length=False) >= 0.5:
                        high.append(entry)
                        continue
            low.append(entry)
        def key(entry):
            coeff = fuzzy_substring(to_alphanumeric(full_prune(entry.name)), q, match_length=False)
            if coeff < 0.5:
                coeff = 0
            return coeff
        out = sorted(high, key=key, reverse=True)
        out.extend(sorted(low, key=key, reverse=True))
        # print(out)
        return out

    # def extract_yt(self, query):
    #     if ":" in query:
    #         query = query.split("v=", 1)[1].split("&", 1)[0]
    #     url = f"https://www.youtube.com/watch?v={query}&gl=US&hl=en&has_verified=1&bpctr=9999999999"
    #     resp = Request(url)
    #     search = b"<script >var ytplayer = ytplayer || {};ytplayer.config = "
    #     try:
    #         data = resp[resp.index(search) + len(search):]
    #     except ValueError:
    #         raise FileNotFoundError("YouTube player data not found.")
    #     search = b";ytplayer.web_player_context_config = "
    #     data = data[:data.index(search)]
    #     player = eval_json(data)
    #     resp = eval_json(player["args"]["player_response"])
    #     entry = {
    #         "formats": [
    #             {

    #             }
    #         ]
    #     }

    # Performs a search, storing and using cached search results for efficiency.
    def search(self, item, force=False, mode=None, count=1):
        item = verify_search(item)
        if mode is None and count == 1 and item in self.searched:
            if utc() - self.searched[item].t < 18000:
                return self.searched[item].data
            else:
                self.searched.pop(item)
        while len(self.searched) > 262144:
            self.searched.pop(next(iter(self.searched)))
        with self.semaphore:
            try:
                obj = cdict(t=utc())
                obj.data = output = self.extract(item, force, mode=mode, count=count)
                self.searched[item] = obj
                return output
            except Exception as ex:
                print_exc()
                return repr(ex)

    # Gets the stream URL of a queue entry, starting download when applicable.
    def get_stream(self, entry, force=False, download=True, callback=None):
        stream = entry.get("stream", None)
        icon = entry.get("icon", None)
        # Use SHA-256 hash of URL to avoid filename conflicts
        h = shash(entry["url"])
        fn = h + ".mp3"
        # Use cached file if one already exists
        if self.cache.get(fn) or not download:
            entry["stream"] = stream
            entry["icon"] = icon
            # Files may have a callback set for when they are loaded
            if callback is not None:
                create_future_ex(callback)
            f = self.cache.get(fn)
            if f is not None:
                entry["file"] = f
                # Assign file duration estimate to queue entry
                # This could be done better, this current implementation is technically not thread-safe
                if f.loaded:
                    entry["duration"] = f.duration()
                else:
                    f.assign.append(entry)
                # Touch file to indicate usage
                f.ensure_time()
                while not f.stream:
                    time.sleep(0.1)
            if f or not force or not download:
                return f
        # "none" indicates stream is currently loading
        if stream == "none" and not force:
            return None
        entry["stream"] = "none"
        searched = False
        # If "research" tag is set, entry does not contain full data and requires another search
        if "research" in entry:
            try:
                self.extract_single(entry)
                searched = True
                entry.pop("research", None)
            except:
                print_exc()
                entry.pop("research", None)
                raise
            else:
                stream = entry.get("stream", None)
                icon = entry.get("icon", None)
        # If stream is still not found or is a soundcloud audio fragment playlist file, perform secondary youtube-dl search
        if stream in (None, "none"):
            data = self.search(entry["url"])
            stream = set_dict(data[0], "stream", data[0].url)
            icon = set_dict(data[0], "icon", data[0].url)
        elif not searched and (stream.startswith("https://cf-hls-media.sndcdn.com/") or stream.startswith("https://www.yt-download.org/download/")):
            data = self.extract(entry["url"])
            stream = set_dict(data[0], "stream", data[0].url)
            icon = set_dict(data[0], "icon", data[0].url)
        # Otherwise attempt to start file download
        try:
            self.cache[fn] = f = AudioFile(fn)
            if stream.startswith("ytsearch:") or stream in (None, "none"):
                self.extract_single(entry, force=True)
                stream = entry.get("stream")
                if stream in (None, "none"):
                    raise FileNotFoundError("Unable to locate appropriate file stream.")
            f.load(stream, check_fmt=entry.get("duration") is None, webpage_url=entry["url"])
            # Assign file duration estimate to queue entry
            f.assign.append(entry)
            entry["stream"] = stream
            entry["icon"] = icon
            entry["file"] = f
            f.ensure_time()
            # Files may have a callback set for when they are loaded
            if callback is not None:
                create_future_ex(callback)
            return f
        except:
            # Remove entry URL if loading failed
            print_exc()
            entry["url"] = ""

    # For ~download
    def download_file(self, url, fmt="ogg", auds=None, fl=8388608):
        # Select a filename based on current time to avoid conflicts
        if fmt[:3] == "mid":
            mid = True
            fmt = "mp3"
            br = 192
            fs = 67108864
        else:
            mid = False
        fn = f"cache/&{ts_us()}.{fmt}"
        info = self.extract(url)[0]
        self.get_stream(info, force=True, download=False)
        stream = info["stream"]
        if not stream:
            raise LookupError(f"No stream URLs found for {url}")
        if not mid:
            # Attempt to automatically adjust output bitrate based on file duration
            duration = get_duration(stream)
            if type(duration) not in (int, float):
                dur = 960
            else:
                dur = duration
            fs = fl - 131072
        else:
            dur = 0
        args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-vn", "-i", stream]
        if auds is not None:
            args.extend(auds.construct_options(full=True))
            dur /= auds.stats.speed / 2 ** (auds.stats.resample / 12)
        if not mid:
            if dur > 960:
                dur = 960
            br = max(32, min(256, floor(((fs - 131072) / dur / 128) / 4) * 4)) * 1024
        if auds and br > auds.stats.bitrate:
            br = max(4096, auds.stats.bitrate)
        args.extend(("-ar", str(SAMPLE_RATE), "-b:a", str(br), "-fs", str(fs), fn))
        try:
            resp = subprocess.run(args)
            resp.check_returncode()
        except subprocess.CalledProcessError as ex:
            # Attempt to convert file from org if FFmpeg failed
            try:
                new = select_and_convert(stream)
            except ValueError:
                if resp.stderr:
                    raise RuntimeError(*ex.args, resp.stderr)
                raise ex
            # Re-estimate duration if file was successfully converted from org
            args[8] = new
            if not mid:
                dur = get_duration(new)
                if dur:
                    if auds:
                        dur /= auds.stats.speed / 2 ** (auds.stats.resample / 12)
                    br = max(32, min(256, floor(((fs - 131072) / dur / 128) / 4) * 4)) * 1024
                    args[-4] = str(br)
                if auds and br > auds.stats.bitrate:
                    br = max(4096, auds.stats.bitrate)
            try:
                resp = subprocess.run(args)
                resp.check_returncode()
            except subprocess.CalledProcessError as ex:
                if resp.stderr:
                    raise RuntimeError(*ex.args, resp.stderr)
                raise ex
            if not is_url(new):
                with suppress():
                    os.remove(new)
        if not mid:
            return fn, f"{info['name']}.{fmt}"
        self.other_x += 1
        with open(fn, "rb") as f:
            resp = Request(
                "https://cts.ofoct.com/upload.php",
                method="post",
                files={"myfile": ("temp.mp3", f)},
                timeout=32,
                decode=True
            )
            resp_fn = ast.literal_eval(resp)[0]
        url = f"https://cts.ofoct.com/convert-file_v2.php?cid=audio2midi&output=MID&tmpfpath={resp_fn}&row=file1&sourcename=temp.ogg&rowid=file1"
        # print(url)
        with suppress():
            os.remove(fn)
        self.other_x += 1
        resp = Request(url, timeout=420)
        self.other_x += 1
        out = Request(f"https://cts.ofoct.com/get-file.php?type=get&genfpath=/tmp/{resp_fn}.mid", timeout=24)
        return io.BytesIO(out), f"{info['name']}.mid"

    # Extracts full data for a single entry. Uses cached results for optimization.
    def extract_single(self, i, force=False):
        item = i.url
        if not force:
            if item in self.searched:
                if utc() - self.searched[item].t < 18000:
                    it = self.searched[item].data[0]
                    i.update(it)
                    if i.get("stream") not in (None, "none"):
                        return True
                else:
                    self.searched.pop(item, None)
            while len(self.searched) > 262144:
                self.searched.pop(next(iter(self.searched)))
        with self.semaphore:
            try:
                data = self.extract_true(item)
                if "entries" in data:
                    data = data["entries"][0]
                elif not issubclass(type(data), collections.abc.Mapping):
                    data = data[0]
                obj = cdict(t=utc())
                obj.data = out = [cdict(
                    name=data["name"],
                    url=data["url"],
                    stream=get_best_audio(data),
                    icon=get_best_icon(data),
                )]
                try:
                    out[0].duration = data["duration"]
                except KeyError:
                    out[0].research = True
                self.searched[item] = obj
                it = out[0]
                i.update(it)
            except:
                i.url = ""
                print_exc()
                return False
        return True

ytdl = AudioDownloader()


class Queue(Command):
    name = ["▶️", "P", "Q", "Play", "Enqueue"]
    alias = name + ["LS"]
    description = "Shows the music queue, or plays a song in voice."
    usage = "<search_link[]> <verbose(?v)> <hide(?h)> <force(?f)> <budge(?b)>"
    flags = "hvfbz"
    no_parse = True

    def __call__(self, flags, name, argv, **void):
        if not argv:
            elapsed = auds.stats.position + (utc() - auds.started) * auds.speed * (auds.reverse * -2 + 1)
            q = auds.queue
            v = "v" in flags
            if not v and len(q) and auds.paused & 1 and "p" in name:
                auds.paused &= -2
                auds.pausec = False
                auds.preparing = False
                if auds.stats.position + (utc() - auds.started) * auds.speed * (auds.reverse * -2 + 1) <= 0:
                    if auds.queue:
                        auds.queue[0].pop("played", None)
                create_future_ex(auds.queue.update_play)
                create_future_ex(auds.ensure_play)
                return "Successfully resumed audio playback."
            if not len(q):
                auds.preparing = False
                create_future_ex(auds.update)
                return "Queue is currently empty."
            return self._callback_()
        resp = ytdl.search(argv)
        if auds.stats.quiet & 2:
            flags.setdefault("h", 1)
        elapsed = auds.stats.position + (utc() - auds.started) * auds.speed * (auds.reverse * -2 + 1)
        q = auds.queue
        if type(resp) is str:
            raise evalEX(resp)
        added = deque()
        names = []
        for e in resp:
            name = e.name
            url = e.url
            temp = {
                "name": name,
                "url": url,
                "duration": e.get("duration"),
            }
            if "research" in e:
                temp["research"] = True
            added.append(cdict(temp))
            names.append(clr_md(name))
        if "b" not in flags:
            total_duration = 0
            for e in q:
                total_duration += e_dur(e.duration)
            if auds.reverse and len(auds.queue):
                total_duration += elapsed - e_dur(q[0].duration)
            else:
                total_duration -= elapsed
        if auds.stats.shuffle:
            added = shuffle(added)
        tdur = 3
        if "f" in flags:
            auds.queue[0].pop("played", None)
            auds.queue.enqueue(added, 0)
            create_future_ex(auds.new)
            total_duration = tdur
        elif "b" in flags:
            auds.queue.enqueue(added, 1)
            total_duration = max(3, e_dur(q[0].duration) - elapsed if q else 0)
        else:
            auds.queue.enqueue(added, -1)
            total_duration = max(total_duration / auds.speed, tdur)
        if not names:
            raise LookupError("No results for " + str(argv) + ".")
        if "v" in flags:
            names = clr_md(alist(i.name + ": " + time_disp(e_dur(i.duration)) for i in added))
        elif len(names) == 1:
            names = names[0]
        else:
            names = str(len(names)) + " items"
        if "h" not in flags:
            return f"Added {sqr_md(names)} to the queue! Estimated time until playing: {sqr_md(sec2time(total_duration))}."

    def _callback_(self, **void):
        q = auds.queue
        pos = 0
        last = max(0, len(q) - 10)
        content = "Queue:\n"
        elapsed = auds.stats.position + (utc() - auds.started) * auds.speed * (auds.reverse * -2 + 1)
        startTime = 0
        if auds.stats.loop or auds.stats.repeat:
            totalTime = inf
        else:
            if auds.reverse and len(auds.queue):
                totalTime = elapsed - e_dur(auds.queue[0].duration)
            else:
                totalTime = -elapsed
            i = 0
            for e in q:
                totalTime += e_dur(e.duration)
                if i < pos:
                    startTime += e_dur(e.duration)
                if not 1 + i & 4095:
                    time.sleep(0.2)
                i += 1
        cnt = len(q)
        info = (
            str(cnt) + " item" + "s" * (cnt != 1) + ", estimated total duration: "
            + sec2time(totalTime / auds.speed) + "\n"
        )
        duration = e_dur(q[0].duration)
        sym = [C.green + "█", C.magenta + "▒"]
        barsize = 24
        r = round(min(1, elapsed / duration) * barsize)
        bar = sym[0] * r + sym[1] * (barsize - r) + C.white
        countstr = "Currently playing " + sqr_md(q[0].name) + " " + C.cyan + "(" + q[0].url + ")" + C.white + "\n"
        countstr += (
            "(" + str(time_disp(elapsed))
            + "/" + str(time_disp(duration)) + ") "
        )
        countstr += bar + "\n"
        emb = content + info + countstr
        if q:
            icon = q[0].get("icon", "")
        else:
            icon = ""
        if icon:
            emb += sqr_md(icon, colour=C.cyan) + "\n"
        embstr = ""
        currTime = startTime
        i = pos
        while i < min(pos + 10, len(q)):
            e = q[i]
            curr = " " * (int(math.log10(len(q))) - int(math.log10(max(1, i))))
            curr += sqr_md(i, colour=C.yellow) + " " + C.blue
            curr += clr_md(e.name)
            curr += " " + C.cyan + "(" + ensure_url(e.url) + ")" + C.white + " ("
            curr += time_disp(e_dur(e.duration)) + ")"
            curr += "\n"
            if len(embstr) + len(curr) > 2048 - len(emb):
                break
            embstr += curr
            if i <= 1 or not auds.stats.shuffle:
                currTime += e_dur(e.duration)
            if not 1 + 1 & 4095:
                time.sleep(0.3)
            i += 1
        emb += embstr
        if pos != last:
            emb += str("And ") + str(len(q) - i) + str(" more...")
        return emb


class Skip(Command):
    name = ["⏭", "🚫", "S", "SK", "CQ", "Remove", "Rem", "ClearQueue", "Clear"]
    description = "Removes an entry or range of entries from the voice channel queue."
    usage = "<0:queue_position[0]> <force(?f)> <vote(?v)> <hide(?h)>"
    flags = "fhv"

    def __call__(self, name, args, argv, flags, **void):
        if name.lower().startswith("c"):
            argv = "inf"
            args = [argv]
            flags["f"] = True
        count = len(auds.queue)
        if not count:
            raise IndexError("Queue is currently empty.")
        if not argv:
            elems = [0]
        elif ":" in argv or ".." in argv:
            while "..." in argv:
                argv = argv.replace("...", "..")
            l = argv.replace("..", ":").split(":")
            it = None
            if len(l) > 3:
                raise ArgumentError("Too many arguments for range input.")
            elif len(l) > 2:
                num = eval_math(l[0])
                it = int(round(float(num)))
            if l[0]:
                num = eval_math(l[0])
                if num > count:
                    num = count
                else:
                    num = round(num) % count
                left = num
            else:
                left = 0
            if l[1]:
                num = eval_math(l[1])
                if num > count:
                    num = count
                else:
                    num = round(num) % count
                right = num
            else:
                right = count
            elems = xrange(left, right, it)
        else:
            elems = [0] * len(args)
            for i in range(len(args)):
                elems[i] = eval_math(args[i])
        if not "f" in flags:
            valid = True
            for e in elems:
                if not is_finite(e):
                    valid = False
                    break
            if not valid:
                elems = range(count)
        required = 0
        response = ""
        i = 1
        for pos in elems:
            pos = float(pos)
            try:
                if not is_finite(pos):
                    if "f" in flags:
                        auds.queue.clear()
                        create_future_ex(auds.new)
                        if "h" not in flags:
                            return "Removed all items from the queue."
                        return
                    raise LookupError
                curr = auds.queue[pos]
            except LookupError:
                response += "\n" + repr(IndexError("Entry " + str(pos) + " is out of range."))
                continue
            curr.skips = None
            i += 1
        pops = set()
        count = 0
        i = 1
        while i < len(auds.queue):
            q = auds.queue
            song = q[i]
            if song.get("skips", True) is None:
                if count <= 3:
                    q.pop(i)
                else:
                    pops.add(i)
                    i += 1
                if count < 4:
                    response += f"{sqr_md(song.name)} has been removed from the queue.\n"
                count += 1
            else:
                i += 1
        if pops:
            auds.queue.pops(pops)
        if auds.queue:
            song = auds.queue[0]
            if song.get("skips", True) is None:
                song.played = True
                # auds.preparing = False
                if name.startswith("r"):
                    auds.queue.popleft()
                elif auds.stats.repeat:
                    auds.queue[0].pop("played", None)
                elif auds.stats.loop:
                    e = auds.queue.popleft()
                    e.pop("played", None)
                    auds.queue.append(e)
                else:
                    auds.queue.popleft()
                auds.new()
                # create_future_ex(auds.queue.advance, looped=r, repeated=r, shuffled=r, timeout=18)
                if count < 4:
                    response += f"{sqr_md(song.name)} has been removed from the queue.\n"
                count += 1
        if "h" not in flags:
            if count >= 4:
                response += f"{sqr_md(song.name)} has been removed from the queue.\n"
            return response.strip()


class Pause(Command):
    name = ["⏸️", "⏯️", "Resume", "Unpause", "Stop"]
    description = "Pauses, stops, or resumes audio playing."
    usage = "<hide(?h)>"
    flags = "h"

    def __call__(self, name, flags, **void):
        auds.preparing = False
        if auds.queue:
            auds.queue[0].pop("played", None)
        if name == "stop":
            auds.stats.position = 0
            auds.pos = 0
            auds.started = 0
        if not auds.paused > 1:
            if name == "⏯️":
                auds.paused ^= 1
            else:
                auds.paused = auds.pausec = name in ("pause", "stop", "⏸️")
            if auds.paused:
                create_future_ex(auds.stop, timeout=18)
        if not auds.paused:
            if auds.queue and not getattr(auds, "proc", None) or not auds.proc.is_running():
                auds.new(ytdl.get_stream(auds.queue[0]), auds.stats.position)
        else:
            auds.pause()
        if auds.player is not None:
            auds.player.time = 1 + utc()
        auds.update()
        if "h" not in flags:
            if name in ("⏸️", "pause"):
                past = "paused"
            elif name == "⏯️":
                past = "paused" if auds.paused & 1 else "resumed"
            elif name == "stop":
                past = "stopped"
            else:
                past = name + "d"
            return f"Successfully {past} audio playback."


class Seek(Command):
    server_only = True
    name = ["↔️", "Replay"]
    min_display = "0~1"
    description = "Seeks to a position in the current audio file."
    usage = "<position[0]> <hide(?h)>"
    flags = "h"
    rate_limit = (0.5, 3)

    def __call__(self, argv, name, flags, **void):
        if name == "replay":
            num = 0
        elif not argv:
            return f"Current audio position: {sqr_md(sec2time(auds.stats.position))}."
        else:
            orig = auds.stats.position
            expr = argv
            num = eval_time(expr, orig)
        pos = auds.seek(num)
        if auds.player is not None:
            auds.player.time = 1 + utc()
        if "h" not in flags:
            return f"Successfully moved audio position to {sqr_md(sec2time(pos))}."


class Dump(Command):
    name = ["Save", "Load", "Dujmpö"]
    description = "Saves or loads the currently playing audio queue state."
    usage = "<data{attached_file}> <append(?a)> <hide(?h)>"
    flags = "ah"

    def __call__(self, name, argv, flags, **void):
        if argv == "" or name.lower() == "save":
            if name.lower() == "load":
                raise ArgumentError("Please input a file, URL or json data to load.")
            resp = auds.get_dump(js=True)
            f = open("dump.json", "wb")
            f.write(bytes(resp, "utf-8"))
            f.close()
            return f"Successfully saved queue data to {sqr_md('dump.json')}."
        try:
            url = verify_url(argv)
            s = Request(url)
        except:
            s = argv
        d = select_and_loads(s)
        if type(d) is list:
            d = dict(queue=d, stats={})
        q = d["queue"]
        for i in range(len(q)):
            e = q[i] = cdict(q[i])
            e.skips = []
        if auds.stats.shuffle:
            shuffle(q)
        for k in d["stats"]:
            if k not in auds.stats:
                d["stats"].pop(k)
            if k in "loop repeat shuffle quiet stay":
                d["stats"][k] = bool(d["stats"][k])
            else:
                d["stats"][k] = float(d["stats"][k])
        if "a" not in flags:
            if auds.queue:
                auds.preparing = True
                auds.stop()
                auds.queue.clear()
            auds.paused = False
            auds.stats.update(d["stats"])
            auds.queue.enqueue(q, -1)
            auds.update()
            auds.queue.update_play()
            if "h" not in flags:
                return (
                    "Successfully loaded audio queue data."
                )
        auds.queue.enqueue(q, -1)
        auds.stats.update(d["stats"])
        if "h" not in flags:
            return (
                "Successfully appended loaded data to queue."
            )
            

class AudioSettings(Command):
    aliasMap = {
        "Volume": "volume",
        "Speed": "speed",
        "Pitch": "pitch",
        "Pan": "pan",
        "BassBoost": "bassboost",
        "Reverb": "reverb",
        "Compressor": "compressor",
        "Chorus": "chorus",
        "NightCore": "resample",
        "Resample": "resample",
        "Bitrate": "bitrate",
        "LoopQueue": "loop",
        "Repeat": "repeat",
        "ShuffleQueue": "shuffle",
        "Quiet": "quiet",
        "Stay": "stay",
        "Reset": "reset",
    }
    aliasExt = {
        "AudioSettings": None,
        "Audio": None,
        "A": None,
        "Vol": "volume",
        "V": "volume",
        "🔉": "volume",
        "🔊": "volume",
        "📢": "volume",
        "SP": "speed",
        "⏩": "speed",
        "rewind": "rewind",
        "⏪": "rewind",
        "PI": "pitch",
        "↕️": "pitch",
        "PN": "pan",
        "BB": "bassboost",
        "🥁": "bassboost",
        "RV": "reverb",
        "📉": "reverb",
        "CO": "compressor",
        "🗜": "compressor",
        "CH": "chorus",
        "📊": "chorus",
        "NC": "resample",
        "Rate": "bitrate",
        "BPS": "bitrate",
        "BR": "bitrate",
        "LQ": "loop",
        "🔁": "loop",
        "LoopOne": "repeat",
        "🔂": "repeat",
        "L1": "repeat",
        "SQ": "shuffle",
        "🤫": "quiet",
        "🔕": "quiet",
        "24/7": "stay",
        "♻": "reset",
    }

    def __init__(self, *args):
        self.alias = list(self.aliasMap) + list(self.aliasExt)[1:]
        self.name = list(self.aliasMap)
        self.description = "Changes the current audio settings for this server."
        self.usage = (
            "<value[]> <volume()(?v)> <speed(?s)> <pitch(?p)> <pan(?e)> <bassboost(?b)> <reverb(?r)> <compressor(?c)> <chorus(?u)> <nightcore(?n)>"
            + " <bitrate(?i)> <loop(?l)> <repeat(?1)> <shuffle(?x)> <quiet(?q)> <stay(?t)> <force_permanent(?f)> <disable_all(?d)> <hide(?h)>"
        )
        self.flags = "vspebrcunilxqtfdh"
        self.map = {k.lower(): self.aliasMap[k] for k in self.aliasMap}
        add_dict(self.map, {k.lower(): self.aliasExt[k] for k in self.aliasExt})
        super().__init__(*args)

    def __call__(self, flags, name, argv, **void):
        ops = alist()
        op1 = self.map[name]
        if op1 == "reset":
            flags.clear()
            flags["d"] = True
        elif op1 is not None:
            ops.append(op1)
        disable = "d" in flags
        if "v" in flags:
            ops.append("volume")
        if "s" in flags:
            ops.append("speed")
        if "p" in flags:
            ops.append("pitch")
        if "e" in flags:
            ops.append("pan")
        if "b" in flags:
            ops.append("bassboost")
        if "r" in flags:
            ops.append("reverb")
        if "c" in flags:
            ops.append("compressor")
        if "u" in flags:
            ops.append("chorus")
        if "n" in flags:
            ops.append("resample")
        if "i" in flags:
            ops.append("bitrate")
        if "l" in flags:
            ops.append("loop")
        if "1" in flags:
            ops.append("repeat")
        if "x" in flags:
            ops.append("shuffle")
        if "q" in flags:
            ops.append("quiet")
        if "t" in flags:
            ops.append("stay")
        if not disable and not argv and (len(ops) != 1 or ops[-1] not in "rewind loop repeat shuffle quiet stay"):
            if len(ops) == 1:
                op = ops[0]
            else:
                key = lambda x: (round(x * 100, 9), x)[type(x) is bool]
                d = dict(auds.stats)
                d.pop("position", None)
                return f"Current audio settings:\n{iter2str(d, key=key)}"
            orig = auds.stats[op]
            num = round(100 * orig, 9)
            return css_md(f"Current audio {op} setting: [{num}].")
        if not ops:
            if disable:
                pos = auds.stats.position + (utc() - auds.started) * auds.speed * (auds.reverse * -2 + 1)
                res = False
                for k, v in auds.defaults.items():
                    if k != "volume" and auds.stats.get(k) != v:
                        res = True
                        break
                auds.stats = cdict(auds.defaults)
                if "f" in flags:
                    if os.path.exists("audiosettings.json"):
                        os.remove("audiosettings.json")
                if auds.queue and res:
                    auds.new(auds.file, pos)
                succ = "Permanently" if "f" in flags else "Successfully"
                return f"{succ} reset all audio settings."
            else:
                ops.append("volume")
        s = ""
        for op in ops:
            if type(op) is str:
                if op in "loop repeat shuffle quiet stay" and not argv:
                    argv = str(not auds.stats[op])
                elif op == "rewind":
                    argv = "100"
            if op == "rewind":
                op = "speed"
                argv = "- " + argv
            if disable:
                val = auds.defaults[op]
                if type(val) is not bool:
                    val *= 100
                argv = str(val)
            origStats = auds.stats
            orig = round(origStats[op] * 100, 9)
            num = eval_math(argv, orig)
            val = round_min(float(num / 100))
            new = round(num, 9)
            if op in "loop repeat shuffle quiet stay":
                origStats[op] = new = bool(val)
                orig = bool(orig)
            else:
                if op == "bitrate":
                    if val > 1966.08:
                        raise PermissionError("Maximum allowed bitrate is 196608.")
                    elif val < 5.12:
                        raise ValueError("Bitrate must be equal to or above 512.")
                elif op == "speed":
                    if abs(val * 2 ** (origStats.get("resample", 0) / 12)) > 16:
                        raise OverflowError("Maximum speed is 1600%.")
                elif op == "resample":
                    if abs(origStats.get("speed", 1) * 2 ** (val / 12)) > 16:
                        raise OverflowError("Maximum speed is 1600%.")
                origStats[op] = val
            if auds.queue:
                if type(op) is str and op not in "loop repeat shuffle quiet stay":
                    auds.new(auds.file, auds.stats.position + (utc() - auds.started) * auds.speed * (auds.reverse * -2 + 1))
            changed = "Permanently changed" if "f" in flags else "Changed"
            s += f"\n{changed} audio {op} setting from {sqr_md(orig)} to {sqr_md(new)}."
        if "f" in flags:
            with open("audiosettings.json", "w") as f:
                f.write(repr(auds.stats))
        if "h" not in flags:
            return s.strip()


class Rotate(Command):
    name = ["Jump"]
    description = "Rotates the queue to the left by a certain amount of steps."
    usage = "<position> <hide(?h)>"
    flags = "h"

    def __call__(self, argv, flags, **void):
        if not argv:
            amount = 1
        else:
            amount = eval_math(argv)
        if len(auds.queue) > 1 and amount:
            auds.queue[0].pop("played", None)
            auds.queue.rotate(-amount)
            auds.seek(inf)
        if "h" not in flags:
            return f"Successfully rotated queue [{amount}] step{'s' if amount != 1 else ''}."


class Shuffle(Command):
    name = ["🔀"]
    description = "Shuffles the audio queue."
    usage = "<force(?f)> <hide(?h)>"
    flags = "fh"

    def __call__(self, flags, **void):
        if len(auds.queue) > 1:
            if "f" in flags:
                auds.queue[0].pop("played", None)
                shuffle(auds.queue)
                auds.seek(inf)
            else:
                temp = auds.queue.popleft()
                shuffle(auds.queue)
                auds.queue.appendleft(temp)
        if "h" not in flags:
            return (
                "Successfully shuffled queue."
            )


class Reverse(Command):
    description = "Reverses the audio queue direction."
    usage = "<hide(?h)>"
    flags = "h"

    def __call__(self, flags, **void):
        if len(auds.queue) > 1:
            reverse(auds.queue)
            auds.queue.rotate(-1)
        if "h" not in flags:
            return (
                "Successfully reversed queue."
            )


def extract_lyrics(s):
    s = s[s.index("JSON.parse(") + len("JSON.parse("):]
    s = s[:s.index("</script>")]
    if "window.__" in s:
        s = s[:s.index("window.__")]
    s = s[:s.rindex(");")]
    data = ast.literal_eval(s)
    d = eval_json(data)
    lyrics = d["songPage"]["lyricsData"]["body"]["children"][0]["children"]
    newline = True
    output = ""
    while lyrics:
        line = lyrics.pop(0)
        if type(line) is str:
            if line:
                if line.startswith("["):
                    output += "\n"
                    newline = False
                if "]" in line:
                    if line == "]":
                        if output.endswith(" ") or output.endswith("\n"):
                            output = output[:-1]
                    newline = True
                output += line + ("\n" if newline else (" " if not line.endswith(" ") else ""))
        elif type(line) is dict:
            if "children" in line:
                # This is a mess, the children objects may or may not represent single lines
                lyrics = line["children"] + lyrics
    return output


def get_lyrics(item):
    url = "https://api.genius.com/search"
    for i in range(2):
        header = {"Authorization": f"Bearer {genius_key}"}
        if i == 0:
            search = item
        else:
            search = "".join(shuffle(item.split()))
        data = {"q": search}
        resp = Request(url, data=data, headers=header, timeout=18)
        rdata = json.loads(resp)
        hits = rdata["response"]["hits"]
        name = None
        path = None
        for h in hits:
            with tracebacksuppressor:
                name = h["result"]["title"]
                path = h["result"]["api_path"]
                break
        if path and name:
            s = "https://genius.com" + path
            page = Request(s, headers=header, decode=True)
            text = page
            html = BeautifulSoup(text, "html.parser", timeout=18)
            lyricobj = html.find('div', class_='lyrics')
            if lyricobj is not None:
                lyrics = lyricobj.get_text().strip()
                print("lyrics_html", s)
                return name, lyrics
            try:
                lyrics = extract_lyrics(text).strip()
                print("lyrics_json", s)
                return name, lyrics
            except:
                if i:
                    raise
                print_exc()
                print(s)
                print(text)
    raise LookupError(f"No results for {item}.")


mmap = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "〝": '"',
    "〞": '"',
    "⸌": "'",
    "⸍": "'",
    "⸢": "'",
    "⸣": "'",
    "⸤": "'",
    "⸥": "'",
    "⸨": "((",
    "⸩": "))",
    "⟦": "[",
    "⟧": "]",
    "〚": "[",
    "〛": "]",
    "「": "[",
    "」": "]",
    "『": "[",
    "』": "]",
    "【": "[",
    "】": "]",
    "〖": "[",
    "〗": "]",
    "（": "(",
    "）": ")",
    "［": "[",
    "］": "]",
    "｛": "{",
    "｝": "}",
    "⌈": "[",
    "⌉": "]",
    "⌊": "[",
    "⌋": "]",
    "⦋": "[",
    "⦌": "]",
    "⦍": "[",
    "⦐": "]",
    "⦏": "[",
    "⦎": "]",
    "⁅": "[",
    "⁆": "]",
    "〔": "[",
    "〕": "]",
    "«": "<<",
    "»": ">>",
    "❮": "<",
    "❯": ">",
    "❰": "<",
    "❱": ">",
    "❬": "<",
    "❭": ">",
    "＜": "<",
    "＞": ">",
    "⟨": "<",
    "⟩": ">",
}
mtrans = "".maketrans(mmap)


class Lyrics(Command):
    time_consuming = True
    name = ["SongLyrics"]
    min_level = 0
    description = "Searches genius.com for lyrics of a song."
    usage = "<0:search_link{queue}> <verbose(?v)>"
    flags = "v"
    lyric_trans = re.compile(
        (
            "[([]+"
            "(((official|full|demo|original|extended) *)?"
            "((version|ver.?) *)?"
            "((w\\/)?"
            "(lyrics?|vocals?|music|ost|instrumental|acoustic|studio|hd|hq) *)?"
            "((album|video|audio|cover|remix) *)?"
            "(upload|reupload|version|ver.?)?"
            "|(feat|ft)"
            ".+)"
            "[)\\]]+"
        ),
        flags=re.I,
    )
    rate_limit = 2

    def __call__(self, argv, flags, **void):
        if not argv:
            try:
                if not auds.queue:
                    raise EOFError
                argv = auds.queue[0].name
            except:
                raise IndexError("Queue not found. Please input a search term, URL, or file.")
        if is_url(argv):
            argv = verify_url(argv)
            resp = ytdl.search(argv)
            search = resp[0].name
        else:
            search = argv
        search = search.translate(mtrans)
        item = verify_search(to_alphanumeric(re.sub(self.lyric_trans, "", search)))
        if not item:
            item = verify_search(to_alphanumeric(search))
            if not item:
                item = search
        name, lyrics = get_lyrics(item)
        text = clr_md(lyrics.strip())
        msg = "Lyrics for " + C.magenta + "{" + name + "}" + C.white + ":"
        s = msg + "\n" + text
        if "v" not in flags and len(s) <= 2000:
            return s.replace("[", C.blue + "[").replace("]", "]" + C.white)
        title = "Lyrics for " + C.magenta + "{" + name + "}" + C.white + ":"
        if len(text) > 6000:
            return (title + "\n\n" + text).strip()
        emb = ""
        curr = ""
        paragraphs = [p + "\n\n" for p in text.split("\n\n")]
        while paragraphs:
            para = paragraphs.pop(0)
            if not emb and len(curr) + len(para) > 2000:
                if len(para) <= 2000:
                    emb = curr.strip() + "\n\n"
                    curr = para
                else:
                    p = [i + "\n" for i in para.split("\n")]
                    if len(p) <= 1:
                        p = [i + "" for i in para.split()]
                        if len(p) <= 1:
                            p = list(para)
                    paragraphs = p + paragraphs
            elif emb and len(curr) + len(para) > 1000:
                if len(para) <= 1000:
                    emb += curr.strip() + "\n\n"
                    curr = para
                else:
                    p = [i + "\n" for i in para.split("\n")]
                    if len(p) <= 1:
                        p = [i + "" for i in para.split()]
                        if len(p) <= 1:
                            p = list(para)
                    paragraphs = p + paragraphs
            else:
                curr += para
        if curr:
            emb += curr.strip()
        emb = title + "\n" + emb
        return emb.strip().replace("[", C.blue + "[").replace("]", "]" + C.white)


Downloaders = deque()


class Download(Command):
    name = ["Search", "YTDL", "Youtube_DL", "AF", "AudioFilter", "ConvertORG", "Org2xm", "Convert"]
    description = "Searches and/or downloads a song from a YouTube/SoundCloud query or audio file link."
    usage = "<0:search_link{queue}> <-1:out_format[ogg]> <apply_settings(?a)> <verbose_search(?v)> <show_debug(?z)>"
    flags = "avz"

    def __call__(self, name, argv, flags, **void):
        if name in ("af", "audiofilter"):
            flags.setdefault("a", 1)
        if not argv:
            try:
                if not auds.queue:
                    raise EOFError
                res = [{"name": e.name, "url": e.url} for e in auds.queue[:10]]
                fmt = "ogg"
                end = "Current items in queue:"
            except:
                raise IndexError("Queue not found. Please input a search term, URL, or file.")
        else:
            if " " in argv:
                try:
                    spl = shlex.split(argv)
                except ValueError:
                    spl = argv.split(" ")
                if len(spl) > 1:
                    fmt = spl[-1]
                    if fmt.startswith("."):
                        fmt = fmt[1:]
                    if fmt.casefold() not in ("mp3", "ogg", "opus", "m4a", "webm", "wav", "mid", "midi"):
                        fmt = "ogg"
                    else:
                        if spl[-2] in ("as", "to"):
                            spl.pop(-1)
                        argv = " ".join(spl[:-1])
                else:
                    fmt = "ogg"
            else:
                fmt = "ogg"
            argv = verify_search(argv)
            res = []
            if is_url(argv):
                argv = verify_url(argv)
                data = ytdl.extract(argv)
                res += data
            if not res:
                sc = min(4, flags.get("v", 0) + 1)
                yt = min(6, sc << 1)
                res = []
                temp = ytdl.search(argv, mode="yt", count=yt)
                res.extend(temp)
                temp = ytdl.search(argv, mode="sc", count=sc)
                res.extend(temp)
            if not res:
                raise LookupError("No results for " + argv + ".")
            res = res[:10]
            end = "Search results for " + argv + ":"
        a = flags.get("a", 0)
        end += "\nDestination format: {." + fmt + "}"
        if a:
            end += ", Audio settings: {ON}"
        end += "\n"
        DownloadData = cdict(urls=[i["url"] for i in res], fmt=fmt, audio=int(bool(a)), func=self._callback_)
        Downloaders.appendleft(DownloadData)
        emb = end + "\n".join(
            [sqr_md(i, colour=C.yellow) + " " + C.blue + e["name"] + " " + C.cyan + "(" + ensure_url(e["url"]) + ")" + C.white for i in range(len(res)) for e in [res[i]]]
        )
        return emb

    def _callback_(self, num, **void):
        data = Downloaders[0]
        if num <= len(data.urls):
            url = data.urls[num]
            fl = 268435456
            print("Downloading " + sqr_md(ensure_url(url)) + "...")
            try:
                if data.audio:
                    a = auds
                else:
                    a = None
            except LookupError:
                a = None
            fn, out = ytdl.download_file(
                url,
                fmt=data.fmt,
                auds=a,
                fl=fl,
            )
            Downloaders.popleft()
            return C.magenta + fn + C.white + ": " + C.green + "Download complete."
        raise ValueError("Invalid entry number.")


for var in tuple(globals().values()):
    if var is not Command:
        load_type = 0
        try:
            if issubclass(var, Command):
                load_type = 1
        except TypeError:
            pass
        if load_type:
            obj = var()


def process_message(msg):
    if Downloaders:
        try:
            i = int(msg)
        except:
            pass
        else:
            return Downloaders[0].func(i)
    if msg.startswith("~"):
        comm = msg[1:]
        if comm and comm[0] == "?":
            check = comm[0]
            i = 1
        else:
            i = len(comm)
            for end in " ?-+":
                if end in comm:
                    i2 = comm.index(end)
                    if i2 < i:
                        i = i2
            check = unicode_prune(comm[:i]).lower().replace("*", "").replace("_", "").replace("||", "")
        if check in commands:
            for command in commands[check]:
                alias = command.__name__
                for a in command.alias:
                    if a.lower() == check:
                        alias = a
                alias = alias.lower()
                argv = comm[i:]
                flags = {}
                if argv:
                    if not hasattr(command, "no_parse"):
                        argv = unicode_prune(argv)
                    argv = argv.strip()
                    if hasattr(command, "flags"):
                        flaglist = command.flags
                        for q in "?-+":
                            if q in argv:
                                for char in flaglist:
                                    flag = q + char
                                    for r in (flag, flag.upper()):
                                        while len(argv) >= 4 and r in argv:
                                            found = False
                                            i = argv.index(r)
                                            if i == 0 or argv[i - 1] == " " or argv[i - 2] == q:
                                                try:
                                                    if argv[i + 2] == " " or argv[i + 2] == q:
                                                        argv = argv[:i] + argv[i + 2:]
                                                        add_dict(flags, {char: 1})
                                                        found = True
                                                except (IndexError, KeyError):
                                                    pass
                                            if not found:
                                                break
                            if q in argv:
                                for char in flaglist:
                                    flag = q + char
                                    for r in (flag, flag.upper()):
                                        while len(argv) >= 2 and r in argv:
                                            found = False
                                            for check in (r + " ", " " + r):
                                                if check in argv:
                                                    argv = argv.replace(check, "")
                                                    add_dict(flags, {char: 1})
                                                    found = True
                                            if argv == r:
                                                argv = ""
                                                add_dict(flags, {char: 1})
                                                found = True
                                            if not found:
                                                break
                if argv:
                    argv = argv.strip()
                if not argv:
                    args = []
                else:
                    argv = argv.replace("\n", " ").replace("\r", "").replace("\t", " ")
                    try:
                        args = shlex.split(argv)
                    except ValueError:
                        args = argv.split(" ")
                return command(args=args, argv=argv, name=alias, flags=flags, callback=process_message)
    else:
        try:
            return eval(msg, globals())
        except SyntaxError:
            return exec(msg, globals())


def update_loop():
    while not DONE:
        try:
            ytdl.update_dl()
            for item in tuple(ytdl.cache.values()):
                item.update()
            auds.update()
            if auds.searching >= 1:
                continue
            auds.searching += 1
            searched = 0
            q = auds.queue
            for i in q:
                if searched >= 32:
                    break
                if "research" in i:
                    try:
                        ytdl.extract_single(i)
                        try:
                            i.pop("research")
                        except KeyError:
                            pass
                        searched += 1
                    except (SystemExit, KeyboardInterrupt):
                        break
                    except:
                        try:
                            i.pop("research")
                        except KeyError:
                            pass
                        break
                if random.random() > 0.99:
                    time.sleep(0.4)
            time.sleep(4)
            auds.searching = max(auds.searching - 1, 0)
        except (RuntimeError, SystemExit, KeyboardInterrupt):
            break
        except:
            print(traceback.format_exc(), end="")


def concur(fut):
    try:
        print(fut.result())
    except (SystemExit, KeyboardInterrupt):
        DONE = True
        raise
    except:
        print(traceback.format_exc(), end="")
    auds.update()


if __name__ == "__main__":
    DONE = False
    if "cache" not in os.listdir():
        os.mkdir("cache")
    cache = os.listdir("cache")
    if not cache:
        if os.path.exists("loaded.json"):
            os.remove("loaded.json")
    else:
        for f in cache:
            fn = "cache/" + f
            if f in LOADED:
                ytdl.cache[f] = AudioFile(f, fn)
            else:
                os.remove(fn)
    create_future_ex(update_loop)
    print("loaded.")
    while not DONE:
        try:
            i = input(C.reset)
            if not i:
                print(end="")
                continue
            elif i.lower() in ("quit", "quit()"):
                raise KeyboardInterrupt
            create_future_ex(concur, create_future_ex(process_message, i))
        except (SystemExit, KeyboardInterrupt):
            DONE = True
            break
        except:
            print(traceback.format_exc())
    auds.dead = True
    create_future_ex(auds.stop)
    time.sleep(1)
    athreads.shutdown(wait=False)
    raise SystemExit