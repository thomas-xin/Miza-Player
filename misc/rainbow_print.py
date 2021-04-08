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

    @property
    def __dict__(self):
        return self

    ___repr__ = lambda self: super().__repr__()
    to_dict = lambda self: dict(**self)
    to_list = lambda self: list(super().values())


def as_str(s):
    if type(s) in (bytes, bytearray, memoryview):
        return bytes(s).decode("utf-8", "replace")
    return str(s)


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
os.system("color")


class RainbowPrint:

    rainbow = (C.red, C.yellow, C.green, C.cyan, C.blue, C.magenta)
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
                spl[i] = next(self.it) + line + C.reset
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