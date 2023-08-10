import os, sys, subprocess, traceback

print("Loading and checking modules...")

def parse_path(path):
	return set(i.rstrip("/\\") for i in path.strip().split(None, 3)[-1].decode("utf-8", "replace").split(os.pathsep) if i)

if "\\AppData\\" in sys.executable:
	p = sys.executable.rsplit("\\", 1)[0]
	paths = [p]
	paths.append(p + "\\Scripts")
	p = p.replace("\\Local\\Programs\\", "\\Roaming\\", 1)
	paths.append(p + "\\Scripts")
	paths.append(p + "\\site-packages")
	# GLOBAL = set(i.rstrip("/\\") for i in os.getenv("PATH", "").split(os.pathsep) if i)
	GLOBAL = parse_path(subprocess.check_output((
		"reg",
		"query",
		r"HKLM\System\CurrentControlSet\Control\Session Manager\Environment",
		"/v",
		"Path",
	)))
	USER = parse_path(subprocess.check_output((
		"reg",
		"query",
		r"HKCU\Environment",
		"/v",
		"Path",
	)))
	intn = USER.intersection(GLOBAL)
	if intn:
		print(f"Optimising PATH...")
		USER.difference_update(GLOBAL)
		s = os.pathsep.join(USER)
		subprocess.run(["setx", "path", s])
		s = os.pathsep.join(USER.union(GLOBAL))
		os.environ["PATH"] = s
	unio = USER.union(GLOBAL)
	for p in paths:
		if not os.path.exists(p):
			continue
		if p not in unio:
			print(f"Adding {p} to PATH...")
			USER.add(p)
			s = os.pathsep.join(USER)
			subprocess.run(["setx", "path", s])
			s = os.pathsep.join(USER.union(GLOBAL))
			os.environ["PATH"] = s

with open("requirements.txt", "rb") as f:
	modlist = f.read().decode("utf-8", "replace").replace("\r", "\n").split("\n")


if os.name == "nt":
	modlist.append("pipwin>=0.5.1")

try:
	import pip
except ModuleNotFoundError:
	subprocess.run([sys.executable, "-m", "ensurepip"])
try:
	import pkg_resources, struct
except ModuleNotFoundError:
	subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "setuptools"])
	import pkg_resources, struct

if sys.version_info.major == 3 and sys.version_info.minor >= 10:
	import collections
	try:
		setattr(collections, "MutableMapping", collections.abc.MutableMapping)
	except AttributeError:
		collections.MutableMapping = collections.abc.Mapping
	collections.Mapping = collections.abc.Mapping

x = sys.version_info[1]
psize = None

modified = False
installing = []
install = lambda m: installing.append(subprocess.Popen([sys.executable, "-m", "pip", "install", "--upgrade", "--user", m]))

def try_int(i):
    if type(i) is str and not i.isnumeric():
        return i
    try:
        return int(i)
    except:
        return i

# Parse requirements.txt
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
				try:
					s = repr([try_int(i) for i in v.split(".")]) + op + repr([try_int(i) for i in version.split(".")])
					assert eval(s, {}, {})
				except TypeError:
					s = repr(v.split(".")) + op + repr(version.split("."))
					assert eval(s, {}, {})
		except:
			# Modules may require an older version, replace current version if necessary
			traceback.print_exc()
			inst = name
			if op in ("==", "<="):
				inst += "==" + version
			install(inst)

# Run pip on any modules that need installing
if installing:
	modified = True
	print("Installing missing or outdated modules, please wait...")
	subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "pip"])
	while installing:
		proc = installing.pop(0)
		if proc.wait():
			if getattr(proc, "attempts", 0) >= 3:
				continue
			proc = subprocess.Popen(proc.args)
			proc.attempts = getattr(proc, "attempts", 0) + 1
			installing.append(proc)

try:
	assert pkg_resources.get_distribution("encodec").version >= "0.1.2a3"
except (pkg_resources.DistributionNotFound, AssertionError):
	subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/encodec", "--user"])
if not os.path.exists("cache/~sample.wav"):
	subprocess.Popen([sys.executable, "-m", "encodec", "-r", "misc/sample.ecdc", "cache/~sample.wav"])

if modified:
	subprocess.Popen([sys.executable] + sys.argv)
	raise SystemExit