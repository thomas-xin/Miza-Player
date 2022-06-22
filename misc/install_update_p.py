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
else:
	modlist.extend((
		"pyopengl>=3.1.5",
		"pyopengl-accelerate>=3.1.5",
		"glfw>=2.3.0",
	))

try:
	import pkg_resources, struct
except ModuleNotFoundError:
	subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "setuptools"])
	import pkg_resources, struct

x = sys.version_info[1]
psize = None

installing = []
install = lambda m: installing.append(subprocess.Popen([sys.executable, "-m", "pip", "install", "--upgrade", "--user", m]))

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
				assert eval(repr(v) + op + repr(version), {}, {})
		except:
			# Modules may require an older version, replace current version if necessary
			traceback.print_exc()
			inst = name
			if op in ("==", "<="):
				inst += "==" + version
			install(inst)

# Run pip on any modules that need installing
if installing:
	print("Installing missing or outdated modules, please wait...")
	subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "pip"])
	for i in installing:
		i.wait()

try:
	v = pkg_resources.get_distribution("yt_dlp").version
	assert v >= "2022.3.8.1"
except:
	print_exc()
	subprocess.run([python, "-m", "pip", "install", "git+https://github.com/yt-dlp/yt-dlp.git", "--upgrade", "--user"])

# if os.name == "nt":
	# for k, v in (
		# ("pyopengl", "3.1.5"),
		# ("pyopengl-accelerate", "3.1.5"),
		# ("glfw", "2.3.0"),
	# ):
		# try:
			# if pkg_resources.get_distribution(k).version < v:
				# raise ValueError
		# except:
			# subprocess.run([sys.executable, "-m", "pipwin", "install", k])
			# try:
				# if pkg_resources.get_distribution(k).version < v:
					# raise ValueError
			# except:
				# subprocess.run([sys.executable, "-m", "pipwin", "refresh"])
				# subprocess.run([sys.executable, "-m", "pip", "-y", "uninstall", k])
				# subprocess.run([sys.executable, "-m", "pipwin", "install", k])