import os, sys, subprocess, traceback

print("Loading and checking modules...")

if "\\AppData\\" in sys.executable:
    p = sys.executable.rsplit("\\", 1)[0]
    paths = [p]
    paths.append(p + "\\Scripts")
    p = p.replace("\\Local\\Programs\\", "\\Roaming\\", 1)
    paths.append(p + "\\Scripts")
    paths.append(p + "\\site-packages")
    for p in paths:
        if not os.path.exists(p):
            continue
        PATH = set(i.rstrip("/\\") for i in os.getenv("PATH", "").split(os.pathsep))
        if p not in PATH:
            print(f"Adding {p} to PATH...")
            PATH.add(p)
            s = os.pathsep.join(PATH) + os.pathsep
            subprocess.run(["setx", "path", s])
            os.environ["PATH"] = s

modlist = """
bs4>=0.0.1
easygui>=0.98.2
easygui_qt>=0.9.3
numpy>=1.21.2
orjson>=3.6.3
pillow>=8.3.2
psutil>=5.8.0
pygame>=2.0.1
pyperclip>=1.8.2
python-magic-bin>=0.4.14
PyQt5>=5.15.4
requests>=2.26.0
samplerate>=0.1.0
scipy>=1.7.1
soundcard>=0.4.1
youtube-dl>=2021.6.6
""".split("\n")

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
if os.name == "nt":
    try:
        if pkg_resources.get_distribution("pyopengl").version < "3.1.5":
            raise ValueError
    except:
        subprocess.run([sys.executable, "-m", "pipwin", "install", "pyopengl"])
    try:
        if pkg_resources.get_distribution("pyopengl-accelerate").version < "3.1.5":
            raise ValueError
    except:
        subprocess.run([sys.executable, "-m", "pipwin", "install", "pyopengl-accelerate"])
    try:
        if pkg_resources.get_distribution("glfw").version < "2.3.0":
            raise ValueError
    except:
        subprocess.run([sys.executable, "-m", "pipwin", "install", "glfw"])