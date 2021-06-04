import sys, subprocess, traceback

print("Loading and checking modules...")

modlist = """
bs4>=0.0.1
easygui>=0.98.2
easygui_qt>=0.9.3
numpy>=1.20.3
psutil>=5.8.0
pygame>=2.0.1
pyperclip>=1.8.2
PyQt5>=5.15.4
requests>=2.25.1
samplerate>=0.1.0
scipy>=1.6.3
youtube-dl>=2021.4.26
""".split("\n")

import pkg_resources, struct
x = sys.version_info[1]
if x >= 9:
    modlist[0] = "pillow>=8.2.0"
psize = None

installing = []
install = lambda m: installing.append(subprocess.Popen(["py", f"-3.{x}", "-m", "pip", "install", "--upgrade", m, "--user"]))

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
    subprocess.run(["py", f"-3.{x}", "-m", "pip", "install", "--upgrade", "pip", "--user"])
    for i in installing:
        i.wait()
if x < 9:
    try:
        pkg_resources.get_distribution("pillow")
    except pkg_resources.DistributionNotFound:
        pass
    else:
        subprocess.run(["py", f"-3.{x}", "-m", "pip", "uninstall", "pillow", "-y"])
    try:
        pkg_resources.get_distribution("pillow-simd")
    except pkg_resources.DistributionNotFound:
        psize = struct.calcsize("P")
        if psize == 8:
            win = "win_amd64"
        else:
            win = "win32"
        subprocess.run(["py", f"-3.{x}", "-m", "pip", "install", f"https://download.lfd.uci.edu/pythonlibs/q4trcu4l/Pillow_SIMD-7.0.0.post3+avx2-cp3{x}-cp3{x}-{win}.whl", "--user"])
try:
    if str(pkg_resources.get_distribution("pyaudio")) < "PyAudio 0.2.11":
        raise ValueError
except (pkg_resources.DistributionNotFound, ValueError):
    if not psize:
        psize = struct.calcsize("P")
        if psize == 8:
            win = "win_amd64"
        else:
            win = "win32"
    subprocess.run(["py", f"-3.{x}", "-m", "pip", "install", f"https://download.lfd.uci.edu/pythonlibs/q4trcu4l/PyAudio-0.2.11-cp3{x}-cp3{x}-{win}.whl", "--user"])
print("Installer terminated.")