import sys
sys.path.append("misc")
import common
globals().update(common.__dict__)


atypes = "wav mp3 ogg opus flac aac m4a webm wma f4a weba mka mp2 oga riff".split()
ftypes = [[f"*.{f}" for f in atypes + "mp4 mov qt mkv avi f4v flv wmv raw".split()]]
ftypes[0].append("All supported audio files")

itypes = "png bmp jpg jpeg webp gif apng afif".split()
iftypes = [[f"*.{f}" for f in itypes + "mp4 mov qt mkv avi f4v flv wmv raw".split()]]
iftypes[0].append("All supported image files")

with open("assoc.bat", "w", encoding="utf-8") as f:
	f.write(f"cd {os.path.abspath('')}\nstart /MIN py -3.{pyv} {sys.argv[0]} %*")
if not os.path.exists("cache"):
	os.mkdir("cache")
if not os.path.exists("playlists"):
	os.mkdir("playlists")


def create_pattern():
	x = 0
	while x in project.patterns:
		x += 1
	pattern = project.settings.union(
		measures={},
		ends={},
		name=f"Pattern {x}",
		colour=tuple(round(i * 255) for i in colorsys.hsv_to_rgb(x / 12 % 1, 1, 1)),
		id=x,
	)
	project.patterns[x] = pattern
	return pattern
project = cdict(
	settings=cdict(
		timesig=(4, 4),
		keysig=0,
		tempo=144,
	),
	instruments={},
	instrument_layout=alist(),
	patterns={},
	playlist=[],
)
project_name = "Untitled"
instruments = project.instruments
patterns = project.patterns

class PlayerWaiter(deque, contextlib.AbstractContextManager):

	def __enter__(self):
		self.append(None)

	def __exit__(self, *args):
		self.pop()

player = cdict(
	paused=False,
	index=0,
	pos=0,
	end=inf,
	amp=0,
	stats=cdict(
		peak=0,
		amplitude=0,
		velocity=0,
	),
	waiting=PlayerWaiter(),
	editor=cdict(
		selection=cdict(
			point=None,
			points=alist(),
			notes=set(),
			orig=(None, None),
		),
		note=cdict(
			instrument=None,
			length=1,
			volume=0.25,
			pan=0,
			effects=[],
		),
		fade=1,
		pattern=0,
		scrolling=False,
		scroll_x=0,
		scroll_y=0,
		targ_x=0,
		targ_y=0,
		zoom_x=1,
		zoom_y=1,
		played=set(),
		undo=[],
		playing_notes=set(),
		held_notes=set(),
		held_update=None,
	),
)
def change_mode(mode):
	player.editor.fade = options.editor.mode == mode
	options.editor.mode = mode
player.editor.change_mode = change_mode
def cancel_selection():
	selection = player.editor.selection
	selection.point = 0
	selection.points.clear()
	selection.notes.clear()
	selection.pop("all", None)
	selection.pop("cpy", None)
player.editor.selection.cancel = cancel_selection
sidebar = cdict(
	queue=alist(),
	instruments=alist(),
	buttons=alist(),
	particles=alist(),
	ripples=alist(),
	scroll=cdict(),
	menu=None
)
toolbar = cdict(
	pause=cdict(
		speed=0,
		angle=0,
		maxspeed=5,
	),
	progress=cdict(
		vis=0,
		angle=0,
		spread=0,
		alpha=0,
		num=0,
		particles=alist(),
		seeking=False,
	),
	downloading=cdict(
		target=0,
	),
	buttons=alist(),
	ripples=alist(),
	editor=0,
)
queue = sidebar.queue
progress = toolbar.progress
downloading = toolbar.downloading
modified = set()


button_sources = (
	"gears",
	"notes",
	"folder",
	"hyperlink",
	"playlist",
	"history",
	"save",
	"plus",
	"edit",
	"waves",
	"repeat",
	"shuffle",
	"back",
	"flip",
	"scramble",
	"unique",
	"record",
	"microphone",
)
button_images = cdict((i, submit(load_surface, "misc/" + i + ".png")) for i in button_sources)

def load_pencil():
	globals()["pencilw"] = load_surface("misc/pencil.png")
	globals()["pencilb"] = pencilw.convert_alpha()
	pencilb.fill((0, 0, 0), special_flags=BLEND_RGB_MULT)
addp = submit(load_pencil)

lyrics_entry = None

def settings_reset():
	options.audio.update(audio_default)
	s = io.StringIO()
	for k, v in options.audio.items():
		s.write(f"~setting #{k} {v}\n")
	s.write(f"~replay\n")
	s.seek(0)
	mixer.submit(s.read())

def transfer_instrument(*instruments):
	try:
		ts = pc()
		if laststart:
			diff = ts - min(laststart)
			if diff < 0.5:
				delay = 0.5 - diff
				laststart.add(ts)
				time.sleep(delay)
				if ts < max(laststart):
					return
			laststart.clear()
		laststart.add(ts)
		for instrument in instruments:
			if instrument.get("wave") is None:
				instrument.wave = synth_gen(**instrument.synth)
			if instrument.get("encoded") is None:
				instrument.encoded = base64.b85encode(bytes2zip(instrument.wave.tobytes()))
			b = instrument.encoded
			opt = json.dumps(instrument.get("opt", default_instrument_opt), separators=(",", ":"))
			a = f"~wave {instrument.id} {opt} ".encode("ascii")
			mixer.submit(a + b)
		if instruments:
			return instrument
	except:
		print_exc()

def select_instrument(i):
	if type(i) is not int:
		i = i.id
	mixer.submit(f"~select {i}")
	return project.instruments[i]

def add_instrument(first=False):
	x = 0
	while x in project.instruments:
		x += 1
	project.instruments[x] = instrument = cdict(
		id=x,
		name=f"Instrument {x}",
		colour=tuple(round(i * 255) for i in colorsys.hsv_to_rgb(x / 12 % 1, 1, 1)),
		synth=cdict(synth_default),
	)
	synth = instrument.synth
	if first:
		synth.shape = 1.5
	submit(transfer_instrument, instrument)
	player.editor.note.instrument = x
	project.instrument_layout.append(x)
	sidebar.instruments.append(cdict())

def setup_buttons():
	try:
		gears = button_images.gears.result()
		notes = button_images.notes.result()
		def settings_toggle():
			sidebar.abspos = not bool(sidebar.abspos)
		def settings_menu():
			sidebar.menu = cdict(
				buttons=(
					("Settings", settings_toggle),
					("Reset", settings_reset),
				),
			)
		sidebar.buttons.append(cdict(
			name="Settings",
			sprite=(gears, notes),
			click=(settings_toggle, settings_menu),
		))
		reset_menu(full=False)
		folder = button_images.folder.result()
		def enqueue_local():
			default = None
			if options.get("path"):
				default = options.path.rstrip("/") + "/"
			files = easygui.fileopenbox(
				"Open an audio or video file here!",
				"Miza Player",
				default=default,
				filetypes=ftypes,
				multiple=True,
			)
			if files:
				submit(_enqueue_local, *files)
		def enqueue_menu():
			def enqueue_folder():
				default = None
				if options.get("path"):
					default = options.path.rstrip("/") + "/"
				files = easygui.diropenbox(
					"Open an audio or video file here!",
					"Miza Player",
					default=default,
				)
				if files:
					submit(_enqueue_local, *(files + "/" + f for f in os.listdir(files)), probe=False)
			sidebar.menu = cdict(
				buttons=(
					("Open file", enqueue_local),
					("Open folder", enqueue_folder),
				),
			)
		def load_project_file():
			default = None
			if options.get("path2"):
				default = options.path.rstrip("/") + "/"
			file = easygui.fileopenbox(
				"Open a music project file here!",
				"Miza Player",
				default=default,
				filetypes=(".mpp",),
				multiple=False,
			)
			if file:
				submit(load_project, file)
		sidebar.buttons.append(cdict(
			name="Open",
			sprite=folder,
			click=(enqueue_local, enqueue_menu),
			click2=load_project_file,
		))
		reset_menu(full=False)
		hyperlink = button_images.hyperlink.result()
		plus = button_images.plus.result()
		def enqueue_search(q=None):
			query = easygui.get_string(
				"Search for one or more songs online!",
				"Miza Player",
				"",
			)
			if query:
				if not is_url(query):
					if q == "sc":
						query = "scsearch:" + query.replace(":", "-")
					elif q:
						query = q + "search:" + query
				submit(_enqueue_search, query)
		def select_search():
			sidebar.menu = cdict(
				buttons=(
					("YouTube", enqueue_search),
					("SoundCloud", enqueue_search, "sc"),
					("Spotify", enqueue_search, "sp"),
					("BandCamp", enqueue_search, "bc"),
				),
			)
		sidebar.buttons.append(cdict(
			name="Search",
			name2="New instrument",
			sprite=hyperlink,
			sprite2=plus,
			click=(enqueue_search, select_search),
			click2=add_instrument,
		))
		reset_menu(full=False)
		playlist = button_images.playlist.result()
		waves = button_images.waves.result()
		def get_playlist():
			items = deque()
			for item in (item for item in os.listdir("playlists") if item.endswith(".json") or item.endswith(".zip")):
				u = unquote(item.rsplit(".", 1)[0])
				fn = "playlists/" + quote(u)[:245] + "." + item.rsplit(".", 1)[-1]
				fn2 = "playlists/" + item
				if fn != fn2 or not os.path.exists(fn):
					os.rename(fn2, fn)
				items.append(u)
			if not items:
				return easygui.show_message("Right click this button to create, edit, or remove a playlist!", "Playlist folder is empty.")
			choice = easygui.get_choice("Select a locally saved playlist here!", "Miza Player", items)
			if choice:
				sidebar.loading = True
				start = len(queue)
				fn = "playlists/" + quote(choice)[:245] + ".zip"
				if not os.path.exists(fn):
					fn = fn[:-4] + ".json"
				if os.path.exists(fn) and os.path.getsize(fn):
					with open(fn, "rb") as f:
						if zipfile.is_zipfile(f):
							f.seek(0)
							data = orjson.loads(zip2bytes(f.read()))
						else:
							f.seek(0)
							data = json.load(f)
					q = data.get("queue", ())
					options.history.appendleft((choice, tuple(e["url"] for e in q)))
					options.history = options.history.uniq(sort=False)[:64]
					entries = [ensure_duration(cdict(**e, pos=start)) for e in q]
					queue.extend(entries)
				if control.shuffle and len(queue) > 1:
					queue[1:].shuffle()
				sidebar.loading = False
		def playlist_menu():
			def create_playlist():
				text = (easygui.textbox(
					"Enter a list of URLs or file paths to include in the playlist!",
					"Miza Player",
				) or "").strip()
				if text:
					urls = text.splitlines()
					entries = deque()
					for url in urls:
						if url:
							name = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
							entries.append(dict(name=name, url=url))
					if entries:
						entries = list(entries)
						ytdl = downloader.result()
						url = entries[0]["url"]
						if is_url(url):
							name = None
							if url in ytdl.searched:
								resp = ytdl.searched[url].data
								if len(resp) == 1:
									name = resp[0].get("name")
							if not name:
								resp = ytdl.downloader.extract_info(url, download=False, process=False)
								name = resp.get("title") or entries[0].get("name")
							if name:
								entries[0]["name"] = name
						else:
							name = entries[0].name
						if len(entries) > 1:
							name += f" +{len(entries) - 1}"
						text = (easygui.get_string(
							"Enter a name for your new playlist!",
							"Miza Player",
							name,
						) or "").strip()
						if text:
							data = dict(queue=entries, stats={})
							fn = "playlists/" + quote(text)[:245] + ".json"
							if len(entries) > 1024:
								fn = fn[:-5] + ".zip"
								b = bytes2zip(orjson.dumps(data))
								with open(fn, "wb") as f:
									f.write(b)
							else:
								with open(fn, "w", encoding="utf-8") as f:
									json.dump(data, f, separators=(",", ":"))
							easygui.show_message(
								f"Playlist {repr(text)} with {len(entries)} item{'s' if len(entries) != 1 else ''} has been added!",
								"Success!",
							)
			def edit_playlist():
				items = [unquote(item[:-5]) for item in os.listdir("playlists") if item.endswith(".json") or item.endswith(".zip")]
				if not items:
					return easygui.show_message("Right click this button to create, edit, or remove a playlist!", "Playlist folder is empty.")
				choice = easygui.get_choice("Select a playlist to edit", "Miza Player", items)
				if choice:
					fn = "playlists/" + quote(choice)[:245] + ".zip"
					if not os.path.exists(fn):
						fn = fn[:-4] + ".json"
					if os.path.exists(fn) and os.path.getsize(fn):
						with open(fn, "rb") as f:
							if zipfile.is_zipfile(f):
								f.seek(0)
								data = orjson.loads(zip2bytes(f.read()))
							else:
								f.seek(0)
								data = json.load(f)
						s = "\n".join(e["url"] for e in data.get("queue", ()) if e.get("url"))
					else:
						print(fn)
						s = ""
					text = easygui.textbox(
						"Enter a list of URLs or file paths to include in the playlist!",
						"Miza Player",
						s,
					)
					if text is not None:
						if not text:
							os.remove(fn)
							easygui.show_message(
								f"Playlist {repr(choice)} has been removed!",
								"Success!",
							)
						else:
							urls = text.splitlines()
							entries = deque()
							for url in urls:
								if url:
									name = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
									entries.append(dict(name=name, url=url))
							if entries:
								entries = list(entries)
								data = dict(queue=entries, stats={})
								fn = "playlists/" + quote(text)[:245] + ".json"
								if len(entries) > 1024:
									fn = fn[:-5] + ".zip"
									b = bytes2zip(orjson.dumps(data))
									with open(fn, "wb") as f:
										f.write(b)
								else:
									with open(fn, "w", encoding="utf-8") as f:
										json.dump(data, f, separators=(",", ":"))
								easygui.show_message(
									f"Playlist {repr(choice)} has been updated!",
									"Success!",
								)
			def delete_playlist():
				items = [unquote(item[:-5]) for item in os.listdir("playlists") if item.endswith(".json")]
				if not items:
					return easygui.show_message("Right click this button to create, edit, or remove a playlist!", "Playlist folder is empty.")
				choice = easygui.get_choice("Select a playlist to delete", "Miza Player", items)
				if choice:
					fn = "playlists/" + quote(choice)[:245] + ".json"
					os.remove(fn)
					easygui.show_message(
						f"Playlist {repr(choice)} has been removed!",
						"Success!",
					)
			sidebar.menu = cdict(
				buttons=(
					("Open playlist", get_playlist),
					("Create playlist", create_playlist),
					("Edit playlist", edit_playlist),
					("Delete playlist", delete_playlist),
				),
			)
		def waves_1():
			raise BaseException
		sidebar.buttons.append(cdict(
			name="Playlist",
			name2="Audio Clip",
			sprite=playlist,
			sprite2=waves,
			click=(get_playlist, playlist_menu),
			click2=waves_1,
		))
		reset_menu(full=False)
		history = button_images.history.result()
		def player_history():
			f = f"%0{len(str(len(options.history)))}d"
			choices = [f % i + ": " + e[0] for i, e in enumerate(options.history)]
			if not choices:
				return easygui.show_message("Play some music to fill up this menu!", "Player history is empty.")
			selected = easygui.get_choice(
				"Player History",
				"Miza Player",
				choices,
			)
			if selected:
				entry = options.history.pop(int(selected.split(":", 1)[0]))
				options.history.appendleft(entry)
				enqueue_auto(*entry[1])
		def history_menu():
			def clear_history():
				options.history.clear()
				return easygui.show_message("History successfully cleared!", "Miza Player")
			def clear_cache():
				futs = deque()
				for fn in os.listdir("cache"):
					futs.append(submit(os.remove, "cache/" + fn))
				for fut in futs:
					try:
						fut.result()
					except (FileNotFoundError, PermissionError):
						pass
				return easygui.show_message("Cache successfully cleared!", "Miza Player")
			sidebar.menu = cdict(
				buttons=(
					("View history", player_history),
					("Clear history", clear_history),
					("Clear cache", clear_cache),
				),
			)
		def project_history():
			raise BaseException
		sidebar.buttons.append(cdict(
			name="History",
			sprite=history,
			click=(player_history, history_menu),
			click2=project_history,
		))
		reset_menu(full=False)
		edit = button_images.edit.result()
		def edit_1():
			pause_toggle(True)
			toolbar.editor ^= 1
			sidebar.scrolling = False
			sidebar.scroll.pos = 0
			sidebar.scroll.target = 0
			player.editor.held_notes.clear()
			player.editor.held_update = 0
			if toolbar.editor:
				mixer.submit(f"~setting spectrogram -1")
				pygame.display.set_caption(f"Miza Player ~ {project_name}")
			else:
				mixer.submit(f"~setting spectrogram {options.spectrogram}")
				pygame.display.set_caption("Miza Player")
		toolbar.buttons.append(cdict(
			name="Editor",
			image=edit,
			click=edit_1,
		))
		reset_menu(full=False)
		repeat = button_images.repeat.result()
		def repeat_1():
			control.loop = (control.loop + 1) % 3
		def repeat_2():
			control.loop = (control.loop - 1) % 3
		toolbar.buttons.append(cdict(
			name="Repeat",
			image=repeat,
			click=(repeat_1, repeat_2),
		))
		reset_menu(full=False)
		shuffle = button_images.shuffle.result()
		def shuffle_1():
			control.shuffle = (control.shuffle + 1) % 3
			if control.shuffle in (0, 2):
				mixer.submit(f"~setting shuffle {control.shuffle}")
			if control.shuffle == 2 and player.get("needs_shuffle") and player.end >= 960:
				seek_abs(player.pos)
		def shuffle_2():
			control.shuffle = (control.shuffle - 1) % 3
			if control.shuffle in (0, 2):
				mixer.submit(f"~setting shuffle {control.shuffle}")
			if control.shuffle == 2 and player.get("needs_shuffle"):
				seek_abs(player.pos)
		toolbar.buttons.append(cdict(
			name="Shuffle",
			image=shuffle,
			click=(shuffle_1, shuffle_2),
		))
		reset_menu(full=False)
		back = button_images.back.result()
		def rleft():
			mixer.clear()
			queue.rotate(1)
			player.fut = None
		toolbar.buttons.append(cdict(
			name="Previous",
			image=back,
			click=rleft,
		))
		front = pygame.transform.flip(back, True, False)
		def rright():
			mixer.clear()
			queue.rotate(-1)
			player.fut = None
		toolbar.buttons.append(cdict(
			name="Next",
			image=front,
			click=rright,
		))
		reset_menu(full=False)
		flip = button_images.flip.result()
		def flip_1():
			mixer.clear()
			queue.reverse()
			player.fut = None
		toolbar.buttons.append(cdict(
			name="Flip",
			image=flip,
			click=flip_1,
		))
		reset_menu(full=False)
		scramble = button_images.scramble.result()
		def scramble_1():
			mixer.clear()
			queue.shuffle()
			player.fut = None
		toolbar.buttons.append(cdict(
			name="Scramble",
			image=scramble,
			click=scramble_1,
		))
		reset_menu(full=False)
		unique = button_images.unique.result()
		def unique_1():
			pops = deque()
			found = set()
			for i, e in enumerate(queue):
				if e.url in found:
					pops.append(i)
				else:
					found.add(e.url)
			queue.pops(pops)
		toolbar.buttons.append(cdict(
			name="Remove Duplicates",
			image=unique,
			click=unique_1,
		))
		reset_menu(full=False)
		microphone = button_images.microphone.result()
		def enqueue_device():
			afut.result().terminate()
			globals()["afut"] = submit(pyaudio.PyAudio)
			globals()["pya"] = afut.result()
			count = pya.get_device_count()
			apis = {}
			ddict = mdict()
			for i in range(count):
				d = cdict(pya.get_device_info_by_index(i))
				a = d.get("hostApi", -1)
				if a not in apis:
					apis[a] = cdict(pya.get_host_api_info_by_index(a))
				if d.maxInputChannels > 0 and apis[a].name in ("MME", "Windows DirectSound", "Windows WASAPI"):
					try:
						if not pya.is_format_supported(
							48000,
							i,
							2,
							pyaudio.paInt16,
						):
							continue
					except:
						continue
					d.id = i
					ddict.add(a, d)
			devices = ()
			for dlist in ddict.values():
				if len(dlist) > len(devices):
					devices = dlist
			f = f"%0{len(str(len(devices)))}d"
			selected = easygui.get_choice(
				"Transfer audio from a sound input device!",
				"Miza Player",
				sorted(f % d.id + ": " + d.name for d in devices),
			)
			if selected:
				submit(_enqueue_local, "<" + selected.split(":", 1)[0].lstrip("0") + ">")
		def microphone_menu():
			sidebar.menu = cdict(
				buttons=(
					("Add input", enqueue_device),
				),
			)
		sidebar.buttons.append(cdict(
			name="Audio input",
			sprite=microphone,
			click=(enqueue_device, microphone_menu),
		))
		reset_menu(full=False)
		record = button_images.record.result()
		def output_device():
			devices = sc.all_speakers()
			f = f"%0{len(str(len(devices)))}d"
			selected = easygui.get_choice(
				"Change the output audio device!",
				"Miza Player",
				[d.name for d in devices],
			)
			if selected:
				globals()["OUTPUT_DEVICE"] = selected
				if DEVICE.name != OUTPUT_DEVICE:
					mixer.submit(f"~output {OUTPUT_DEVICE}")
					globals()["DEVICE"] = get_device(OUTPUT_DEVICE)
		def end_recording():
			mixer.submit("~record")
			try:
				if not os.path.getsize(sidebar.recording):
					raise FileNotFoundError
			except FileNotFoundError:
				pass
			else:
				fmt = ".ogg" if not sidebar.get("filming") else ".mp4"
				fn = sidebar.recording.split("/", 1)[-1][1:].split(".", 1)[0] + fmt
				fn = easygui.filesavebox(
					"Save As",
					"Miza Player",
					fn.translate(safe_filenames),
					filetypes=ftypes,
				)
				if fn:
					args = f"{ffmpeg} -hide_banner -y -v error"
					if sidebar.get("filming"):
						args += f" -i {sidebar.filming}"
						globals()["video-render"].result()
					args += f" -f f32le -ar 48k -ac 2 -i {sidebar.recording} -b:a 256k"
					if sidebar.get("filming"):
						args += " -c:v copy"
					args += f" {fn}"
					print(args)
					os.system(args)
			sidebar.recording = ""
			sidebar.filming = ""
		def record_audio():
			if not sidebar.get("recording"):
				sidebar.recording = f"cache/\x7f{ts_us()}.pcm"
				mixer.submit(f"~record {sidebar.recording}")
			else:
				end_recording()
		def record_video():
			if not sidebar.get("recording"):
				sidebar.recording = f"cache/\x7f{ts_us()}.pcm"
				sidebar.filming = f"cache/\x7f{ts_us()}.mp4"
				globals()["video-render"] = concurrent.futures.Future()
				mixer.submit(f"~record {sidebar.recording} {sidebar.filming}")
			else:
				end_recording()
		def record_menu():
			sidebar.menu = cdict(
				buttons=(
					("Record audio", record_audio),
					("Record video", record_video),
					("Change device", output_device),
				),
			)
		sidebar.buttons.append(cdict(
			name="Audio output",
			sprite=record,
			click=(record_audio, record_menu),
		))
		reset_menu(full=False)
	except:
		print_exc()

def send_status():
	pass

cached_fns = {}
def _enqueue_local(*files, probe=True):
	try:
		if not files:
			return
		sidebar.loading = True
		start = len(queue)
		title = None
		for fn in files:
			entries = None
			if fn[0] == "<" and fn[-1] == ">":
				pya = afut.result()
				dev = pya.get_device_info_by_index(int(fn.strip("<>")))
				entry = cdict(
					url=fn,
					stream=fn,
					name=dev.get("name"),
					duration=inf,
				)
			elif fn.endswith(".zip"):
				with open(fn, "rb") as f:
					b = f.read()
				data = orjson.loads(zip2bytes(b))
				q = data.get("queue", ())
				entries = [ensure_duration(cdict(**e, pos=start)) for e in q]
				queue.extend(entries)
				entry = entries[0]
			elif fn.endswith(".json"):
				with open(fn, "rb") as f:
					data = json.load(f)
				q = data.get("queue", ())
				entries = [ensure_duration(cdict(**e, pos=start)) for e in q]
				queue.extend(entries)
				entry = entries[0]
			else:
				fn = fn.replace("\\", "/")
				if "/" not in fn:
					fn = "/" + fn
				options.path, name = fn.rsplit("/", 1)
				name = name.rsplit(".", 1)[0]
				try:
					try:
						dur, cdc = cached_fns[fn]
					except KeyError:
						if probe:
							dur, cdc = cached_fns[fn] = get_duration_2(fn)
						else:
							dur, cdc = None, None
				except:
					print_exc()
					dur, cdc = (None, "N/A")
				entry = cdict(
					url=fn,
					stream=fn,
					name=name,
					duration=dur,
					cdc=cdc,
					pos=start,
				)
			if not title:
				title = entry.name
				if len(files) > 1:
					title += f" +{len(files) - 1}"
				options.history.appendleft((title, files))
				options.history = options.history.uniq(sort=False)[:64]
			if not entries:
				queue.append(entry)
		if control.shuffle:
			queue[1:].shuffle()
		sidebar.loading = False
	except:
		sidebar.loading = False
		print_exc()

eparticle = dict(colour=(255,) * 3)
def _enqueue_search(query):
	try:
		if not query:
			return
		sidebar.loading = True
		start = len(queue)
		ytdl = downloader.result()
		try:
			entries = ytdl.search(query)
		except:
			print_exc()
			sidebar.particles.append(cdict(eparticle))
		else:
			if entries:
				entry = entries[0]
				name = entry.name
				if len(entries) > 1:
					name += f" +{len(entries) - 1}"
				url = query if is_url(query) and len(entries) > 1 else entry.url
				options.history.appendleft((name, (url,)))
				options.history = options.history.uniq(sort=False)[:64]
				entries = [cdict(**e, pos=start) for e in entries]
				queue.extend(entries)
			else:
				sidebar.particles.append(cdict(eparticle))
		if control.shuffle and len(queue) > 1:
			queue[bool(start):].shuffle()
		sidebar.loading = False
	except:
		sidebar.loading = False
		print_exc()

def enqueue_auto(*queries):
	futs = deque()
	start = len(queue)
	for i, query in enumerate(queries):
		q = query.strip()
		if not q:
			continue
		if is_url(q) or not os.path.exists(q):
			if i < 1:
				futs.append(submit(_enqueue_search, q))
			else:
				for fut in futs:
					fut.result()
				name = q.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
				queue.append(cdict(name=name, url=q, duration=None, pos=start))
		else:
			futs.append(submit(_enqueue_local, q, probe=i < 1))

def load_project(fn, switch=True):
	if switch and not toolbar.editor:
		toolbar.editor = 1
		mixer.submit(f"~setting spectrogram -1")
	player.editor_surf = None
	globals()["mclick"] = globals()["mc2"] = globals()["mc3"] = globals()["mc4"] = (None,) * 5
	try:
		f = open(fn, "rb") if type(fn) is str else io.BytesIO(fn) if type(fn) is bytes else fn
		pdata = None
		try:
			with f:
				b = f.read(7)
				if b != b">~MPP~>":
					raise TypeError("Invalid project file header.")
				if type(fn) is bytes:
					b2 = memoryview(fn)[7:]
				else:
					b2 = f.read()
				with zipfile.ZipFile(io.BytesIO(b2), allowZip64=True, strict_timestamps=False) as z:
					with z.open("<~MPP~<", force_zip64=True) as f:
						pdata = pickle.load(f)
		except BufferError:
			if not pdata:
				raise
		project.update(pdata)
		if not project.instruments:
			add_instrument(True)
		submit(transfer_instrument, *project.instruments.values())
		instruments = [cdict() for _ in range(len(project.instrument_layout))]
		sidebar.instruments.fill(instruments)
		if not player.paused:
			player.broken = True
		player.editor_surf = None
		FRESH_PATTERNS.clear()
		globals()["instruments"] = project.instruments
		globals()["patterns"] = project.patterns
		if type(fn) is str:
			globals()["project_name"] = fn.rsplit(".", 1)[0]
		return project
	except:
		print_exc()

def save_project(fn=None):
	player.editor_surf = None
	globals()["mclick"] = globals()["mc2"] = globals()["mc3"] = globals()["mc4"] = (None,) * 5
	if type(fn) is str and not fn.endswith(".mpp"):
		return render_project(fn)
	try:
		for instrument in project.instruments.values():
			if "synth" in instrument:
				instrument.pop("wave", None)
				instrument.pop("encoded", None)
		pdata = io.BytesIO()
		with zipfile.ZipFile(pdata, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=7, allowZip64=True) as z:
			with z.open("<~MPP~<", "w", force_zip64=True) as f:
				pickle.dump(project, f)
		f = open(fn, "wb") if type(fn) is str else fn or io.BytesIO()
		f.write(b">~MPP~>")
		f.write(pdata.getbuffer())
		if type(fn) is str:
			f.close()
			globals()["project_name"] = fn.rsplit(".", 1)[0]
		else:
			f.seek(0)
		return f
	except:
		print_exc()


if len(sys.argv) > 1:
	if len(sys.argv) == 2 and not is_url(sys.argv[1]) and os.path.exists(sys.argv[1]):
		fi = sys.argv[1]
		with open(fi, "rb") as f:
			b = f.read(7)
		if b == b">~MPP~>":
			submit(load_project, fi)
		else:
			submit(enqueue_auto, fi)
	else:
		submit(enqueue_auto, *sys.argv[1:])


sidebar.abspos = 0
osize = None
ssize = None
def reset_menu(full=True, reset=False):
	global osize, ssize
	if full:
		globals().update(options)
		common.__dict__.update(options)
		if reset:
			DISP.fill(0)
			modified.add(tuple(screensize))
	ssize2 = (screensize[0] - sidebar_width, screensize[1] - toolbar_height)
	if ssize != ssize2 or full and reset:
		ssize = ssize2
		mixer.submit(f"~ssize {' '.join(map(str, ssize))}")
		mixer.new = False
	player.rect = (0,) * 2 + ssize
	sidebar.colour = None
	sidebar.updated = False
	sidebar.rect = (screensize[0] - sidebar_width, 0, sidebar_width, screensize[1] - toolbar_height)
	sidebar.rect2 = sidebar.rect #(screensize[0] - sidebar_width, 0, sidebar_width, screensize[1] - toolbar_height + 4)
	for i, button in enumerate(sidebar.buttons, -1):
		if i < 0:
			button.pos = (screensize[0] - 48, sidebar.rect[1] + 8)
		else:
			button.pos = (sidebar.rect[0] + 8 + 44 * i, sidebar.rect[1] + 8)
		button.rect = button.pos + (40, 40)
	sidebar.resizing = False
	sidebar.resizer = False
	sidebar.scrolling = False
	sidebar.scroll.setdefault("pos", 0)
	sidebar.scroll.setdefault("target", 0)
	toolbar.colour = None
	toolbar.updated = False
	toolbar.rect = (0, screensize[1] - toolbar_height, screensize[0], toolbar_height)
	toolbar.pause.radius = min(64, toolbar_height // 2 - 2)
	toolbar.pause.pos = (toolbar.pause.radius + 2, screensize[1] - toolbar_height + toolbar.pause.radius + 2)
	progress.pos = (round(toolbar.pause.pos[0] + toolbar.pause.radius * 1.5 + 4), screensize[1] - toolbar_height + toolbar.pause.radius * 2 // 3 + 1)
	progress.box = toolbar.pause.radius * 6 // 2 + 8
	progress.length = max(0, screensize[0] - progress.pos[0] - toolbar.pause.radius // 2 - progress.box)
	progress.width = min(16, toolbar.pause.radius // 3)
	progress.rect = (progress.pos[0] - progress.width // 2 - 3, progress.pos[1] - progress.width // 2 - 3, progress.length + 6, progress.width + 6)
	progress.seeking = False
	bsize = min(40, toolbar_height // 3)
	for i, button in enumerate(toolbar.buttons):
		if i:
			button.pos = (toolbar.pause.radius * 2 + 8 + (bsize + 4) * (i - 1), screensize[1] - toolbar_height + toolbar.pause.radius * 2 - bsize - 4)
		else:
			button.pos = (screensize[0] - bsize - 4, screensize[1] - bsize - 4)
		rect = button.pos + (bsize,) * 2
		sprite = button.get("sprite", button.image)
		isize = (bsize - 6,) * 2
		if sprite.get_size() != isize:
			sprite = pygame.transform.smoothscale(button.image, isize)
		button.sprite = sprite
		if 0 < i < 3:
			button.on = button.sprite.convert_alpha()
			button.on.fill((0, 255, 255), special_flags=BLEND_RGB_MULT)
			button.off = button.sprite.convert_alpha()
			button.off.fill((0,) * 3, special_flags=BLEND_RGB_MULT)
		button.rect = rect
	toolbar.resizing = False
	toolbar.resizer = False
	osize2 = (progress.box, toolbar.pause.radius * 4 // 3 - 3)
	if osize != osize2 or full and reset:
		osize = osize2
		mixer.submit(f"~osize {' '.join(map(str, osize))}")
		mixer.new = False
	globals()["last-cond"] = True


submit(setup_buttons)


is_active = lambda: pc() - player.get("last", 0) <= max(player.get("lastframe", 0), 1 / 30) * 4
e_dur = lambda d: float(d) if type(d) is str else (d if d is not None else nan)

def ensure_duration(e):
	e.duration = e.get("duration")
	return e

def copy_entry(e):
	d = cdict(url=e["url"])
	try:
		d.name = e["name"]
	except AttributeError:
		d.name = e["url"].rsplit("/", 1)[1].split("?", 1)[0].rsplit(".", 1)[0]
	try:
		if not e["duration"]:
			raise KeyError
		d.duration = e["duration"]
	except KeyError:
		pass
	return d

lyrics_cache = {}
lyrics_renders = {}
def render_lyrics(entry):
	try:
		name = entry.name
		try:
			lyrics = lyrics_cache[name]
		except KeyError:
			try:
				lyrics = lyrics_scraper.result()(name)
			except:
				lyrics = ""
			if lyrics:
				lyrics_cache[name] = lyrics
			else:
				entry.lyrics = lyrics
				return
		if options.spectrogram:
			return
		rect = (player.rect[2] - 8, player.rect[3] - 92)
		if entry.get("lyrics") and entry.lyrics[1].get_size() == rect:
			return entry.pop("lyrics_loading", None)
		try:
			render = lyrics_renders[name]
			if render[1].get_size() != rect:
				raise KeyError
		except KeyError:
			render = None
			while len(lyrics_renders) > 6:
				n = next(iter(lyrics_renders))
				lyrics_cache.pop(n, None)
				render = lyrics_renders.pop(n, None)
				if render:
					ref = sys.getrefcount(render)
					if ref > 2 or render[1].get_size() != rect:
						# print(f"Deleting surface {render[1]} ({ref})...")
						render = None
					# else:
					#	 print(f"Reusing surface {render[1]} ({ref})...")
			if not render:
				lyrics_surf = pygame.Surface(rect, FLAGS)
				render = [None, lyrics_surf]
			else:
				render[1].fill((0, 0, 0))
			render[0] = lyrics[0]
			lyrics_renders[name] = render
			mx = 0
			x = 0
			y = 0
			for para in lyrics[1].split("\n\n"):
				lines = para.splitlines()
				if mx and y + 42 > render[1].get_height():
					y = 0
					x += mx
					mx = 0
				for line in lines:
					line = line.strip()
					if not line:
						continue
					if mx and y + 28 > render[1].get_height():
						y = 0
						x += mx
						mx = 0
					col = (255,) * 3
					if line[0] in "([" and line[-1] in "])":
						if line[0] != "(":
							name = line.split("(", 1)[0][1:]
						else:
							name = line[1:]
						name = name.rstrip("]) ").casefold().strip()
						if name.startswith("pre-") or name.startswith("pre "):
							pre = True
							name = name[4:]
						elif name.startswith("post-") or name.startswith("post "):
							pre = True
							name = name[5:]
						else:
							pre = False
						if name.startswith("intro") or name.startswith("outro"):
							col = (255, 0, 0)
						elif name.startswith("verse"):
							col = (191, 127, 255)
						elif name.startswith("chorus"):
							col = (0, 255, 255)
						elif name.startswith("bridge"):
							col = (255, 127, 191)
						elif name.startswith("breakdown"):
							col = (255, 0, 255)
						elif name.startswith("refrain"):
							col = (255, 255, 0)
						elif name.startswith("interlude"):
							col = (0, 255, 0)
						elif name.startswith("instrumental"):
							col = (255, 127, 0)
						elif line[0] == "[":
							col = (0, 0, 255)
						if pre:
							col = tuple(i >> 1 for i in col)
					s = text_size(line, 12)
					if s[0] <= render[1].get_width() >> 1:
						rect = message_display(
							line,
							12,
							(x, y),
							col,
							align=0,
							surface=render[1],
						)
						mx = max(mx, rect[2] + 8)
						y += 14
					else:
						p = 0
						words = line.split()
						curr = ""
						while words:
							w = words.pop(0)
							orig = curr
							curr = curr + " " + w if curr else w
							if orig:
								s = text_size(curr, 12)
								if s[0] > render[1].get_width() // 2 - p:
									rect = message_display(
										orig,
										12,
										(x + p, y),
										col,
										align=0,
										surface=render[1],
									)
									mx = max(mx, rect[2] + 8 + p)
									y += 14
									p = 8
									curr = w
						if curr:
							rect = message_display(
								curr,
								12,
								(x + p, y),
								col,
								align=0,
								surface=render[1],
							)
							mx = max(mx, rect[2] + 8 + p)
							y += 14
				if y:
					y += 7
		entry.lyrics = render
		entry.pop("lyrics_loading", None)
	except:
		print_exc()

def prepare(entry, force=False, download=False):
	if not entry.url:
		return
	stream = entry.get("stream", "")
	if not stream or stream.startswith("ytsearch:") or force and (stream.startswith("https://cf-hls-media.sndcdn.com/") or expired(stream)):
		if not is_url(entry.url):
			entry.stream = os.path.exists(entry.url) and entry.url
			duration = entry.duration
			if not duration:
				info = get_duration_2(stream)
				duration = info[0]
				if info[0] in (None, nan) and info[1] in ("N/A", "auto"):
					fi = stream
					fn = "cache/~" + shash(entry.url) + ".pcm"
					if not os.path.exists(fn):
						fn = select_and_convert(fi)
					duration = get_duration_2(fn)[0]
					stream = entry.stream = fn
				globals()["queue-length"] = -1
			entry.duration = duration or entry.duration
			return entry.stream
		ytdl = downloader.result()
		try:
			resp = ytdl.search(entry.url)
			data = resp[0]
			if force:
				stream = ytdl.get_stream(entry, force=True, download=False)
		except requests.ConnectionError:
			return time.sleep(2)
		except:
			fn = "cache/~" + shash(entry.url) + ".pcm"
			if os.path.exists(fn):
				duration = get_duration_2(fn)[0]
				stream = entry.stream = fn
				entry.duration = duration or entry.duration
				globals()["queue-length"] = -1
				return stream
			entry.url = ""
			print_exc()
			return
		else:
			if entry.name != data.get("name"):
				entry.pop("lyrics", None)
				entry.pop("surf", None)
				entry.pop("lyrics_loading", None)
				if len(resp) == 1:
					submit(render_lyrics, queue[0])
			globals()["queue-length"] = -1
			entry.update(data)
			if len(resp) > 1:
				try:
					i = queue.index(entry) + 1
				except:
					i = len(queue)
				q1, q3 = queue.view[:i - 1], queue.view[i:]
				q2 = alist(cdict(**e, pos=i) for e in resp)
				if control.shuffle and len(q2) > 1:
					q2.shuffle()
				queue.fill(np.concatenate((q1, q2, q3)))
				submit(render_lyrics, queue[0])
	elif (force > 1 or force and not stream) and is_url(entry.get("url")):
		ytdl = downloader.result()
		if force > 1:
			data = ytdl.extract(entry.url)
			stream = data[0].setdefault("stream", data[0].url)
		else:
			stream = ytdl.get_stream(entry, force=True, download=False)
		globals()["queue-length"] = -1
	elif not is_url(stream) and not os.path.exists(stream):
		entry.stream = os.path.exists(entry.url) and entry.url
		duration = entry.duration
		if not duration:
			info = get_duration_2(stream)
			duration = info[0]
			if info[0] in (None, nan) and info[1] in ("N/A", "auto"):
				fi = stream
				fn = "cache/~" + shash(entry.url) + ".pcm"
				if not os.path.exists(fn):
					fn = select_and_convert(fi)
				duration = get_duration_2(fn)[0]
				stream = entry.stream = fn
			globals()["queue-length"] = -1
		entry.duration = duration or entry.duration
		return entry.stream
	else:
		stream = entry.stream
	stream = stream.strip()
	duration = entry.duration
	if not duration:
		info = get_duration_2(stream)
		duration = info[0]
		if info[0] in (None, nan) and info[1] in ("N/A", "auto"):
			fi = stream
			fn = "cache/~" + shash(entry.url) + ".pcm"
			if not os.path.exists(fn):
				fn = select_and_convert(fi)
			duration = get_duration_2(fn)[0]
			stream = entry.stream = fn
		globals()["queue-length"] = -1
	elif stream and is_url(stream) and download:
		es = base64.b85encode(stream.encode("utf-8")).decode("ascii")
		mixer.submit(f"~download {es} cache/~{shash(entry.url)}.pcm")
	entry.duration = duration or entry.duration
	return stream

def start_player(pos=None, force=False):
	try:
		entry = queue[0]
	except IndexError:
		return skip()
	if len(queue) > 1 and control.loop < 2:
		ensure_next(1)
	duration = entry.duration or 300
	if pos is None:
		if audio.speed >= 0:
			pos = 0
	elif pos >= duration:
		if audio.speed > 0:
			return skip()
		pos = duration
	elif pos <= 0:
		if audio.speed < 0:
			return skip()
		pos = 0
	with player.waiting:
		if pos is not None:
			player.pos = pos
			player.index = player.pos * 30
		if force and is_url(queue[0].url):
			queue[0].stream = None
			queue[0].research = True
			downloader.result().cache.pop(queue[0].url, None)
		player.last = 0
		player.amp = 0
		player.pop("osci", None)
		stream = prepare(queue[0], force=force + 1)
		if not queue[0].url:
			return skip()
		stream = prepare(queue[0], force=force + 1)
		entry = queue[0]
		if not entry.url:
			return skip()
		if not stream:
			player.fut = None
			return None, inf
		duration = entry.duration
		if not duration:
			info = get_duration_2(stream)
			duration = info[0]
			if info[0] in (None, nan) and info[1] in ("N/A", "auto"):
				fi = stream
				fn = "cache/~" + shash(fi) + ".pcm"
				if not os.path.exists(fn):
					fn = select_and_convert(fi)
				duration = get_duration_2(fn)[0]
				stream = entry.stream = fn
		entry.duration = duration or entry.duration
		duration = entry.duration or 300
		if duration is None:
			player.fut = None
			return None, inf
		if pos is None:
			if audio.speed >= 0:
				pos = 0
			else:
				pos = duration
		elif pos >= duration:
			if audio.speed > 0:
				return skip()
			pos = duration
		elif pos <= 0:
			if audio.speed < 0:
				return skip()
			pos = 0
		if control.shuffle == 2:
			player.needs_shuffle = False
		else:
			player.needs_shuffle = not is_url(stream)
		es = base64.b85encode(stream.encode("utf-8")).decode("ascii")
		s = f"{es}\n{pos} {duration} {entry.get('cdc', 'auto')} {shash(entry.url)}\n"
		mixer.submit(s, force=False)
		player.pos = pos
		player.index = player.pos * 30
		player.end = duration or inf
		return stream, duration

def start():
	if queue:
		return enqueue(queue[0])
	player.last = 0
	player.pos = 0
	player.end = inf
	return None, inf

last_save = 0
def skip():
	if queue:
		e = queue.popleft()
		if control.shuffle:
			queue[1:].shuffle()
		if control.loop == 2:
			queue.appendleft(e)
		elif control.loop == 1:
			queue.append(e)
		else:
			if len(queue) > sidebar.maxitems - 1:
				queue[sidebar.maxitems - 1].pos = sidebar.maxitems - 1
			sidebar.particles.append(e)
		t = pc()
		if t >= last_save + 10:
			submit(save_settings)
			globals()["last_save"] = t
		if queue:
			return enqueue(queue[0])
	mixer.clear()
	player.last = 0
	player.pos = 0
	player.end = inf
	return None, inf

def seek_abs(pos):
	return start_player(pos) if queue else (None, inf)

def seek_rel(pos):
	if not pos:
		return
	player.last = 0
	player.amp = 0
	player.pop("osci", None)
	if pos + player.pos >= player.end:
		if audio.speed > 0:
			return skip()
		pos = player.end - player.pos
	if pos + player.pos <= 0:
		if audio.speed < 0:
			return skip()
		pos = -player.pos
	progress.num += pos
	progress.alpha = 255
	if audio.speed > 0 and pos > 0 and pos <= 180:
		with player.waiting:
			mixer.drop(pos)
	elif audio.speed < 0 and pos < 0 and pos >= -180:
		with player.waiting:
			mixer.drop(pos)
	else:
		seek_abs(max(0, player.pos + pos))
	player.pos += pos

def restart_mixer():
	global mixer
	if not mixer:
		return
	if mixer.is_running():
		for p in mixer.children(True):
			p.kill()
		mixer.kill()
	mixer = start_mixer()
	mixer.state(player.paused)
	submit(transfer_instrument, *project.instruments.values())
	return seek_abs(player.pos)

def play():
	try:
		while True:
			b = as_str(mixer.stderr.readline()).rstrip()
			if "~" not in b:
				if b:
					print(b)
				else:
					time.sleep(0.1)
					if not mixer.is_running():
						restart_mixer()
				continue
			if b[0] == "o":
				b = b[1:]
				osize = list(map(int, b.split("~")))
				req = int(np.prod(osize) * 4)
				b = mixer.stderr.read(req)
				while len(b) < req:
					if not b:
						time.sleep(0.1)
						if not mixer.is_running():
							restart_mixer()
						continue
					b += mixer.stderr.read(req - len(b))
				osci = pygame.image.frombuffer(b, osize, "RGBA")
				player.osci = osci
			else:
				b = b[1:]
				ssize = list(map(int, b.split("~")))
				req = int(np.prod(ssize) * 3)
				b = mixer.stderr.read(req)
				while len(b) < req:
					if not b:
						time.sleep(0.1)
						if not mixer.is_running():
							restart_mixer()
						continue
					b += mixer.stderr.read(req - len(b))
				if not options.spectrogram:
					continue
				spec = pygame.image.frombuffer(b, ssize, "RGB").convert()
				player.spec = spec
				player.pop("spec_used", None)
	except:
		if not mixer.is_running():
			print(as_str(mixer.stderr.read()))
		print_exc()

def reevaluate():
	time.sleep(2)
	while not player.pos and not pygame.closed:
		print("Re-evaluating file stream...")
		start_player(0, True)
		time.sleep(2)

device_waiting = None
def wait_on():
	if not device_waiting:
		return
	try:
		d = get_device(OUTPUT_DEVICE, default=False)
		if d:
			globals()["device_waiting"] = None
			print("Device target found.")
			mixer.submit(f"~output {OUTPUT_DEVICE}")
			return
	except:
		print_exc()

def get_device(name, default=True):
	try:
		return sc.get_speaker(name)
	except (IndexError, RuntimeError):
		if default:
			return sc.default_speaker()

PG_USED = None
SC_EMPTY = np.zeros(3200, dtype=np.float32)
def sc_player(d):
	cc = d.channels
	try:
		if not PG_USED:
			raise RuntimeError
		player = d.player(SR, cc, 2048)
	except RuntimeError:
		if PG_USED:
			pygame.mixer.Channel(0).stop()
		else:
			pygame.mixer.init(SR, 32, cc, 512, devicename=d.name)
		globals()["PG_USED"] = (d.name, cc)
		player = pygame.mixer
		player.type = "pygame"
		player.dtype = np.float32
		player.peak = 32767
	else:
		player.__enter__()
		player.type = "soundcard"
		player.dtype = np.float32
		player.peak = 1
	player.closed = False
	player.playing = None
	player.fut = None
	player._data_ = ()
	player.channels = cc
	# a monkey-patched play function that has a better buffer
	# (soundcard's normal one is insufficient for continuous playback)
	def play(self):
		while True:
			if player.paused:
				if len(self._data_) > 3200 * cc:
					self._data_ = self._data_[-3200 * cc:]
				return
			towrite = self._render_available_frames()
			if towrite < 50 * cc:
				async_wait()
				continue
			if self.fut:
				self.fut.result()
			self.fut = concurrent.futures.Future()
			if not len(self._data_):
				self._data_ = SC_EMPTY[:cc * 1600]
			b = self._data_[:towrite << 1].data
			buffer = self._render_buffer(towrite)
			CFFI.memmove(buffer[0], b, b.nbytes)
			self._render_release(towrite)
			self._data_ = self._data_[towrite << 1:]
			if self.closed:
				return
			self.fut.set_result(None)
	def write(data):
		if player.closed:
			return
		if cc < 2:
			if data.dtype == np.float32:
				data = data[::2] + data[1::2]
				data *= 0.5
			else:
				data >>= 1
				data = data[::2] + data[1::2]
		player.wait()
		if player.type == "pygame":
			if cc >= 2:
				data = data.reshape((len(data) // cc, cc))
			sound = pygame.sndarray.make_sound(data)
			return player.Channel(0).queue(sound)
		if not len(player._data_):
			player._data_ = data
			return verify()
		player.fut = concurrent.futures.Future()
		player._data_ = np.concatenate((player._data_, data))
		player.fut.set_result(None)
		return verify()
	player.write = write		
	def close():
		player.closed = True
		if player.type == "pygame":
			return player.Channel(0).stop()
		try:
			player.__exit__(None, None, None)
		except:
			print_exc()
	player.close = close
	def wait():
		if player.type == "pygame":
			while player.Channel(0).get_queue():
				async_wait()
			return
		if not len(player._data_):
			return
		verify()
		while len(player._data_) > 3200 * cc:
			async_wait()
		while player.fut and not player.fut.done():
			player.fut.result()
	def verify():
		if not player.playing or player.playing.done():
			player.playing = submit(play, player)
	player.wait = wait
	return player

get_channel = lambda: sc_player(get_device(OUTPUT_DEVICE))
channel = get_channel()

def pos():
	try:
		while True:
			s = None
			while not s and mixer.is_running():
				s = as_str(mixer.stdout.readline()).rstrip()
				if s and s[0] != "~":
					if s[0] in "'\"":
						s = ast.literal_eval(s)
					print(s, end="")
					s = ""
			if not s:
				time.sleep(0.05)
				continue
			if mixer.new:
				submit(reset_menu, reset=True)
			player.last = pc()
			s = s[1:]
			if s == "s":
				submit(skip)
				player.last = 0
				continue
			if s[0] == "o":
				globals()["OUTPUT_DEVICE"] = s[2:]
				continue
			if s[0] == "w":
				globals()["OUTPUT_DEVICE"] = s[2:]
				if not device_waiting:
					globals()["device_waiting"] = OUTPUT_DEVICE
					print(f"Waiting on {OUTPUT_DEVICE}...")
				continue
			if s == "W":
				globals()["device_waiting"] = None
				continue
			if s[0] == "n":
				player.note = float(s[2:])
				continue
			if s == "r":
				submit(start_player, 0, True)
				print("Re-evaluating file stream...")
				submit(reevaluate)
				continue
			if s == "V":
				globals()["video-render"].set_result(None)
				continue
			if s[0] == "x":
				spl = s[2:].split()
				player.stats.peak = spl[0]
				player.stats.amplitude = spl[1]
				player.stats.velocity = spl[2]
				player.amp = float(spl[-1])
				continue
			elif s[0] == "y":
				player.amp = float(s[2:])
				continue
			elif s[0] == "t":
				i = int(s[2:])
				transfer_instrument(i)
				select_instrument(i)
				continue
			if player.waiting:
				continue
			i, dur = map(float, s.split(" ", 1))
			if not progress.seeking:
				player.index = i
				player.pos = round(player.index / 30, 4)
			if dur >= 0:
				player.end = dur or inf
	except:
		if not mixer.is_running():
			print(mixer.stderr.read().decode("utf-8", "replace"))
		print_exc()

submit(play)
submit(pos)

def ensure_next(i=1):
	if i <= 1 or all(e.done() for e in enext):
		enext.clear()
		if len(queue) > i:
			e = queue[i]
			e.duration = e.get("duration") or False
			e.pop("research", None)
			enext.add(submit(prepare, e, force=i <= 1, download=i <= 1))
			if i <= 1 and not e.get("lyrics_loading") and not e.get("lyrics"):
				e.lyrics_loading = True
				enext.add(submit(render_lyrics, e))

def enqueue(entry, start=True):
	try:
		if not queue:
			return None, inf
		while queue[0] is None:
			time.sleep(0.5)
		queue[0].lyrics_loading = True
		submit(render_lyrics, queue[0])
		flash_window()
		stream, duration = start_player()
		progress.num = 0
		progress.alpha = 0
		return stream, duration
	except:
		print_exc()
	return None, inf

ffmpeg_start = (ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-err_detect", "ignore_err", "-hwaccel", "auto", "-vn")
concat_buff = b"\x00" * (48000 * 2 * 2)

def download(entries, fn, settings=False):
	try:
		downloading.fn = fn
		downloading.target = sum(e.duration or 300 for e in entries)
		if len(entries) > 1:
			downloading.target += len(entries)
		cmd = ffmpeg_start + ("-f", "s16le", "-ar", "48k", "-ac", "2", "-i", "-")
		if fn.startswith(".pcm"):
			cmd += ("-f", "s16le")
		else:
			cmd += ("-b:a", "224k")
		cmd += ("-ar", "48k", "-ac", "2", fn)
		print(cmd)
		saving = psutil.Popen(cmd, stdin=subprocess.PIPE, bufsize=192000)
		ots = ts = time.time_ns() // 1000
		procs = deque()
		for entry in entries:
			if len(procs) > 2:
				while procs:
					p = procs.popleft()
					if len(entries) > 1:
						saving.stdin.write(concat_buff)
					p.wait()
					if os.path.exists(p.fn):
						if os.path.getsize(p.fn):
							with open(p.fn, "rb") as sf:
								while True:
									b = sf.read(1048576)
									if not b:
										break
									saving.stdin.write(b)
						submit(os.remove(p.fn))
			st = prepare(entry, force=True)
			sh = "cache/~" + shash(entry.url) + ".pcm"
			if os.path.exists(sh) and os.path.getsize(sh) >= (entry.duration - 1) * 48000 * 2 * 2:
				st = sh
			downloading.target = sum(e.duration or 300 for e in entries)
			if len(entries) > 1:
				downloading.target += len(entries)
			fn3 = f"cache/\x7f{ts}.pcm"
			cmd = ffmpeg_start + ("-nostdin",)
			if not is_url(st) and st.endswith(".pcm"):
				cmd += ("-f", "s16le", "-ar", "48k", "-ac", "2")
			cmd += ("-i", st)
			if settings:
				cmd += tuple(construct_options())
			if settings and control.silenceremove:
				cmd += ("-af", "silenceremove=start_periods=1:start_duration=1:start_threshold=-50dB:start_silence=0.5:stop_periods=-9000:stop_threshold=-50dB:window=0.015625")
			cmd += ("-f", "s16le", "-ar", "48k", "-ac", "2", fn3)
			print(cmd)
			p = psutil.Popen(cmd)
			p.fn = fn3
			procs.append(p)
			ts += 1
		while procs:
			p = procs.popleft()
			saving.stdin.write(concat_buff)
			p.wait()
			if os.path.exists(p.fn):
				if os.path.getsize(p.fn):
					with open(p.fn, "rb") as sf:
						while True:
							b = sf.read(1048576)
							if not b:
								break
							saving.stdin.write(b)
				os.remove(p.fn)
		saving.stdin.close()
		saving.wait()
		downloading.target = 0
	except:
		print_exc()


def pause_toggle(state=None):
	globals()["tick"] = -2
	if state is None:
		player.paused ^= True
	else:
		player.paused = state
	if toolbar.editor:
		player.broken = player.paused
	mixer.state(player.paused or toolbar.editor)
	toolbar.pause.speed = toolbar.pause.maxspeed
	sidebar.menu = 0
	player.editor.played.clear()


def update_menu():
	global sidebar_width, toolbar_height
	ts = toolbar.pause.setdefault("timestamp", 0)
	t = pc()
	player.lastframe = duration = max(0.001, min(t - ts, 0.125))
	player.flash_s = max(0, player.get("flash_s", 0) - duration * 60)
	player.flash_i = max(0, player.get("flash_i", 0) - duration * 60)
	player.flash_o = max(0, player.get("flash_o", 0) - duration * 60)
	toolbar.pause.timestamp = pc()
	ratio = 1 + 1 / (duration * 12)
	progress.alpha *= 0.998 ** (duration * 480)
	if progress.alpha < 16:
		progress.alpha = progress.num = 0
	progress.angle = -t * pi
	if progress.seeking:
		player.amp = 0.5
	elif not is_active():
		player.amp = 0
		player.pop("osci", None)
	toolbar.pause.angle = (toolbar.pause.angle + (toolbar.pause.speed + 1) * duration * (-2 if player.paused else 2))
	toolbar.pause.angle %= tau
	toolbar.pause.speed *= 0.995 ** (duration * 480)
	sidebar.scroll.target = max(0, min(sidebar.scroll.target, len(queue) * 32 - screensize[1] + options.toolbar_height + 36 + 32))
	r = ratio if sidebar.scrolling else (ratio - 1) / 3 + 1
	if abs(sidebar.scroll.pos - sidebar.scroll.target) < 1 / 32:
		sidebar.scroll.pos = sidebar.scroll.target
	else:
		sidebar.scroll.pos = (sidebar.scroll.pos * (r - 1) + sidebar.scroll.target) / r
	sidebar.maxitems = mi = ceil(screensize[1] - options.toolbar_height - 36 + sidebar.scroll.pos % 32) // 32
	sidebar.base = max(0, int(sidebar.scroll.pos // 32))
	# print(sidebar.scroll.target, sidebar.scroll.pos, sidebar.maxitems, sidebar.base)
	m2 = mi + 1
	if toolbar.editor:
		for i, entry in enumerate(sidebar.instruments[sidebar.base:sidebar.base + mi], sidebar.base):
			if i == entry.setdefault("pos", 0):
				continue
			i2 = (entry.pos * (ratio - 1) + i) / ratio
			d = abs(i2 - i)
			if d > m2:
				i2 = i + (m2 if i2 > i else -m2)
			elif d < 1 / 32:
				i2 = i
			entry.pos = i2
	else:
		for i, entry in enumerate(queue[sidebar.base:sidebar.base + mi], sidebar.base):
			if i == entry.setdefault("pos", 0):
				continue
			i2 = (entry.pos * (ratio - 1) + i) / ratio
			d = abs(i2 - i)
			if d > m2:
				i2 = i + (m2 if i2 > i else -m2)
			elif d < 1 / 32:
				i2 = i
			entry.pos = i2
	if kclick[K_SPACE]:
		pause_toggle()
		if player.paused:
			c = (255, 0, 0)
		else:
			c = (0, 255, 0)
		if control.ripples:
			toolbar.ripples.append(cdict(
				pos=toolbar.pause.pos,
				radius=0,
				colour=c,
				alpha=255,
			))
	if toolbar.resizing:
		toolbar_height = max(64, screensize[1] - mpos2[1] + 2)
		if options.toolbar_height != toolbar_height:
			options.toolbar_height = toolbar_height
			reset_menu()
			toolbar.resizing = True
			modified.add(toolbar.rect)
	if toolbar_height > screensize[1] - 64:
		options.toolbar_height = toolbar_height = screensize[1] - 64
		reset_menu()
		toolbar.resizing = True
		modified.add(toolbar.rect)
	if progress.seeking:
		orig = player.pos
		if player.end < inf:
			player.pos = max(0, min(1, (mpos2[0] - progress.pos[0] + progress.width // 2) / progress.length) * player.end)
			progress.num += (player.pos - orig)
		progress.alpha = 255
		player.index = player.pos * 30
		if not mheld[0]:
			progress.seeking = False
			if queue and isfinite(e_dur(queue[0].duration)):
				submit(seek_abs, player.pos)
	if sidebar.resizing:
		sidebar_width = min(screensize[0] - 8, max(144, screensize[0] - mpos2[0] + 2))
		if options.sidebar_width != sidebar_width:
			options.sidebar_width = sidebar_width
			reset_menu()
			sidebar.resizing = True
			modified.add(sidebar.rect)
	if sidebar_width > screensize[0] - 8:
		options.sidebar_width = sidebar_width = screensize[0] - 8
		reset_menu()
		sidebar.resizing = True
		modified.add(sidebar.rect)
	if queue and not toolbar.editor and isfinite(e_dur(queue[0].duration)):
		if kspam[K_PAGEUP]:
			submit(seek_rel, 300)
		elif kspam[K_PAGEDOWN]:
			submit(seek_rel, -300)
		elif kspam[K_UP]:
			submit(seek_rel, 30)
		elif kspam[K_DOWN]:
			submit(seek_rel, -30)
		elif kspam[K_RIGHT]:
			if CTRL[kheld]:
				submit(seek_abs, inf)
			else:
				submit(seek_rel, 5)
		elif kspam[K_LEFT]:
			if CTRL[kheld]:
				submit(seek_abs, 0)
			else:
				submit(seek_rel, -5)
	reset = False
	menu = sidebar.menu
	if menu and menu.get("scale", 0) >= 1:
		menu.selected = -1
		if in_rect(mpos, menu.rect):
			i = (mpos2[1] - menu.rect[1]) // 20
			if mclick[0]:
				button = menu.buttons[i]
				sidebar.menu = None
				button[1](*button[2:])
				reset = True
				mclick[0] = False
			menu.selected = i
			globals()["mpos"] = (nan, nan)
	if any(mclick):
		sidebar.menu = None
	if in_rect(mpos, toolbar.rect[:3] + (5,)):
		if mclick[0]:
			toolbar.resizing = True
		else:
			toolbar.resizer = True
	if in_circ(mpos, toolbar.pause.pos, max(4, toolbar.pause.radius - 2)):
		if any(mclick):
			pause_toggle()
		toolbar.pause.outer = 255
		toolbar.pause.inner = 191
		toolbar.updated = False
		sidebar.updated = False
	else:
		toolbar.pause.outer = 191
		toolbar.pause.inner = 127
	if in_rect(mpos, progress.rect) and not toolbar.editor:
		if mclick[0]:
			progress.seeking = True
			if queue and isfinite(e_dur(queue[0].duration)):
				mixer.clear()
		elif mclick[1]:
			sidebar.menu = 0
			enter = easygui.get_string(
				"Seek to position",
				"Miza Player",
				time_disp(player.pos),
			)
			if enter:
				pos = time_parse(enter)
				submit(seek_abs, pos)
	c = options.get("toolbar_colour", (64, 0, 96))
	if toolbar.resizing or in_rect(mpos, toolbar.rect):
		hls = colorsys.rgb_to_hls(*(i / 255 for i in c))
		hls = (hls[0], max(0, hls[1] + 1 / 24),  hls[2] / 1.2)
		c = verify_colour(round(i * 255) for i in colorsys.hls_to_rgb(*hls))
	toolbar.updated = False#toolbar.colour != c
	toolbar.colour = c
	if any(mclick):
		for button in toolbar.buttons:
			try:
				if in_rect(mpos, button.rect):
					button.flash = 64
					sidebar.menu = 0
					if callable(button.click):
						button.click()
					else:
						button.click[min(mclick.index(1), len(button.click) - 1)]()
			except AttributeError:
				while "rect" not in button:
					time.sleep(0.01)
			if toolbar.editor:
				break
	else:
		for button in toolbar.buttons:
			if "flash" in button:
				button.flash = max(0, button.flash - duration * 64)
	maxb = (options.sidebar_width - 12) // 44
	if any(mclick):
		for button in sidebar.buttons[:maxb]:
			if in_rect(mpos, button.rect):
				button.flash = 32
				click = button.click if not toolbar.editor else button.get("click2") or button.click
				sidebar.menu = 0
				if callable(click):
					click()
				else:
					click[min(mclick.index(1), len(click) - 1)]()
	else:
		for button in sidebar.buttons:
			if "flash" in button:
				button.flash = max(0, button.flash - duration * 64)
	if in_rect(mpos, sidebar.rect[:2] + (5, sidebar.rect[3])):
		if not toolbar.resizing and mclick[0]:
			sidebar.resizing = True
		else:
			sidebar.resizer = True
	c = options.get("sidebar_colour", (64, 0, 96))
	if sidebar.resizing or in_rect(mpos, sidebar.rect):
		hls = colorsys.rgb_to_hls(*(i / 255 for i in c))
		hls = (hls[0], max(0, hls[1] + 1 / 24),  hls[2] / 1.2)
		c = verify_colour(round(i * 255) for i in colorsys.hls_to_rgb(*hls))
	sidebar.updated = False#sidebar.colour != c
	sidebar.colour = c
	sidebar.relpos = (sidebar.get("relpos", 0) * (ratio - 1) + bool(sidebar.abspos)) / ratio
	scroll_height = screensize[1] - toolbar_height - 72
	sidebar.scroll.rect = (screensize[0] - 20, 52 + 16, 12, scroll_height)
	scroll_rat = max(12, min(scroll_height, scroll_height / max(1, len(queue)) * (screensize[1] - toolbar_height - 36) / 32))
	scroll_pos = sidebar.scroll.pos / (32 * max(1, len(queue)) - screensize[1] + toolbar_height + 52 + 16) * (scroll_height - scroll_rat) + 52 + 16
	sidebar.scroll.select_rect = (sidebar.scroll.rect[0], scroll_pos, sidebar.scroll.rect[2], scroll_rat)
	c = options.get("sidebar_colour", (64, 0, 96))
	hls = colorsys.rgb_to_hls(*(i / 255 for i in c))
	light = 1 - (1 - hls[1]) / 4
	if hls[2]:
		sat = 1 - (1 - hls[2]) / 2
	else:
		sat = 0
	c1 = tuple(round(i * 255) for i in colorsys.hls_to_rgb(hls[0], light - 1 / 4, sat))
	c2 = tuple(round(i * 255) for i in colorsys.hls_to_rgb((hls[0] + 1 / 12) % 1, light, sat))
	progress.select_colour = c1
	if sidebar.scrolling or in_rect(mpos, sidebar.scroll.rect):
		c1 += (191,)
		if mclick[0]:
			sidebar.scrolling = True
		elif mclick[1]:
			enter = easygui.get_string(
				"Scroll to position",
				"Miza Player",
				str(round(sidebar.scroll.target / 32)),
			)
			if enter:
				pos = float(enter) * 32
				sidebar.scroll.target = pos
		if sidebar.scrolling:
			if not mheld[0]:
				sidebar.scrolling = False
			r = min(max(0, mpos2[1] - 52 - 16 - scroll_rat / 2) / max(1, scroll_height - scroll_rat), 1)
			sidebar.scroll.target = r * (32 * max(1, len(queue)) - screensize[1] + toolbar_height + 52 + 16)
	else:
		c1 += (127,)
		c2 += (191,)
	sidebar.scroll.background = c1
	sidebar.scroll.colour = c2
	if common.font_reload:
		common.font_reload = False
		[e.pop("surf", None) for e in queue]
		[e.pop("surf", None) for e in sidebar.instruments]
		lyrics_cache.clear()
		lyrics_renders.clear()
	if toolbar.editor:
		update_piano()


def get_ripple(i, mode="toolbar"):
	if i >= 4:
		return (255,) * 3
	if i == 3:
		return (0,) * 3
	if mode == "toolbar":
		c = options.get("toolbar_colour", (64, 0, 96))
	else:
		c = options.get("sidebar_colour", (64, 0, 96))
	if not i:
		d = 0
	else:
		d = i * 2 - 3
	h, l, s = colorsys.rgb_to_hls(*(i / 255 for i in c))
	h = (h + d * 5 / 12) % 1
	s = 1 - (1 - s) / 2
	l = 1 - (1 - l) / 2
	return [round(i * 255) for i in colorsys.hls_to_rgb(h, l, s)]


# importing doesn't work; target files are simply functions that used to be here and cannot independently run without sharing global variables
with open("misc/sidebar.py", "rb") as f:
	b = f.read()
exec(compile(b, "sidebar.py", "exec"))
with open("misc/sidebar2.py", "rb") as f:
	b = f.read()
exec(compile(b, "sidebar2.py", "exec"))
with open("misc/piano.py", "rb") as f:
	b = f.read()
exec(compile(b, "piano.py", "exec"))


no_lyrics_path = options.get("no_lyrics_path", "misc/Default/no_lyrics.png")
no_lyrics_fut = submit(load_surface, no_lyrics_path)

globals()["h-cache"] = {}
globals()["h-timer"] = 0

def load_bubble(bubble_path):
	try:
		with Image.open(bubble_path) as im:
			if "RGB" not in im.mode:
				im = im.convert("RGBA")
			if im.mode == "RGBA":
				A = im.getchannel("A")
			im2 = ImageOps.grayscale(im)
			if any(x > 256 for x in im.size):
				size = limit_size(*im.size, 256, 256)
				im2 = im2.resize(size, Image.LANCZOS)
				if im.mode == "RGBA":
					A = A.resize(size, Image.LANCZOS)
			if im2.mode == "L":
				im2 = im2.point(lambda x: (x / 255) ** 0.8 * 255)
			if "RGB" not in im2.mode:
				im2 = im2.convert("RGB")
			if im.mode == "RGBA":
				im2.putalpha(A)
			globals()["h-img"] = im2
		globals()["h-cache"].clear()
		globals()["h-timer"] = 0

		def bubble_ripple(dest, colour, pos, radius, alpha=255, **kwargs):
			alpha = round_random(alpha / 3) * 3
			diameter = round_random(radius * 2)
			if not diameter > 0:
				return
			try:
				surf = globals()["h-cache"][diameter]
			except KeyError:
				im = globals()["h-img"].resize((diameter,) * 2, resample=Image.BILINEAR)
				surf = pil2pyg(im, convert=True)
				im.close()
				globals()["h-cache"][diameter] = surf
			globals()["h-timer"] = pc()
			return blit_complex(
				dest,
				surf,
				[round_random(x - y / 2) for x, y in zip(pos, surf.get_size())],
				alpha=alpha,
				colour=colour,
			)
		globals()["h-ripple"] = bubble_ripple
	except:
		print_exc()

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
				im = globals()["s-img"].resize((diameter,) * 2, resample=Image.BICUBIC)
				if "RGB" not in im.mode:
					im = im.convert("RGBA")
				surf = pil2pyg(im, convert=True)
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

bubble_path = options.get("bubble_path")
if bubble_path:
	submit(load_bubble, bubble_path)

spinner_path = "misc/Default/bubble.png"
submit(load_spinner, spinner_path)

def get_spinny_life(t):
	try:
		return t[1].life
	except (TypeError, AttributeError, IndexError):
		return inf

def spinnies():
	ts = 0
	while "pos" not in progress:
		time.sleep(0.5)
	t = pc()
	while True:
		while is_minimised():
			time.sleep(0.1)
		dur = max(0.001, min(t - ts, 0.125))
		ts = t
		try:
			ratio = 1 + 1 / (dur * 8)
			progress.vis = (progress.vis * (ratio - 1) + player.pos) / ratio
			progress.spread = min(1, (progress.spread * (ratio - 1) + player.amp) / ratio)
			progress.angle = -t * pi
			pops = deque()
			for i, p in sorted(enumerate(progress.particles), key=get_spinny_life):
				if not p:
					break
				p.life -= dur * 2.5
				if p.life <= 6:
					p.angle += dur
					p.rad = max(0, p.rad - 12 * dur)
					p.hsv[2] = max(p.hsv[2] - dur / 5, 0)
				if p.life < 3:
					pops.append(i)
			try:
				progress.particles.pops(pops)
			except IndexError:
				pass
			x = progress.pos[0] + round_random(progress.length * progress.vis / player.end) - progress.width // 2 if not progress.seeking or player.end < inf else mpos2[0]
			x = min(progress.pos[0] - progress.width // 2 + progress.length, max(progress.pos[0] - progress.width // 2, x))
			d = abs(pc() % 2 - 1)
			hsv = [0.5 + d / 4, 1 - 0.75 + abs(d - 0.75), 1]
			r = progress.spread * toolbar.pause.radius
			if r >= 1:
				a = progress.angle
				point = [cos(a) * r, sin(a) * r]
				p = (x + point[0], progress.pos[1] + point[1])
				progress.particles.append(cdict(
					centre=(x, progress.pos[1]),
					angle=a,
					rad=r,
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
	length = progress.length
	width = progress.width
	x = progress.pos[0] + round(length * progress.vis / player.end) - width // 2 if not progress.seeking or player.end < inf else mpos2[0]
	x = min(progress.pos[0] - width // 2 + length, max(progress.pos[0] - width // 2, x))
	r = progress.spread * toolbar.pause.radius
	if r:
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
			ri = max(1, round_random(p.life ** 1.2 * toolbar.pause.radius / 72))
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
	d = abs(pc() % 2 - 1)
	hsv = [0.5 + d / 4, 1 - 0.75 + abs(d - 0.75), 1]
	col = [round_random(i * 255) for i in colorsys.hsv_to_rgb(*hsv)]
	for i in shuffle(range(3)):
		a = progress.angle + i / 3 * tau
		point = [cos(a) * r, sin(a) * r]
		p = (x + point[0], progress.pos[1] + point[1])
		ri = max(7, progress.width // 2 + 2)
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


def change_bubble():
	bubble_path = easygui.fileopenbox(
		"Open an image file here!",
		"Miza Player",
		default=options.get("bubble_path", globals()["bubble_path"]),
		filetypes=iftypes,
	)
	if bubble_path:
		options.bubble_path = bubble_path
		submit(load_bubble, bubble_path)

def reset_bubble():
	options.pop("bubble_path", None)
	globals().pop("h-ripple", None)
	globals()["h-cache"] = {}
	globals()["h-timer"] = 0


def render_settings(dur, ignore=False):
	global crosshair, hovertext
	offs = round(sidebar.setdefault("relpos", 0) * -sidebar_width)
	sc = sidebar.colour or (64, 0, 96)
	sub = (sidebar.rect2[2], sidebar.rect2[3] - 52)
	subp = (screensize[0] - sidebar_width, 52)
	DISP2 = DISP.subsurface(subp + sub)
	in_sidebar = in_rect(mpos, sidebar.rect)
	offs2 = offs + sidebar_width
	c = options.get("sidebar_colour", (64, 0, 96))
	c = high_colour(c)
	for i, opt in enumerate(asettings):
		message_display(
			opt.capitalize(),
			12,
			(offs2 + 8, i * 32),
			colour=c,
			surface=DISP2,
			align=0,
			cache=True,
			font="Comic Sans MS",
		)
		# numrect = (screensize[0] + offs + sidebar_width - 8, 68 + i * 32)
		s = str(round(options.audio.get(opt, 0) * 100, 2)) + "%"
		message_display(
			s,
			12,
			(offs2 + sidebar_width - 8, 16 + i * 32),
			colour=c,
			surface=DISP2,
			align=2,
			cache=True,
			font="Comic Sans MS",
		)
		srange = asettings[opt]
		w = (sidebar_width - 16)
		x = (options.audio.get(opt, 0) - srange[0]) / (srange[1] - srange[0])
		if opt in ("speed", "pitch", "nightcore"):
			x = min(1, max(0, x))
		else:
			x = min(1, abs(x))
		x = round(x * w)
		brect = (screensize[0] + offs + 6, 67 + i * 32, sidebar_width - 12, 13)
		brect2 = (offs2 + 6, 17 + i * 32, sidebar_width - 12, 13)
		hovered = in_sidebar and in_rect(mpos, brect) or aediting[opt]
		crosshair |= bool(hovered) << 1
		v = max(0, min(1, (mpos2[0] - (screensize[0] + offs + 8)) / (sidebar_width - 16))) * (srange[1] - srange[0]) + srange[0]
		if len(srange) > 2:
			v = round_min(math.round(v / srange[2]) * srange[2])
		else:
			rv = round_min(math.round(v * 32) / 32)
			if type(rv) is int and rv not in srange:
				v = rv
		if hovered and not hovertext:
			hovertext = cdict(
				text=str(round(v * 100, 2)) + "%",
				size=10,
				colour=(255, 255, 127),
			)
			if aediting[opt]:
				if not mheld[0]:
					aediting[opt] = False
			elif mclick[0]:
				aediting[opt] = True
			elif mclick[1]:
				enter = easygui.get_string(
					opt.capitalize(),
					"Miza Player",
					str(round_min(options.audio[opt] * 100)),
				)
				if enter:
					v = round_min(float(safe_eval(enter)) / 100)
					aediting[opt] = True
			if aediting[opt]:
				orig, options.audio[opt] = options.audio[opt], v
				if orig != v:
					mixer.submit(f"~setting {opt} {v}", force=ignore or opt == "volume" or not queue)
		z = max(0, x - 4)
		rect = (offs2 + 8 + z, 17 + i * 32, sidebar_width - 16 - z, 9)
		col = (48 if hovered else 32,) * 3
		bevel_rectangle(
			DISP2,
			col,
			rect,
			3,
		)
		rainbow = quadratic_gradient((w, 9), pc() / 2 + i / 4)
		DISP2.blit(
			rainbow,
			(offs2 + 8 + x, 17 + i * 32),
			(x, 0, w - x, 9),
			special_flags=BLEND_RGB_MULT,
		)
		rect = (offs2 + 8, 17 + i * 32, x, 9)
		col = (223 if hovered else 191,) * 3
		bevel_rectangle(
			DISP2,
			col,
			rect,
			3,
		)
		rainbow = quadratic_gradient((w, 9), pc() + i / 4)
		DISP2.blit(
			rainbow,
			(offs2 + 8, 17 + i * 32),
			(0, 0, x, 9),
			special_flags=BLEND_RGB_MULT,
		)
		if hovered:
			bevel_rectangle(
				DISP2,
				progress.select_colour,
				brect2,
				2,
				filled=False,
			)
	rect = (offs2 + sidebar_width // 2 - 32, 304, 64, 32)
	crect = (screensize[0] - sidebar_width // 2 - 32, 355) + rect[2:]
	hovered = in_rect(mpos, crect)
	if hovered and mclick[0]:
		settings_reset()
	col = (0, 191, 191) if not hovered else (191, 255, 255)
	bevel_rectangle(
		DISP2,
		col,
		rect,
		4,
	)
	message_display(
		"Reset",
		16,
		[offs2 + sidebar_width // 2, 318],
		(255,) * 3,
		font="Comic Sans MS",
		surface=DISP2,
		cache=True,
	)
	if "more" not in sidebar:
		sidebar.more = sidebar.more_angle = 0
	if sidebar.more_angle > 0.001:
		more = (
			("silenceremove", "Skip silence", "Skips over silent or extremely quiet frames of audio."),
			("unfocus", "Reduce unfocus FPS", "Greatly reduces FPS of display when window is left unfocused."),
			("presearch", "Preemptive search", "Pre-emptively searches up and displays duration of songs in a playlist.\nIncreases amount of requests being sent, and may also cause lag spikes."),
			("preserve", "Preserve sessions", "Preserves sessions and automatically reloads them."),
			("ripples", "Ripples", "Clicking anywhere on the sidebar or toolbar produces a visual ripple effect."),
			("autoupdate", "Auto update", "Automatically and silently updates Miza Player in the background when an update is detected."),
		)
		mrect = (offs2 + 8, 376, sidebar_width - 16, 192)
		if sidebar.more_angle < 63 / 64:
			surf = HWSurface.any(mrect[2:], FLAGS | pygame.SRCALPHA)
			surf.fill((0, 0, 0, 0))
		else:
			surf = DISP2.subsurface(mrect)
		for i, t in enumerate(more):
			s, name, description = t
			apos = (screensize[0] - sidebar_width + offs2 + 24, 427 + i * 32 + 16)
			hovered = hypot(*(np.array(mpos) - apos)) < 16
			if hovered and mclick[0]:
				options.control[s] ^= 1
				if s in ("silenceremove", "unfocus"):
					mixer.submit(f"~setting {s} {options.control[s]}")
			ripple_f = globals().get("s-ripple", concentric_circle)
			if options.control.get(s):
				col = (96, 255, 96)
			else:
				col = (127, 0, 0)
			pos = (16, i * 32 + 16)
			reg_polygon_complex(
				surf,
				pos,
				(255,) * 3 if hovered else col,
				0,
				14,
				14,
				0,
				255 if hovered else 255 * (abs((pc() - i / 2) / 4 % 1 - 0.5) + 0.5),
				2,
				9,
				True,
				soft=True,
			)
			ripple_f(
				surf,
				colour=col,
				pos=pos,
				radius=16,
				fill_ratio=0.5,
			)
			c = options.get("sidebar_colour", (64, 0, 96))
			c = high_colour(c, 255 if hovered else 223)
			message_display(
				name,
				16,
				(36, i * 32 + 4),
				colour=c,
				align=0,
				surface=surf,
				font="Comic Sans MS",
				cache=True,
			)
		if sidebar_width >= 192:
			r = (sidebar_width - 80, 160, 64, 32)
			r2 = (screensize[0] - 68 + offs + sidebar_width, 589, 64, 32)
			if in_rect(mpos, r2):
				if mclick[0]:
					submit(update_collections2)
					common.repo_fut = submit(update_repo, force=True)
					if common.repo_fut.result():
						easygui.show_message(
							"No new updates found.",
							"Miza Player",
						)
				c = (112, 127, 64, 223)
			else:
				c = (96, 112, 80, 127)
			fut = common.__dict__.get("repo-update")
			if fut and not isinstance(fut, bool):
				c2 = verify_colour(x * sin(pc() * tau / 4) for x in c[:3])
				c2.append(c[-1])
				c = c2
			bevel_rectangle(surf, c, r, bevel=4)
			message_display(
				"Update",
				16,
				rect_centre(r),
				colour=(255,) * 3,
				alpha=c[-1] + 32,
				surface=surf,
				font="Comic Sans MS",
				cache=True,
			)
		if sidebar.more_angle < 63 / 64:
			arr = np.linspace(sidebar.more_angle * 510, sidebar.more_angle * 510 - 255, mrect[3])
			np.clip(arr, 0, 255, out=arr)
			arr = arr.astype(np.uint8)
			im = pyg2pil(surf)
			a = im.getchannel("A")
			a2 = Image.fromarray(arr, "L").resize(mrect[2:], resample=Image.NEAREST)
			A = ImageChops.multiply(a, a2)
			im.putalpha(A)
			surf = pil2pyg(im, convert=False)
			DISP2.blit(
				surf,
				mrect[:2],
			)
	rect = (offs2 + sidebar_width // 2 - 32, 344, 64, 32)
	crect = (screensize[0] - sidebar_width // 2 - 32, 395) + rect[2:]
	hovered = in_rect(mpos, crect)
	if hovered and any(mclick):
		sidebar.more = not sidebar.more
	rat = 0.05 ** dur
	sidebar.more_angle = sidebar.more_angle * rat + sidebar.more * (1 - rat)
	lum = 223 if hovered else 191
	c = options.get("sidebar_colour", (64, 0, 96))
	hls = list(colorsys.rgb_to_hls(*(i / 255 for i in c)))
	light = 1 - (1 - hls[1]) / 4
	if hls[2]:
		sat = 1 - (1 - hls[2]) / 2
	else:
		sat = 0
	hls[1] = lum / 255 * light
	hls[2] = sat
	if not sidebar.get("more"):
		fut = common.__dict__.get("repo-update")
		if fut and not isinstance(fut, bool):
			hls[1] = sin(pc() * tau / 4)
	col = [round(i * 255) for i in colorsys.hls_to_rgb(*hls)]
	bevel_rectangle(
		DISP2,
		col,
		rect,
		4,
	)
	kwargs = {}
	if not toolbar.ripples:
		kwargs["soft"] = sidebar.colour
	reg_polygon_complex(
		DISP2,
		(offs2 + sidebar_width // 2 - 48, 357 + sidebar.more_angle * 6),
		(255,) * 3,
		3,
		12,
		12,
		pi/2 - sidebar.more_angle * pi,
		255,
		2,
		9,
		filled=True,
		**kwargs,
	)
	text = "More" if not sidebar.get("more") else "Less"
	message_display(
		text,
		16,
		[offs2 + sidebar_width // 2, 358],
		(255,) * 3,
		font="Comic Sans MS",
		surface=DISP2,
		cache=True,
	)

def draw_menu():
	global crosshair, hovertext
	ts = toolbar.progress.setdefault("timestamp", 0)
	t = pc()
	dur = max(0.001, min(t - ts, 0.125))
	if not tick & 7:
		toolbar.progress.timestamp = pc()
		pops = set()
		for i, ripple in enumerate(toolbar.ripples):
			ripple.radius += dur * 128
			ripple.alpha -= dur * 128
			if ripple.alpha <= 2:
				pops.add(i)
		if pops:
			toolbar.ripples.pops(pops)
		pops = set()
		for i, ripple in enumerate(sidebar.ripples):
			ripple.radius += dur * 128
			ripple.alpha -= dur * 128
			if ripple.alpha <= 2:
				pops.add(i)
		if pops:
			sidebar.ripples.pops(pops)
	crosshair = False
	hovertext = None
	if any(mclick) and in_rect(mpos, sidebar.rect):
		if control.ripples:
			sidebar.ripples.append(cdict(
				pos=mpos,
				radius=0,
				colour=get_ripple(mclick.index(1), mode="sidebar"),
				alpha=255,
			))
		if mclick[1] and sidebar.menu is None:

			def set_colour():
				colour = easygui.get_color_rgb()
				if colour:
					options.sidebar_colour = colour

			sidebar.menu = cdict(
				buttons=(
					("Set colour", set_colour),
					("Change ripples", change_bubble),
					("Reset ripples", reset_bubble),
				),
			)
			if not toolbar.get("editor") and not sidebar.abspos:
				p = pyperclip.paste()
				if p:
					paste_queue = lambda: submit(enqueue_auto, *p.splitlines())
					sidebar.menu.buttons = (("Paste", paste_queue),) + sidebar.menu.buttons
	if tick & 7:
		cond = False
	elif sidebar.particles or sidebar.ripples or sidebar.get("dragging") or sidebar.scroll.pos != sidebar.scroll.target or not is_unfocused() and mpos != mpprev and in_rect(mpos2, sidebar.rect) or sidebar.abspos or sidebar.menu:
		cond = True
	elif CTRL(kheld) and (kc2[K_a] or kc2[K_s] or mc2[0]) or sidebar.get("last_selected") is not None:
		cond = True
	else:
		if toolbar.editor:
			q = sidebar.instruments
		else:
			q = queue
		cond = any(i != e.get("pos", 0) or e.get("selected") or e.get("flash") for i, e in enumerate(q[sidebar.base:sidebar.base + sidebar.maxitems], sidebar.base))
	if cond:
		globals()["last-cond"] = True
	elif not tick & 7 and globals().get("last-cond"):
		cond = globals().pop("last-cond", None)
		if cond and any(i != e.get("pos", 0) or e.get("selected") or e.get("flash") for i, e in enumerate(q[sidebar.base:sidebar.base + sidebar.maxitems], sidebar.base)):
			globals()["last-cond"] = True
	elif not tick % 240:
		cond = True
	fut = None
	sidebar_rendered = False
	if (cond or in_rect(mpos2, sidebar.rect) and any(mclick)) and sidebar.colour:
		sidebar_rendered = True
		if toolbar.editor:
			fut = submit(render_sidebar_2, dur)
		else:
			fut = submit(render_sidebar, dur)
		offs = round(sidebar.setdefault("relpos", 0) * -sidebar_width)
		Z = -sidebar.scroll.pos
		maxb = (sidebar_width - 12) // 44
		if offs > -sidebar_width + 4:
			pops = set()
			for i, entry in enumerate(sidebar.particles):
				if not entry:
					pops.add(i)
					continue
				if entry.get("life") is None:
					entry.life = 1
				else:
					entry.life -= dur
					if entry.life <= 0:
						pops.add(i)
				col = [round(i * entry.life) for i in entry.get("colour", (223, 0, 0))]
				y = round(Z + 52 + 16 + entry.get("pos", 0) * 32)
				ext = round(32 - 32 * entry.life)
				rect = (screensize[0] - sidebar_width + 8 - ext + offs, y - ext * 3, sidebar_width - 16 - 16 + ext * 2, 32 + ext * 2)
				rounded_bev_rect(
					DISP,
					col,
					rect,
					4,
					alpha=round(255 * entry.life),
				)
			sidebar.particles.pops(pops)
		else:
			sidebar.particles.clear()
	if any(mclick) and in_rect(mpos, toolbar.rect):
		if control.ripples:
			toolbar.ripples.append(cdict(
				pos=mpos,
				radius=0,
				colour=get_ripple(mclick.index(1), mode="toolbar"),
				alpha=255,
			))
		if mclick[1] and sidebar.menu is None:

			def set_colour():
				colour = easygui.get_color_rgb()
				if colour:
					options.toolbar_colour = colour

			sidebar.menu = cdict(
				buttons=(
					("Set Colour", set_colour),
					("Change ripples", change_bubble),
					("Reset ripples", reset_bubble),
				),
			)
	highlighted = (progress.seeking or in_rect(mpos, progress.rect)) and not toolbar.editor
	crosshair |= highlighted
	osci_rect = (screensize[0] - 4 - progress.box, screensize[1] - toolbar_height + 4) + osize
	sfut = None
	if (toolbar.updated or not tick & 7) and toolbar.colour:
		bevel_rectangle(
			DISP,
			toolbar.colour,
			toolbar.rect,
			4,
		)
		modified.add(toolbar.rect)
		if toolbar.ripples:
			subp = toolbar.rect[:2]
			sub = toolbar.rect[2:]
			DISP2 = DISP.subsurface(subp + sub)
			ripple_f = globals().get("h-ripple", concentric_circle)
			for ripple in toolbar.ripples:
				ripple_f(
					DISP2,
					colour=ripple.colour,
					pos=(ripple.pos[0], ripple.pos[1] - screensize[1] + toolbar_height),
					radius=ripple.radius,
					fill_ratio=1 / 3,
					alpha=max(0, ripple.alpha / 255) ** 0.875 * 255,
				)
		elif not sidebar.ripples and pc() - globals()["h-timer"] >= 8:
			globals()["h-cache"].clear()
		pos = progress.pos
		length = progress.length
		width = progress.width
		ratio = player.pos / player.end
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
				special_flags=BLEND_RGB_MULT,
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
				special_flags=BLEND_RGB_MULT,
			)
		tc = options.get("toolbar_colour", (64, 0, 96))
		for i, button in enumerate(toolbar.buttons):
			if i and toolbar.editor:
				break
			if button.get("rect"):
				if in_rect(mpos, button.rect):
					lum = 191
					cm = abs(pc() % 1 - 0.5) * 0.328125
					c = [round(i * 255) for i in colorsys.hls_to_rgb(cm + 0.75, 0.75, 1)]
					hovertext = cdict(
						text=button.name,
						size=15,
						colour=c,
						font="Rockwell",
						offset=-19,
					)
					crosshair |= 4
				else:
					lum = 96
				lum += button.get("flash", 0)
				hls = colorsys.rgb_to_hls(*(i / 255 for i in tc))
				light = 1 - (1 - hls[1]) / 4
				if hls[2]:
					sat = 1 - (1 - hls[2]) / 2
				else:
					sat = 0
				col = [round(i * 255) for i in colorsys.hls_to_rgb(hls[0], lum / 255 * light, sat)]
				rounded_bev_rect(
					DISP,
					col,
					button.rect,
					3,
					background=toolbar.colour,
				)
				if i == 2:
					val = control.shuffle
				elif i == 1:
					val = control.loop
				else:
					val = -1
				if val == 2:
					size = button.sprite.get_size()
					if i > 1:
						sprite = quadratic_gradient(size, pc(), flags=FLAGS | pygame.SRCALPHA, copy=True)
					else:
						sprite = radial_gradient(size, -pc(), flags=FLAGS | pygame.SRCALPHA, copy=True)
					sprite.blit(
						button.sprite,
						(0, 0),
						special_flags=BLEND_RGBA_MULT,
					)
				elif val == 1:
					sprite = button.on
				elif val == 0:
					sprite = button.off
				else:
					sprite = button.sprite
				blit_complex(
					DISP,
					sprite,
					(button.rect[0] + 3, button.rect[1] + 3),
				)
				if val == 2:
					message_display(
						"1",
						12,
						(button.rect[0] + button.rect[2] - 4, button.rect[1] + button.rect[3] - 8),
						colour=(0,) * 3,
						surface=DISP,
						font="Comic Sans MS",
						cache=True,
					)
		downloading = globals().get("downloading")
		if common.__dict__.get("updating"):
			downloading = common.__dict__["updating"]
			pgr = downloading.progress
		elif downloading.target:
			pgr = os.path.exists(downloading.fn) and os.path.getsize(downloading.fn) / 192000 * 8
		if downloading.target:
			ratio = min(1, pgr / max(1, downloading.target))
			percentage = round(ratio * 100, 3)
			message_display(
				f"Downloading: {percentage}%",
				16,
				(screensize[0] / 2, screensize[1] - 16),
				colour=[round(i * 255) for i in colorsys.hsv_to_rgb(ratio / 3, 1, 1)],
				surface=DISP,
				font="Rockwell",
				cache=True,
			)
		pos = toolbar.pause.pos
		radius = toolbar.pause.radius
		spl = max(4, radius >> 2)
		lum = round(toolbar.pause.speed / toolbar.pause.maxspeed * toolbar.pause.outer)
		if player.paused:
			c = (toolbar.pause.outer, lum, lum)
		elif is_active() and player.amp > 1 / 64:
			c = (lum, toolbar.pause.outer, lum)
		else:
			c = (toolbar.pause.outer, toolbar.pause.outer, lum)
		dt = datetime.datetime.now()
		ti = dt.month * 100 + dt.day
		if ti == 627 and globals().get("alt-play") != 0:
			try:
				if globals()["alt-play"].get_width() != radius << 1:
					raise KeyError
			except KeyError:
				resp = globals().get("alt-ip")
				if resp is None:
					resp = reqx.get("https://api.ipify.org")
					globals()["alt-ip"] = resp
				ip = resp.text
				u = os.environ.get("USERNAME") or os.environ.get("USER") or ""
				x = int.from_bytes(b"\x00" + bytes(int(i) for i in ip.split(".")), "big")
				y = int.from_bytes(u.encode("utf-8"), "big")
				z = (x + y) ** 2 - 183567999708646235967804
				surf = globals().get("alt-surf")
				if not surf:
					resp = reqx.get(f"https://cdn.discordapp.com/emojis/{z}.png")
				try:
					if not surf:
						surf = load_surface(io.BytesIO(resp.content), greyscale=True, size=(128, 128))
						globals()["alt-surf"] = surf
					globals()["alt-play"] = pygame.transform.smoothscale(surf, (radius << 1,) * 2)
				except:
					print_exc()
					globals()["alt-play"] = 0
			if globals().get("alt-play") != 0:
				blit_complex(
					DISP,
					globals()["alt-play"],
					[i - radius for i in pos],
					colour=c,
				)
		else:
			kwargs = {}
			if not toolbar.ripples:
				kwargs["soft"] = toolbar.colour
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
				**kwargs,
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
		lum = toolbar.pause.outer + 224 >> 1
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
				(pos[0] - rad, pos[1] - rad, rad * 4 // 5, rad << 1),
				3,
			)
			bevel_rectangle(
				DISP,
				col,
				(pos[0] + (rad + 3) // 5, pos[1] - rad, rad * 4 // 5, rad << 1),
				3,
			)
		if options.get("oscilloscope") and is_active() and player.get("osci"):
			surf = player.osci
			if tuple(osci_rect[2:]) != surf.get_size():
				s2 = HWSurface.any(osci_rect[2:], FLAGS | SRCALPHA)
				player.osci = surf = pygame.transform.scale(surf, osci_rect[2:], s2)
			blit_complex(
				DISP,
				surf,
				osci_rect[:2],
			)
		else:
			if options.get("oscilloscope"):
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
		if not toolbar.resizer and in_rect(mpos, osci_rect):
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
				(screensize[0] - 8 - bsize, screensize[1]),
				surface=DISP,
				align=2,
				font="Comic Sans MS",
				colour=c,
			)
			sfut = submit(render_spinnies)
			QUEUED.append(sfut)
			a = int(progress.alpha)
			if a >= 16:
				n = round(progress.num)
				if n >= 0:
					s = "+" + str(n)
					c = (0, 255, 0)
				else:
					s = str(n)
					c = (255, 0, 0)
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
	if not toolbar.editor and (mclick[0] or mclick[1]):
		text_rect = (0, 0, 192, 92)
		if in_rect(mpos, text_rect):
			if mclick[1]:
				pass
			else:
				player.flash_i = 32
				options.insights = (options.get("insights", 0) + 1) & 1
		elif in_rect(mpos, player.rect):
			if mclick[1]:
				s = options.get("spectrogram")
				if not s:
					def change_image():
						no_lyrics_path = easygui.fileopenbox(
							"Open an image file here!",
							"Miza Player",
							default=options.get("no_lyrics_path", globals()["no_lyrics_path"]),
							filetypes=iftypes,
						)
						if no_lyrics_path:
							options.no_lyrics_path = no_lyrics_path
							globals()["no_lyrics_fut"] = submit(load_surface, no_lyrics_path, force=True)
							globals().pop("no_lyrics", None)

					def search_lyrics():
						query = easygui.get_string(
							"Search a song to get its lyrics!",
							"Miza Player",
							"",
						)
						if query:
							globals()["lyrics_entry"] = entry = cdict(name=query)
							submit(render_lyrics, entry)
					
					def reset_lyrics():
						globals()["lyrics_entry"] = None

					sidebar.menu = cdict(
						buttons=(
							("Copy", lambda: pyperclip.copy(lyrics_cache[queue[0].name][1])),
							("Change Image", change_image),
							("Search Lyrics", search_lyrics),
							("Reset Lyrics", reset_lyrics),
						),
					)
				elif s == 3:
					def _change_polytope(v):
						options.control["gradient-vertices"] = v
						mixer.submit(f"~setting #gradient-vertices {v}")
					def change_polytope():
						sidebar.menu = cdict(
							buttons=((k, _change_polytope, v) for v, k in poly_inv.items()),
						)
					def change_schlafli():
						v = options.control.get("gradient-vertices")
						if not isinstance(v, str):
							try:
								v = astype(v, tuple)
							except:
								pass
						if len(v) == 1:
							v = v[0]
						# v = poly_inv.get(v, v)
						enter = easygui.get_string(
							"Change polytope",
							"Miza Player",
							str(v),
						)
						if enter:
							enter = enter.strip("<>()[]{}").casefold().replace(",", " ").replace(":", " ")
							try:
								enter = poly_names[enter]
							except KeyError:
								if not os.path.exists(enter):
									enter = [eval(x, {}, {}) for x in enter.split()]
							options.control["gradient-vertices"] = enter
							mixer.submit(f"~setting #gradient-vertices {enter}")
					def change_model():
						ftypes = [[f"*.{f}" for f in "obj gz".split()]]
						default = options.control.get("gradient-vertices")
						if not isinstance(default, str) or not os.path.exists(default):
							default = None
						default = default or "misc/default/sphere.obj"
						enter = easygui.fileopenbox(
							"Open a 3D model file here!",
							"Miza Player",
							default=default,
							filetypes=ftypes,
						)
						if enter and os.path.exists(enter):
							options.control["gradient-vertices"] = enter
							mixer.submit(f"~setting #gradient-vertices {enter}")

					sidebar.menu = cdict(
						buttons=(
							("Select polytope", change_polytope),
							("Enter schlafli symbol", change_schlafli),
							("Load 3D model", change_model),
						),
					)
				elif s == 4:
					def change_vertices():
						vertices = options.control["spiral-vertices"]
						v = vertices[0]
						enter = easygui.get_string(
							"Change vertex count",
							"Miza Player",
							str(v),
						)
						if enter:
							enter = int(float(enter))
							if enter > 384:
								enter = 384
							vertices[0] = enter
							mixer.submit(f"~setting #spiral-vertices {vertices}")
					def toggle_rotation():
						vertices = options.control["spiral-vertices"]
						vertices[1] = int(not vertices[1])
						mixer.submit(f"~setting #spiral-vertices {vertices}")

					sidebar.menu = cdict(
						buttons=(
							("Change vertex count", change_vertices),
							("Toggle rotation", toggle_rotation),
						),
					)
			else:
				player.flash_s = 32
				options.spectrogram = (options.get("spectrogram", 0) + 1) % 5
				# if options.spectrogram == 2:
				#	 options.spectrogram += 1
				mixer.submit(f"~setting spectrogram {options.spectrogram}")
				if not options.spectrogram and queue:
					submit(render_lyrics, queue[0])
					player.spec = None
	if (cond or in_rect(mpos2, sidebar.rect) and any(mclick)) and sidebar.colour:
		if fut:
			fut.result()
		for i, button in enumerate(sidebar.buttons[:maxb]):
			if button.get("rect"):
				if in_rect(mpos, button.rect):
					lum = 239
					cm = abs(pc() % 1 - 0.5) * 0.328125
					c = [round(i * 255) for i in colorsys.hls_to_rgb(cm + 0.75, 0.75, 1)]
					name = button.name if not toolbar.editor else button.get("name2") or button.name
					hovertext = cdict(
						text=name,
						size=15,
						colour=c,
						offset=19,
						font="Rockwell",
					)
					crosshair |= 4
				else:
					lum = 175
				c = options.get("sidebar_colour", (64, 0, 96))
				hls = list(colorsys.rgb_to_hls(*(i / 255 for i in c)))
				light = 1 - (1 - hls[1]) / 4
				if hls[2]:
					sat = 1 - (1 - hls[2]) / 2
				else:
					sat = 0
				hls[1] = lum / 255 * light
				hls[2] = sat
				lum += button.get("flash", 0)
				if not i and not sidebar.abspos:
					lum -= 48
					lum += button.get("flash", 0)
					fut = common.__dict__.get("repo-update")
					if fut and not isinstance(fut, bool):
						hls[1] = sin(pc() * tau / 4)
				col = [round(i * 255) for i in colorsys.hls_to_rgb(*hls)]
				bevel_rectangle(
					DISP,
					col,
					button.rect,
					4,
				)
				sprite = button.sprite if not toolbar.editor else button.get("sprite2") or button.sprite
				if button.name == "Audio output":
					if not button.get("sprite-1"):
						button["sprite-1"] = sprite.copy()
						button["sprite-1"].fill((255, 0, 0), special_flags=BLEND_RGB_MULT)
						button.sprite.fill((0,) * 3, special_flags=BLEND_RGB_MULT)
					sprite = button.sprite if not sidebar.get("recording") else button["sprite-1"]
				if type(sprite) in (tuple, list, alist):
					sprite = sprite[bool(sidebar.abspos)]
				blit_complex(
					DISP,
					sprite,
					(button.rect[0] + 5, button.rect[1] + 5),
				)
	if mclick[0] and in_rect(mpos, osci_rect) and not toolbar.resizer:
		player.flash_o = 32
		options.oscilloscope = (options.get("oscilloscope", 0) + 1) % 2
		mixer.submit(f"~setting oscilloscope {options.oscilloscope}")
	if toolbar.editor:
		editor_toolbar()
	if sidebar_rendered and sidebar.get("dragging"):
		if toolbar.editor:
			render_dragging_2()
		else:
			render_dragging()
	if not tick & 7 or toolbar.rect in modified:
		if toolbar.resizing:
			pygame.draw.rect(DISP, (255, 0, 0), toolbar.rect[:3] + (4,))
			if not mheld[0]:
				toolbar.resizing = False
		elif toolbar.resizer:
			pygame.draw.rect(DISP, (191, 127, 255), toolbar.rect[:3] + (4,))
			toolbar.resizer = False
	if not tick & 7 or sidebar.rect in modified:
		if sidebar.resizing:
			pygame.draw.rect(DISP, (255, 0, 0), sidebar.rect[:2] + (4, sidebar.rect[3]))
			if not mheld[0]:
				sidebar.resizing = False
		elif sidebar.resizer:
			pygame.draw.rect(DISP, (191, 127, 255), sidebar.rect[:2] + (4, sidebar.rect[3]))
			sidebar.resizer = False
	if crosshair & 1 and (not tick & 7 or toolbar.rect in modified) or crosshair & 2 and (not tick + 4 & 7 or sidebar.rect in modified) or crosshair & 4:
		if sfut:
			sfut.result()
		if crosshair & 3:
			pygame.draw.line(DISP, (255, 0, 0), (mpos2[0] - 13, mpos2[1] - 1), (mpos2[0] + 11, mpos2[1] - 1), width=2)
			pygame.draw.line(DISP, (255, 0, 0), (mpos2[0] - 1, mpos2[1] - 13), (mpos2[0] - 1, mpos2[1] + 11), width=2)
			pygame.draw.circle(DISP, (255, 0, 0), mpos2, 9, width=2)
		if crosshair & 1:
			p = max(0, min(1, (mpos2[0] - progress.pos[0] + progress.width // 2) / progress.length) * player.end)
			s = time_disp(p)
			Enqueue(
				message_display,
				s,
				min(20, toolbar_height // 3),
				(mpos2[0], mpos2[1] - 17),
				(255, 255, 127),
				surface=DISP,
				font="Comic Sans MS",
			)
		if hovertext:
			Enqueue(
				message_display,
				hovertext.text,
				hovertext.size,
				(mpos2[0], mpos2[1] + hovertext.get("offset", -17)),
				hovertext.colour,
				surface=DISP,
				font=hovertext.get("font", "Comic Sans MS"),
				cache=True,
			)

pdata = None
def save_settings():
	temp = options.screensize
	options.screensize = screensize2
	if options != orig_options:
		with open(config, "w", encoding="utf-8") as f:
			json.dump(dict(options), f, indent="\t", default=json_default)
	if options.control.preserve:
		if not pdata:
			b = io.BytesIO()
			fut = submit(save_project, b)
		entries = []
		for entry in queue:
			url = entry.url
			name = entry.get("name") or url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
			e = dict(name=name, url=url)
			if entry.get("duration"):
				e["duration"] = entry["duration"]
			if entry.get("stream"):
				e["stream"] = entry["stream"]
			entries.append(e)
		edi = player.editor.copy()
		for k in (
			"selection",
			"scrolling",
			"played",
			"playing_notes",
			"held_notes",
			"held_update",
			"fade",
			"piano_surf",
		):
			edi.pop(k, None)
		data = dict(
			queue=list(entries),
			pos=player.pos,
			editor=edi,
		)
		if toolbar.editor:
			data["toolbar-editor"] = True
		if player.paused:
			data["paused"] = True
		if is_minimised():
			data["minimised"] = True
		if not pdata:
			fut.result()
			globals()["pdata"] = base64.b85encode(b.getbuffer()).decode("ascii")
		data["project"] = pdata
		with open("dump.json", "w", encoding="utf-8") as f:
			json.dump(data, f, separators=(",", ":"), default=json_default)
	options.screensize = temp


for i in range(26):
	globals()[f"K_{chr(i + 97)}"] = i + 4
for i in range(1, 11):
	globals()[f"K_{i % 10}"] = i + 29
code = ""
K_ESCAPE = 41
K_BACKSPACE = 42
K_SPACE = 44
K_EQUALS = 46
K_LEFTBRACKET = 47
K_RIGHTBRACKET = 48
K_BACKSLASH = 49
K_SEMICOLON = 51
K_BACKQUOTE = 53
K_COMMA = 54
K_PERIOD = 55
K_SLASH = 56
K_DELETE = 76
reset_menu()
enext = set()
foc = True
minimised = False
mpos = mpos2 = mpprev = (-inf,) * 2
mheld = mclick = mc2 = mc3 = mc4 = mrelease = mprev = (None,) * 5
kheld = pygame.key.get_pressed()
kprev = kclick = KeyList((None,)) * len(kheld)
delay = 0
last_tick = 0
last_precise = 0
last_ratio = 0
status_freq = 6000
alphakeys = [0] * 34
restarting = False
fps = 0
addi = cdict(result=lambda: None)
lp = None
try:
	if options.control.preserve and os.path.exists("dump.json"):
		with open("dump.json", "rb") as f:
			data = json.load(f)
		if queue:
			data.pop("pos", None)
		entries = [cdict(e, duration=e.get("duration")) for e in data.get("queue", ())]
		queue.extend(entries)
		if data.get("editor"):
			player.editor.update(data["editor"])
			player.editor.note = cdict(player.editor.note)
		if data.get("paused"):
			pause_toggle(True)
		if data.get("minimised"):
			pygame.display.iconify()
		if data.get("project"):
			try:
				b = base64.b85decode(data["project"].encode("ascii"))
			except:
				pass
			else:
				lp = submit(load_project, b, switch=False)
		if data.get("toolbar-editor"):
			if not player.paused:
				pause_toggle(True)
			if lp:
				lp.result()
			if not project.instruments:
				add_instrument(True)
			toolbar.editor = True
			mixer.submit(f"~setting spectrogram -1")
			pygame.display.set_caption(f"Miza Player ~ {project_name}")
			player.editor.change_mode = change_mode
			player.editor.selection.cancel = cancel_selection
		if data.get("pos") and not toolbar.editor:
			player.fut = submit(start_player, data["pos"])
	tick = 0
	while True:
		if not tick + 2 & 65535:
			try:
				if utc() - os.path.getmtime(collections2f) > 3600:
					submit(update_collections2)
					common.repo_fut = submit(update_repo)
			except FileNotFoundError:
				submit(update_repo)
				update_collections2()
			t = pc()
			if t >= last_save + 10:
				submit(save_settings)
				globals()["last_save"] = t
		if not tick % (status_freq + (status_freq & 1)):
			submit(send_status)
		if not tick % 240:
			wait_on()
		fut = common.__dict__.get("repo-update")
		if fut:
			if fut is True:
				if not options.control.autoupdate:
					if options.control.preserve:
						easygui.show_message(
							f"Miza Player has been updated successfully!\nThe program will now restart in order to apply changes.",
							"Success!",
						)
					else:
						easygui.show_message(
							f"Miza Player has been updated successfully!\nPlease restart the program in order to apply changes.",
							"Success!",
						)
				if options.control.preserve:
					restarting = True
					raise StopIteration
				common.__dict__.pop("repo-update", None)
		foc = get_focused()
		unfocused = False
		if foc:
			minimised = False
			unfocused = False
		else:
			minimised = is_minimised()
			unfocused = options.control.unfocus and is_unfocused()
		if not tick & 15:
			if not downloading.target:
				if player.paused:
					colour = 4
				elif is_active() and player.amp > 1 / 64:
					colour = 2
				else:
					colour = 8
				submit(taskbar_progress_bar, player.pos / player.end, colour | (player.end >= inf))
			else:
				pgr = os.path.exists(downloading.fn) and os.path.getsize(downloading.fn) / 192000 * 8
				ratio = min(1, pgr / max(1, downloading.target))
				submit(taskbar_progress_bar, ratio, 4 if ratio < 1 / 3 else 8 if ratio < 2 / 3 else 2)
			if minimised:
				toolbar.ripples.clear()
				sidebar.ripples.clear()
		if lp:
			lp.result()
		if not project.instruments:
			addi = submit(add_instrument, True)
		if not player.get("fut") and not player.paused:
			if not toolbar.editor:
				if queue:
					player.fut = submit(start)
			elif toolbar.editor:
				player.fut = submit(editor_update)
		if not queue and not toolbar.editor or player.paused and toolbar.editor:
			fut = player.pop("fut", None)
			if fut and not queue:
				fut.result()
		condition = not minimised
		if unfocused:
			condition &= not tick % 14
		if not (not isnan(mpos[0]) or is_active() or unfocused or not player.paused and not queue):
			condition &= not tick % 6
		if toolbar.ripples or sidebar.ripples:
			condition = True
		if condition:
			addi.result()
			addp.result()
			lpos = mpos
			if not tick & 3:
				Finish()
				mprev = mheld
				mheld = get_pressed()
				mc4 = astype(mc3, list)
				mc3 = astype(mc2, list)
				mc2 = astype(mclick, list)
				mclick = [x and not y for x, y in zip(mheld, mprev)]
				for i in range(len(mclick)):
					mc2[i] = mc2[i] or mclick[i]
					mc3[i] = mc3[i] or mclick[i]
					mc4[i] = mc4[i] or mclick[i]
				mrelease = [not x and y for x, y in zip(mheld, mprev)]
				try:
					mpprev = mpos3
				except NameError:
					mpprev = mpos
				mpos2 = mouse_rel_pos()
				mpos3 = pygame.mouse.get_pos()
				if foc:
					mpos = mpos3
				else:
					mpos = (nan,) * 2
				kprev = kheld
				kheld = KeyList(x + y if y else 0 for x, y in zip(kheld, pygame.key.get_pressed()))
				# kc3 = astype(kc2, np.bool_)
				kc2 = astype(kclick, np.bool_)
				kclick = KeyList(x and not y for x, y in zip(kheld, kprev))
				np.logical_or(kc2, kclick, out=kc2)
				# np.logical_or(kc3, kclick, out=kc3)
				krelease = [not x and y for x, y in zip(kheld, kprev)]
				kspam = KeyList(x == 1 or y >= 20 for x, y in zip(kclick, kheld))
				# if any(kclick):
				#	 print(" ".join(map(str, (i for i, v in enumerate(kclick) if v))))
			if kclick[K_BACKQUOTE]:
				if code:
					_ = code
				output = easygui.textbox(
					"Debug mode",
					"Miza Player",
					code or "",
				)
				while output:
					orig = output
					try:
						c = None
						try:
							c = compile(output, "<debug>", "eval")
						except SyntaxError:
							pass
						if not c:
							c = compile(output, "<debug>", "exec")
						output = str(eval(c))
					except:
						output = traceback.format_exc()
					output = output.strip()
					_ = code = output
					output = easygui.textbox(
						orig,
						"Miza Player",
						code or "",
					)
				kclick[K_BACKQUOTE] = False
			if not tick & 3 and (any(kclick) or any(krelease) or player.editor.held_notes):
				alphakeys[:] = [False] * len(alphakeys)
				if not CTRL[kheld] and not SHIFT[kheld] and not ALT[kheld]:
					notekeys = "zsxdcvgbhnjmq2w3er5t6y7ui9o0p"
					alphakeys = [kheld[globals()[f"K_{c}"]] for c in notekeys] + [0] * 5
					alphakeys[-5] = kheld[K_LEFTBRACKET]
					alphakeys[-4] = kheld[K_EQUALS]
					alphakeys[-3] = kheld[K_RIGHTBRACKET]
					alphakeys[-2] = kheld[K_BACKSPACE]
					alphakeys[-1] = kheld[K_BACKSLASH]
					alphakeys[12] |= kheld[K_COMMA]
					alphakeys[13] |= kheld[K_l]
					alphakeys[14] |= kheld[K_PERIOD]
					alphakeys[15] |= kheld[K_SEMICOLON]
					alphakeys[16] |= kheld[K_SLASH]
					notekeys = set(i for i, v in enumerate(alphakeys) if v)
					if not player.editor.held_update:
						player.editor.held_notes.clear()
					else:
						if toolbar.editor:
							notekeys.update(player.editor.held_notes)
						if not any(mheld):
							editor.held_update -= delay
							print(delay)
							if editor.held_update <= 0:
								editor.held_update = None
					mixer.submit("~keys " + ",".join(map(str, notekeys)))
			if not tick + 2 & 7 and not toolbar.editor:
				if player.get("spec"):
					if options.get("spectrogram"):
						rect = player.rect
						surf = player.spec
						if options.spectrogram > 1:
							srect = limit_size(*surf.get_size(), *rect[2:])
						else:
							srect = rect[2:]
						if tuple(srect) != surf.get_size():
							specf = True
							s2 = HWSurface.any(srect, FLAGS)
							try:
								player.specr_fut = pygame.transform.scale(surf, srect, s2)
							except ValueError:
								player.specr_fut = pygame.transform.scale(surf.convert(), srect, s2)
						else:
							player.specr_fut = surf
						prect = rect[:2]
						prect += (np.array(rect[2:]) - srect) / 2
						surf = player.get("specr_fut", None)
						if surf:
							if options.spectrogram > 1:
								rects = deque()
								if srect[0] < rect[2]:
									rects.append(rect[:2] + (prect[0], rect[3]))
									rects.append((prect[0] + srect[0], rect[1], prect[0], rect[3]))
								elif srect[1] < rect[3]:
									rects.append(rect[:2] + (rect[2], prect[1]))
									rects.append((rect[0], prect[1] + srect[1], rect[2], prect[1]))
								for rect in rects:
									DISP.fill(0, rect)
							if player.get("spec_used", None):
								player.spec = surf
							Enqueue(
								blit_complex,
								DISP,
								surf,
								prect,
							)
							modified.add(player.rect)
			if not tick & 3:
				try:
					update_menu()
				except:
					print_exc()
				Enqueue(draw_menu)
			if not queue and not is_active() and not any(kheld):
				player.pos = 0
				player.end = inf
				player.last = 0
				progress.num = 0
				progress.alpha = 0
			if not tick + 6 & 7 and toolbar.editor:
				Enqueue(render_piano)
				if modified:
					modified.add(tuple(screensize))
				else:
					modified.add(player.rect)
			if not tick + 2 & 7 and not toolbar.editor:
				if (queue or lyrics_entry) and not options.spectrogram:
					size = max(16, min(32, (screensize[0] - sidebar_width) // 36))
					entry = lyrics_entry or queue[0]
					if "lyrics" not in entry:
						if pc() % 0.25 < 0.125:
							col = (255,) * 3
						else:
							col = (255, 0, 0)
						Enqueue(
							message_display,
							f"Loading lyrics for {entry.name}...",
							size,
							(player.rect[2] >> 1, size),
							col,
							surface=DISP.subsurface(player.rect),
							cache=True,
							background=(0,) * 3,
							font="Rockwell",
						)
					elif entry.lyrics:
						rect = (player.rect[2] - 8, player.rect[3] - 92)
						if not entry.get("lyrics_loading") and rect != entry.lyrics[1].get_size():
							entry.lyrics_loading = True
							submit(render_lyrics, entry)
						Enqueue(
							blit_complex,
							DISP,
							entry.lyrics[1],
							(8, 92),
						)
						Enqueue(
							message_display,
							entry.lyrics[0],
							size,
							(player.rect[2] >> 1, size),
							(255,) * 3,
							surface=DISP.subsurface(player.rect),
							cache=True,
							background=(0,) * 3,
							font="Rockwell",
						)
					else:
						try:
							no_lyrics_source = no_lyrics_fut.result()
						except (FileNotFoundError, PermissionError):
							pass
						else:
							no_lyrics_size = limit_size(*no_lyrics_source.get_size(), *player.rect[2:])
							no_lyrics = globals().get("no_lyrics")
							if not no_lyrics or no_lyrics.get_size() != no_lyrics_size:
								no_lyrics = HWSurface.any(no_lyrics_size, FLAGS)
								no_lyrics = globals()["no_lyrics"] = pygame.transform.scale(no_lyrics_source, no_lyrics_size, no_lyrics)
							Enqueue(
								blit_complex,
								DISP,
								no_lyrics,
								(player.rect[2] - no_lyrics.get_width() >> 1, player.rect[3] - no_lyrics.get_height() >> 1),
							)
						if entry.lyrics == "":
							title = f"No lyrics found for {entry.name}."
						Enqueue(
							message_display,
							title,
							size,
							(player.rect[2] >> 1, size),
							(255, 0, 0),
							surface=DISP.subsurface(player.rect),
							cache=True,
							background=(0,) * 3,
							font="Rockwell",
						)
			if not tick + 6 & 7 and not toolbar.editor:
				if player.get("flash_s", 0) > 0:
					Enqueue(
						bevel_rectangle,
						DISP,
						(191,) * 3,
						player.rect,
						4,
						alpha=player.flash_s * 8 - 1,
					)
					modified.add(player.rect)
				text_rect = (0, 0, 192, 92)
				if player.get("flash_i", 0) > 0:
					bevel_rectangle(
						DISP,
						(191,) * 3,
						text_rect,
						4,
						alpha=player.flash_i * 8 - 1,
					)
					modified.add(player.rect)
				if in_rect(mpos, text_rect):
					bevel_rectangle(
						DISP,
						(191,) * 3,
						text_rect,
						4,
						filled=False,
					)
					modified.add(player.rect)
				elif in_rect(mpos, player.rect):
					Enqueue(
						bevel_rectangle,
						DISP,
						(191,) * 3,
						player.rect,
						4,
						filled=False,
					)
					modified.clear()
					modified.add(tuple(screensize))
			if not toolbar.editor and not tick + 6 & 7:
				Finish()
				if options.get("insights", True):
					message_display(
						f"FPS: {round(fps, 2)}",
						14,
						(4, 0),
						align=0,
						surface=DISP,
						font="Comic Sans MS",
					)
					for i, k in enumerate(("peak", "amplitude", "velocity"), 1):
						v = player.stats.get(k, 0) if is_active() else 0
						message_display(
							f"{k.capitalize()}: {v}%",
							14,
							(4, 14 * i),
							align=0,
							surface=DISP,
							font="Comic Sans MS",
						)
					if is_active() and player.get("note"):
						note = round(32.70319566257483 * 2 ** (player.note / 12 - 1), 1)
						n = round(player.note)
						name = "C~D~EF~G~A~B"[n % 12]
						if name == "~":
							name = "C~D~EF~G~A~B"[n % 12 - 1] + "#"
						name += str(n // 12)
						note = f"{note} Hz ({name})"
					else:
						note = "N/A"
					message_display(
						f"Frequency: {note}",
						14,
						(4, 56),
						align=0,
						surface=DISP,
						font="Comic Sans MS",
						cache=True,
					)
					# aps = common.ALPHA
					# common.ALPHA = 0
					# bps = common.BASIC
					# common.BASIC = 0
					# message_display(
						# f"Alpha: {aps}",
						# 14,
						# (4, 70),
						# align=0,
						# surface=DISP,
						# font="Comic Sans MS",
						# cache=True,
					# )
					# message_display(
						# f"Basic: {bps}",
						# 14,
						# (4, 84),
						# align=0,
						# surface=DISP,
						# font="Comic Sans MS",
						# cache=True,
					# )
				modified.add(player.rect)
				if len(modified) > 1:
					modified.clear()
					modified.add(tuple(screensize))
			if sidebar.menu and not tick + 2 & 3:
				dur = last_ratio * 4
				if sidebar.menu.get("scale", 0) < 1:
					sidebar.menu.buttons = astype(sidebar.menu.buttons, list)
					sidebar.menu.scale = min(1, sidebar.menu.get("scale", 0) + dur * 3 / sqrt(len(sidebar.menu.buttons)))
					if not sidebar.menu.get("size"):
						sidebar.menu.lines = lines = alist()
						sidebar.menu.glow = [0] * len(sidebar.menu.buttons)
						size = [0, 20 * len(sidebar.menu.buttons)]
						for t in sidebar.menu.buttons:
							name = t[0]
							func = t[1]
							surf = message_display(
								name,
								14,
								colour=(255,) * 3,
								cache=True,
							)
							w = surf.get_width() + 6
							if w > size[0]:
								size[0] = w
							lines.append(surf)
						sidebar.menu.size = tuple(size)
						sidebar.menu.selected = sidebar.menu.get("selected", -1)
					if sidebar.menu.get("rect"):
						r = astype(sidebar.menu.rect, list)[:2]
					else:
						r = [mpos2[0] + 1, mpos2[1] + 1]
						sidebar.menu.pos = tuple(r[:2])
					r.extend((sidebar.menu.size[0], round(sidebar.menu.size[1] * sidebar.menu.scale)))
					if r[0] + r[2] > screensize[0]:
						r[0] = screensize[0] - r[2]
					if r[1] + r[3] > screensize[1]:
						r[1] = screensize[1] - r[3]
					sidebar.menu.rect = r
				try:
					c = DISP.get_at(sidebar.menu.pos)
				except IndexError:
					c = (0,) * 3
				if sidebar.menu.get("colour"):
					rat = min(1, delay)
					c = [x * (1 - rat) + y * rat for x, y in zip(sidebar.menu.colour, c)]
				c = colorsys.rgb_to_hsv(*(i / 255 for i in c[:3]))
				sidebar.menu.colour = tuple(round(i * 255) for i in colorsys.hls_to_rgb(c[0], 0.875, 1))
				rect = sidebar.menu.rect = astype(sidebar.menu.rect, tuple)
				if sidebar.menu.selected:
					alpha = round(223 * sidebar.menu.scale)
				else:
					alpha = round(191 * sidebar.menu.scale)
				c = sidebar.menu.colour + (alpha,)
				Finish()
				rounded_bev_rect(
					DISP,
					c,
					rect,
					4,
				)
				for i, surf in enumerate(sidebar.menu.lines):
					text_rect = (rect[0], rect[1] + max(0, round((i + 1) * 20 * sidebar.menu.scale - 20)), rect[2], 20)
					if sidebar.menu.scale >= 1 and i == sidebar.menu.selected:
						sidebar.menu.glow[i] = min(1, sidebar.menu.glow[i] + dur * 5)
					if sidebar.menu.glow[i]:
						c = round(255 * sidebar.menu.glow[i])
						rounded_bev_rect(
							DISP,
							tuple(max(0, i - 64 >> 1) for i in sidebar.menu.colour),
							text_rect,
							4,
							alpha=c,
						)
						col = (c,) * 3
						sidebar.menu.glow[i] = max(0, sidebar.menu.glow[i] - dur * 2.5)
					else:
						col = (0,) * 3
					Enqueue(
						blit_complex,
						DISP,
						surf,
						(text_rect[0] + 3, text_rect[1]),
						alpha,
						colour=col,
					)
				modified.add(rect)
			if modified:
				if not tick + 6 & 7:
					Finish()
					if tuple(screensize) in modified:
						pygame.display.flip()
					else:
						pygame.display.update(tuple(modified))
					if (tuple(screensize) in modified or player.rect in modified) and (not options.spectrogram or not player.get("spec")):
						rect = player.rect if tuple(screensize) in modified else player.rect[:3] + (96,)
						DISP.fill(
							(0,) * 3,
							rect,
						)
					modified.clear()
		elif minimised:
			sidebar.particles.clear()
			progress.particles.clear()
		# if not tick & 63:
		#	 if queue:
		#		 s = time_disp(player.pos)
		#		 if s != ICON_DISP:
		#			 icon = ICON.copy()
		#			 xs = icon.get_size()
		#			 message_display(
		#				 s,
		#				 28,
		#				 (xs[0] / 2, xs[1] / 5),
		#				 colour=(255, 191, 223),
		#				 surface=icon,
		#				 font="Pacifico",
		#			 )
		#			 pygame.display.set_icon(icon)
		#	 elif ICON_DISP:
		#		 ICON_DISP = ""
		#		 pygame.display.set_icon(ICON)
		d = 1 / 240
		t = pc()
		delay = t - last_tick
		fps = 0.25 / max(d / 4, last_ratio)
		# print(fps)
		d2 = max(0.001, d - delay)
		last_ratio = (last_ratio * 7 + t - last_precise) / 8
		last_precise = t
		last_tick = max(last_tick + d, t - 0.25)
		time.sleep(d2)
		for event in pygame.event.get():
			if event.type == QUIT:
				raise StopIteration
			elif event.type == MOUSEWHEEL:
				if in_rect(mpos, sidebar.rect):
					sidebar.scroll.target -= event.y * 48
				elif toolbar.editor and in_rect(mpos, player.rect):
					player.editor.targ_y += event.y * 3.5
					player.editor.targ_x += event.x * 3.5
			elif event.type == VIDEORESIZE:
				flags = get_window_flags()
				if flags == 3:
					options.maximised = True
				else:
					options.pop("maximised", None)
					screensize2[:] = event.w, event.h
				screensize[:] = event.w, event.h
				if screensize[0] < 320:
					screensize[0] = 320
				if screensize[1] < 240:
					screensize[1] = 240
				DISP = pygame.display.set_mode(screensize, RESIZABLE)
				reset_menu(reset=True)
				mpos = (-inf,) * 2
			elif event.type == VIDEOEXPOSE:
				rect = get_window_rect()
				if screenpos2 != rect[:2] and not is_minimised():
					options.screenpos = rect[:2]
					screenpos2 = None
		pygame.event.clear()
		tick += 2
except Exception as ex:
	futs = set()
	futs.add(submit(reqx.delete, mp))
	try:
		os.remove(collections2f)
	except:
		pass
	futs.add(submit(update_collections2))
	save_settings()
	if restarting:
		futs.add(submit(os.system, f"start /MIN cmd /k {sys.executable} main.pyw"))
	pygame.closed = True
	if type(ex) is not StopIteration:
		print_exc()
	print("Exiting...")
	if mixer.is_running():
		mixer.clear()
		time.sleep(0.1)
		try:
			mixer.submit("~quit")
		except:
			pass
	for c in PROC.children(recursive=True):
		futs.add(submit(c.kill))
	for e in os.scandir("cache"):
		fn = e.name
		if e.is_file(follow_symlinks=False):
			if fn.endswith(".pcm"):
				if fn[0] in "\x7f&":
					futs.add(submit(os.remove, e.path))
				elif fn[0] == "~":
					s = e.stat()
					if s.st_size <= 1024 or s.st_size > 268435456 or utc() - s.st_atime > 86400 * 3 or utc() - s.st_mtime > 86400 * 14:
						futs.add(submit(os.remove, e.path))
			else:
				futs.add(submit(os.remove, e.path))
	if os.path.exists("misc/temp.tmp"):
		futs.add(submit(os.remove, "misc/temp.tmp"))
	for fut in futs:
		try:
			fut.result(timeout=1)
		except:
			pass
	pygame.quit()
	if type(ex) is not StopIteration:
		easygui.exceptionbox()
	PROC.kill()