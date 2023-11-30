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
if not os.path.exists("persistent"):
	os.mkdir("persistent")
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
	offpos=-inf,
	extpos=lambda: pc() + player.offpos if player.offpos > -inf or is_active() else player.pos,
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
	shuffler=2147483647,
	previous=None,
	video=None,
	sprite=None,
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
		pos=None,
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
	globals()["pencilb"] = pencilw.copy()
	pencilb.fill((0, 0, 0), special_flags=BLEND_RGB_MULT)
addp = submit(load_pencil)

has_api = False
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

def playlist_sync():
	# print("Syncing playlists...")
	if utc() - has_api >= 60:
		return
	try:
		try:
			t1 = max(max(st.st_mtime, st.st_ctime) for f in os.scandir("playlists") for st in (f.stat(),))
		except ValueError:
			t1 = 0
		if control.playlist_sync and is_url(control.playlist_sync):
			url = control.playlist_sync.replace("/p/", "/fileinfo/")
			try:
				resp = requests.get(url)
				resp.raise_for_status()
			except requests.exceptions.HTTPError as ex:
				print_exc()
				if ex.args:
					err = ex.args[0]
					if isinstance(err, str):
						err = int(err.split(None, 1)[0])
					if err in (401, 403, 404):
						control.playlist_sync = ""
					else:
						globals()["last_sync"] = time.time() + 1800
						return
				else:
					globals()["last_sync"] = time.time() + 1800
					return
			except:
				globals()["last_sync"] = time.time() + 1800
				print_exc()
			else:
				try:
					info = cdict(resp.json())
				except:
					print_exc()
					control.playlist_sync
				else:
					t2 = info.timestamp
					# print(control.playlist_sync, t1, t2)
					if t2 >= t1 + 20 and info.size != control.playlist_size:
						print(f"Downloading playlists from {url}...")
						resp = requests.get(control.playlist_sync.replace("/p/", "/d/"))
						b = io.BytesIO(resp.content)
						with zipfile.ZipFile(b) as z:
							for fn in os.listdir("playlists"):
								os.remove("playlists/" + fn)
							z.extractall()
						control.playlist_files = len(os.listdir("playlists"))
						control.playlist_size = len(resp.content)
						print("Extracted to playlists folder")
						return
					if control.playlist_files == len(os.listdir("playlists")):
						# print("Playlists match, skipping...")
						return
		if not os.listdir("playlists"):
			return
		print("Uploading playlists...")
		b = io.BytesIO()
		with zipfile.ZipFile(b, mode="w", compression=zipfile.ZIP_STORED) as z:
			for fn in os.listdir("playlists"):
				z.write("playlists/" + fn)
		b.seek(0)
		resp = requests.post(
			"https://api.mizabot.xyz/upload_chunk",
			headers={"X-File-Name": "playlists.zip", "X-File-Size": str(b.getbuffer().nbytes)},
			data=b,
		)
		resp.raise_for_status()
		url = "https://api.mizabot.xyz/merge"
		if is_url(control.playlist_sync) and "?key=" in control.playlist_sync:
			url = url.rsplit("/", 1)[0] + "/edit/" + control.playlist_sync.split("/p/", 1)[-1]
		resp = requests.patch(
			url,
			data={"x-file-name": "playlists.zip", "index": "0"},
		)
		resp.raise_for_status()
		url = "https://api.mizabot.xyz" + resp.text
		control.playlist_sync = url
		control.playlist_files = len(os.listdir("playlists"))
		control.playlist_size = b.getbuffer().nbytes
		print("Uploaded to", url)
	except:
		globals()["last_sync"] = time.time() + 1800
		print_exc()


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
			name="Settings (ESC)",
			sprite=(gears, notes),
			click=(settings_toggle, settings_menu),
			hotkey=(K_ESCAPE,),
		))
		reset_menu(full=False)
		folder = button_images.folder.result()
		def enqueue_local():
			default = None
			if options.get("path"):
				default = options.path.rstrip("/") + "/"
			def enqueue_local_a(files):
				if files:
					submit(_enqueue_local, *files, index=sidebar.get("lastsel"))
			easygui2.fileopenbox(
				enqueue_local_a,
				"Open an audio or video file here!",
				"Miza Player",
				default=default,
				filetypes=ftypes,
				multiple=True,
			)
		def enqueue_menu():
			def enqueue_folder():
				default = None
				if options.get("path"):
					default = options.path.rstrip("/") + "/"
				def enqueue_folder_a(files):
					if files:
						submit(
							_enqueue_local,
							*(files + "/" + f for f in os.listdir(files)),
							probe=False,
							index=sidebar.get("lastsel"),
						)
				easygui2.diropenbox(
					enqueue_folder_a,
					"Open an audio or video file here!",
					"Miza Player",
					default=default,
				)
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
			def load_project_file_a(file):
				if file:
					submit(load_project, file)
			easygui2.fileopenbox(
				load_project_file_a,
				"Open a music project file here!",
				"Miza Player",
				default=default,
				filetypes=(".mpp",),
				multiple=False,
			)
		sidebar.buttons.append(cdict(
			name="Open (CTRL+O)",
			sprite=folder,
			click=(enqueue_local, enqueue_menu),
			click2=load_project_file,
			hotkey=(CTRL, K_o),
		))
		reset_menu(full=False)
		hyperlink = button_images.hyperlink.result()
		plus = button_images.plus.result()
		def enqueue_search(q=""):
			def enqueue_search_a(query):
				def enqueue_search_b(query):
					print(query)
					if query and query in easygui2.tmap:
						url = easygui2.tmap[query]
						submit(_enqueue_search, url, index=sidebar.get("lastsel"))
				if query:
					if not is_url(query):
						if not q:
							ytdl = downloader.result()
							try:
								q2 = "ytsearch:" + query.replace(":", "-")
								entries = ytdl.search(q2, count=20)
							except:
								print_exc()
								sidebar.particles.append(cdict(eparticle))
								return
							matches = [fuzzy_substring(query.upper(), e.name.upper()) for e in entries]
							M = max(matches)
							print(matches)
							if M < 0.5:
								futs = []
								for mode in ("sc",):
									q2 = query.replace(":", "-")
									fut = submit(ytdl.search, q2, mode=mode, count=12)
									futs.append(fut)
								for fut in futs:
									res = fut.result()
									# print(fut, res)
									entries.extend(res)
							if len(entries) > 1:
								nchoices = {e.name: e.url for e in entries}
								easygui2.tmap = nchoices
								cnames = sorted(nchoices, key=lambda n: fuzzy_substring(query.upper(), n.upper()), reverse=True)[:12]
								if len(cnames) > 1:
									easygui2.choicebox(
										enqueue_search_b,
										"Select a search result here!",
										title="Miza Player",
										choices=cnames,
									)
									return
							query = entries[0].url
						elif q:
							query = q + "search:" + query.replace(":", "-")
					submit(_enqueue_search, query, index=sidebar.get("lastsel"))
			easygui2.enterbox(
				enqueue_search_a,
				"Search for one or more songs online!",
				title="Miza Player",
				default="",
			)
		def select_search():
			sidebar.menu = cdict(
				buttons=(
					("YouTube", enqueue_search, "yt"),
					("SoundCloud", enqueue_search, "sc"),
					("Spotify", enqueue_search, "sp"),
					("BandCamp", enqueue_search, "bc"),
				),
			)
		sidebar.buttons.append(cdict(
			name="Search (CTRL+F)",
			name2="New instrument (CTRL+F)",
			sprite=hyperlink,
			sprite2=plus,
			click=(enqueue_search, select_search),
			click2=add_instrument,
			hotkey=(CTRL, K_f),
		))
		reset_menu(full=False)
		playlist = button_images.playlist.result()
		waves = button_images.waves.result()
		def get_playlist():
			items = deque()
			for item in (item for item in os.listdir("playlists") if item.endswith(".json") or item.endswith(".zip")):
				u = unquote(item.rsplit(".", 1)[0])
				fn = "playlists/" + quote(u)[:244] + "." + item.rsplit(".", 1)[-1]
				fn2 = "playlists/" + item
				if fn != fn2 or not os.path.exists(fn):
					os.rename(fn2, fn)
				items.append(u)
			if not items:
				return easygui2.msgbox(
					None,
					"Right click this button to create, edit, or remove a playlist!",
					title="Playlist folder is empty.",
				)
			def get_playlist_a(choice):
				if choice:
					sidebar.loading = True
					start = len(queue)
					fn = "playlists/" + quote(choice)[:244] + ".zip"
					if not os.path.exists(fn):
						fn = fn[:-4] + ".json"
					if os.path.exists(fn) and os.path.getsize(fn):
						fi = fn
					else:
						fi = "playlists/" + [item for item in os.listdir("playlists") if (item.endswith(".json") or item.endswith(".zip")) and unquote(item.rsplit(".", 1)[0]) == choice][0]
					with open(fi, "rb") as f:
						if zipfile.is_zipfile(f):
							f.seek(0)
							data = orjson.loads(zip2bytes(f.read()))
						else:
							f.seek(0)
							data = json.load(f)
					ytdl = downloader.result()
					for e in data.get("queue", ()):
						if e.get("url"):
							url = e["url"]
							if url not in ytdl.searched:
								ytdl.searched[url] = cdict(t=time.time(), data=[astype(cdict, e)])
					q = data.get("queue", ())
					options.history.appendleft((choice, tuple(e["url"] for e in q)))
					options.history = options.history.uniq(sort=False)[:64]
					entries = [ensure_duration(cdict(**e, pos=start)) for e in q]
					queue.extend(entries)
					index = sidebar.get("lastsel")
					if control.shuffle and len(queue) > 1:
						queue[bool(start):].shuffle()
					elif index is not None:
						temp = list(queue[start:])
						queue[index + len(temp):] = queue[index:-len(temp)]
						queue[index:index + len(temp)] = temp
						if index < 1:
							submit(enqueue, queue[0])
					sidebar.loading = False
			easygui2.choicebox(
				get_playlist_a,
				"Select a locally saved playlist here!",
				title="Miza Player",
				choices=items,
			)
		def playlist_menu():
			def create_playlist():
				def create_playlist_a(text):
					text = (text or "").strip()
					if text:
						urls = text.splitlines()
						entries = deque()
						futs = deque()
						futm = {}
						for url in urls:
							if url:
								name = duration = None
								ytdl = downloader.result()
								if url in ytdl.searched:
									resp = ytdl.searched[url].data
									if len(resp) == 1:
										name = resp[0].get("name")
										duration = resp[0].get("duration")
								else:
									if len(futs) >= 8:
										try:
											futs.popleft().result()
										except:
											print_exc()
									fut = submit(ytdl.search, url)
									futs.append(fut)
									futm[len(entries)] = fut
								if not name:
									name = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
								entries.append(dict(name=name, url=url))
								if duration:
									entries[-1]["duration"] = duration
						if entries:
							for k, v in futm.items():
								try:
									entry = v.result()[0]
								except:
									print_exc()
								else:
									entries[k]["name"] = entry["name"]
									if entry.get("duration"):
										entries[k]["duration"] = entry["duration"]
									entries[k]["url"] = entry["url"]
							entries = list(entries)
							ytdl = downloader.result()
							# url = entries[0]["url"]
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
								name = entries[0].get("name") or name
							if len(entries) > 1:
								name += f" +{len(entries) - 1}"
							def create_playlist_b(text):
								text = (text or "").strip()
								if text:
									data = dict(queue=entries, stats={})
									fn = "playlists/" + quote(text)[:244] + ".json"
									if len(entries) > 1024:
										fn = fn[:-5] + ".zip"
										b = bytes2zip(orjson.dumps(data))
										with open(fn, "wb") as f:
											f.write(b)
									else:
										with open(fn, "w", encoding="utf-8") as f:
											json.dump(data, f, separators=(",", ":"))
									submit(
										easygui2.msgbox,
										None,
										f"Playlist {json.dumps(text)} with {len(entries)} item{'s' if len(entries) != 1 else ''} has been added!",
										title="Success!",
									)
							submit(
								easygui2.enterbox,
								create_playlist_b,
								"Enter a name for your new playlist!",
								title="Miza Player",
								default=name,
							)
				easygui2.textbox(
					create_playlist_a,
					"Enter a list of URLs or file paths to include in the playlist!",
					"Miza Player",
				)
			def edit_playlist():
				items = [unquote(item.rsplit(".", 1)[0]) for item in os.listdir("playlists") if item.endswith(".json") or item.endswith(".zip")]
				if not items:
					return easygui2.msgbox(
						None,
						"Right click this button to create, edit, or remove a playlist!",
						title="Playlist folder is empty.",
					)
				def edit_playlist_a(choice):
					if choice:
						fn = "playlists/" + quote(choice)[:244] + ".zip"
						if not os.path.exists(fn):
							fn = fn[:-4] + ".json"
						if os.path.exists(fn) and os.path.getsize(fn):
							fi = fn
						else:
							fi = "playlists/" + [item for item in os.listdir("playlists") if (item.endswith(".json") or item.endswith(".zip")) and unquote(item.rsplit(".", 1)[0]) == choice][0]
						with open(fi, "rb") as f:
							if zipfile.is_zipfile(f):
								f.seek(0)
								data = orjson.loads(zip2bytes(f.read()))
							else:
								f.seek(0)
								data = json.load(f)
						ytdl = downloader.result()
						for e in data.get("queue", ()):
							if e.get("url"):
								url = e["url"]
								if url not in ytdl.searched:
									ytdl.searched[url] = cdict(t=time.time(), data=[astype(cdict, e)])
						s = "\n".join(e["url"] for e in data.get("queue", ()) if e.get("url"))
						def edit_playlist_b(text):
							if text is not None:
								if not text:
									os.remove(fi)
									submit(
										easygui2.msgbox,
										None,
										f"Playlist {json.dumps(choice)} has been removed!",
										title="Success!",
									)
								else:
									if fi != fn:
										os.remove(fi)
									urls = text.splitlines()
									entries = deque()
									futs = deque()
									futm = {}
									for url in urls:
										if url:
											name = duration = None
											if url in ytdl.searched:
												resp = ytdl.searched[url].data
												if len(resp) == 1:
													name = resp[0].get("name")
													duration = resp[0].get("duration")
											else:
												if len(futs) >= 8:
													try:
														futs.popleft().result()
													except:
														print_exc()
												fut = submit(ytdl.search, url)
												futs.append(fut)
												futm[len(entries)] = fut
											if not name:
												name = url.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
											entries.append(dict(name=name, url=url))
											if duration:
												entries[-1]["duration"] = duration
									if entries:
										for k, v in futm.items():
											try:
												entry = v.result()[0]
											except:
												print_exc()
											else:
												entries[k]["name"] = entry["name"]
												if entry.get("duration"):
													entries[k]["duration"] = entry["duration"]
												# entries[k]["url"] = entry["url"]
										entries = list(entries)
										data = dict(queue=entries, stats={})
										out = "playlists/" + quote(choice)[:244] + ".json"
										if len(entries) > 1024:
											out = out[:-5] + ".zip"
											b = bytes2zip(orjson.dumps(data))
											with open(out, "wb") as f:
												f.write(b)
										else:
											with open(out, "w", encoding="utf-8") as f:
												json.dump(data, f, separators=(",", ":"))
										submit(
											easygui2.msgbox,
											None,
											f"Playlist {json.dumps(choice)} has been updated!",
											title="Success!",
										)
						submit(
							easygui2.textbox,
							edit_playlist_b,
							"Enter a list of URLs or file paths to include in the playlist!",
							title="Miza Player",
							text=s,
						)
				easygui2.choicebox(
					edit_playlist_a,
					"Select a playlist to edit",
					title="Miza Player",
					choices=items,
				)
			def delete_playlist():
				items = [unquote(item.rsplit(".", 1)[0]) for item in os.listdir("playlists") if item.endswith(".json") or item.endswith(".zip")]
				if not items:
					return easygui2.msgbox(
						None,
						"Right click this button to create, edit, or remove a playlist!",
						title="Playlist folder is empty.",
					)
				def delete_playlist_a(choice):
					if choice:
						fn = "playlists/" + quote(choice)[:244] + ".zip"
						if not os.path.exists(fn):
							fn = fn[:-4] + ".json"
						if os.path.exists(fn) and os.path.getsize(fn):
							fi = fn
						else:
							fi = "playlists/" + [item for item in os.listdir("playlists") if (item.endswith(".json") or item.endswith(".zip")) and unquote(item.rsplit(".", 1)[0]) == choice][0]
						os.remove(fi)
						submit(
							easygui2.msgbox,
							None,
							f"Playlist {json.dumps(choice)} has been removed!",
							title="Success!",
						)
				easygui2.choicebox(
					delete_playlist_a,
					"Select a playlist to delete",
					title="Miza Player",
					choices=items,
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
			name="Playlist (CTRL+P)",
			name2="Audio Clip (CTRL+P)",
			sprite=playlist,
			sprite2=waves,
			click=(get_playlist, playlist_menu),
			click2=waves_1,
			hotkey=(CTRL, K_p),
		))
		reset_menu(full=False)
		history = button_images.history.result()
		def player_history():
			f = f"%0{len(str(len(options.history)))}d"
			choices = [f % i + ": " + e[0] for i, e in enumerate(options.history)]
			if not choices:
				return easygui2.msgbox(
					None,
					"Play some music to fill up this menu!",
					title="Player history is empty.",
				)
			def player_history_a(selected):
				if selected:
					entry = options.history.pop(int(selected.split(":", 1)[0]))
					options.history.appendleft(entry)
					enqueue_auto(*entry[1])
			easygui2.choicebox(
				player_history_a,
				"Player History",
				title="Miza Player",
				choices=choices,
			)
		def history_menu():
			def clear_history():
				options.history.clear()
				return easygui2.msgbox(
					None,
					"History successfully cleared!",
					title="Miza Player",
				)
			def clear_cache():
				futs = deque()
				for fn in os.listdir("cache"):
					futs.append(submit(os.remove, "cache/" + fn))
				if os.path.exists("misc/cache"):
					for fn in os.listdir("misc/cache"):
						try:
							os.remove("misc/cache/" + fn)
						except:
							pass
				for fut in futs:
					try:
						fut.result()
					except (FileNotFoundError, PermissionError):
						pass
				ytdl = downloader.result()
				ytdl.searched.clear()
				return easygui2.msgbox(
					None,
					"Cache successfully cleared!",
					title="Miza Player",
				)
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
			name="History (CTRL+H)",
			sprite=history,
			click=(player_history, history_menu),
			click2=project_history,
			hotkey=(CTRL, K_h),
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
				mixer.submit(f"~setting spectrogram {options.spectrogram - 1}")
				pygame.display.set_caption("Miza Player")
		toolbar.buttons.append(cdict(
			name="Editor (CTRL+E)",
			image=edit,
			click=None,
			hotkey=(CTRL, K_e),
		))
		reset_menu(full=False)
		repeat = button_images.repeat.result()
		def repeat_1():
			control.loop = (control.loop + 1) % 3
			globals()["last-cond"] = 2
		def repeat_2():
			control.loop = (control.loop - 1) % 3
			globals()["last-cond"] = 2
		toolbar.buttons.append(cdict(
			name="Repeat (ALT+Z)",
			image=repeat,
			click=(repeat_1, repeat_2),
			hotkey=(ALT, K_z),
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
			name="Shuffle (ALT+X)",
			image=shuffle,
			click=(shuffle_1, shuffle_2),
			hotkey=(ALT, K_x),
		))
		reset_menu(full=False)
		back = button_images.back.result()
		def rleft():
			if player.previous:
				queue.appendleft(player.previous)
				player.previous = None
			else:
				queue.rotate(1)
			mixer.clear()
			player.fut = None
		toolbar.buttons.append(cdict(
			name="Previous (ALT+LEFT)",
			image=back,
			click=rleft,
			hotkey=(ALT, K_LEFT),
		))
		front = pygame.transform.flip(back, True, False)
		def rright():
			player.previous = None
			queue.rotate(-1)
			mixer.clear()
			player.fut = None
		toolbar.buttons.append(cdict(
			name="Next (ALT+RIGHT)",
			image=front,
			click=rright,
			hotkey=(ALT, K_RIGHT),
		))
		reset_menu(full=False)
		flip = button_images.flip.result()
		def flip_1():
			queue.reverse()
			mixer.clear()
			player.fut = None
		toolbar.buttons.append(cdict(
			name="Flip (ALT+F)",
			image=flip,
			click=flip_1,
			hotkey=(ALT, K_f),
		))
		reset_menu(full=False)
		scramble = button_images.scramble.result()
		def scramble_1():
			queue.shuffle()
			mixer.clear()
			player.fut = None
		toolbar.buttons.append(cdict(
			name="Scramble (ALT+S)",
			image=scramble,
			click=scramble_1,
			hotkey=(ALT, K_s),
		))
		reset_menu(full=False)
		unique = button_images.unique.result()
		def unique_1():
			if not queue:
				return
			orig = queue[0]
			pops = deque()
			found = set()
			for i, e in enumerate(queue):
				if e.url in found:
					pops.append(i)
				else:
					found.add(e.url)
			queue.pops(pops)
			if orig != queue[0]:
				mixer.clear()
				player.fut = None
		toolbar.buttons.append(cdict(
			name="Remove Duplicates (ALT+U)",
			image=unique,
			click=unique_1,
			hotkey=(ALT, K_u),
		))
		reset_menu(full=False)
		microphone = button_images.microphone.result()
		# def enqueue_device():
			# afut.result().terminate()
			# globals()["afut"] = submit(pyaudio.PyAudio)
			# globals()["pya"] = afut.result()
			# count = pya.get_device_count()
			# apis = {}
			# ddict = mdict()
			# for i in range(count):
				# d = cdict(pya.get_device_info_by_index(i))
				# a = d.get("hostApi", -1)
				# if a not in apis:
					# apis[a] = cdict(pya.get_host_api_info_by_index(a))
				# if d.maxInputChannels > 0 and apis[a].name in ("MME", "Windows DirectSound", "Windows WASAPI"):
					# try:
						# if not pya.is_format_supported(
							# 48000,
							# i,
							# 2,
							# pyaudio.paInt16,
						# ):
							# continue
					# except:
						# continue
					# d.id = i
					# ddict.add(a, d)
			# devices = ()
			# for dlist in ddict.values():
				# if len(dlist) > len(devices):
					# devices = dlist
			# f = f"%0{len(str(len(devices)))}d"
			# selected = easygui.get_choice(
				# "Transfer audio from a sound input device!",
				# "Miza Player",
				# sorted(f % d.id + ": " + d.name for d in devices),
			# )
			# if selected:
				# submit(_enqueue_local, "<" + selected.split(":", 1)[0].lstrip("0") + ">")
		# def microphone_menu():
			# sidebar.menu = cdict(
				# buttons=(
					# ("Add input", enqueue_device),
				# ),
			# )
		sidebar.buttons.append(cdict(
			name="Audio input (CTRL+I)",
			sprite=microphone,
			click=(lambda: None),
			# click=enqueue_device,
			hotkey=(CTRL, K_i),
		))
		reset_menu(full=False)
		record = button_images.record.result()
		def output_device():
			devices = sc.all_speakers()
			f = f"%0{len(str(len(devices)))}d"
			def output_device_a(selected):
				if selected:
					if "\x7f" in selected:
						selected = None
					common.OUTPUT_DEVICE = selected
					# globals()["DEVICE"] = get_device(common.OUTPUT_DEVICE)
					print("Audio device changed.")
					restart_mixer()
			devicenames = ["Default\x7f" + sc.default_speaker().name]
			devicenames.extend(d.name for d in devices)
			easygui2.choicebox(
				output_device_a,
				"Change the output audio device!",
				title="Miza Player",
				choices=devicenames,
			)
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
				def end_recording_a(fn):
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
				easygui2.filesavebox(
					end_recording_a,
					"Save As",
					"Miza Player",
					fn.translate(safe_filenames),
					filetypes=ftypes,
				)
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
					# ("Change device", output_device),
				),
			)
		sidebar.buttons.append(cdict(
			name="Audio output (CTRL+R)",
			sprite=record,
			click=(record_audio, record_menu),
			hotkey=(CTRL, K_r),
		))
		hotkey=(ALT, K_p),
		reset_menu(full=False)
	except:
		print_exc()

def send_status():
	pass

cached_fns = {}
def _enqueue_local(*files, probe=True, index=None, allowshuffle=True):
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
			elif fn.endswith(".ecdc"):
				fn = fn.replace("\\", "/")
				if "/" not in fn:
					fn = "/" + fn
				options.path, name = fn.rsplit("/", 1)
				dur = None
				url = fn
				try:
					try:
						name, dur, url = cached_fns[fn]
					except KeyError:
						args = [sys.executable, "misc/ecdc_stream.py", "-i", url]
						info = subprocess.check_output(args).decode("utf-8", "replace").splitlines()
						assert info
						info = cdict(line.split(": ", 1) for line in info if line)
						if info.get("Name"):
							name = orjson.loads(info["Name"]) or name
						if info.get("Duration"):
							dur = orjson.loads(info["Duration"]) or dur
						if info.get("Source"):
							url = orjson.loads(info["Source"]) or url
						cached_fns[fn] = name, dur, url
				except:
					print_exc()
					dur = None
				entry = cdict(
					url=url,
					stream=fn,
					name=name,
					duration=dur,
					cdc="ecdc",
					pos=start,
				)
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
					icon=fn,
					video=fn,
				)
			if not title:
				title = entry.name
				if len(files) > 1:
					title += f" +{len(files) - 1}"
				options.history.appendleft((title, files))
				options.history = options.history.uniq(sort=False)[:64]
			if not entries:
				queue.append(entry)
		if index is not None:
			temp = list(queue[start:])
			queue[index + len(temp):] = queue[index:-len(temp)]
			queue[index:index + len(temp)] = temp
			if index < 1:
				submit(enqueue, queue[0])
		elif allowshuffle and control.shuffle and len(queue) > 1:
			queue[bool(start):].shuffle()
		sidebar.loading = False
	except:
		sidebar.loading = False
		print_exc()

eparticle = dict(colour=(255,) * 3)
def _enqueue_search(query, index=None, allowshuffle=True):
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
				name = entry["name"]
				if len(entries) > 1:
					name += f" +{len(entries) - 1}"
				url = query if is_url(query) and len(entries) > 1 else entry["url"]
				options.history.appendleft((name, (url,)))
				options.history = options.history.uniq(sort=False)[:64]
				entries = [(cdict(**e, pos=start) if "pos" not in e else cdict(e)) for e in entries]
				queue.extend(entries)
			else:
				sidebar.particles.append(cdict(eparticle))
		if index is not None:
			temp = list(queue[start:])
			queue[index + len(temp):] = queue[index:-len(temp)]
			queue[index:index + len(temp)] = temp
			if index < 1:
				submit(enqueue, queue[0])
		elif allowshuffle and control.shuffle and len(queue) > 1:
			queue[bool(start):].shuffle()
		sidebar.loading = False
	except:
		sidebar.loading = False
		print_exc()

def enqueue_auto(*queries, index=None):
	futs = deque()
	start = len(queue)
	for i, query in enumerate(queries):
		q = query.strip()
		if not q:
			continue
		if is_url(q) or not os.path.exists(q):
			if i < 1:
				futs.append(submit(_enqueue_search, q, allowshuffle=index is None))
			else:
				for fut in futs:
					fut.result()
				futs.clear()
				name = q.rsplit("/", 1)[-1].split("?", 1)[0].rsplit(".", 1)[0]
				queue.append(cdict(name=name, url=q, duration=None, pos=start))
		else:
			futs.append(submit(_enqueue_local, q, probe=i < 1, allowshuffle=index is None))
	for fut in futs:
		fut.result()
	if index is not None:
		temp = list(queue[start:])
		queue[index + len(temp):] = queue[index:-len(temp)]
		queue[index:index + len(temp)] = temp
		if index < 1:
			submit(enqueue, queue[0])
	elif control.shuffle and len(queue) > 1:
		queue[bool(start):].shuffle()

def load_project(fn, switch=True):
	if switch and not toolbar.editor:
		toolbar.editor = 1
		mixer.submit(f"~setting spectrogram -1")
	player.editor_surf = None
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


argv = [arg for arg in sys.argv if not arg.startswith("-")]
if len(argv) > 1:
	if len(argv) == 2 and not is_url(argv[1]) and os.path.exists(argv[1]):
		fi = argv[1]
		with open(fi, "rb") as f:
			b = f.read(7)
		if b == b">~MPP~>":
			submit(load_project, fi)
		else:
			submit(enqueue_auto, fi)
	else:
		submit(enqueue_auto, *argv[1:])


def load_video(url, pos=0, bak=None, sig=None, iterations=0):
	try:
		if player.video and player.video.is_running():
			try:
				player.video.terminate()
			except psutil.NoSuchProcess:
				pass
		# print("Loading", url)
		cmd = ("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,duration", "-of", "csv=s=x:p=0", url)
		print(cmd)
		p = psutil.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		cmd2 = ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-hwaccel", hwaccel]
		if is_url(url):
			cmd2 += ["-reconnect", "1", "-reconnect_at_eof", "0", "-reconnect_streamed", "1", "-reconnect_delay_max", "240"]
		if not iterations:
			cmd2 += ["-ss", str(pos)]
		cmd2 += ["-i", url, "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0"]
		if iterations:
			cmd2 += ["-vframes", "1"]
		cmd2 += ["-"]
		print(cmd2)
		proc = psutil.Popen(cmd2, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1048576)
		proc.url = sig
		proc.pos = pos
		bcount = 3
		mode = "RGB"
		try:
			res = as_str(p.stdout.read()).strip()
			if not res:
				print("Iteration", iterations, url)
				if iterations < 1 and is_url(url) and queue and sig == queue[0].get("url"):
					e = queue[0]
					prepare(e, force=2)
					url = e.get("video") or bak
					return load_video(bak, pos=pos, sig=proc.url, iterations=iterations + 1)
				if iterations < 2 and is_url(url):
					if bak and bak != url:
						try:
							return load_video(bak, pos=pos, sig=proc.url, iterations=inf)
						except:
							pass
					h = header()
					resp = reqs.get(url, headers=h, stream=True)
					url = resp.url
					head = resp.headers
					ctype = [t.strip() for t in head.get("Content-Type", "").split(";")]
					if "text/html" in ctype:
						it = resp.iter_content(65536)
						data = next(it)
						s = data.decode("utf-8")
						try:
							s = s[s.index("<meta") + 5:]
							search = 'http-equiv="refresh" content="'
							try:
								s = s[s.index(search) + len(search):]
								s = s[:s.index('"')]
								res = None
								for k in s.split(";"):
									temp = k.strip()
									if temp.casefold().startswith("url="):
										res = temp[4:]
										break
								if not res:
									raise ValueError
							except ValueError:
								search ='property="og:image" content="'
								s = s[s.index(search) + len(search):]
								res = s[:s.index('"')]
						except ValueError:
							pass
						else:
							return load_video(res, pos=pos, bak=bak, sig=proc.url, iterations=iterations + 1)
				raise TypeError(f'File "{url}" is not supported.')
			info = res.split("x", 3)
		except:
			print(as_str(p.stderr.read()), end="")
			raise
		size = tuple(map(int, info[:2]))
		try:
			fps = eval(info[2], {}, {})
		except (ValueError, TypeError, SyntaxError, ZeroDivisionError, NameError):
			fps = 30
		try:
			dur = eval(info[3], {}, {})
		except (ValueError, TypeError, SyntaxError, ZeroDivisionError, NameError):
			dur = 0
		bcount *= int(np.prod(size))
		proc.fps = fps
		proc.size = size
		print(size, bcount, fps, pos, dur)
		proc.tex = proc.im = proc.im2 = None
		player.video = proc
		i = 0
		im = None
		while bcount:
			b = proc.stdout.read(bcount)
			while len(b) < bcount:
				if not b or not proc.is_running():
					break
				b += proc.stdout.read(bcount - len(b))
			if len(b) < bcount:
				break
			curr = i / fps + pos
			proc.pos = curr
			while im and proc.is_running() and curr > player.pos or is_minimised():
				time.sleep(0.04)
			# print(curr, len(b), i, fps)
			if not im:
				im = pyglet.image.ImageData(*size, "RGB", b)
				player.video_loading = None
			else:
				im.set_data("RGB", size[0] * 3, b)
			proc.im = im
			proc.im2 = None
			if player.sprite:
				player.sprite.changed += 1
			i += 1
			if not proc.is_running():
				break
			async_wait()
		if not bcount:
			with reqs.get(url, verify=False, stream=True) as resp:
				if resp.headers.get("Content-Length", 1) <= 0:
					raise EOFError(resp)
				if resp.headers.get("Content-Type", "image").split("/", 1)[0] not in ("image", "video"):
					raise TypeError(resp.headers["Content-Type"])
				im = Image.open(resp.raw)
				proc.im = pil2pgl(im, flip=False)
				proc.im2 = None
		if pos >= dur - 1 or not bcount:
			while pos < player.pos + 1 and queue and proc.url == queue[0].url:
				pos = player.pos
				proc.pos = pos
				time.sleep(0.08)
				if im and not proc.im:
					proc.im = im
					proc.im2 = None
		print("Video exited!")
		proc.pos = inf
	except:
		print_exc()
		if queue and proc.url == queue[0].url:
			queue[0].novid = True
	finally:
		player.video_loading = None


sidebar.abspos = 0
osize = None
ssize = None
def reset_menu(full=True, reset=False):
	global osize, ssize
	if full:
		globals().update(options)
		common.__dict__.update(options)
	ssize2 = (screensize[0] - sidebar_width, screensize[1] - toolbar_height)
	if ssize != ssize2 or full and reset:
		sps = np.frombuffer(globals()["stat-mem"].buf[16:24], dtype=np.uint32)
		sps[:] = ssize = ssize2
		mixer.new = False
	if reset:
		globals()["mpos"] = (-inf,) * 2
	player.rect = (0,) * 2 + ssize
	sidebar.colour = None
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
	toolbar.rect = (0, screensize[1] - toolbar_height, screensize[0], toolbar_height)
	toolbar.pause.radius = min(64, toolbar_height // 2 - 2)
	toolbar.pause.pos = (toolbar.pause.radius + 2, screensize[1] - toolbar_height + toolbar.pause.radius + 2)
	progress.pos = (round(toolbar.pause.pos[0] + toolbar.pause.radius * 1.5 + 4), screensize[1] - toolbar_height + toolbar.pause.radius * 2 // 3 + 1)
	progress.box = toolbar.pause.radius * 6 // 2 + 8
	progress.length = max(0, screensize[0] - progress.pos[0] - toolbar.pause.radius // 2 - progress.box)
	progress.width = min(16, toolbar.pause.radius // 3)
	progress.rect = (progress.pos[0] - progress.width // 2 - 3, progress.pos[1] - progress.width // 2 - 3, progress.length + 6, progress.width + 6)
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
			button.on = button.sprite.copy()
			button.on.fill((0, 255, 255), special_flags=BLEND_RGB_MULT)
			button.off = button.sprite.copy()
			button.off.fill((0,) * 3, special_flags=BLEND_RGB_MULT)
		button.rect = rect
	toolbar.resizing = False
	toolbar.resizer = False
	osize2 = (progress.box, toolbar.pause.radius * 4 // 3 - 3)
	if osize != osize2 or full and reset:
		sps = np.frombuffer(globals()["stat-mem"].buf[8:16], dtype=np.uint32)
		sps[:] = osize = osize2
		mixer.new = False
	globals()["last-cond"] = 2


submit(setup_buttons)


is_active = lambda: not player.paused and pc() - player.get("last", 0) <= max(player.get("lastframe", 0), 1 / 30) * 3
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
	lyrics_size = control.lyrics_size
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
		if options.spectrogram != 1:
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
			z = lyrics_size // 2 + 1
			for para in lyrics[1].split("\n\n"):
				lines = para.splitlines()
				if mx and y + z * 3 > render[1].get_height():
					y = 0
					x += mx
					mx = 0
				for line in lines:
					line = line.strip()
					if not line:
						continue
					if mx and y + z * 2 > render[1].get_height():
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
					s = text_size(line, lyrics_size)
					if s[0] <= render[1].get_width() >> 1:
						rect = message_display(
							line,
							lyrics_size,
							(x, y),
							col,
							align=0,
							surface=render[1],
						)
						mx = max(mx, rect[2] + 8)
						z = rect[3]
						y += z
					else:
						p = 0
						words = line.split()
						curr = ""
						while words:
							w = words.pop(0)
							orig = curr
							curr = curr + " " + w if curr else w
							if orig:
								s = text_size(curr, lyrics_size)
								if s[0] > render[1].get_width() // 2 - p:
									rect = message_display(
										orig,
										lyrics_size,
										(x + p, y),
										col,
										align=0,
										surface=render[1],
									)
									mx = max(mx, rect[2] + 8 + p)
									z = rect[3]
									y += z
									p = 8
									curr = w
						if curr:
							rect = message_display(
								curr,
								lyrics_size,
								(x + p, y),
								col,
								align=0,
								surface=render[1],
							)
							mx = max(mx, rect[2] + 8 + p)
							z = rect[3]
							y += z
				if y:
					y += z >> 1
		print(entry)
		entry.lyrics = render
		entry.pop("lyrics_loading", None)
	except:
		print_exc()

def reset_entry(entry):
	entry.pop("lyrics", None)
	entry.pop("surf", None)
	entry.pop("lyrics_loading", None)
	return entry

has_api = 0
api_wait = set()
ecdc_wait = {}
def prepare(entry, force=False, download=False, delay=0):
	# print("PREPARE", entry, force, delay)
	reset_entry(entry)
	if not entry.url:
		return
	url = entry.url
	url = unyt(url)
	fn = ofn = "cache/~" + shash(url) + ".webm"
	# print(fn)
	if delay:
		time.sleep(delay)
	if utc() - has_api < 60 and is_url(entry.url) and (entry.get("stream") or "").endswith(".ecdc"):
		entry.pop("stream", None)
	try:
		# raise
		if (entry.get("stream") or "").startswith("https://api.mizabot.xyz/ytdl"):
			raise StopIteration("API Passthrough...")
		if force > 2 and is_url(entry.get("url")):
			raise StopIteration("Downloads blocked, attempting backup...")
		if force > 1 and not entry.get("icon"):
			entry.novid = False
			if is_url(entry.url):
				ytdl = downloader.result()
				try:
					e = ytdl.extract(entry.url)[0]
				except:
					print_exc()
					entry.novid = True
				else:
					try:
						ytdl.searched[e.url].data[0] = e
					except KeyError:
						pass
					entry.update(e)
					entry.novid = False
			else:
				entry.icon = entry.video = entry.url
			print(entry)
		if os.path.exists(fn) and os.path.getsize(fn):
			if time.time() - os.path.getmtime(fn) > 60 and is_url(entry.url) and not fn.endswith(".ecdc"):
				url = entry.url
				url = unyt(url)
				ofn = "persistent/~" + shash(url) + ".ecdc"
				if not os.path.exists(ofn) or not os.path.getsize(ofn):
					ecdc_submit(entry, fn)
			dur = entry.get("duration")
			if dur and not isinstance(dur, str) and isfinite(dur):
				entry.stream = fn
				ytdl = downloader.result()
				if entry.url in ytdl.searched:
					resp = ytdl.searched[entry.url].data
					if len(resp) == 1:
						entry.name = resp[0].get("name")
						entry.video = resp[0].get("video")
						entry.icon = resp[0].get("icon")
				return fn
			elif force:
				ytdl = downloader.result()
				stream = ytdl.get_stream(entry, force=True, download=False)
				return stream
		stream = entry.get("stream", "")
		# print("STREAM:", stream)
		if stream and not is_url(stream) and not os.path.exists(stream) and stream != entry.url:
			entry.pop("stream", None)
			stream = ""
		if not stream or stream.startswith("ytsearch:") or force and (stream.startswith("https://cf-hls-media.sndcdn.com/") or is_youtube_url(stream) or expired(stream)):
			if not is_url(entry.url):
				duration = entry.duration
				if os.path.exists(entry.url):
					stream = entry.stream = entry.url
					if not duration:
						info = get_duration_2(stream)
						duration = info[0]
						if info[0] in (None, nan) and info[1] in ("N/A", "auto"):
							fi = stream
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
				raise
			except:
				if os.path.exists(fn):
					duration = get_duration_2(fn)[0]
					stream = entry.stream = fn
					entry.duration = duration or entry.duration
					globals()["queue-length"] = -1
					return stream
				raise
				# entry.url = ""
				# print_exc()
				# return
			else:
				if entry.name != data.get("name"):
					reset_entry(entry)
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
			stream = entry.get("stream") or ""
			# print("STREAM2:", stream, entry, resp)
		elif stream.startswith("https://api.mizabot.xyz/ytdl"):
			if download:
				with reqs.get(stream, stream=True) as resp:
					it = resp.iter_content(65536)
					with open(ofn, "wb") as f:
						try:
							while True:
								b = next(it)
								if not b:
									break
								f.write(b)
						except StopIteration:
							pass
				print(steam)
				return ofn
		elif (force > 1 or force and not stream) and is_url(entry.get("url")):
			ytdl = downloader.result()
			if force > 1:
				data = ytdl.extract(entry.url)
				entry.name = data[0].name
				stream = entry.stream = data[0].setdefault("stream", data[0].url)
			else:
				stream = ytdl.get_stream(entry, force=True, download=False)
			globals()["queue-length"] = -1
		elif not is_url(stream) and not os.path.exists(stream):
			entry.stream = entry.url
			duration = entry.duration
			if not duration:
				info = get_duration_2(stream)
				duration = info[0]
				if info[0] in (None, nan) and info[1] in ("N/A", "auto"):
					fi = stream
					if not os.path.exists(fn):
						fn = select_and_convert(fi)
					duration = get_duration_2(fn)[0]
					stream = entry.stream = fn
				globals()["queue-length"] = -1
			entry.duration = duration or entry.duration
			return entry.stream
	except:
		if force and is_url(entry.url) and utc() - has_api < 60 and utc() - entry.get("tried_api", 0) > 120:
			try:
				resp = requests.head("https://api.mizabot.xyz/ip")
				resp.raise_for_status()
			except:
				# print_exc()
				globals()["has_api"] = 0
		if is_url(entry.url) and utc() - has_api < 60 and utc() - entry.get("tried_api", 0) > 120:
			stream = f"https://api.mizabot.xyz/ytdl?d={entry.url}&fmt=webm"
			if not download:
				return stream
			if os.path.exists(ofn) and os.path.getsize(ofn):
				entry.stream = ofn
				return ofn
			if stream in api_wait:
				return stream
			if force:
				stream += "&asap=1"
			if stream in api_wait:
				return stream
			api_wait.add(stream)
			entry["tried_api"] = utc()
			try:
				print("API:", stream)
				with reqs.get(stream, stream=True) as resp:
					resp.raise_for_status()
					it = resp.iter_content(65536)
					with open(ofn, "wb") as f:
						try:
							while True:
								b = next(it)
								if not b:
									break
								f.write(b)
						except StopIteration:
							pass
				print(stream, ofn)
				entry.stream = ofn
				return ofn
			except:
				print_exc()
			finally:
				api_wait.discard(stream)
		url = entry.url
		if not url:
			return
		url = unyt(url)
		cfn = "persistent/~" + shash(url) + ".ecdc"
		# out = "cache/~" + shash(url) + ".wav"
		# if os.path.exists(out):
			# return out
		# while ecdc_wait.get(cfn):
			# time.sleep(1)
		# if os.path.exists(out):
			# return out
		if os.path.exists(cfn) and os.path.getsize(cfn):
			entry.stream = cfn
			return cfn
		entry.url = ""
		print_exc()
		print("DELETING:", entry, cfn)
		return
	stream = stream.strip()
	duration = entry.duration
	if not duration:
		info = (None, None)
		try:
			info = get_duration_2(stream)
			duration = info[0]
		except:
			print_exc()
			duration = None
		if info[0] in (None, nan) and info[1] in ("N/A", "auto"):
			fi = stream
			if not os.path.exists(fn):
				fn = select_and_convert(fi)
			duration = get_duration_2(fn)[0]
			stream = entry.stream = fn
		globals()["queue-length"] = -1
	elif stream and is_url(stream) and download:
		es = base64.b85encode(stream.encode("utf-8")).decode("ascii")
		url = entry.url
		url = unyt(url)
		mixer.submit(f"~download {es} cache/~{shash(url)}.webm")
	entry.duration = duration or entry.duration
	if stream:
		entry.stream = stream
	return stream

def start_player(pos=None, force=False):
	try:
		print("start_player", queue[0], pos, force)
		if options.get("spectrogram", 0) == 0 and not queue[0].get("video"):
			submit(prepare, queue[0], force=force + 1)
		try:
			entry = queue[0]
		except IndexError:
			return skip()
		if control.loop < 2 and len(queue) > 1:
			thresh = min(8, max(2, len(queue) // 8)) + 1
			# if control.shuffle > 1 or player.shuffler >= thresh:
				# ensure_next(queue[1])
				# thresh = 0
			# elif control.shuffle:
				# thresh -= player.shuffler
			# print(thresh)
			for i, e in enumerate(queue[1:min(len(queue), thresh + 1)]):
				# print("EN", e, i)
				ensure_next(e, delay=i + 1)
		duration = entry.duration or 300
		if pos is None:
			if audio.speed >= 0:
				pos = 0
		elif pos >= duration and not player.paused:
			if audio.speed > 0:
				return skip()
			pos = duration
		elif pos <= 0:
			if audio.speed < 0 and not player.paused:
				return skip()
			pos = 0
		with player.waiting:
			if pos is not None:
				player.pos = pos
				player.offpos = pos - pc() if pos > 0 else -inf
				player.index = player.pos * 30
			if force and queue and is_url(queue[0].url):
				queue[0].stream = None
				queue[0].research = True
				downloader.result().cache.pop(queue[0].url, None)
			player.last = 0
			player.amp = 0
			stream = prepare(queue[0], force=force + 1)
			if not queue or not queue[0].url:
				return skip()
			stream = prepare(queue[0], force=force + 1)
			entry = queue[0]
			if not entry.url:
				return skip()
			if not stream:
				player.fut = None
				return None, inf
			duration = entry.duration
			print("ENTRY:", stream)
			if not duration:
				try:
					info = get_duration_2(stream)
				except:
					print_exc()
					info = (None, None)
				duration = info[0]
				if info[0] in (None, nan) and info[1] in ("N/A", "auto"):
					fi = stream
					fn = "cache/~" + shash(fi) + ".webm"
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
			elif pos >= duration and not player.paused:
				if audio.speed > 0:
					return skip()
				pos = duration
			elif pos <= 0:
				if audio.speed < 0 and not player.paused:
					return skip()
				pos = 0
			if control.shuffle == 2:
				player.needs_shuffle = False
			else:
				player.needs_shuffle = not is_url(stream)
			if is_url(stream) and expired(stream):
				ytdl = downloader.result()
				data = ytdl.extract(entry.url)
				entry.name = data[0].name
				stream = entry["stream"] = data[0].setdefault("stream", data[0].url)
			if is_url(entry.url):
				if not is_url(stream) and os.path.exists(stream) and not stream.endswith(".ecdc") and time.time() - os.path.getmtime(stream) > 30:
					ecdc_submit(entry, stream)
				else:
					ecdc_submit(entry, stream, force=True)
			es = base64.b85encode(stream.encode("utf-8")).decode("ascii")
			url = entry.url
			url = unyt(url)
			s = f"{es}\n{pos} {duration} {entry.get('cdc', 'auto')} {shash(url)}\n"
			print("SUBMIT:", s)
			mixer.submit(s, force=False)
			player.pos = pos
			player.offpos = pos - pc() if pos > 0 else -inf
			player.index = player.pos * 30
			player.end = duration or inf
			player.stream = stream
			return stream, duration
	finally:
		if reevaluating:
			reevaluating.mut.append(None)
			reevaluating.cancel()
		mut = []
		globals()["reevaluating"] = submit(reevaluate_in, 20, mut=mut)
		reevaluating.mut = mut


ECDC_CURR = None
ECDC_QUEUE = {}
ECDC_RUNNING = set()
ECDC_TRIED = set()

def ecdc_submit(entry, stream="", force=False):
	url = entry.url
	url = unyt(url)
	ofn = "persistent/~" + shash(url) + ".ecdc"
	if not force and os.path.exists(ofn) and os.path.getsize(ofn):
		return
	if force is None and ofn in ECDC_TRIED:
		return
	try:
		ECDC_QUEUE[ofn][1] = stream or ECDC_QUEUE[ofn][1]
		ECDC_QUEUE[ofn][2] = force or ECDC_QUEUE[ofn][2]
	except KeyError:
		ECDC_QUEUE[ofn] = [entry, stream, force]
		ECDC_TRIED.add(ofn)

def ecdc_compress(entry, stream, force=False):
	url = entry.url
	url = unyt(url)
	ofn = "persistent/~" + shash(url) + ".ecdc"
	exists = os.path.exists(ofn) and os.path.getsize(ofn)
	try:
		if exists:
			if utc() - has_api >= 60:
				return ofn
			with open(ofn, "rb") as f:
				b = f.read(5)
			if len(b) < 5 or b[-1] < 192:
				exists = False
		if force is not None or not stream:
			if not stream or is_url(stream) and expired(stream):
				ytdl = downloader.result()
				data = ytdl.extract(entry.url)
				entry.name = data[0].name
				stream = entry["stream"] = data[0].setdefault("stream", data[0].url)
			try:
				dur, bps, cdc = _get_duration_2(stream)
			except:
				print_exc()
				bps = 196608
		else:
			bps = 64000
		if not bps:
			br = 24
		elif bps < 48000:
			br = 6
		elif bps < 96000:
			br = 12
		else:
			br = 24
			print("BPS:", bps)
		try:
			import pynvml
			pynvml.nvmlInit()
			dc = pynvml.nvmlDeviceGetCount()
		except:
			dc = 0
		if exists or force is None or not force and psutil.cpu_count() >= 8 and (dc > len(ECDC_RUNNING) / 2) or utc() - has_api >= 60:
			i = None
		else:
			i = False
		ECDC_RUNNING.add("!" + ofn)
		try:
			name = entry.get("name") or ""
			b = "auto" if i is None else br
			query = f"bitrate={b}&name={urllib.parse.quote_plus(name)}&source={urllib.parse.quote_plus(url)}"
			api = f"https://api.mizabot.xyz/encodec?{query}&inference={i}&url={url}"
			if i is not None:
				print(api)
			ifn = "cache/~" + shash(url) + ".webm"
			if force is None:
				b = b""
			elif exists:
				with open(ofn, "rb") as f:
					b = f.read()
			elif os.path.exists(ifn):
				with open(ifn, "rb") as f:
					b = f.read()
			else:
				b = b""
			try:
				# print(api)
				if utc() - has_api >= 60:
					raise EOFError
				with requests.post(api, data=b, stream=True) as resp:
					resp.raise_for_status()
					b = resp.content
				if not b:
					print(api)
					raise EOFError
				if not exists or len(b) > os.path.getsize(ofn):
					with open(ofn, "wb") as f:
						f.write(b)
				return
			except Exception as ex:
				if force is None:
					return
				if force:
					raise
				# print_exc()
				if not os.path.exists(ofn) or not os.path.getsize(ofn):
					if ofn in ECDC_RUNNING:
						raise
					ECDC_RUNNING.add(ofn)
					try:
						out = ecdc_encode(stream, bitrate=br, name=name, source=url)
						os.rename(out, ofn)
					finally:
						ECDC_RUNNING.discard(ofn)
				if os.path.exists(ofn) and os.path.getsize(ofn) and utc() - has_api < 60:
					with open(ofn, "rb") as f:
						b = f.read()
					query = f"bitrate={br}&name={urllib.parse.quote_plus(name)}&source={urllib.parse.quote_plus(url)}"
					api = f"https://api.mizabot.xyz/encodec?{query}&inference=None&url={url}"
					# print(api)
					with requests.post(api, data=b, stream=True) as resp:
						print(resp)
						if resp.status_code not in range(200, 400):
							print(api, resp.content)
							globals()["has_api"] = utc()
				return ofn
		finally:
			ECDC_RUNNING.discard("!" + ofn)
	except:
		print_exc()

def persist():
	try:
		q2 = list(queue)
		random.shuffle(q2)
		api = "https://api.mizabot.xyz/encodec"
		urls = " ".join(e.url for e in q2)
		with requests.post(api, data=dict(urls=urls)) as resp:
			print(resp)
		qt = resp.json()
		if len(qt) != len(q2):
			print("Persist Mismatch:", len(qt), len(q2))
			q2 = q2[:len(qt)]
		q3 = [e for i, e in enumerate(q2) if qt[i]]
		print("SCAN:", f"{len(q3)}/{len(q2)}")
		if "-d" in sys.argv:
			q4 = [t[1] for t in sorted(enumerate(q2), key=lambda t: qt[t[0]], reverse=True)]
		else:
			q4 = q2
		for i, entry in enumerate(q4):
			if is_url(entry.url) and qt[i] or "-d" in sys.argv:
				ecdc_submit(entry, entry.get("stream") or "", force=None if qt[i] else False)
			if not i + 1 & 511:
				time.sleep(0.5)
	except:
		print_exc()

def start():
	if queue:
		return enqueue(queue[0])
	player.last = 0
	player.pos = 0
	player.offpos = -inf
	player.end = inf
	return None, inf

def delete_entry(e):
	if e.get("surf"):
		del e["surf"]
	if e.get("lyrics_surf"):
		del e["lyrics_surf"]

last_save = -inf
last_sync = -inf
def skip():
	import inspect
	curframe = inspect.currentframe()
	calframe = inspect.getouterframes(curframe, 2)
	print('Skipped:', calframe)
	if player.video:
		player.video.url = None
	if queue:
		e = queue.popleft()
		if control.shuffle > 1:
			queue[1:].shuffle()
		elif control.shuffle:
			thresh = min(8, max(2, len(queue) / 8))
			if player.shuffler >= thresh:
				queue[thresh:].shuffle()
				player.shuffler = 0
			else:
				player.shuffler += 1
		if control.loop == 2:
			player.previous = None
			queue.appendleft(e)
		elif control.loop == 1:
			player.previous = None
			queue.append(e)
		else:
			player.previous = e
			if len(queue) > sidebar.maxitems - 1:
				queue[sidebar.maxitems - 1].pos = sidebar.maxitems - 1
			sidebar.particles.append(e)
		t = pc()
		if t >= last_save + 10:
			globals()["last_save_fut"] = submit(save_settings)
			globals()["last_save"] = t
		if queue:
			return enqueue(queue[0])
	mixer.clear()
	player.last = 0
	player.pos = 0
	player.offpos = -inf
	player.end = inf
	return None, inf

def seek_abs(pos, force=False):
	if not force:
		if pos <= 0 and player.pos <= 0 and audio.speed > 0 and not player.paused:
			return (player.get("stream"), player.end)
		if pos >= player.end and player.pos >= player.end and audio.speed < 0 and not player.paused:
			return (player.get("stream"), player.end)
	return start_player(pos) if queue else (None, inf)

def seek_rel(pos):
	if not pos:
		return
	player.last = 0
	player.amp = 0
	if pos + player.extpos() >= player.end and not player.paused:
		if audio.speed > 0:
			return skip()
		if player.extpos() >= player.end:
			return
		pos = player.end - player.extpos()
	if pos + player.extpos() <= 0:
		if audio.speed < 0 and not player.paused:
			return skip()
		if player.extpos() <= 0:
			return
		pos = -player.extpos()
	progress.num += pos
	progress.alpha = 255
	if audio.speed > 0 and pos > 0 and pos <= 180:
		with player.waiting:
			mixer.drop(pos)
	elif audio.speed < 0 and pos < 0 and pos >= -180:
		with player.waiting:
			mixer.drop(pos)
	else:
		seek_abs(max(0, player.extpos() + pos))
	player.pos = player.extpos() + pos

def restart_mixer(devicename=None):
	global mixer
	if not mixer:
		return
	if mixer.is_running():
		for p in mixer.children(True):
			p.kill()
		mixer.kill()
	mixer = start_mixer(devicename)
	if not mixer or not mixer.is_running():
		return
	mixer.state(player.paused)
	submit(transfer_instrument, *project.instruments.values())
	return seek_abs(player.extpos())

reevaluating = False
def reevaluate():
	time.sleep(2)
	while not player.pos and not pygame.closed:
		print("Re-evaluating file stream...")
		if not queue:
			break
		url = queue[0]["url"]
		force = True
		if is_url(url):
			if (queue[0].get("stream") or "").startswith("https://api.mizabot.xyz/ytdl"):
				queue[0].pop("stream", None)
				ytdl = downloader.result()
				ytdl.cache.pop(url, None)
			else:
				queue[0]["stream"] = f"https://api.mizabot.xyz/ytdl?d={url}&fmt=webm&asap=1"
				force = False
				try:
					url = f"https://api.mizabot.xyz/ytdl?q={url}&count=1"
					resp = reqs.get(url)
					resp.raise_for_status()
					queue[0].update(resp.json()[0])
				except:
					url = queue[0].url
					url = unyt(url)
					cfn = "persistent/~" + shash(url) + ".ecdc"
					if os.path.exists(cfn) and os.path.getsize(cfn):
						queue[0].stream = cfn
						queue[0].url = cfn
					else:
						print_exc()
						queue[0].url = ""
						print("REDELETING:", entry, cfn)
		start_player(0, force=force)
		time.sleep(2)

def reevaluate_in(delay=0, mut=()):
	if delay:
		time.sleep(delay)
	if mut:
		return
	a = is_active()
	print("WATCHDOG:", a)
	if not queue or a or player.paused:
		return
	url = queue[0]["url"]
	force = True
	if is_url(url):
		if (queue[0].get("stream") or "").startswith("https://api.mizabot.xyz/ytdl"):
			queue[0].pop("stream", None)
			ytdl = downloader.result()
			ytdl.cache.pop(url, None)
		else:
			queue[0]["stream"] = f"https://api.mizabot.xyz/ytdl?d={url}&fmt=webm&asap=1"
			force = False
			try:
				url = f"https://api.mizabot.xyz/ytdl?q={url}&count=1"
				resp = reqs.get(url)
				resp.raise_for_status()
				queue[0].update(resp.json()[0])
			except:
				url = queue[0].url
				url = unyt(url)
				cfn = "persistent/~" + shash(url) + ".ecdc"
				if os.path.exists(cfn) and os.path.getsize(cfn):
					queue[0].stream = cfn
					queue[0].url = cfn
				else:
					print_exc()
					queue[0].url = ""
					print("REDELETING:", entry, cfn)
	return start_player(0, force=force)

def distribute_in(delay):
	if delay:
		time.sleep(delay)
		while utc() - has_api > 60:
			time.sleep(600)
	if os.name == "nt":
		resp = reqs.get("https://raw.githubusercontent.com/thomas-xin/Miza/master/x-distribute.py")
		with open("x-distribute.py", "wb") as f:
			f.write(resp.content)
		args = [sys.executable, "x-distribute.py"]
		print(args)
		psutil.Popen(args)

device_waiting = None
def wait_on():
	if not device_waiting:
		return
	d = get_device(common.OUTPUT_DEVICE)
	if d and d.name != common.OUTPUT_DEVICE:
		restart_mixer(d.name)
		print("Using", d.name, "for now.")
	while True:
		try:
			d = get_device(common.OUTPUT_DEVICE, default=False)
			if d:
				print(d)
				print("Device target found.")
				restart_mixer()
				globals()["device_waiting"] = None
				# mixer.submit(f"~output {OUTPUT_DEVICE}")
				return
		except:
			print_exc()
		time.sleep(1)

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

	emptybuff = b"\x00" * 6400
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
		# print(num_bytes, len(data))
		pos = self.position
		ts = max(0.004, len(data) / (audio_format.sample_rate * audio_format.sample_size * audio_format.channels / 8))
		self.position += ts
		return pyglet.media.codecs.AudioData(data, len(data), pos, inf, [])

class Player(pyglet.media.Player):

	type = "pyglet"
	peak = 32767
	dtype = np.int16
	channels = 2
	re_paused = 0

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
		self.re_paused = 0
		if not self.paused:
			self.entry.buffer.append(data)

	def wait(self):
		while self.playing and not self.paused and self.source and len(self.entry.buffer) >= 4:
			async_wait()
		if not self.entry.buffer:
			if not self.re_paused:
				self.re_paused = 1
			elif self.re_paused == 1:
				for i in range(3):
					self.entry.buffer.append(self.entry.emptybuff)
				self.re_paused = 2
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

get_channel = lambda: sc_player(get_device(common.OUTPUT_DEVICE))
player.channel = get_channel()
print(player.channel)

def mixer_audio():
	try:
		while mixer and mixer.is_running():
			mixer.client.sendall(b"\x7f")
			async_wait()
			player.channel.wait()
			if not mixer:
				break
			try:
				b = mixer.client.recv(262144)
			except ConnectionResetError:
				continue
			a = np.frombuffer(b, dtype=np.int16)
			player.channel.write(a)
	except:
		print_exc()
threading.Thread(target=mixer_audio).start()

def mixer_stdout():
	try:
		while mixer:
			s = None
			try:
				while not s and mixer and mixer.is_running():
					s = as_str(mixer.stdout.readline())
					if s:
						if s[0] != "~":
							if s[0] in "'\"":
								s = ast.literal_eval(s)
							sys.stdout.write(s)
							s = ""
						else:
							s = s.rstrip()
							break
					elif mixer and not mixer.is_running():
						time.sleep(2)
						if mixer and not mixer.is_running():
							print("Mixer has crashed.")
							restart_mixer()
					async_wait()
				else:
					if not mixer or not mixer.is_running():
						time.sleep(2)
						if mixer and not mixer.is_running():
							print("Mixer has crashed.")
							restart_mixer()
			except:
				print_exc()
			if not s:
				time.sleep(0.05)
				continue
			if not mixer:
				break
			if mixer.new:
				submit(reset_menu, reset=True)
			player.last = pc()
			# print(s)
			s = s[1:]
			if s == "s":
				print("ENTERED ~S")
				submit(skip)
				player.last = 0
				continue
			if s[0] == "o":
				common.OUTPUT_DEVICE = s[2:]
				continue
			if s[0] == "w":
				common.OUTPUT_DEVICE = s[2:]
				if not device_waiting:
					globals()["device_waiting"] = common.OUTPUT_DEVICE
					print(f"Waiting on {common.OUTPUT_DEVICE}...")
					submit(wait_on)
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
				if reevaluating:
					reevaluating.mut.append(None)
					reevaluating.cancel()
				mut = []
				globals()["reevaluating"] = submit(reevaluate, mut=mut)
				reevaluating.mut = mut
				continue
			if s[0] == "R":
				url = s[2:]
				for e in queue:
					if e.url == url or e.get("stream") == url:
						break
				else:
					continue
				submit(prepare, e, force=2, download=True)
				print("Re-evaluating file download...")
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
				player.pos = pos = round(player.index / 30, 4)
				offpos = pos - pc() if pos > 0 else -inf
				if abs(offpos - player.get("offpos", 0)) > 0.25:
					player.offpos = offpos
			if dur >= 0:
				player.end = dur or inf
	except:
		print_exc()
threading.Thread(target=mixer_stdout).start()

CACHE_LIMITS = cdict(
	md_font=32768,
	rp_surf=256,
	br_surf=4096,
	rb_surf=1024,
)
def garbage_collector():
	try:
		for k, v in CACHE_LIMITS.items():
			c = getattr(common, k, None)
			if c:
				x = common.garbage_collect(c, v)
				if x:
					print(f"{k}: deleted {x} surfaces!")
	except:
		print_exc()

enext = {}
def ensure_next(e, delay=0):
	i = e.get("url")
	if not i:
		return
	if i in enext:
		return
	enext[i] = submit(prepare, e, force=delay <= 4, download=True, delay=delay)
	e.duration = e.get("duration") or False
	e.pop("research", None)
	# if not e.get("lyrics_loading") and not e.get("lyrics"):
		# submit(render_lyrics, e)

def enqueue(entry, start=True):
	# import inspect
	# curframe = inspect.currentframe()
	# calframe = inspect.getouterframes(curframe, 2)
	# print('caller name:', calframe[1][3])
	try:
		if not queue:
			return None, inf
		while queue[0] is None:
			time.sleep(0.5)
		# submit(render_lyrics, queue[0])
		stream, duration = start_player()
		progress.num = 0
		progress.alpha = 0
		return stream, duration
	except:
		print_exc()
	return None, inf

ffmpeg_start = (ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-fflags", "+discardcorrupt+genpts+igndts+flush_packets", "-err_detect", "ignore_err", "-hwaccel", hwaccel, "-vn")
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
						os.remove(p.fn)
			st = prepare(entry, force=True)
			url = entry.url
			url = unyt(url)
			sh = "cache/~" + shash(url) + ".pcm"
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
				srf = "silenceremove=start_periods=1:start_duration=1:start_threshold=-50dB:start_silence=0.5:stop_periods=-9000:stop_threshold=-50dB:window=0.015625"
				if "-filter_complex" in cmd:
					i = cmd.index("-filter_complex") + 1
					cmd = cmd[:i] + (cmd[i] + "," + srf,) + cmd[i + 1:]
				else:
					cmd += ("-af", srf)
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
	if player.paused:
		player.channel.pause()
	else:
		player.channel.resume()
	if toolbar.editor:
		player.broken = player.paused
	mixer.state(player.paused or toolbar.editor)
	if not player.paused:
		if reevaluating:
			reevaluating.mut.append(None)
			reevaluating.cancel()
		mut = []
		globals()["reevaluating"] = submit(reevaluate_in, 5, mut=mut)
		reevaluating.mut = mut
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
	ratio = 1 + 1 / (duration * 16)
	progress.alpha *= 0.998 ** (duration * 480)
	if progress.alpha < 16:
		progress.alpha = progress.num = 0
	progress.angle = -t * pi
	if progress.seeking:
		player.amp = 0.5
	elif not is_active():
		player.amp = 0
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
			while len(toolbar.ripples) >= 64:
				toolbar.ripples.pop(0)
			toolbar.ripples.append(cdict(
				pos=tuple(toolbar.pause.pos),
				radius=0,
				colour=c,
				alpha=255,
			))
	if toolbar.resizing:
		toolbar_height = max(64, screensize[1] - mpos2[1] + 2)
		if toolbar.get("snapped", 0) < 2 and abs(toolbar_height - 132) <= 12:
			toolbar.snapped = 1
			toolbar_height = 132
		elif toolbar.get("snapped", 0) == 1:
			toolbar.snapped = 2
		if options.toolbar_height != toolbar_height:
			options.toolbar_height = toolbar_height
			reset_menu()
			toolbar.resizing = True
	if toolbar_height > screensize[1] - 64:
		options.toolbar_height = toolbar_height = screensize[1] - 64
		reset_menu()
		toolbar.resizing = True
	if progress.seeking:
		orig = player.extpos()
		if player.end < inf:
			player.pos = pos = max(0, min(1, (mpos2[0] - progress.pos[0] + progress.width // 2) / progress.length) * player.end)
			player.offpos =-inf
			progress.num += (player.pos - orig)
		progress.alpha = 255
		player.index = player.pos * 30
		if not mheld[0]:
			progress.seeking = False
			if queue and isfinite(e_dur(queue[0].duration)):
				submit(seek_abs, player.extpos(), force=True)
	if sidebar.resizing:
		sidebar_width = min(screensize[0] - 8, max(144, screensize[0] - mpos2[0] + 2))
		if options.sidebar_width != sidebar_width:
			options.sidebar_width = sidebar_width
			reset_menu()
			sidebar.resizing = True
	if sidebar_width > screensize[0] - 8:
		options.sidebar_width = sidebar_width = screensize[0] - 8
		reset_menu()
		sidebar.resizing = True
	if queue and not toolbar.editor and isfinite(e_dur(queue[0].duration)) and not kheld[K_LSUPER] and not kheld[K_RSUPER]:
		if not player.get("seeking") or player.seeking.done():
			if kspam[K_UP]:
				player.seeking = submit(seek_rel, 30)
			elif kspam[K_DOWN]:
				player.seeking = submit(seek_rel, -30)
			elif kspam[K_RIGHT]:
				if CTRL[kheld]:
					player.seeking = submit(seek_abs, inf)
				else:
					player.seeking = submit(seek_rel, 5)
			elif kspam[K_LEFT]:
				if CTRL[kheld]:
					player.seeking = submit(seek_abs, 0)
				else:
					player.seeking = submit(seek_rel, -5)
			elif kspam[K_HOME]:
				player.seeking = submit(seek_abs, 0)
			elif kspam[K_END]:
				player.seeking = submit(seek_abs, inf)
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
	if in_rect(mpos, toolbar.rect[:3] + (6,)):
		if mclick[0]:
			toolbar.resizing = True
		else:
			toolbar.resizer = True
	if in_circ(mpos, toolbar.pause.pos, max(4, toolbar.pause.radius - 2)):
		if any(mclick):
			pause_toggle()
		toolbar.pause.outer = 255
		toolbar.pause.inner = 191
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
			def seek_position_a(enter):
				if enter:
					pos = time_parse(enter)
					submit(seek_abs, pos)
			easygui2.enterbox(
				seek_position_a,
				"Seek to position",
				title="Miza Player",
				default=time_disp(player.extpos()),
			)
	c = options.get("toolbar_colour", (64, 0, 96))
	if toolbar.resizing or in_rect(mpos, toolbar.rect):
		hls = colorsys.rgb_to_hls(*(i / 255 for i in c[:3]))
		hls = (hls[0], max(0, hls[1] + 1 / 24),  hls[2] / 1.2)
		tc = verify_colour(round(i * 255) for i in colorsys.hls_to_rgb(*hls))
		if len(c) > 3:
			tc.append(c[3])
			c = tc
		else:
			c = tc
	toolbar.colour = c
	maxb = (options.sidebar_width - 12) // 44 + len(toolbar.buttons)
	buttons = toolbar.buttons.concat(sidebar.buttons)
	for i, button in enumerate(buttons):
		try:
			if button.get("flash"):
				# print(i, button.flash)
				button.flash = max(0, button.flash - duration * 64)
			if in_rect(mpos, button.rect):
				button.flash = 16
				clicked = i < maxb and any(mclick)
			else:
				clicked = False
		except AttributeError:
			print_exc()
			continue
		if not clicked:
			hotkey = button.get("hotkey")
			if hotkey:
				clicked = True
				for k in hotkey[:-1]:
					if callable(k):
						if not k(kheld):
							clicked = False
							break
					else:
						if not kheld[k]:
							clicked = False
							break
				else:
					k = hotkey[-1]
					if callable(k):
						if not k(kclick):
							clicked = False
					else:
						if not kclick[k]:
							clicked = False
		if clicked:
			button.flash = 64
			click = button.click if not toolbar.editor else button.get("click2") or button.click
			sidebar.menu = 0
			if callable(click):
				click()
			elif 1 not in mclick:
				click[0]()
			elif click:
				click[min(mclick.index(1), len(click) - 1)]()

	if in_rect(mpos, sidebar.rect[:2] + (6, sidebar.rect[3])):
		if not toolbar.resizing and mclick[0]:
			sidebar.resizing = True
		else:
			sidebar.resizer = True
	c = options.get("sidebar_colour", (64, 0, 96))
	if sidebar.resizing or in_rect(mpos, sidebar.rect):
		hls = colorsys.rgb_to_hls(*(i / 255 for i in c[:3]))
		hls = (hls[0], max(0, hls[1] + 1 / 24),  hls[2] / 1.2)
		sc = verify_colour(round(i * 255) for i in colorsys.hls_to_rgb(*hls))
		if len(c) > 3:
			sc.append(c[3])
			c = sc
		else:
			c = sc
	sidebar.colour = c
	sidebar.relpos = (sidebar.get("relpos", 0) * (ratio - 1) + bool(sidebar.abspos)) / ratio
	scroll_height = screensize[1] - toolbar_height - 72
	sidebar.scroll.rect = (screensize[0] - 20, 52 + 16, 12, scroll_height)
	scroll_rat = max(12, min(scroll_height, scroll_height / max(1, len(queue)) * (screensize[1] - toolbar_height - 36) / 32))
	scroll_pos = sidebar.scroll.pos / (32 * max(1, len(queue)) - screensize[1] + toolbar_height + 52 + 16) * (scroll_height - scroll_rat) + 52 + 16
	sidebar.scroll.select_rect = (sidebar.scroll.rect[0], scroll_pos, sidebar.scroll.rect[2], scroll_rat)
	c = options.get("sidebar_colour", (64, 0, 96))
	hls = colorsys.rgb_to_hls(*(i / 255 for i in c[:3]))
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
			def scroll_position_a(enter):
				if enter:
					pos = float(enter) * 32
					sidebar.scroll.target = pos
			easygui2.enterbox(
				scroll_position_a,
				"Scroll to position",
				title="Miza Player",
				default=str(round(sidebar.scroll.target / 32)),
			)
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
		[e.pop("lyrics", None) for e in queue]
		# lyrics_cache.clear()
		lyrics_renders.clear()
		# for i in range(2):
			# if i >= len(queue):
				# break
			# submit(render_lyrics, queue[i])
	if toolbar.editor:
		update_piano()


def get_ripple(i, mode="toolbar"):
	if i >= 4:
		return (255,) * 3
	if i == 3:
		return (0,) * 3
	c = options.get(f"{mode}_colour", (64, 0, 96))
	if not i:
		d = 0
	else:
		d = i * 2 - 3
	h, l, s = colorsys.rgb_to_hls(*(i / 255 for i in c[:3]))
	h = (h + d * 5 / 12) % 1
	s = 1 - (1 - s) / 2
	l = 1 - (1 - l) / 2
	return [round(i * 255) for i in colorsys.hls_to_rgb(h, l, s)]


# importing doesn't work; target files are simply functions that used to be here and cannot independently run without sharing global variables
for fn in ("sidebar", "sidebar2", "render", "toolbar", "piano"):
	with open(f"misc/{fn}.py", "rb") as f:
		b = f.read()
	exec(compile(b, f"{fn}.py", "exec"), globals())


no_lyrics_path = options.get("no_lyrics_path", "misc/Default/no_lyrics.png")
no_lyrics_fut = submit(load_surface, no_lyrics_path)

globals()["h-cache"] = {}
globals()["h-timer"] = 0

def load_bubble(bubble_path=None):
	try:
		if not bubble_path:
			surf = concentric_circle(
				None,
				(255,) * 3,
				(0, 0),
				256,
				fill_ratio=1 / 3,
			)
			globals()["h-ripi"] = pyg2pgl(surf)
			return
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
				im2 = im2.point(lambda x: round((x / 255) ** 0.8 * 255))
			if "RGB" not in im2.mode:
				im2 = im2.convert("RGB")
			if im.mode == "RGBA":
				im2.putalpha(A)
			globals()["h-img"] = im2
		globals()["h-ripi"] = pil2pgl(globals()["h-img"])
	except:
		print_exc()

load_bubble()
bubble_path = options.get("bubble_path")
if bubble_path:
	submit(load_bubble, bubble_path)

def load_spinner(spinner_path):
	try:
		globals()["s-img"] = Image.open(spinner_path)
		globals()["s-cache"] = {}

		def spinner_ripple(dest, colour, pos, radius, alpha=255, z=0, **kwargs):
			diameter = round_random(radius * 2)
			if not diameter > 0:
				return
			try:
				surf = globals()["s-cache"][diameter]
			except KeyError:
				im = globals()["s-img"].resize((diameter,) * 2, resample=Resampling.LANCZOS)
				if "RGB" not in im.mode:
					im = im.convert("RGBA")
				surf = pil2pyg(im)
				im.close()
				globals()["s-cache"][diameter] = surf
			blit_complex(
				dest,
				surf,
				[round_random(x - y / 2) for x, y in zip(pos, surf.get_size())],
				alpha=alpha,
				colour=colour,
				z=z,
			)
		globals()["s-ripple"] = spinner_ripple
	except:
		print_exc()

spinner_path = "misc/Default/bubble.png"
submit(load_spinner, spinner_path)


def change_bubble():
	def change_bubble_a(bubble_path):
		if bubble_path:
			options.bubble_path = bubble_path
			submit(load_bubble, bubble_path)
	easygui2.fileopenbox(
		change_bubble_a,
		"Open an image file here!",
		"Miza Player",
		default=options.get("bubble_path", globals()["bubble_path"]),
		filetypes=iftypes,
	)

def reset_bubble():
	options.pop("bubble_path", None)
	load_bubble()


@DISP.event
def on_mouse_scroll(x, y, vx, vy):
	if in_rect(mpos, sidebar.rect):
		sidebar.scroll.target -= vy * 48
	elif toolbar.editor and in_rect(mpos, player.rect):
		player.editor.targ_y += vy * 3.5
		player.editor.targ_x += vx * 3.5

@DISP.event
def on_file_dropped(x, y, paths):
	print(paths)

# event_logger = pyglet.window.event.WindowEventLogger()
# DISP.push_handlers(event_logger)

def render_settings(dur, ignore=False):
	global crosshair, hovertext
	offs = round(sidebar.setdefault("relpos", 0) * -sidebar_width)
	sc = tuple(sidebar.colour or (64, 0, 96))
	if DISP.transparent:
		sc += (223,)
	sub = (sidebar.rect2[2], sidebar.rect2[3] - 52)
	subp = (screensize[0] - sidebar_width, 52)
	DISP2 = DISP.subsurf(subp + sub)
	in_sidebar = in_rect(mpos, sidebar.rect)
	offs2 = offs + sidebar_width
	c = sc
	hc = high_colour(c)
	if sidebar.scroll.get("colour"):
		rounded_bev_rect(
			DISP2,
			sidebar.scroll.background,
			(sidebar.scroll.rect[0] + offs - screensize[0] + sidebar_width, sidebar.scroll.rect[1]) + sidebar.scroll.rect[2:],
			4,
			z=129,
		)
		rounded_bev_rect(
			DISP2,
			sidebar.scroll.colour,
			(sidebar.scroll.select_rect[0] + offs - screensize[0] + sidebar_width, sidebar.scroll.select_rect[1]) + sidebar.scroll.select_rect[2:],
			4,
			z=129,
		)
	for i, opt in enumerate(asettings):
		message_display(
			opt.capitalize(),
			15,
			(offs2 + 8, i * 32 - 2),
			colour=hc,
			surface=DISP2,
			align=0,
			cache=True,
			font="Comic Sans MS",
			z=129,
		)
		# numrect = (screensize[0] + offs + sidebar_width - 8, 68 + i * 32)
		s = str(round(options.audio.get(opt, 0) * 100, 2)) + "%"
		message_display(
			s,
			13,
			(offs2 + sidebar_width - 8, 17 + i * 32),
			colour=hc,
			surface=DISP2,
			align=2,
			cache=True,
			font="Comic Sans MS",
			z=129,
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
			rv = round_min(math.round(v * 32) / 16)
			if type(rv) is int and rv not in srange:
				v = rv / 2
		if hovered and not hovertext or aediting[opt]:
			hovertext = cdict(
				text=str(round(v * 100, 2)) + "%",
				size=13,
				colour=(255, 255, 127),
			)
			if aediting[opt]:
				if not mheld[0]:
					aediting[opt] = False
			elif mclick[0]:
				aediting[opt] = True
			elif mclick[1]:
				def opt_set_a(enter, opt):
					if enter:
						v = round_min(float(safe_eval(enter)) / 100)
						aediting[opt] = True
						orig, options.audio[opt] = options.audio[opt], v
						if orig != v:
							mixer.submit(f"~setting {opt} {v}", force=ignore or opt == "volume" or not queue)
				easygui2.enterbox(
					opt_set_a,
					opt.capitalize(),
					"Miza Player",
					str(round_min(options.audio[opt] * 100)),
					args=(opt,),
				)
			if aediting[opt]:
				orig, options.audio[opt] = options.audio[opt], v
				if orig != v:
					mixer.submit(f"~setting {opt} {v}", force=ignore or opt == "volume" or not queue)
		z = max(0, x - 4)
		if x < w:
			rect = (offs2 + 8 + z, 19 + i * 32, sidebar_width - 16 - z, 9)
			col = (48 if hovered else 32,) * 3
			bevel_rectangle(
				DISP2,
				col,
				rect,
				3,
				z=129,
			)
			rainbow = quadratic_gradient((w, 9), pc() / 2 + i / 4, unique=True)
			DISP2.blit(
				rainbow,
				(offs2 + 8 + x, 19 + i * 32),
				(x, 0, w - x, 9),
				special_flags=BLEND_RGB_MULT,
				z=130,
			)
		if x > 0:
			rect = (offs2 + 8, 19 + i * 32, x, 9)
			col = (223 if hovered else 191,) * 3
			bevel_rectangle(
				DISP2,
				col,
				rect,
				3,
				z=131,
			)
			rainbow = quadratic_gradient((w, 9), pc() + i / 4, unique=True)
			DISP2.blit(
				rainbow,
				(offs2 + 8, 19 + i * 32),
				(0, 0, x, 9),
				special_flags=BLEND_RGB_MULT,
				z=132,
			)
		if hovered:
			bevel_rectangle(
				DISP2,
				progress.select_colour,
				brect2,
				2,
				filled=False,
				z=132,
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
		z=129,
	)
	message_display(
		"Reset",
		16,
		[offs2 + sidebar_width // 2, 318],
		(255,) * 3,
		font="Comic Sans MS",
		surface=DISP2,
		cache=True,
		z=130,
	)
	if "more" not in sidebar:
		sidebar.more = sidebar.more_angle = 0
	if sidebar.more_angle > 0.001:
		more = (
			("silenceremove", "Skip silence", "Skips over silent or extremely quiet frames of audio."),
			("unfocus", "Reduce unfocus FPS", "Greatly reduces FPS of display when the window is left unfocused."),
			("subprocess", "Subprocess", "Audio is transferred through subprocess. More stable and efficient, but difficult to record or stream using third party programs."),
			("presearch", "Preemptive search", "Pre-emptively searches up and displays duration of songs in a playlist when applicable. Increases amount of requests being sent."),
			("preserve", "Preserve sessions", "Preserves sessions and automatically reloads them when the program is restarted."),
			("blur", "Motion blur", "Makes animations look smoother, but consume slightly more resources."),
			("transparency", "Window transparency", "Makes the window see-through (Requires restart)."),
			("ripples", "Ripples", "Clicking anywhere on the sidebar or toolbar produces a visual ripple effect."),
			("autobackup", "Auto backup", "Automatically backs up and/or restores your playlist folder on mizabot.xyz."),
			("autoupdate", "Auto update", "Automatically updates Miza Player in the background when an update is detected."),
		)
		mrect = (offs2 + 8, 376, sidebar_width - 16, 32 * 10)
		surf = HWSurface.any(mrect[2:], FLAGS | pygame.SRCALPHA)
		surf.fill((0, 0, 0, 0))
		for i, t in enumerate(more):
			s, name, description = t
			apos = (screensize[0] - sidebar_width + offs2 + 24, 427 + i * 32 + 16)
			hovered = hypot(*(np.array(mpos) - apos)) < 16
			if hovered:
				hovertext = cdict(
					text=description,
					size=13,
					colour=(255, 255, 255),
					background=(0, 0, 0, 128),
					offset=0,
					align=2,
				)
				if mclick[0]:
					try:
						options.control[s] ^= 1
					except KeyError:
						options.control[s] = 1
					if s in ("silenceremove", "unfocus", "subprocess"):
						mixer.submit(f"~setting {s} {options.control[s]}")
					elif options.control[s]:
						if s == "autobackup":
							globals()["last_sync"] = -inf
						elif s == "autoupdate":
							globals()["last_save"] = -inf
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
				z=129,
			)
			ripple_f(
				surf,
				colour=col,
				pos=pos,
				radius=16,
				fill_ratio=0.5,
				z=130,
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
				z=129,
			)
		if sidebar_width >= 192 and screensize[1] - toolbar_height - mrect[1] >= 32 * 10:
			r = (sidebar_width - 80, 256, 64, 32)
			r2 = (screensize[0] - 68 + offs + sidebar_width, 685, 64, 32)
			if in_rect(mpos, r2):
				if mclick[0]:
					def edit_sync_url_a(enter):
						if enter is None:
							return
						enter = enter.strip()
						try:
							assert is_url(enter)
						except:
							print_exc()
							enter = ""
						else:
							enter = enter.replace("/file/", "/p/").replace("/f/", "/p/").replace("/d/", "/p/").replace("/view/", "/p/")
						control.playlist_sync = enter
						globals()["last_sync"] = -inf
					easygui2.enterbox(
						edit_sync_url_a,
						"Edit sync URL",
						title="Miza Player",
						default=control.playlist_sync,
					)
				c = (112, 127, 64, 223)
			else:
				c = (96, 112, 80, 127)
			bevel_rectangle(surf, c, r, bevel=4, z=129)
			message_display(
				"Edit",
				16,
				rect_centre(r),
				colour=(255,) * 3,
				alpha=c[-1] + 32,
				surface=surf,
				font="Comic Sans MS",
				cache=True,
				z=130,
			)
		if sidebar_width >= 192 and screensize[1] - toolbar_height - mrect[1] >= 32 * 11:
			r = (sidebar_width - 80, 288, 64, 32)
			r2 = (screensize[0] - 68 + offs + sidebar_width, 717, 64, 32)
			if in_rect(mpos, r2):
				if mclick[0]:
					submit(update_collections2)
					common.repo_fut = submit(update_repo, force=True)
					if common.repo_fut.result():
						easygui2.msgbox(
							None,
							"No new updates found.",
							title="Miza Player",
						)
				c = (112, 127, 64, 223)
			else:
				c = (96, 112, 80, 127)
			fut = common.__dict__.get("repo-update")
			if fut and not isinstance(fut, bool):
				c2 = verify_colour(round_random(x * (sin(pc() * tau / 4) + 0.5)) for x in c[:3])
				c2.append(c[-1])
				c = c2
			bevel_rectangle(surf, c, r, bevel=4, z=129)
			message_display(
				"Update",
				16,
				rect_centre(r),
				colour=(255,) * 3,
				alpha=c[-1] + 32,
				surface=surf,
				font="Comic Sans MS",
				cache=True,
				z=130,
			)
		if sidebar.more_angle < 63 / 64:
			arr = np.linspace(sidebar.more_angle * 510, sidebar.more_angle * 510 - 255, mrect[3])
			np.clip(arr, 0, 255, out=arr)
			arr = arr.astype(np.uint8)
			im = pyg2pil(surf)
			a = im.getchannel("A")
			a2 = Image.fromarray(arr, "L").resize(mrect[2:], resample=Resampling.NEAREST)
			A = ImageChops.multiply(a, a2)
			im.putalpha(A)
			surf = pil2pyg(im)
		DISP2.blit(
			surf,
			mrect[:2],
			redraw=True,# sidebar.more_angle < 63 / 64 or in_rect((mpos[0] - DISP2.rect[0], mpos[1] - DISP2.rect[1]), mrect),
			z=131,
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
	hls = list(colorsys.rgb_to_hls(*(i / 255 for i in c[:3])))
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
		z=129,
	)
	kwargs = {}
	if not sidebar.ripples and not DISP.transparent:
		kwargs["soft"] = sidebar.colour
	pos = (offs2 + sidebar_width // 2 - 48 - 6, 357 + sidebar.more_angle * 6)
	reg_polygon_complex(
		DISP2,
		pos,
		(255,) * 3,
		3,
		12,
		12,
		pi / 4 + sidebar.more_angle * pi,
		255,
		2,
		9,
		filled=True,
		z=130,
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
		z=130,
	)

def draw_menu():
	global crosshair, hovertext
	ts = toolbar.progress.setdefault("timestamp", 0)
	t = pc()
	dur = max(0.001, min(t - ts, 0.125))
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
	if any(mclick) and in_rect(mpos, sidebar.rect):
		if control.ripples:
			while len(sidebar.ripples) >= 64:
				sidebar.ripples.pop(0)
			sidebar.ripples.append(cdict(
				pos=tuple(mpos),
				radius=0,
				colour=get_ripple(mclick.index(1), mode="sidebar"),
				alpha=255,
			))
		if mclick[1] and sidebar.menu is None:

			def set_colour():
				def get_colour_a(enter):
					if enter:
						enter = int(enter.lstrip("#"), 16)
						if enter >= 16777216:
							enter = ((enter >> 24) & 255, (enter >> 16) & 255, (enter >> 8) & 255, enter & 255)
						else:
							enter = ((enter >> 16) & 255, (enter >> 8) & 255, enter & 255)
						if isinstance(enter, tuple) and len(enter) in (3, 4):
							options.sidebar_colour = enter
				v = options.get("sidebar_colour", (64, 0, 96))
				easygui2.enterbox(
					get_colour_a,
					"Change sidebar colour",
					title="Miza Player",
					default="#" + "".join(("0" + hex(i)[2:])[-2:] for i in v).upper(),
				)

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
	cond = False
	if sidebar.particles or sidebar.ripples or sidebar.get("dragging") or sidebar.scroll.pos != sidebar.scroll.target or sidebar.abspos or sidebar.menu:
		cond = True
	elif not is_unfocused() and in_rect(mpos2, sidebar.rect) and (mpos != mpprev or CTRL(kheld) and (kclick[K_a] or kclick[K_s] or mclick[0])) or sidebar.get("last_selected") is not None:
		cond = True
	else:
		if toolbar.editor:
			q = sidebar.instruments
		else:
			q = queue
		cond = any(i != e.get("pos", 0) or e.get("selected") or e.get("flash") for i, e in enumerate(q[sidebar.base:sidebar.base + sidebar.maxitems], sidebar.base))
	if cond:
		globals()["last-cond"] = 2
	elif globals().get("last-cond"):
		globals()["last-cond"] -= 1
		if globals()["last-cond"] <= 0:
			globals().pop("last-cond")
		cond = True
	elif tick < 3 or DISP.lastclear or options.control.get("blur") and DISP.transparent:
		cond = globals()["last-cond"] = True
	if cond and sidebar.colour:
		if toolbar.editor:
			render_sidebar_2(dur)
		else:
			render_sidebar(dur)
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
					z=288,
				)
			sidebar.particles.pops(pops)
		else:
			sidebar.particles.clear()
	if any(mclick) and in_rect(mpos, toolbar.rect):
		if control.ripples:
			while len(toolbar.ripples) >= 64:
				toolbar.ripples.pop(0)
			toolbar.ripples.append(cdict(
				pos=tuple(mpos),
				radius=0,
				colour=get_ripple(mclick.index(1), mode="toolbar"),
				alpha=255,
			))
		if mclick[1] and sidebar.menu is None:

			def set_colour():
				def get_colour_a(enter):
					if enter:
						enter = int(enter.lstrip("#"), 16)
						if enter >= 16777216:
							enter = ((enter >> 24) & 255, (enter >> 16) & 255, (enter >> 8) & 255, enter & 255)
						else:
							enter = ((enter >> 16) & 255, (enter >> 8) & 255, enter & 255)
						if isinstance(enter, tuple) and len(enter) in (3, 4):
							options.toolbar_colour = enter
				v = options.get("toolbar_colour", (64, 0, 96))
				easygui2.enterbox(
					get_colour_a,
					"Change toolbar colour",
					title="Miza Player",
					default="#" + "".join(("0" + hex(i)[2:])[-2:] for i in v).upper(),
				)


			sidebar.menu = cdict(
				buttons=(
					("Set Colour", set_colour),
					("Change ripples", change_bubble),
					("Reset ripples", reset_bubble),
				),
			)
	if not toolbar.editor and (mclick[0] or mclick[1]):
		text_rect = (0, 0, 192, 92)
		if in_rect(mpos, text_rect):
			if mclick[1]:
				pass
			else:
				player.flash_i = 32
				options.insights = (options.get("insights", 0) + 1) & 1
				mixer.submit(f"~setting insights {options.insights}")
		elif in_rect(mpos, player.rect):
			if mclick[1]:
				s = options.get("spectrogram")
				if s == 1:
					def change_font_size():
						def change_font_size_a(enter):
							if enter:
								enter = eval(enter, {}, {})
								options.control.lyrics_size = enter
								common.font_reload = True
						v = options.control.lyrics_size
						easygui2.enterbox(
							change_font_size_a,
							"Change font size",
							title="Miza Player",
							default=str(v),
						)
					def change_image():
						def change_image_a(no_lyrics_path):
							if no_lyrics_path:
								options.no_lyrics_path = no_lyrics_path
								globals()["no_lyrics_fut"] = submit(load_surface, no_lyrics_path, force=True)
								globals().pop("no_lyrics", None)
						easygui2.fileopenbox(
							change_image_a,
							"Open an image file here!",
							"Miza Player",
							default=options.get("no_lyrics_path", globals()["no_lyrics_path"]),
							filetypes=iftypes,
						)
					def search_lyrics():
						def search_lyrics_a(query):
							if query:
								globals()["lyrics_entry"] = entry = cdict(name=query)
								submit(render_lyrics, entry)
						easygui2.enterbox(
							search_lyrics_a,
							"Search a song to get its lyrics!",
							title="Miza Player",
							default="",
						)
					def reset_lyrics():
						globals()["lyrics_entry"] = None

					sidebar.menu = cdict(
						buttons=(
							("Copy", lambda: pyperclip.copy(lyrics_cache[queue[0].name][1])),
							("Change image", change_image),
							("Change font size", change_font_size),
							("Search lyrics", search_lyrics),
							("Reset lyrics", reset_lyrics),
						),
					)
				elif s == 4:
					def _change_polytope(v):
						options.control["gradient-vertices"] = v
						mixer.submit(f"~setting #gradient-vertices {v}")
					def change_polytope():
						sidebar.menu = cdict(
							buttons=((k, _change_polytope, v) for v, k in tuple(poly_inv.items())[:44]),
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
						def change_schlafli_a(enter):
							if enter:
								enter = enter.strip("<>()[]{}").casefold().replace(",", " ").replace(":", " ")
								try:
									enter = poly_names[enter]
								except KeyError:
									if not os.path.exists(enter):
										enter = [eval(x, {}, {}) for x in enter.split()]
								options.control["gradient-vertices"] = enter
								mixer.submit(f"~setting #gradient-vertices {enter}")
						easygui2.enterbox(
							change_schlafli_a,
							"Change polytope",
							title="Miza Player",
							default=str(v),
						)
					def change_model():
						ftypes = [[f"*.{f}" for f in "obj gz".split()]]
						default = options.control.get("gradient-vertices")
						if not isinstance(default, str) or not os.path.exists(default):
							default = None
						default = default or "misc/default/sphere.obj"
						def change_model_a(enter):
							if enter and os.path.exists(enter):
								options.control["gradient-vertices"] = enter
								mixer.submit(f"~setting #gradient-vertices {enter}")
						easygui2.fileopenbox(
							change_model_a,
							"Open a 3D model file here!",
							"Miza Player",
							default=default,
							filetypes=ftypes,
						)

					sidebar.menu = cdict(
						buttons=(
							("Select polytope", change_polytope),
							("Enter schlafli symbol", change_schlafli),
							("Load 3D model", change_model),
						),
					)
				elif s == 5:
					def change_vertices():
						vertices = options.control["spiral-vertices"]
						v = vertices[0]
						def change_vertices_a(enter):
							if enter:
								enter = int(float(enter))
								if enter > 384:
									enter = 384
								vertices[0] = enter
								mixer.submit(f"~setting #spiral-vertices {vertices}")
						easygui2.enterbox(
							change_vertices_a,
							"Change vertex count",
							title="Miza Player",
							default=str(v),
						)
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
				options.spectrogram = (options.get("spectrogram", 0) + 1) % 6
				mixer.submit(f"~setting spectrogram {options.spectrogram - 1}")
				if options.spectrogram == 1 and queue:
					# submit(render_lyrics, queue[0])
					if player.get("spec"):
						player.spec.fill((0, 0, 0))
	if sidebar.colour:
		# if globals().get("sidebar_rendered"):
			# sidebar_rendered.result()
			# globals()["sidebar_rendered"] = None
		maxb = (sidebar_width - 12) // 44
		c = options.get("sidebar_colour", (64, 0, 96))
		hls = list(colorsys.rgb_to_hls(*(i / 255 for i in c[:3])))
		light = 1 - (1 - hls[1]) / 4
		if hls[2]:
			sat = 1 - (1 - hls[2]) / 2
		else:
			sat = 0
		lum = 175
		hls[1] = lum / 255 * light
		hls[2] = sat
		col = [round(i * 255) for i in colorsys.hls_to_rgb(*hls)]
		for i, button in enumerate(sidebar.buttons[:maxb]):
			lum = 175
			if not button.get("rect"):
				continue
			redraw = False
			if in_rect(mpos, button.rect):
				lum = 239
				cm = abs(pc() % 1 - 0.5) * 0.328125
				c2 = [round(i * 255) for i in colorsys.hls_to_rgb(cm + 0.75, 0.75, 1)]
				name = button.name if not toolbar.editor else button.get("name2") or button.name
				hovertext = cdict(
					text=name,
					size=15,
					background=high_colour(c2),
					colour=c2,
					offset=19,
					font="Rockwell",
				)
				crosshair |= 4
				redraw = True
			elif button.get("flash"):
				redraw = True
			elif not i and not sidebar.abspos:
				redraw = True
			if redraw:
				hls[1] = lum / 255 * light
				lum += button.get("flash", 0)
				if not i and not sidebar.abspos:
					lum -= 48
					lum += button.get("flash", 0)
					fut = common.__dict__.get("repo-update")
					if fut and not isinstance(fut, bool):
						hls[1] = (sin(pc() * tau / 4) + 1) / 2
					elif not cond:
						break
				tcol = [round(i * 255) for i in colorsys.hls_to_rgb(*hls)]
			else:
				tcol = col
			bevel_rectangle(
				DISP,
				tcol,
				button.rect,
				4,
				z=257,
				cache=not redraw,
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
				z=258,
			)
			if not cond:
				break
	osci_rect = (screensize[0] - 4 - progress.box, screensize[1] - toolbar_height + 4) + osize
	if mclick[0] and in_rect(mpos, osci_rect) and not toolbar.resizer:
		player.flash_o = 32
		options.oscilloscope = (options.get("oscilloscope", 0) + 1) % 2
		mixer.submit(f"~setting oscilloscope {options.oscilloscope}")
	if toolbar.editor:
		editor_toolbar()
	if sidebar.get("dragging"):
		if toolbar.editor:
			render_dragging_2()
		else:
			render_dragging()
	if toolbar.resizing:
		if toolbar.get("snapped", 0) == 1:
			c = (0, 255, 0)
		else:
			c = (255, 0, 0)
		draw_rect(DISP, c, toolbar.rect[:3] + (6,), z=640)
		if not mheld[0]:
			toolbar.resizing = False
			toolbar.snapped = 0
	elif toolbar.resizer:
		draw_rect(DISP, (191, 127, 255), toolbar.rect[:3] + (6,), z=640)
		toolbar.resizer = False
	if sidebar.resizing:
		draw_rect(DISP, (255, 0, 0), sidebar.rect[:2] + (6, sidebar.rect[3]), z=640)
		if not mheld[0]:
			sidebar.resizing = False
	elif sidebar.resizer:
		draw_rect(DISP, (191, 127, 255), sidebar.rect[:2] + (6, sidebar.rect[3]), z=640)
		sidebar.resizer = False
	if crosshair:
		if crosshair & 3:
			DISP.shape(pyglet.shapes.Line, *(mpos2[0] - 13, mpos2[1] - 1), *(mpos2[0] + 11, mpos2[1] - 1), width=2, color=(255, 0, 0), z=641)
			DISP.shape(pyglet.shapes.Line, *(mpos2[0] - 1, mpos2[1] - 13), *(mpos2[0] - 1, mpos2[1] + 11), width=2, color=(255, 0, 0), z=641)
			draw_circle(DISP, (255, 0, 0), mpos2, 9, width=2, z=641)
		if crosshair & 1:
			p = max(0, min(1, (mpos2[0] - progress.pos[0] + progress.width // 2) / progress.length) * player.end)
			s = time_disp(p)
			message_display(
				s,
				min(20, toolbar_height // 3),
				(mpos2[0], mpos2[1] - 17),
				(255, 255, 127),
				surface=DISP,
				font="Comic Sans MS",
				z=642,
			)
	if hovertext:
		message_display(
			hovertext.text,
			hovertext.size,
			(mpos2[0], mpos2[1] + hovertext.get("offset", -17)),
			hovertext.colour,
			background=hovertext.get("background", None),
			surface=DISP,
			font=hovertext.get("font", "Comic Sans MS"),
			align=hovertext.get("align", 1),
			cache=True,
			z=642,
			clip=True,
		)

pdata = None
def save_settings(closing=False):
	# print("autosaving...")
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
			pos=player.extpos(),
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
		with open("dump2.json", "w", encoding="utf-8") as f:
			json.dump(data, f, separators=(",", ":"), default=json_default)
		if os.path.exists("dump.json"):
			os.remove("dump.json")
		os.rename("dump2.json", "dump.json")
	options.screensize = temp
	for e in os.scandir("cache"):
		fn = e.name
		if e.is_file(follow_symlinks=False):
			if fn.endswith(".webm") or fn.endswith(".wav"):
				if fn[0] in "\x7f&":
					submit(os.remove, e.path)
				elif fn[0] == "~":
					s = e.stat()
					if s.st_size <= 1024 or s.st_size > 268435456 or utc() - s.st_atime > 86400 * 3 or utc() - s.st_mtime > 86400 * 14 or closing and utc() - s.st_mtime < 5:
						submit(os.remove, e.path)
			else:
				try:
					os.remove(e.path)
				except:
					pass
	if os.path.exists("misc/cache"):
		for fn in os.listdir("misc/cache"):
			try:
				os.remove("misc/cache/" + fn)
			except:
				pass
	if os.path.exists("misc/temp.tmp"):
		try:
			os.remove("misc/temp.tmp")
		except:
			pass


orig = code = ""
reset_menu()
foc = True
minimised = False
mpos = mpos2 = mpprev = (-inf,) * 2
mheld = mclick = mrelease = mprev = (None,) * 5
kheld = pygame.key.get_pressed()
kprev = kclick = KeyList((None,)) * len(kheld)
delay = 0
last_tick = 0
last_played = 0
last_precise = 0
last_ratio = 0
status_freq = 6000
alphakeys = [False] * 34
restarting = False
DISP.fps = 0
fps = 60
lp = None
addp.result()

try:
	if "-nd" not in sys.argv:
		submit(distribute_in, 0 if "-d" in sys.argv else 300)
	if options.control.preserve and os.path.exists("dump.json"):
		ytdl = downloader.result()
		with open("dump.json", "rb") as f:
			data = json.load(f)
		if queue:
			data.pop("pos", None)
		for e in data.get("queue", ()):
			e.pop("novid", None)
			if "mizabot.xyz/ytdl" in (e.get("stream") or ""):
				e.pop("stream", None)
				e.pop("video", None)
			if e.get("url"):
				url = e["url"]
				if url not in ytdl.searched:
					ytdl.searched[url] = cdict(t=time.time(), data=[astype(cdict, e)])
		entries = [cdict(e, duration=e.get("duration")) for e in data.get("queue", ())]
		queue.extend(entries)
		if data.get("editor"):
			player.editor.update(data["editor"])
			player.editor.note = cdict(player.editor.note)
		if data.get("paused"):
			pause_toggle(True)
		if data.get("minimised"):
			DISP.minimize()
		DISP.set_visible(True)
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
			# submit(render_lyrics, queue[0])
	else:
		DISP.set_visible(True)
	toolbar.editor = 0
	tick = 0
	while True:
		foc = get_focused()
		unfocused = False
		if foc:
			minimised = False
		else:
			minimised = is_minimised()
		unfocused = is_unfocused()
		if not tick + 1 & 31:
			try:
				if utc() - max(os.path.getmtime(collections2f), os.path.getatime(collections2f)) > 3600:
					raise FileNotFoundError
			except FileNotFoundError:
				common.repo_fut = submit(update_repo)
				submit(update_collections2)
			# print(t, last_save, last_sync)
			t = pc()
			if t >= last_save + 20 and is_active():
				submit(save_settings)
				globals()["last_save"] = max(last_save, t - 10)
			if t >= last_sync + 730 and control.autobackup:
				submit(playlist_sync)
				globals()["last_sync"] = max(last_sync, t - 10)
		if not (tick << 3) % (status_freq + (status_freq & 1)) or minimised and not tick + 1 & 7:
			submit(send_status)
		if not tick + 1 & 127 and pc() - globals().get("last_persist", -inf) > 3600:
			if queue and utc() - has_api < 60:
				last_persist = pc()
				submit(persist)
			# print("UTC:", utc())
		if not tick & 7 or minimised:
			if ECDC_QUEUE and len(ECDC_RUNNING) < 5:
				entry, stream, force = ECDC_QUEUE.pop(next(iter(ECDC_QUEUE)))
				fut = submit(ecdc_compress, entry, stream, force=force)
				if force is not None:
					ECDC_CURR = fut
		fut = common.__dict__.get("repo-update")
		if fut:
			if fut is True:
				if not options.control.autoupdate:
					if options.control.preserve:
						easygui2.msgbox(
							None,
							"Miza Player has been updated successfully!\nThe program will now restart in order to apply changes.",
							title="Success!",
						)
					else:
						easygui2.msgbox(
							None,
							"Miza Player has been updated successfully!\nPlease restart the program in order to apply changes.",
							title="Success!",
						)
				if options.control.preserve:
					restarting = True
					raise StopIteration
				common.__dict__.pop("repo-update", None)
		if not minimised and not tick % 2401:
			garbage_collector()
		v = minimised | unfocused << 1
		globals()["stat-mem"].buf[0] = v
		if not downloading.target:
			if player.paused:
				colour = 4
			elif is_active() and player.amp > 1 / 64:
				colour = 2
			else:
				colour = 8
			taskbar_progress_bar(player.extpos() / player.end, colour | (player.end >= inf))
		else:
			pgr = os.path.exists(downloading.fn) and os.path.getsize(downloading.fn) / 192000 * 8
			ratio = min(1, pgr / max(1, downloading.target))
			taskbar_progress_bar(ratio, 4 if ratio < 1 / 3 else 8 if ratio < 2 / 3 else 2)
		if minimised:
			toolbar.ripples.clear()
			sidebar.ripples.clear()
		if lp:
			lp.result()
		if not project.instruments:
			add_instrument(True)
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
		condition = not minimised or any(DISP.krelease) or any(DISP.mrelease)
		if toolbar.ripples or sidebar.ripples:
			condition = True
		if condition:
			if getattr(DISP, "reset", False):
				reset_menu(reset=True)
				DISP.reset = False
			lpos = mpos
			mprev = mheld
			mheld = list(DISP.mheld)
			mclick = list(DISP.mclick)
			DISP.mclick[:] = [0] * len(DISP.mclick)
			mrelease = list(DISP.mrelease)
			DISP.mrelease[:] = [0] * len(DISP.mrelease)
			mpprev = tuple(mpos2)
			mpos2 = mouse_rel_pos()
			if foc:
				mpos = mpos2
			else:
				mpos = (nan,) * 2
			kprev = kheld
			kheld = KeyList(DISP.kheld)
			kclick = KeyList(DISP.kclick)
			DISP.kclick[:] = [0] * len(DISP.kclick)
			krelease = KeyList(DISP.krelease)
			DISP.krelease[:] = [0] * len(DISP.krelease)
			kspam = KeyList(x or y and pc() - y >= 0.5 for x, y in zip(kclick, kheld))
			# if any(kclick):
			#	 print(" ".join(map(str, (i for i, v in enumerate(kclick) if v))))
			if kclick[K_BACKQUOTE]:
				if code:
					_ = code
				def debug_code_a(output):
					if not output:
						return
					global orig, code
					orig = output
					try:
						c = None
						try:
							c = compile(output, "<debug>", "eval")
						except SyntaxError:
							pass
						if not c:
							c = compile(output, "<debug>", "exec")
						output = str(eval(c, globals()))
					except:
						output = traceback.format_exc()
					globals()["output"] = output.strip()
					_ = code = output
					submit(
						easygui2.textbox,
						debug_code_a,
						orig or "Debug mode",
						title="Miza Player",
						text=code or "",
					)
				easygui2.textbox(
					debug_code_a,
					orig or "Debug mode",
					title="Miza Player",
					text=code or "",
				)
			if any(kclick) or any(krelease) or player.editor.held_notes:
				alphakeys[:] = [False] * len(alphakeys)
				if not CTRL[kheld] and not SHIFT[kheld] and not ALT[kheld]:
					notekeys = "zsxdcvgbhnjmq2w3er5t6y7ui9o0p"
					alphakeys = [bool(kheld[globals()[f"K_{c}"]]) for c in notekeys] + [False] * 5
					alphakeys[-5] = kheld[K_LEFTBRACKET]
					alphakeys[-4] = kheld[K_EQUALS]
					alphakeys[-3] = kheld[K_RIGHTBRACKET]
					alphakeys[-2] = kheld[K_BACKSPACE]
					alphakeys[-1] = kheld[K_BACKSLASH]
					alphakeys[12] |= bool(kheld[K_COMMA])
					alphakeys[13] |= bool(kheld[K_l])
					alphakeys[14] |= bool(kheld[K_PERIOD])
					alphakeys[15] |= bool(kheld[K_SEMICOLON])
					alphakeys[16] |= bool(kheld[K_SLASH])
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
			crosshair = False
			hovertext = None
			video_sourced = False
			if not toolbar.editor:
				if options.get("spectrogram", 0) > 1:
					rect = player.rect
					render_spectrogram(rect)
				elif queue and queue[0] and not queue[0].get("novid") and options.get("spectrogram", 0) == 0 and (not player.video or player.video.url):
					video_sourced = True
					if player.video:
						if player.video.url and player.video.url != queue[0].url:
							print("Video changed", player.video.url, queue[0].url)
							player.video.url = None
							# video_sourced = False
						if not player.get("video_loading"):
							if player.video.pos > player.pos + 1:
								if isfinite(player.video.pos):
									print("Video seeked backwards", player.video.pos, player.pos)
								if player.video.is_running():
									try:
										player.video.terminate()
									except psutil.NoSuchProcess:
										pass
								# player.video_loading = None
								player.video.url = None
								# video_sourced = False
							elif player.video.pos < player.pos - 15:
								print("Video seeked forwards", player.video.pos, player.pos)
								if player.video.is_running():
									try:
										player.video.terminate()
									except psutil.NoSuchProcess:
										pass
								# player.video_loading = None
								player.video.url = None
								# video_sourced = False
					if video_sourced:
						if player.video and (player.video.is_running() or abs(player.video.pos - player.pos) < 1) and player.video.im:
							im = player.video.im
							if im.width > 0 and im.height > 0:
								if not player.video.tex:
									# player.video.tex = None
									# tex = pyglet.image.Texture.create(im.width, im.height, rectangle=True)
									tex = im.get_texture()
									player.video.tex = tex = tex.get_transform(flip_y=True)
									tex.anchor_y = 0
								tex = player.video.tex
								if player.video.im2 != player.video.im:
									# tex = im.get_texture()
									# player.video.tex = tex = tex.get_transform(flip_y=True)
									try:
										tex.blit_into(player.video.im, 0, 0, 0)
									except:
										print_exc()
									# print(tex)
									player.video.im2 = player.video.im
								size = limit_size(im.width, im.height, *player.rect[2:])
								x = player.rect[2] - size[0] >> 1
								y = (player.rect[3] - size[1] >> 1) + toolbar_height
								scale = size[0] / tex.width
								batch = DISP.get_batch(0)
								if not player.sprite:
									player.sprite = sp = pyglet.sprite.Sprite(
										tex,
										x=x,
										y=y,
										batch=batch,
										group=DISP.get_group(2),
									)
									sp.scale = scale
									sp.changed = 1
								else:
									sp = player.sprite
									if sp.image != tex:
										sp.image = tex
									if sp.x != x:
										sp.x = x
									if sp.y != y:
										sp.y = y
									if sp.scale != scale:
										sp.scale = scale
								batch.used = True
								sp.changed = max(0, sp.changed - 1)
						else:
							if (not player.video or not player.video.is_running() and not abs(player.video.pos - player.pos) < 1) and not player.get("video_loading"):
								url = queue[0].get("video") or queue[0].get("icon")
								if url:
									print("Loading", url)
									player.video_loading = submit(load_video, url, pos=player.pos, bak=queue[0].get("icon"), sig=queue[0].url)
								elif "novid" not in queue[0]:
									print(queue[0])
									submit(prepare, queue[0], force=2)
									queue[0].novid = True
							if not player.sprite:
								# try:
									# no_lyrics_source = no_lyrics_fut.result()
								# except (FileNotFoundError, PermissionError):
									# pass
								# else:
									# no_lyrics_size = limit_size(*no_lyrics_source.get_size(), *player.rect[2:])
									# no_lyrics = globals().get("no_lyrics")
									# if not no_lyrics or no_lyrics.get_size() != no_lyrics_size:
										# no_lyrics = globals()["no_lyrics"] = pygame.transform.scale(no_lyrics_source, no_lyrics_size)
									# blit_complex(
										# DISP,
										# no_lyrics,
										# (player.rect[2] - no_lyrics.get_width() >> 1, player.rect[3] - no_lyrics.get_height() >> 1),
										# z=1,
									# )
								if pc() % 0.25 < 0.125:
									col = (255,) * 3
								else:
									col = (255, 0, 0)
								s = f"Loading video for {queue[0].name}..."
								size = max(20, min(40, (screensize[0] - sidebar_width) // len(s)))
								message_display(
									s,
									size,
									(player.rect[2] >> 1, size),
									col,
									surface=DISP.subsurf(player.rect),
									cache=True,
									background=(0,) * 3,
									font="Rockwell",
									z=2,
								)
				elif queue or lyrics_entry:
					novid = queue and queue[0] and queue[0].get("novid") and options.get("spectrogram", 0) == 0
					entry = lyrics_entry or queue[0]
					rect = (player.rect[2] - 8, player.rect[3] - 92)
					if not entry.get("lyrics_loading") and (not entry.get("lyrics") or rect != entry.lyrics[1].get_size()):
						print(entry)
						entry.lyrics_loading = submit(render_lyrics, entry)
					if not novid and "lyrics" not in entry:
						if pc() % 0.25 < 0.125:
							col = (255,) * 3
						else:
							col = (255, 0, 0)
						s = f"Loading lyrics for {entry.name}..."
						size = max(20, min(40, (screensize[0] - sidebar_width) // len(s)))
						message_display(
							s,
							size,
							(player.rect[2] >> 1, size),
							col,
							surface=DISP.subsurf(player.rect),
							cache=True,
							background=(0,) * 3,
							font="Rockwell",
							z=2,
						)
					elif not novid and entry.get("lyrics"):
						s = entry.lyrics[0]
						blit_complex(
							DISP,
							entry.lyrics[1],
							(8, 92),
							z=1,
						)
						size = max(20, min(40, (screensize[0] - sidebar_width) // len(s)))
						message_display(
							s,
							size,
							(player.rect[2] >> 1, size),
							(255,) * 3,
							surface=DISP.subsurf(player.rect),
							cache=True,
							background=(0,) * 3,
							font="Rockwell",
							z=2,
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
								no_lyrics = globals()["no_lyrics"] = pygame.transform.scale(no_lyrics_source, no_lyrics_size)
							blit_complex(
								DISP,
								no_lyrics,
								(player.rect[2] - no_lyrics.get_width() >> 1, player.rect[3] - no_lyrics.get_height() >> 1),
								z=1,
							)
						if novid:
							s = f"No video found for {entry.name}."
						elif entry.get("lyrics") == "":
							s = f"No lyrics found for {entry.name}."
						elif "lyrics" in entry:
							s = entry.lyrics[0]
						else:
							s = ""
						if s:
							size = max(20, min(40, (screensize[0] - sidebar_width) // len(s)))
							message_display(
								s,
								size,
								(player.rect[2] >> 1, size),
								(255, 0, 0),
								surface=DISP.subsurf(player.rect),
								cache=True,
								background=(0,) * 3,
								font="Rockwell",
								z=2,
							)
			if not video_sourced and player.video:
				if player.sprite:
					player.sprite.delete()
				player.sprite = None
				if player.video and player.video.is_running():
					try:
						player.video.terminate()
					except psutil.NoSuchProcess:
						pass
				player.video = None
				player.video_loading = None
			update_menu()
			if toolbar.colour:
				render_toolbar()
			draw_menu()
			if not queue and not is_active() and not any(kheld):
				player.pos = 0
				player.offpos = -inf
				player.end = inf
				player.last = 0
				progress.num = 0
				progress.alpha = 0
			if toolbar.editor:
				render_piano()
			if not toolbar.editor:
				if player.get("flash_s", 0) > 0:
					bevel_rectangle(
						DISP,
						(191,) * 3,
						player.rect,
						4,
						alpha=player.flash_s * 8 - 1,
						z=3,
						cache=False,
					)
				text_rect = (0, 0, 192, 92)
				if player.get("flash_i", 0) > 0:
					bevel_rectangle(
						DISP,
						(191,) * 3,
						text_rect,
						4,
						alpha=player.flash_i * 8 - 1,
						z=5,
						cache=False,
					)
				if in_rect(mpos, text_rect):
					bevel_rectangle(
						DISP,
						(191,) * 3,
						text_rect,
						4,
						filled=False,
						z=6,
						cache=False,
					)
				elif not sidebar.get("dragging") and in_rect(mpos, player.rect):
					bevel_rectangle(
						DISP,
						(191,) * 3,
						player.rect,
						4,
						filled=False,
						z=4,
						cache=False,
					)
			if not toolbar.editor:
				if options.get("insights", True):
					message_display(
						f"FPS: {round(DISP.fps, 2)}",
						14,
						(4, 0),
						align=0,
						surface=DISP,
						font="Comic Sans MS",
						z=5,
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
							z=5,
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
						cache=False,
						z=5,
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
			if sidebar.menu:
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
						DISP.get_at(sidebar.menu.pos)
					r.extend((sidebar.menu.size[0], round(sidebar.menu.size[1] * sidebar.menu.scale)))
					if r[0] + r[2] > screensize[0]:
						r[0] = screensize[0] - r[2]
					if r[1] + r[3] > screensize[1]:
						r[1] = screensize[1] - r[3]
					sidebar.menu.rect = r
				lines = sidebar.menu.get("lines")
				if lines:
					try:
						c = DISP.get_at(sidebar.menu.pos, force=False)
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
					rounded_bev_rect(
						DISP,
						c,
						rect,
						4,
						z=656,
					)
					for i, surf in enumerate(lines):
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
								z=657,
							)
							col = (c,) * 3
							sidebar.menu.glow[i] = max(0, sidebar.menu.glow[i] - dur * 2.5)
						else:
							col = (0,) * 3
						blit_complex(
							DISP,
							surf,
							(text_rect[0] + 3, text_rect[1]),
							alpha,
							colour=col,
							z=658,
						)
			# Finish()
			DISP.update()
			if tick < 2 or not tick % 15:
				DISP.switch_to()
				# DISP.clear()
				DISP.update_keys()
			elif options.spectrogram in (0, 1):
				DISP.fill(
					(0,) * 4,
					player.rect,
				)
		elif unfocused:
			DISP.mrelease[:] = DISP.mheld
			DISP.krelease[:] = DISP.kheld
			DISP.mheld[:] = [0] * len(DISP.mheld)
			DISP.kheld[:] = [0] * len(DISP.kheld)
			sidebar.particles.clear()
		DISP.dispatch_events()
		if not tick & 3:
			DISP.update_held()
		if unfocused:
			if options.control.unfocus:
				fps = 7.5
			else:
				fps = 24
		elif getattr(DISP, "mmoved", False):
			fps = 60
		elif player.video and player.sprite and player.sprite.changed > 0 and player.video.fps > 35:
			fps = 60
			player.sprite.changed = 0
		else:
			fps = 30
		globals()["fps"] = fps
		d = 1 / fps
		DISP.mmoved = False
		t = pc()
		if (not unfocused and not minimised) or is_active():
			last_played = t
		delplay = t - last_played
		delay = t - last_tick
		if delay > 5 / fps:
			DISP.lastclear = True
		if delay >= 30 or delplay >= 3600 and common.OUTPUT_DEVICE:
			restart_mixer()
			last_tick = last_played = t = pc()
		DISP.fps = 1 / max(d / 4, last_ratio)
		d2 = max(0.004, d - delay)
		last_ratio = (last_ratio * 7 + t - last_precise) / 8
		last_precise = t
		last_tick = max(last_tick + d, t - 0.125)
		time.sleep(d2)
		if getattr(DISP, "cLoSeD", False):
			raise StopIteration
		if minimised:
			for i in range(50):
				time.sleep(0.02)
				if not is_minimised():
					break
				DISP.dispatch_events()
			DISP.update_held()
		tick += 1
except Exception as ex:
	futs = set()
	futs.add(submit(reqs.delete, mp))
	futs.add(submit(update_collections2))
	# futs.add(submit(DISP.close))
	if restarting:
		futs.add(submit(os.execl, sys.executable, "python", *sys.argv))
	pygame.closed = True
	if type(ex) is not StopIteration:
		print_exc()
	if mixer.is_running():
		try:
			mixer.submit("~quit")
		except:
			pass
	for c in PROC.children(recursive=True):
		futs.add(submit(c.terminate))
	mixer = None
	for fut in futs:
		try:
			fut.result(timeout=1)
		except:
			pass
	if globals().get("last_save_fut"):
		last_save_fut.result()
	save_settings(closing=True)
	if not restarting and type(ex) is not StopIteration:
		easygui.exceptionbox()
	print("Exiting...")
	DISP.close()
	pygame.quit()
	PROC.kill()