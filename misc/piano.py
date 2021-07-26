editor = player.editor
note_names = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
def note_colour(i, s=1, v=1):
    return verify_colour(x * 255 for x in colorsys.hsv_to_rgb(i / 12, s, v))

def render_piano():
    ts = editor.setdefault("timestamp", 0)
    t = pc()
    duration = max(0.001, min(t - ts, 0.125))
    ratio = 1 + 1 / (duration * 8)
    r = ratio# if editor.scrolling else (ratio - 1) / 3 + 1
    y = editor.scroll_y
    editor.scroll_y = (editor.scroll_y * (r - 1) + editor.targ_y) / r
    if not project.patterns:
        pattern = project.create_pattern()
    else:
        pattern = project.patterns[0]
    timesig = pattern.timesig

    note_width = 60 * editor.zoom_x
    note_height = 12 * editor.zoom_y
    note_spacing = note_height + 1
    if abs(y - editor.scroll_y) >= 0.25 / note_spacing:
        player.editor_surf = None
    bars = ceil(ssize[1] / note_spacing + 1)
    offs = editor.scroll_y % 1 * note_spacing
    offs = round(offs - note_spacing / 2)
    centre = editor.scroll_y // 1 + 48 + (bars + 1 >> 1)
    surf = player.get("editor_surf")
    if not surf or surf.get_size() != player.rect[2:]:
        surf = player["editor_surf"] = pygame.Surface(player.rect[2:], SRCALPHA)

        for i in range(bars + 1):
            note = round(centre - i)
            name = note_names[note % 12]
            rect = (0, offs - note_spacing, player.rect[2], note_spacing)
            c = (127,) * 3
            if i != bars:
                draw_hline(surf, 0, player.rect[2], offs, c)
            c = note_colour(note % 12, 0.5, 1)
            r = (0, offs - note_spacing, 48, note_height)
            bevel_rectangle(surf, c, r, bevel=ceil(note_height / 5))
            octave = note // 12
            if not note % 12 and note_height > 6:
                s = ceil(note_height * 0.75)
                message_display(f"C{octave}", s, (46, offs - note_spacing + note_height), colour=(0,) * 3, surface=DISP, align=2)
            if name.endswith("#"):
                c = note_colour(note % 12, 0.75, 1 / 3)
                r = (0, offs - note_spacing + ceil(note_height / 10), 30, note_height - ceil(note_height / 10) * 2)
                bevel_rectangle(surf, c, r, bevel=ceil(note_height / 5))
                c = note_colour(note % 12, 0.75, 0.125)
                r = (48, offs - note_spacing + 1, player.rect[2], note_height)
                surf.fill(c, r)
            else:
                c = note_colour(note % 12, 0.5, 0.25)
                r = (48, offs - note_spacing + 1, player.rect[2], note_height)
                surf.fill(c, r)
            for i in range(ceil((player.rect[2] - 48) / note_width * timesig[1])):
                c = 64 if i % timesig[1] else 127 if i % (timesig[0] * timesig[1]) else 255
                x = 48 + i * note_width / timesig[1]
                draw_vline(surf, round(x), 0, player.rect[3], (c,) * 3)
            offs += note_spacing
    DISP.blit(surf, player.rect[:2])

    offs = editor.scroll_y % 1 * note_spacing
    offs = round(offs - note_spacing / 2)
    for i in range(bars + 1):
        note = round(centre - i)
        name = note_names[note % 12]
        rect = (0, offs - note_spacing, player.rect[2], note_spacing)
        selected = in_rect(mpos, rect)
        if selected:
            c = note_colour(note % 12, 0.75, 1)
            r = (0, offs - note_spacing, 48, note_height)
            bevel_rectangle(DISP, c, r, bevel=ceil(note_height / 5))
            if name.endswith("#"):
                c = note_colour(note % 12, 0.75, 0.6)
                r = (0, offs - note_spacing + ceil(note_height / 10), 30, note_height - ceil(note_height / 10) * 2)
                bevel_rectangle(DISP, c, r, bevel=ceil(note_height / 5))
                c = note_colour(note % 12, 0.75, 0.4)
                r = (48, offs - note_spacing + 1, player.rect[2], note_height)
                DISP.fill(c, r)
            else:
                c = note_colour(note % 12, 0.5, 0.5)
                r = (48, offs - note_spacing + 1, player.rect[2], note_height)
                DISP.fill(c, r)
            for i in range(ceil(player.rect[2] / 24 - 2)):
                x = round(24 * i + 48 + (pc() * 72) % 24)
                draw_vline(DISP, x, offs - note_spacing + 1, offs - 1, (255,) * 3)
        offs += note_spacing