def extract_text_from_vtt(vtt: str) -> str:
    lines = []
    if isinstance(vtt, str):
        vtt = vtt.split('\n')
    else:
        return ""
    for line in vtt:
        line = line.strip()
        if not line:
            continue
        if '-->' in line:
            continue
        if line.startswith('WEBVTT'):
            continue
        lines.append(line)
    return ' '.join(lines)