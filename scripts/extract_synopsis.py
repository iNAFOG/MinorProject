"""Extract text fields from the supplied Pages IWA archive, without dependencies."""
from pathlib import Path
import zipfile


def varint(data, pos):
    value = shift = 0
    while True:
        b = data[pos]
        pos += 1
        value |= (b & 127) << shift
        if b < 128:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError('invalid varint')


def snappy(data):
    size, pos = varint(data, 0)
    out = bytearray()
    while pos < len(data):
        tag = data[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:
            n = tag >> 2
            if n >= 60:
                width = n - 59
                n = int.from_bytes(data[pos:pos+width], 'little')
                pos += width
            n += 1
            out.extend(data[pos:pos+n])
            pos += n
        else:
            if kind == 1:
                n = 4 + ((tag >> 2) & 7)
                offset = ((tag & 224) << 3) | data[pos]
                pos += 1
            else:
                n = 1 + (tag >> 2)
                width = 2 if kind == 2 else 4
                offset = int.from_bytes(data[pos:pos+width], 'little')
                pos += width
            for _ in range(n):
                out.append(out[-offset])
    assert len(out) == size
    return bytes(out)


def fields(data, depth=0):
    if depth > 25:
        return []
    pos = 0
    found = []
    try:
        while pos < len(data):
            key, pos = varint(data, pos)
            if not key >> 3:
                return []
            wire = key & 7
            if wire == 0:
                _, pos = varint(data, pos)
            elif wire in (1, 5):
                pos += 8 if wire == 1 else 4
            elif wire == 2:
                n, pos = varint(data, pos)
                part = data[pos:pos+n]
                if len(part) != n:
                    return []
                pos += n
                try:
                    text = part.decode('utf-8')
                    if len(text) > 3 and sum(c.isprintable() or c.isspace() for c in text) / len(text) > 0.95:
                        found.append(text)
                        continue
                except UnicodeDecodeError:
                    pass
                found.extend(fields(part, depth + 1))
            else:
                return []
    except (IndexError, ValueError):
        return []
    return found


def extract(path):
    parts = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('.iwa') or 'Stylesheet' in name or 'Metadata' in name:
                continue
            data = z.read(name)
            raw = bytearray()
            pos = 0
            while pos < len(data):
                kind = data[pos]
                n = int.from_bytes(data[pos+1:pos+4], 'little')
                pos += 4
                assert kind == 0
                raw.extend(snappy(data[pos:pos+n]))
                pos += n
            pos = 0
            texts = []
            while pos < len(raw):
                n, pos = varint(raw, pos)
                header = raw[pos:pos+n]
                pos += n
                # ArchiveInfo: field 2 contains MessageInfo; field 3 is payload length.
                hpos = 0
                lengths = []
                while hpos < len(header):
                    key, hpos = varint(header, hpos)
                    if key & 7 == 0:
                        _, hpos = varint(header, hpos)
                    elif key & 7 == 2:
                        length, hpos = varint(header, hpos)
                        info = header[hpos:hpos+length]
                        hpos += length
                        if key >> 3 == 2:
                            ipos = 0
                            while ipos < len(info):
                                ikey, ipos = varint(info, ipos)
                                if ikey & 7 == 0:
                                    value, ipos = varint(info, ipos)
                                    if ikey >> 3 == 3:
                                        lengths.append(value)
                                elif ikey & 7 == 2:
                                    value, ipos = varint(info, ipos)
                                    ipos += value
                                else:
                                    raise ValueError('unexpected header')
                    else:
                        raise ValueError('unexpected archive')
                for length in lengths:
                    texts.extend(fields(raw[pos:pos+length]))
                    pos += length
            if texts:
                parts.append(f'\n## {name}\n' + '\n'.join(dict.fromkeys(texts)))
    return '\n'.join(parts)


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    source = next(Path('.').glob('*.pages'))
    result = extract(source)
    Path('docs/synopsis-extracted.txt').write_text(result, encoding='utf-8')
    print(result)
