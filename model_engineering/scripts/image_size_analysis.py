#!/usr/bin/env python
# coding: utf-8
"""
Analise de tamanho das imagens brutas + simulacao do pipeline de resize/crop.

Lê dimensões diretamente dos metadados JPEG (SOF) dentro do zip,
sem extrair as imagens. Rápido: lê ~64KB por arquivo.

Uso:
    python scripts/image_size_analysis.py /caminho/psoriase.zip
    python scripts/image_size_analysis.py /caminho/psoriase.zip --sample 200
"""

import argparse
import json
import re
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def jpeg_dimensions(data: bytes):
    """Extrai (altura, largura) do primeiro SOF do JPEG."""
    if len(data) < 4 or data[0:2] != b'\xff\xd8':
        return None
    off = 2
    while off + 9 < len(data):
        if data[off] != 0xFF:
            return None
        marker = data[off + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            off += 2
            continue
        seg_len = struct.unpack('>H', data[off + 2:off + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            h, w = struct.unpack('>HH', data[off + 5:off + 9])
            return h, w
        if marker == 0xDA:
            return None
        off += 2 + seg_len
    return None


# IFD entry da tag 0x0112 (Orientation), type SHORT (3), count 1.
#   big-endian  (MM): 01 12 | 00 03 | 00 00 00 01 | 00 XX 00 00
#   little-endian (II): 12 01 | 03 00 | 01 00 00 00 | XX 00 00 00
def _orient_pattern(val: int):
    be = bytes([0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, val, 0x00, 0x00])
    le = bytes([0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, val, 0x00, 0x00, 0x00])
    return re.compile(be + b'|' + le)


ORIENT_PATTERNS = {v: _orient_pattern(v) for v in (3, 6, 7, 8)}


def scan_zip(zip_path: str, sample: int = 0):
    z = zipfile.ZipFile(zip_path)
    infos = [i for i in z.infolist()
             if not i.is_dir() and i.filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if sample:
        step = max(1, len(infos) // sample)
        infos = infos[::step]

    records = []
    for n, info in enumerate(infos):
        with z.open(info) as src:
            head = src.read(131072)
        dims = jpeg_dimensions(head)
        if dims is None:
            continue
        h, w = dims
        orient = None
        for val, pat in ORIENT_PATTERNS.items():
            if pat.search(head):
                orient = val
                break
        records.append({
            'file': info.filename,
            'w': w, 'h': h,
            'mp': round(w * h / 1e6, 3),
            'aspect': round(w / h, 4),
            'exif_orient': orient,
            'bytes': info.file_size,
        })
        if (n + 1) % 500 == 0:
            print(f'  {n + 1}/{len(infos)} processadas...', flush=True)

    return records


def simulate(records):
    """Quanta area original cada estrategia do pipeline preserva."""
    for r in records:
        w, h = r['w'], r['h']
        a = w / h
        # Resize(shortest=512) + CenterCrop(512): fracao = min(a, 1/a)
        kept = min(a, 1 / a)
        r['crop512_kept'] = round(kept, 4)
        # Letterbox: 100% do conteudo, escala do lado maior
        r['lb512_scale'] = round(min(512 / w, 512 / h), 4)
        r['eff_mp_512'] = round(r['mp'] * kept, 2)
    return records


def summarize(records):
    import statistics as st

    def q(vals, p):
        s = sorted(vals)
        return s[int(round(p * (len(s) - 1)))]

    ws = [r['w'] for r in records]
    hs = [r['h'] for r in records]
    mps = [r['mp'] for r in records]
    asps = [r['aspect'] for r in records]
    kept = [r['crop512_kept'] for r in records]
    eff = [r['eff_mp_512'] for r in records]
    res = Counter((r['w'], r['h']) for r in records)

    rot = Counter(r['exif_orient'] for r in records if r['exif_orient'])
    n_rot = sum(rot.values())
    tall = sum(1 for r in records if r['h'] > r['w'])
    wide = sum(1 for r in records if r['w'] > r['h'])

    return {
        'n': len(records),
        'width':  {'min': min(ws), 'p25': q(ws, .25), 'median': q(ws, .5), 'p75': q(ws, .75), 'max': max(ws)},
        'height': {'min': min(hs), 'p25': q(hs, .25), 'median': q(hs, .5), 'p75': q(hs, .75), 'max': max(hs)},
        'aspect': {'min': min(asps), 'median': q(asps, .5), 'max': max(asps)},
        'mp':     {'min': min(mps), 'p25': q(mps, .25), 'median': q(mps, .5), 'p75': q(mps, .75), 'max': max(mps)},
        'orientation': {'retrato': tall, 'paisagem': wide,
                        'retrato_pct': round(100 * tall / len(records), 1),
                        'paisagem_pct': round(100 * wide / len(records), 1)},
        'top_resolutions': [(f'{w}x{h}', c) for (w, h), c in res.most_common(10)],
        'exif_rotated': {'n': n_rot, 'pct': round(100 * n_rot / len(records), 1),
                         'por_orient': dict(rot)},
        'crop512_kept': {'min': min(kept), 'p25': q(kept, .25), 'median': q(kept, .5),
                         'p75': q(kept, .75), 'max': max(kept)},
        'eff_mp_512': {'median': q(eff, .5)},
        'mean_original_mp': round(st.mean(mps), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='arquivo .zip (ou .7z extraído via zip) com as imagens')
    ap.add_argument('--sample', type=int, default=0,
                    help='analisar apenas uma amostra estratificada (ex: 200)')
    ap.add_argument('-o', '--out', default='image_size_analysis.json')
    args = ap.parse_args()

    print(f'Escaneando {args.source} ...')
    records = simulate(scan_zip(args.source, sample=args.sample))
    summary = summarize(records)

    out = Path(args.out)
    out.write_text(json.dumps({'summary': summary}, indent=2))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\nResumo salvo em: {out.resolve()}')


if __name__ == '__main__':
    main()
