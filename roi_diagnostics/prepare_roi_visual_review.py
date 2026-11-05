#!/usr/bin/env python3
"""Collect original/reference-ROI/predicted-ROI images for selected Fold 5 cases.

Run from the repository root, after downloading roi_fold5_priority_cases.csv:

python prepare_roi_visual_review.py --cases roi_fold5_priority_cases.csv

Produces roi_fold5_visual_review.zip with original photos and side-by-side sheets.
This is a non-destructive read-only export; it never reruns inference.
"""
from __future__ import annotations
import argparse
import io
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def draw_sheet(rows):
    """Rows contain (case metadata, original image, reference ROI, predicted ROI)."""
    panel_width, panel_height = 240, 188
    left_width, top_height, gap = 320, 34, 8
    canvas = Image.new('RGB', (left_width + 3*panel_width, len(rows)*(panel_height+top_height+gap)+28), '#ffffff')
    d=ImageDraw.Draw(canvas)
    font=ImageFont.load_default()
    d.text((left_width+12, 7), 'Original', fill='black',font=font)
    d.text((left_width+panel_width+12, 7), 'Reference ROI',fill='black',font=font)
    d.text((left_width+2*panel_width+12, 7),'Predicted ROI',fill='black',font=font)
    y=28
    for case,images in rows:
        label=f"{case['image_id']} [{case['dx']}]"
        change=f"ref: {case['reference_class']} -> pred: {case['predicted_class']}"
        d.text((8,y+4),label,fill='black',font=font)
        d.text((8,y+24),change,fill='#555555',font=font)
        d.text((8,y+44),case['outcome'],fill='#333333',font=font)
        for col,img in enumerate(images):
            tile=ImageOps.contain(img.convert('RGB'),(panel_width-10,panel_height),Image.Resampling.LANCZOS)
            x=left_width+col*panel_width+(panel_width-tile.width)//2
            canvas.paste(tile,(x,y+top_height))
        y += panel_height+top_height+gap
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,default=Path('roi_fold5_priority_cases.csv'))
    p.add_argument('--metadata',type=Path,default=Path('splits/ham10000_dev_cv_v1.csv'))
    p.add_argument('--images-root',type=Path,default=Path('data/images'))
    p.add_argument('--reference-root',type=Path,default=Path('data/processed_roi_margin15_v1'))
    p.add_argument('--predicted-root',type=Path,default=Path('data/processed_roi_predicted_fold5_v1'))
    p.add_argument('--output',type=Path,default=Path('roi_fold5_visual_review.zip'))
    args=p.parse_args()
    if args.output.exists():
        p.error(f'{args.output} already exists; choose another output to avoid overwriting.')
    cases=pd.read_csv(args.cases,dtype={'image_id':str})
    metadata=pd.read_csv(args.metadata,dtype={'image_id':str}).set_index('image_id')
    if not metadata.index.is_unique:
        p.error('Metadata contains duplicate image IDs.')
    required={'image_id','dx','reference_class','predicted_class','outcome','fold'}
    if not required.issubset(cases):
        p.error(f'Cases CSV missing {required-set(cases.columns)}')
    if cases.image_id.duplicated().any():
        p.error('Cases list contains duplicate image IDs.')
    assembled=[]
    missing=[]
    searched_predicted=0
    for case in cases.to_dict('records'):
        image_id=case['image_id']
        if image_id not in metadata.index:
            p.error(f'Unknown image_id: {image_id}')
        row=metadata.loc[image_id]
        if int(row['fold']) != int(case['fold']) or int(case['fold']) != 5 or row['dx'] != case['dx']:
            p.error(f'Fold/label metadata mismatch: {image_id}')
        stem=f'{image_id}_224x224.jpg'
        original_path = args.images_root / f'{image_id}.jpg'
        reference_path = args.reference_root / row['source_split'] / row['dx'] / stem
        # Some fold-specific exporters place ALL held-out fold images under
        # 'val/', independently of the original HAM10000 source_split.
        predicted_path = args.predicted_root / row['source_split'] / row['dx'] / stem
        if not predicted_path.is_file():
            candidates = [
                path for path in args.predicted_root.rglob(stem)
                if path.parent.name == row['dx']
            ] if args.predicted_root.is_dir() else []
            if len(candidates) == 1:
                predicted_path = candidates[0]
                searched_predicted += 1
            elif len(candidates) > 1:
                p.error(f'Ambiguous predicted crop for {image_id}: {candidates}')
        paths = [original_path, reference_path, predicted_path]
        for path in paths:
            if not path.is_file():
                missing.append(str(path))
        assembled.append((case,paths))
    if missing:
        print('Missing files (first 20):')
        for path in missing[:20]:print('  ',path)
        p.error(f'{len(missing)} required input images missing. Check --reference-root and --predicted-root paths.')

    if searched_predicted:
        print(f'Located {searched_predicted} predicted crops using image-ID search (folder layout differs from source_split).')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    groups={}
    with ZipFile(args.output,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
        z.write(args.cases,arcname='priority_cases.csv')
        for case,paths in assembled:
            files=[]
            for label,path in zip(('original','reference_roi','predicted_roi'),paths):
                z.write(path,arcname=f"images/{case['image_id']}_{label}{path.suffix.lower()}")
                with Image.open(path) as im: files.append(im.convert('RGB').copy())
            groups.setdefault(case['dx'],[]).append((case,files))
        for name,rows in groups.items():
            sheet=draw_sheet(rows)
            buffer=io.BytesIO()
            sheet.save(buffer,format='PNG',optimize=True)
            z.writestr(f'sheets/{name}_comparison.png',buffer.getvalue())
    print(f'Saved {len(assembled)} cases to {args.output}')
    print('Grouped visual sheets:',', '.join(groups))

if __name__=='__main__':main()
