"""Non-destructive U-Net component audit for a canonical validation fold.

Run from the Mole-Classification-HAM10000 repository root:
  python -m evaluation.mask_component_audit --fold 5

This script does NOT save cropped images or modify checkpoints. It compares the
current thresholded mask with a hypothetical largest-connected-component-only mask.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import torch

from core.canonical_cv import get_fold_metadata
from core.segmentation_unet import SmallUNet
from preprocessing.roi_crop import crop_lesion_with_margin


def bbox_fill(binary: np.ndarray) -> float:
    ys, xs = np.where(binary)
    if not xs.size:
        return 0.0
    bbox_area = (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)
    return float(binary.sum() / bbox_area)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--geometry-csv', type=Path, default=None,
                        help='Optional per-image geometry CSV, for outcome join and status verification')
    args = parser.parse_args()

    fold = args.fold
    output_dir = args.output_dir or Path(f'reports/segmentation/small_unet_fold{fold}_seed123')
    csv_path = output_dir / f'mask_component_audit_fold{fold}.csv'
    summary_path = output_dir / f'mask_component_audit_fold{fold}_summary.json'
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError('Audit output exists; refusing to overwrite it.')

    geometry_path = args.geometry_csv or output_dir / f'roi_geometry_diagnostic_fold{fold}.csv'
    geometry = None
    if geometry_path.is_file():
        geometry = pd.read_csv(geometry_path, dtype={'image_id': str})
        if geometry.image_id.duplicated().any():
            raise ValueError('Duplicate image_id in geometry report.')
        geometry = geometry.set_index('image_id')

    _, val = get_fold_metadata(fold)
    if val.image_id.duplicated().any():
        raise ValueError('Duplicate image_id in validation metadata.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(f'checkpoints/segmentation/small_unet_fold{fold}_seed123_best.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    rows: list[dict] = []
    with torch.inference_mode():
        for i, row in enumerate(val.itertuples(index=False), 1):
            img_path = Path('data/images') / f'{row.image_id}.jpg'
            with Image.open(img_path) as source:
                image = source.convert('RGB')
            resized = image.resize((256, 192), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            tensor = torch.from_numpy(array.copy()).unsqueeze(0).to(device)
            predicted = (torch.sigmoid(model(tensor))[0, 0] >= 0.5).to(torch.uint8).cpu().numpy()
            mask = Image.fromarray(predicted * 255).resize(image.size, Image.Resampling.NEAREST)
            binary = np.asarray(mask, dtype=np.uint8) > 127

            labels, n_components = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
            if n_components:
                sizes = np.bincount(labels.ravel(), minlength=n_components + 1)[1:]
                largest_index = int(sizes.argmax()) + 1
                largest_size = int(sizes.max())
                second_size = int(np.partition(sizes, -2)[-2]) if n_components >= 2 else 0
                cleaned_binary = (labels == largest_index)
            else:
                largest_size = second_size = 0
                cleaned_binary = binary
            original_foreground = int(binary.sum())
            removed = original_foreground - largest_size
            cleaned_mask = Image.fromarray((cleaned_binary.astype(np.uint8) * 255))
            old_crop, old_status = crop_lesion_with_margin(image, mask)
            new_crop, new_status = crop_lesion_with_margin(image, cleaned_mask)
            changed_pixels = not np.array_equal(np.asarray(old_crop), np.asarray(new_crop))

            result = {
                'fold': fold,
                'image_id': row.image_id,
                'dx': row.dx,
                'components_8connected': int(n_components),
                'largest_component_pixels': largest_size,
                'second_largest_pixels': second_size,
                'removed_foreground_pixels': removed,
                'removed_foreground_fraction': (removed / original_foreground) if original_foreground else 0.,
                'bbox_fill_original': bbox_fill(binary),
                'bbox_fill_largest_only': bbox_fill(cleaned_binary),
                'original_status': old_status,
                'largest_only_status': new_status,
                'original_crop_area_fraction': (old_crop.width * old_crop.height) / (image.width * image.height),
                'largest_only_crop_area_fraction': (new_crop.width * new_crop.height) / (image.width * image.height),
                'crop_changed_by_largest_only': bool(changed_pixels),
                'status_changed_by_largest_only': old_status != new_status,
            }
            if geometry is not None:
                if row.image_id not in geometry.index:
                    raise ValueError(f'{row.image_id} absent from geometry CSV.')
                old_record = geometry.loc[row.image_id]
                if old_record['predicted_status'] != old_status:
                    raise ValueError(f'Status mismatch for {row.image_id}: {old_record["predicted_status"]} != {old_status}')
                result['outcome'] = str(old_record['outcome'])
                result['roi_transition'] = str(old_record['roi_vs_fallback_transition'])
                result['abs_crop_area_delta_vs_reference'] = abs(float(old_record['crop_fraction_delta']))
            rows.append(result)
            if i % 250 == 0 or i == len(val):
                print(f'Fold {fold}: component audit {i}/{len(val)}')

    audited = pd.DataFrame(rows)
    if len(audited) != len(val):
        raise AssertionError('Audit row count does not match validation count.')
    summary = {
        'fold': fold,
        'n_images': len(audited),
        'n_multicomponent_masks': int((audited.components_8connected > 1).sum()),
        'n_changed_crops_with_largest_only': int(audited.crop_changed_by_largest_only.sum()),
        'n_changed_statuses_with_largest_only': int(audited.status_changed_by_largest_only.sum()),
        'n_5percent_foreground_removed': int((audited.removed_foreground_fraction >= 0.05).sum()),
        'original_status_counts': dict(Counter(audited.original_status)),
        'largest_only_status_counts': dict(Counter(audited.largest_only_status)),
        'safety_notes': [
            'Largest-component filtering is an exploratory hypothetical change; no classifier performance is measured.',
            'Small true lesion components can be removed; do not adopt without downstream evaluation.',
            'Reference ROI crops, saved images, checkpoints and existing reports are unchanged.',
        ],
    }
    if 'outcome' in audited.columns:
        subset = audited[audited.crop_changed_by_largest_only]
        summary['changed_crops_by_outcome'] = dict(Counter(subset.outcome))
        summary['all_outcomes'] = dict(Counter(audited.outcome))
    output_dir.mkdir(parents=True, exist_ok=True)
    audited.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Saved: {csv_path} and {summary_path}')


if __name__ == '__main__':
    main()
