import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from ast import literal_eval

CROP_SIZE = 512
STRIDE = CROP_SIZE // 2
VISIBILITY_THRESHOLD = 0.7

def process_segments(img_dir, og_bbox_df, save_as_dir, save_as_csv):
    """

    :param img_dir:
    :param save_as_csv:
    :param save_as_dir:
    :param og_bbox_df:
    :return:
    """
    # Make a copy of the bbox df and turn bbox coords into arrays
    bbox_df = og_bbox_df.copy()
    bbox_df.relative_bbox = bbox_df.relative_bbox.apply(literal_eval).apply(np.array)

    # Empty List to hold rows from df in
    all_rows = []

    # If the folder in the path does not exist, create it
    os.makedirs(save_as_dir, exist_ok=True)

    # Create column with name of image segment for that row
    bbox_df['image_name'] = bbox_df['tablet_CDLI'].astype(str) + "_" + bbox_df['segm_idx'].astype(str) + ".jpg"

    # Loop through unique image names
    for img_name in bbox_df.image_name.unique():
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Copy the rows of the df where image name matches
        img_boxes = bbox_df[bbox_df.image_name == img_name].copy()

        # Get other data from df
        if not img_boxes.empty:
            segm_idx = img_boxes.iloc[0]['segm_idx']
            tablet_CDLI = img_boxes.iloc[0]['tablet_CDLI']
            view_desc = img_boxes.iloc[0]['view_desc']
        else:
            segm_idx = None
            tablet_CDLI = None
            view_desc = None


        # Pad so img dims are multiples of CROP_SIZE
        img, img_boxes = pad_images(img=img, bbox_df=img_boxes)

        # Crop and save
        crop_rows = crop_and_save_images(
            img=img, bbox_df=img_boxes,
            og_name=img_name, save_as_dir=save_as_dir,
            segm_idx=segm_idx, tablet_CDLI=tablet_CDLI,
            view_desc=view_desc
        )

        # Add rows to end of List
        all_rows.extend(crop_rows)

    big_df = pd.DataFrame(all_rows)
    big_df.to_csv(save_as_csv, index=False)
    print(f"Saved {len(big_df)} annotations to {save_as_csv}")

    return big_df

def pad_images(img, bbox_df):
    """
    Add padding to the image so it is evenly divisible by 512 pixels
    :param img: The image to pad
    :param bbox_df: The df containing the bboxes for that image
    :return: The padded image and updated bbox_df
    """
    h, w = img.shape[:2]
    size = max(h, w)
    new_size = int(np.ceil(size / CROP_SIZE) * CROP_SIZE)

    # Calculate where to add the padding to
    pad_top = (new_size - h) // 2
    pad_bottom = new_size - h - pad_top
    pad_left = (new_size - w) // 2
    pad_right = new_size - w - pad_left

    # Add padding evenly
    img_padded = cv2.copyMakeBorder(
        src=img,  top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Adjust bboxes relative to the new dimensions
    bbox_df = bbox_df.copy()
    if not bbox_df.empty:
        bbox_df.relative_bbox = bbox_df.relative_bbox.apply(
        lambda b: [
            int(b[0]) + pad_left,    # xmin
            int(b[1]) + pad_top,     # ymin
            int(b[2]) + pad_left,    # xmax
            int(b[3]) + pad_top      # ymax
        ]
    )

    return img_padded, bbox_df

def crop_and_save_images(img, bbox_df, og_name, save_as_dir,
                         segm_idx, tablet_CDLI, view_desc):
    """
    Crop images into 512 x 512 and save them to specified directory
    :param save_as_dir: Where to save the crops
    :param og_name: Name of the original image
    :param view_desc: View of tablet (e.g. Obv)
    :param tablet_CDLI: CDLI number of the tablet
    :param segm_idx: Image segment number
    :param img: Image to crop
    :param bbox_df: Bbox df related to the image
    :return: List of df rows for new crops
    """
    crop_bbox_rows = []
    h, w = img.shape[:2]

    # Iterate over vertical positions first (height)
    for y in range(0, h - CROP_SIZE + 1, STRIDE):
        # Iterate over horizontal positions (width)
        for x in range(0, w - CROP_SIZE + 1, STRIDE):
            # Crop the image
            crop = img[y:y + CROP_SIZE, x:x + CROP_SIZE]
            crop_name = f"{Path(og_name).stem}_{y}_{x}.jpg"
            cv2.imwrite(os.path.join(save_as_dir, crop_name), crop)

            crop_bboxes = []
            for _, row in bbox_df.iterrows():
                xmin, ymin, xmax, ymax = row["relative_bbox"]

                # Calculate intersection between bbox and crop area
                ixmin, iymin = max(xmin, x), max(ymin, y)
                ixmax, iymax = min(xmax, x + CROP_SIZE), min(ymax, y + CROP_SIZE)

                inter_w = max(0, ixmax - ixmin)
                inter_h = max(0, iymax - iymin)
                inter_area = inter_w * inter_h

                og_area = (xmax - xmin) * (ymax - ymin)
                visibility = inter_area / og_area if og_area > 0 else 0

                # Include bbox if enough of it is visible inside the crop
                if visibility >= VISIBILITY_THRESHOLD:
                    crop_bboxes.append({
                        'image_path': crop_name,
                        'xmin': ixmin - x,
                        'ymin': iymin - y,
                        'xmax': ixmax - x,
                        'ymax': iymax - y,
                        'mzl_label': row["mzl_label"],
                        'segm_idx': segm_idx,
                        'tablet_CDLI': tablet_CDLI,
                        'view_desc': view_desc
                    })

            if crop_bboxes:
                crop_bbox_rows.extend(crop_bboxes)
            else:
                crop_bbox_rows.append({
                    'image_path': crop_name,
                    'xmin': None,
                    'ymin': None,
                    'xmax': None,
                    'ymax': None,
                    'mzl_label': None,
                    'segm_idx': segm_idx,
                    'tablet_CDLI': tablet_CDLI,
                    'view_desc': view_desc
                })

    return crop_bbox_rows














