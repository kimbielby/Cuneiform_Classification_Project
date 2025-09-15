import copy
import cv2
import os
from Utils import get_filepaths
from pathlib import Path

def check_dims(img_list):
    """

    :param img_list:
    :return:
    """
    # Get dimensions of first image
    first_img = img_list[0]
    first_size = first_img.shape[:2]

    # Check if other images in list have same dimensions
    for img in img_list[1:]:
        if img.shape[:2] != first_size:
            return False
    return True

def copy_images(top_dir, og_img_name_list, save_dir):
    """

    :param top_dir:
    :param og_img_name_list:
    :param save_dir:
    :return:
    """
    img_name_list = copy.deepcopy(og_img_name_list)
    # Create folder if it doesn't already exist
    os.makedirs(save_dir, exist_ok=True)

    try:
        for img_name in img_name_list:
            file_name = f"{img_name}.jpg"
            read_path = os.path.join(top_dir, file_name)
            save_path = os.path.join(save_dir, file_name)

            if os.path.exists(read_path):
                img = cv2.imread(read_path)
                print(f"Read in {read_path}")
                cv2.imwrite(save_path, img)
                print(f"Saved {save_path}")
                print("")
            else:
                print(f"No file {read_path}")
                print("")

    except Exception as e:
        print(e)

def save_segments_and_bboxes(img_top_dir, og_seg_df, og_bbox_df, save_dirs):
    """

    :param img_top_dir:
    :param og_seg_df:
    :param og_bbox_df:
    :param save_dirs:
    :return:
    """
    # Deepcopy df's
    seg_df = copy.deepcopy(og_seg_df)
    bbox_df = copy.deepcopy(og_bbox_df)

    # Extract save directories
    img_save_dir = save_dirs[0]
    os.makedirs(img_save_dir, exist_ok=True)
    # Create those directories if they don't already exist
    bbox_save_dir = save_dirs[1]
    os.makedirs(bbox_save_dir, exist_ok=True)

    # Get names of each image in top directory
    filepaths = get_filepaths(dir_name=img_top_dir)
    # For each image in the list, slice the segment df and the bbox df
    for img_name in filepaths:
        # get tablet CDLI number
        cdli = os.path.splitext(img_name)[0]
        tablet_path = os.path.join(img_top_dir, img_name)
        if os.path.exists(tablet_path):
            # Read in the full tablet image
            img = cv2.imread(tablet_path)
            # Get the relevant slices of segments and bbox df's
            seg_df_slice = seg_df[seg_df.tablet_CDLI == cdli]
            bbox_df_slice = bbox_df[bbox_df.tablet_CDLI == cdli]

            for i, (si, s_rec) in enumerate(seg_df_slice.iterrows()):
                # Crop the segment from the og image (obv or rev usually)
                tablet_seg = img[s_rec.bbox[1]:s_rec.bbox[3],
                             s_rec.bbox[0]:s_rec.bbox[2]]
                # Slice that segment's bboxes from the bbox df
                bbox_seg = bbox_df_slice[bbox_df_slice.segm_idx == s_rec.segm_idx].copy()
                # Put commas back into array
                bbox_seg.relative_bbox = bbox_seg.relative_bbox.apply(lambda arr: [int(x) for x in arr])
                # Save tablet segment
                tablet_save_path = os.path.join(img_save_dir, f"{cdli}_{s_rec.segm_idx}.jpg")
                cv2.imwrite(tablet_save_path, tablet_seg)
                # Save bbox segment slice
                bbox_save_path = os.path.join(bbox_save_dir, f"{cdli}_{s_rec.segm_idx}_bbox.csv")
                bbox_seg.to_csv(bbox_save_path, index=False)

def save_segments_with_bboxes(img, seg_df_slice, bbox_df_slice, save_dir):
    """

    :param img:
    :param seg_df_slice:
    :param bbox_df_slice:
    :param save_dir:
    :return:
    """
    # Make sure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Go through each segment for that tablet
    for _, s_rec in seg_df_slice.iterrows():
        # Crop segment
        x_min, y_min, x_max, y_max = map(int, s_rec.bbox)
        tablet_seg = img[y_min:y_max, x_min:x_max].copy()

        # Get bboxes for this segment
        selected_segment = bbox_df_slice[bbox_df_slice.segm_idx == s_rec.segm_idx]
        for _, bbox_rec in selected_segment.iterrows():
            bx_min, by_min, bx_max, by_max = map(int, bbox_rec.relative_bbox)
            label = str(bbox_rec.mzl_label)

            # Draw bboxes
            cv2.rectangle(img=tablet_seg,
                          pt1=(bx_min, by_min),
                          pt2=(bx_max, by_max),
                          color=(255, 0, 0), thickness=2)

            # Draw labels
            cv2.putText(img=tablet_seg, text=label,
                        org=(bx_min, max(0, by_min - 5)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 0, 0), thickness=1,
                        lineType=cv2.LINE_AA)

        # Save segment
        filename = f"{s_rec.tablet_CDLI}_{s_rec.segm_idx}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, tablet_seg)
        print(f"Saved: {save_path}")

def resolve_best_ckpt(run_root: Path, metric_name: str = "map50") -> Path:
    run_root = Path(run_root)

    patterns = [
        f"*/best_{metric_name}.pth", f"best_{metric_name}.pth",
        "*/best*.pth", f"best*.pth"
    ]
    candidate_paths = []
    for p in patterns:
        candidate_paths.extend(run_root.glob(p))
    if not candidate_paths:
        # Fall back to last.pth
        last = sorted(run_root.glob("*/last.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if last:
            return last[0]
        raise FileNotFoundError(f"No best checkpoint under {run_root}")

    return max(candidate_paths, key=lambda p: p.stat().st_mtime)










