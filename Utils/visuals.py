import os.path
import numpy as np
import matplotlib.pyplot as plt
import math

def visualise_crops_with_bboxes(img, bbox_slice, save_as=None):
    img_name = os.path.splitext(bbox_slice.image_path.iloc[0])[0]

    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    # Show main image
    ax.imshow(img)
    # Set the title of the plot
    ax.set_title(f"{img_name}")
    # Get bboxes and mzl labels
    bboxes = bbox_slice[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
    labels = bbox_slice.mzl_label.values
    # Plot the bboxes and labels
    plot_boxes(boxes=bboxes, labels=labels, ax=ax)

    if save_as:
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        fig.savefig(save_as, bbox_inches="tight")
        print(f"Saved: {os.path.abspath(save_as)}")
        print("")
        plt.close(fig)
    else:
        plt.show()

""" Before Cropping (with PIL) """
def display_basic_image(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def visualise_segments(seg_df_slice, pil_img, annot_df_slice=None):
    """
    Crops and displays segments of a tablet. Can also display sign bboxes.
    :param save_dir:
    :param annot_df_slice: If bbox is True, slice of annotation df will be provided
    :param bbox: True if bounding boxes should be displayed
    :param seg_df_slice: Slice of tablet segments df
    :param pil_img: Image relating to CDLI number in seg_slice
    """
    num_items = len(seg_df_slice)
    cols = 2
    rows = math.ceil(num_items / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    ax = ax.flatten()

    for i, (si, s_rec) in enumerate(seg_df_slice.iterrows()):
        # Crop segment
        tablet_seg = crop_segments(img=pil_img, bbox_list=s_rec.bbox)
        # Plot segment
        ax[i].imshow(tablet_seg)
        title = f"{s_rec.tablet_CDLI} {s_rec.view_desc}"
        ax[i].set_title(title)

        if annot_df_slice:
            # Select sign bbox annotations
            selected_seg = (annot_df_slice.segm_idx == s_rec.segm_idx)
            sign_bboxes = np.stack(annot_df_slice[selected_seg].relative_bbox.values)
            sign_labels = annot_df_slice[selected_seg].mzl_label.values

            # Plot boxes
            plot_boxes(boxes=sign_bboxes, labels=sign_labels, ax=ax[i])

    for j in range(i+1, len(ax)):
        ax[j].axis('off')

    plt.show()

def visualise_line_annotations(seg_df_slice, pil_img, line_df):
    """
    Crops and displays segments of a tablet with line annotations
    :param seg_df_slice: Slice of tablet segments df
    :param pil_img: Image relating to CDLI number in seg_slice
    :param line_df: Line annotations df
    """
    num_items = len(seg_df_slice)
    cols = 2
    rows = math.ceil(num_items / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    ax = ax.flatten()
    for i, (si, s_rec) in enumerate(seg_df_slice.iterrows()):
        # Crop segment
        tablet_seg = crop_segments(img=pil_img, bbox_list=s_rec.bbox)

        # Select line annotations
        line_df_slice = line_df[line_df.segm_idx == s_rec.segm_idx]

        # Plot line annotations
        grouped = line_df_slice.groupby('line_idx')
        for li, line_rec in grouped:
            # Plot single line as piece-wise linear function
            ax[i].plot(line_rec.x.values, line_rec.y.values, linewidth=3)
            # Annotate line with line index
            ax[i].text(line_rec.x.values[0], line_rec.y.values[0],
                       f"{line_rec.line_idx.values[0]}",
                       bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')

        # Plot image
        ax[i].imshow(tablet_seg, cmap='gray')
        ax[i].set_title(f"{s_rec.tablet_CDLI} {s_rec.view_desc}")

    for j in range(i+1, len(ax)):
        ax[j].axis('off')

    plt.show()

def crop_segments(img, bbox_list):
    """
    :param img: PIL image
    :param bbox_list: List of segment bbox coords (xmin, ymin, xmax, ymax)
    :return: Cropped segment of og image as PIL image
    """
    return img.crop((bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3]))

def plot_boxes(boxes, labels, ax=None):
    """
    Plots bounding box annotations on a PIL image
    :param boxes: Sign bounding boxes
    :param labels: Sign labels
    :param ax: Axis number
    """
    # Set up figure if necessary
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 12))
    # Iterate over bounding boxes
    for ii, bbox in enumerate(boxes):
        # Plot box
        ax.add_patch(
            plt.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor='blue',
                alpha=0.8,
                linewidth=1.0
            )
        )
        # Plot label
        ax.text(
            bbox[0],
            bbox[1] - 2,
            f"{labels[ii]}",
            bbox=dict(facecolor='blue', alpha=0.2),
            fontsize=6,
            color='white'
        )



