import ssl
import urllib.request as request
from urllib.error import HTTPError
from PIL import Image
import os
from Utils import *

def get_images(top_dir, base_url, save_dir):
    """

    :param top_dir:
    :param base_url:
    :param save_dir:
    :return:
    """
    # Get one df from all csv files
    segments_df = read_in_csv(top_dir=top_dir)

    # Get just the tablet_CDLI column
    cdli_col = segments_df["tablet_CDLI"]

    # Get list of unique cdli numbers
    unique_cdli = cdli_col.drop_duplicates()

    # For each line, get cdli number then fetch from cdli site
    for i in range(len(unique_cdli)):
        selected_cdli = str(unique_cdli.iloc[i]).strip()
        try:
            if "VAT" not in selected_cdli:  # Exclude images starting with "VAT"
                img_url = f"{base_url}{selected_cdli}.jpg"
                context = ssl._create_unverified_context()  # type: ignore[attr-defined]
                # Get image from cdli site
                with request.urlopen(img_url, context=context) as response:
                    img = Image.open(response)
                    img.load()
                # Save image
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{selected_cdli}.jpg")
                img.save(save_path)
        except HTTPError as e:
            print(e)
        except Exception as e:
            print(e)

    return segments_df
