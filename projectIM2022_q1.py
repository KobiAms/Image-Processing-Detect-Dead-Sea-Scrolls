from functions import read_images_from_dir, clear_margins, find_mask, detect_scrolls
import cv2
from pathlib import Path

files_names, images = read_images_from_dir("./images/")


if images and len(images) > 0:

    print("FOUND: ", len(images), "files")
    print("create output directory in ./out/bounding_boxes")
    Path("./out/bounding_boxes/").mkdir(parents=True, exist_ok=True)
    

    for i, image in enumerate(images):
        
        cleared = clear_margins(image)
        mask = find_mask(cleared)
        detected = detect_scrolls(image, mask, files_names[i])
        cv2.imwrite("./out/bounding_boxes/"+files_names[i]+".jpg", detected)

else:
    print("ERROR: files not found")



