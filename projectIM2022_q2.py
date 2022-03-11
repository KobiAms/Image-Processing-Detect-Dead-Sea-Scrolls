from functions import read_images_from_dir, clear_margins, find_mask, get_scroll_contour
import cv2
from pathlib import Path


files_names, images = read_images_from_dir("./images/")
if images and len(images) > 0:
    
    print("FOUND: ", len(images), "files")
    print("create output directory in ./out/contours")
    Path("./out/contours/").mkdir(parents=True, exist_ok=True)



    for i in range(len(files_names)):
        print("READ:", files_names[i], " SHAPE:", images[i].shape)

    for i, image in enumerate(images):

        cleared = clear_margins(image)
        mask = find_mask(cleared)
        detected = get_scroll_contour(image, mask, files_names[i])
        cv2.imwrite("./out/contours/"+files_names[i]+".jpg", detected)

else:
    print("ERROR: files not found")