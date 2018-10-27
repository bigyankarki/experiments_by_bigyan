from PIL import Image
import base64
import numpy as np
from io import BytesIO
import cv2

def processed_image(img_b64):
    # decode the image
    encoded_image = img_b64.split(",")[1]
    # print(encoded_image)
    # print(encoded_image)
    decoded_image = base64.b64decode(encoded_image)

    # read the decoded image
    pil_img = Image.open(BytesIO(decoded_image)).convert('L')
    open_cv_image = np.array(pil_img)
    resized_image = cv2.resize(255-open_cv_image, (28, 28))  # Resizing the image into 28x28 matrix


    norm_image = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # ---------------Fit the image into 20x20 pixel box-------------
    # remove unnecessary black background
    while np.sum(norm_image[0]) == 0:
        norm_image = norm_image[1:]

    while np.sum(norm_image[:,0]) == 0:
        norm_image = np.delete(norm_image, 0, 1)

    while np.sum(norm_image[-1]) == 0:
        norm_image = norm_image[:-1]

    while np.sum(norm_image[:,-1]) == 0:
        norm_image = np.delete(norm_image, -1, 1)

    rows,cols = norm_image.shape

    # Now fit the image to 20x20 box.
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        norm_image = cv2.resize(norm_image, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        norm_image = cv2.resize(norm_image, (cols, rows))

    # now let's resize the image into 28x28.
    colsPadding = (int(np.math.ceil((28 - cols) / 2.0)), int(np.math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(np.math.ceil((28-rows)/2.0)),int(np.math.floor((28-rows)/2.0)))
    norm_image = np.lib.pad(norm_image, (rowsPadding,colsPadding), 'constant')

    return norm_image.flatten()


if __name__ == "__main__":
    processed_image(img)
