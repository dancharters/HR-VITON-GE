import cv2
import numpy as np
import os


def create_mask(image_path):
    # Global variables
    drawing = False
    brush_radius = 15  # initial brush radius
    last_point = None

    # Load an image
    img = cv2.imread(image_path)

    # Resize the image to 1024x768 if it's not already that size
    # img = cv2.resize(img, (768, 1024))

    img_copy = img.copy()

    # Initialize mask as a white array of the same shape as the image
    mask = np.ones_like(img_copy)

    def draw(event, x, y, flags, param):
        nonlocal drawing, mask, last_point

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            cv2.circle(mask, (x, y), brush_radius, (255, 255, 255), -1)
            last_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(mask, last_point, (x, y), (255, 255, 255), brush_radius*2)
                last_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(mask, last_point, (x, y), (255, 255, 255), brush_radius*2)
            last_point = None

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    while True:
        img_masked = cv2.bitwise_and(img_copy, mask)
        display_image = cv2.addWeighted(img_copy, 0.3, img_masked, 0.7, 0)

        cv2.imshow('image', display_image)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('[') and brush_radius > 1:
            brush_radius -= 1
        elif k == ord(']') and brush_radius < 100:
            brush_radius += 1
        elif k == 27:  # ESC key to exit
            break

    # Save the mask
    base = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join('masks', f'{base}_mask.png')
    cv2.imwrite(mask_path, cv2.bitwise_not(mask))

    # Load the mask back
    mask_red = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Create a colored mask [B, G, R] with red being 255 and rest zero.
    colored_mask = np.zeros_like(img)
    colored_mask[mask_red == 0] = [28, 28, 151]  # Red where the mask is black
    colored_mask[mask_red == 255] = [0, 0, 255]  # Black where the mask is white

    # Define weights for overlay
    overlay_weight = 1
    image_weight = 1

    # Overlay the colored mask onto the original image with transparency
    img_with_mask = cv2.addWeighted(img_copy, image_weight, colored_mask, overlay_weight, 0)

    # Save the final image
    final_image_path = os.path.join('overlays', f'{base}_masked_image.png')
    cv2.imwrite(final_image_path, img_with_mask)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    create_mask('images/02244_00_09148_00.png')
    # create_mask('images/00458_00_12082_00.png')
    # create_mask('images/00620_00_00112_00.png')
    # create_mask('images/01008_00_12704_00.png')
    # create_mask('images/01175_00_08584_00.png')
    # create_mask('images/01352_00_00339_00.png')
