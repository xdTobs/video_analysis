import numpy as np
import cv2

def display_color_swatches(colors):
    # Create a blank image
    height = 100
    width = 200
    image = np.zeros((height * len(colors), width * 2, 3), dtype="uint8")

    # Fill the image with the color swatches
    for index, (color, (lower, upper)) in enumerate(colors.items()):
        # Calculate the y position for the rectangles
        y = index * height
        
        # Draw rectangles for lower and upper BGR values
        image[y:y+height, 0:width] = lower
        image[y:y+height, width:width*2] = upper

        # Put text on the rectangles
        cv2.putText(image, f'{color} lower', (5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f'{color} upper', (width + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the image
    cv2.imshow('Color Swatches', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# BGR values from the input text
colors = {
    'white': ([146, 171, 191], [255, 255, 255]),
    'orange': ([148, 192, 202], [255, 255, 255]),
    'green': ([143, 174, 152], [224, 255, 233]),
    'red': ([0, 0, 189], [4, 2, 255]),
    'border': ([0, 14, 168], [30, 62, 255])
}
colors2 = {
    'white': (np.array([158, 161, 170]), np.array([255, 255, 255])),
    'orange': (np.array([58, 117, 187]), np.array([137, 212, 255])),
    'green': (np.array([128, 145, 101]), np.array([202, 222, 156])),
    'red': (np.array([34, 24, 164]), np.array([64, 47, 255])),
    'border': (np.array([19, 23, 129]), np.array([88, 82, 224]))
}

# Display the color swatches using the defined function
display_color_swatches(colors2)