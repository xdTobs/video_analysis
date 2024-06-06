import numpy as np
import cv2

# Define BGR values from the input text
colors = {
    'white': ([181, 168, 169], [255, 255, 255]),
    'orange': ([119, 166, 200], [187, 255, 255]),
    'green': ([162, 168, 101], [246, 255, 152]),
    'red': ([57, 46, 188], [131, 109, 255]),
    'border': ([52, 51, 147], [107, 109, 255])
}

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
