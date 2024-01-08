import cv2
import numpy as np

# Initialize variables
points = []
image = None

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.imshow("Select ROI", image)
        else:
            print("Processing points...")
            process_points(points)
            points = []  # Clear points for the next selection
            cv2.imshow("Select ROI", image)

# Function to process points
def process_points(points):
    # Order the points clockwise
    points = np.array(points, dtype=np.float32)
    ordered_points = np.zeros((4, 2), dtype=np.float32)

    # Find the top-left, top-right, bottom-right, and bottom-left points
    s = points.sum(axis=1)
    ordered_points[0] = points[np.argmin(s)]
    ordered_points[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    ordered_points[1] = points[np.argmin(diff)]
    ordered_points[3] = points[np.argmax(diff)]

    # Set the destination points for the perspective transformation
    w, h = 500, 500
    destination_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    # Calculate the perspective transformation matrix and apply it to the original image
    matrix = cv2.getPerspectiveTransform(ordered_points, destination_points)
    result = cv2.warpPerspective(image, matrix, (w, h))

    # Convert the ROI to HSV
    roi_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)

    # Calculate min, max, and avg HSV values over height and width of image
    min_hsv = np.min(roi_hsv, axis=(0, 1))
    max_hsv = np.max(roi_hsv, axis=(0, 1))
    avg_hsv = np.mean(roi_hsv, axis=(0, 1))

    print("Min HSV:", min_hsv)
    print("Max HSV:", max_hsv)
    print("Avg HSV:", avg_hsv)

    # Display the selected ROI and the result
    cv2.imshow("Selected ROI", result)

# Load the image
image_path = "/home/allen/catkin_ws/src/franka_irom/src/imageAnalysis.png"  # image path
image = cv2.imread(image_path)
cv2.imshow("Select ROI", image)

# Set the callback function for mouse events
cv2.setMouseCallback("Select ROI", mouse_callback)

# Continuous loop for point selection and processing
while True:
    cv2.waitKey(1) # wait call for responsiveness

    if len(points) == 4:
        print("Processing points...")
        process_points(points)
        points = []  # Clear points for the next selection
        cv2.imshow("Select ROI", image)

    key = cv2.waitKey(1) # most recent wait captures event
    if key == 27:  # Check if the user pressed the 'Esc' key to exit
        break

# Cleanup
cv2.destroyAllWindows()
