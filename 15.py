import cv2
import numpy as np

def detect_road_margins(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (0, height),
        (width * 0.1, height * 0.5),
        (width * 0.9, height * 0.5),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=50,
        maxLineGap=30
    )

    line_image = np.copy(image)
    left_lines, right_lines = [], []

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.2:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0.2:
                    right_lines.append((x1, y1, x2, y2))

    for x1, y1, x2, y2 in left_lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for x1, y1, x2, y2 in right_lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    cv2.imshow('Original Image', image)
    cv2.imshow('Detected Road Margins', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

if __name__ == "__main__":
    image_path = 'road.png'
    result_image = detect_road_margins(image_path)
    cv2.imwrite('road_with_margins.jpg', result_image)
