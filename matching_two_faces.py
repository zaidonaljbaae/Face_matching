import cv2
import numpy as np
import dlib


def extract_landmarks(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks_points = []

    for face in faces:
        landmarks = predictor(gray, face)
        points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points.append((x, y))
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        landmarks_points.append(points)

    return landmarks_points, image


def perform_delaunay_triangulation(points, img_color):
    rect = cv2.boundingRect(np.array(points))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        cv2.line(img_color, pt1, pt2, (0, 0, 255), 2)
        cv2.line(img_color, pt2, pt3, (0, 0, 255), 2)
        cv2.line(img_color, pt1, pt3, (0, 0, 255), 2)


def calculate_distance(landmarks1, landmarks2):
    if len(landmarks1) != len(landmarks2):
        return float('inf')
    distances = []
    for p1, p2 in zip(landmarks1, landmarks2):
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        distances.append(dist)
    return np.mean(distances)


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    img1 = cv2.imread("1.jpg")
    img2 = cv2.imread("2.jpg")

    landmarks1, img1_with_landmarks = extract_landmarks(img1, detector, predictor)
    landmarks2, img2_with_landmarks = extract_landmarks(img2, detector, predictor)

    for points in landmarks1:
        perform_delaunay_triangulation(points, img1_with_landmarks)

    for points in landmarks2:
        perform_delaunay_triangulation(points, img2_with_landmarks)

    if landmarks1 and landmarks2:
        distance = calculate_distance(landmarks1[0], landmarks2[0])
        print(f"Average distance between landmarks: {distance}")
        threshold = 200
        if distance < threshold:
            print("Faces match!")
        else:
            print("Faces do not match.")

    cv2.imshow("Image 1 with landmarks", img1_with_landmarks)
    cv2.imshow("Image 2 with landmarks", img2_with_landmarks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
