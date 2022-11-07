from Functions import landmarks
from Functions import triangle


def capital_case(x):
    return x.capitalize()


def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'


def test_wrapper():
    image_paths, images = landmarks.landmark(False, "Images/face1.jpg", "Images/face2.jpg", "shape_predictor_68_face_landmarks.dat")
    triangle.helper(images[0], images[1], "img1", "img2.jpg", "./")
    # print(
    #     "Finished Test"
    # )
