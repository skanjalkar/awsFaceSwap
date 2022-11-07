from Functions import landmarks
from Functions import triangle


def capital_case(x):
    return x.capitalize()


def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'


def test_wrapper():
    image_paths, images = landmarks.landmark(False, "/home/ubuntu/awsFaceSwap/Images/face1.jpg", "/home/ubuntu/awsFaceSwap/Images/face2.jpg")
    triangle.helper(images[0], images[1], "img1", "img2.jpg", "./")
    # print(
    #     "Finished Test"
    # )
