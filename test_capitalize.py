from Functions import landmarks
from Functions import triangle


def capital_case(x):
    return x.capitalize()


def test_capital_case():
    assert capital_case('semaphore') == 'Semaphore'


def test_wrapper():
    image_paths, images = landmarks.landmark(True)
    triangle.helper(images[0], images[1])
    print(
        "Finished Test"
    )
