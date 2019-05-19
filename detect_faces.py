import cv2
import os
import numpy as np


def detect(path, filename, scaleFactor=1.5, cascade_file='cascade/lbpcascade_animeface.xml'):
    if not os.path.isfile(cascade_file):
        raise RuntimeError('%s: not found' % cascade_file)
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(24, 24))
        for i, (x, y, w, h) in enumerate(faces):
            # cv2.rectangle(image, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)
            xcenter, ycenter = (x + w / 2, y + h / 2)
            w, h = (w, h) * np.array([scaleFactor, scaleFactor])
            x1 = int(xcenter - w / 2)
            y1 = int(ycenter - h / 2)
            x2 = int(x1 + w)
            y2 = int(y1 + h)
            try:
                roi = image[y1:y2, x1:x2]
                roi = cv2.resize(roi, (128, 128))
                cv2.imwrite('dataset/scaled_faces/{}-{}.jpg'.format(filename, i), roi)
                print('success in scaling {}-{}.jpg'.format(filename, i))
            except Exception:
                print('failure in scaling {}-{}.jpg'.format(filename, i))
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)


if __name__ == '__main__':
    try:
        os.makedirs('dataset/scaled_faces')
    except FileExistsError:
        pass
    faces_dir = 'dataset/faces'
    for img in os.listdir(faces_dir):
        detect('dataset/faces/{}'.format(img), '{}'.format(img[:-4]))
