import PIL
from PIL import Image
import cv2
import numpy as np
import torchvision


class CropBorder(object):
    """
    Crops image so that it doesn't contain excess border
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray)
        Returns:
            PIL Image or numpy.ndarray
        """
        convert_back_to_pil = False

        if isinstance(pic, Image.Image):
            pic = np.array(pic)
            convert_back_to_pil = True

        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # find contours in the edge map
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        cropped = self._crop(img=pic, contours=contours)

        if convert_back_to_pil:
            cropped = Image.fromarray(cropped)

        return cropped

    @staticmethod
    def _crop(img, contours):
        idx = np.argmax([cv2.contourArea(c) for c in contours])
        x, y, w, h = cv2.boundingRect(contours[idx])
        return img[y:y + h, x:x + w]

    def __repr__(self):
        return self.__class__.__name__


class ResizeMaxLength:
    """
    Resizes image so that larger edge has length of size and other edge is
    scaled by size * smaller / larger
    """
    def __init__(self,size,
                      interpolation=torchvision.transforms.InterpolationMode
                      .BILINEAR):
        self._size = size
        self._interpolation = interpolation

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or torch.Tensor)
        Returns:
            (PIL Image or torch.Tensor)
        """
        if isinstance(pic, Image.Image):
            shape = pic.size
        else:
            shape = pic.shape

        largest_edge = np.argmax(shape)
        aspect_ratio = shape[int(not largest_edge)] / shape[largest_edge]

        if largest_edge == 0:
            new_size = (int(self._size * aspect_ratio), self._size)
        else:
            new_size = (self._size, int(self._size * aspect_ratio))
        return torchvision.transforms.Resize(new_size,
                                             interpolation=self._interpolation)(pic)


def main(from_ndarray=False):
    from matplotlib import pyplot as plt
    from PIL import Image
    import time
    import imgaug.augmenters as iaa

    f, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 10))

    cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1,2)]
    for i, cell in enumerate(cells):
        start = time.time()
        if from_ndarray:
            pic = cv2.imread(
                '/Users/adam.amster/herbarium-2020-fgvc7/nybg2020/train'
                '/images/000/01/515057.jpg')
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        else:
            pic = Image.open('/Users/adam.amster/herbarium-2020-fgvc7/nybg2020/train'
                '/images/000/01/515057.jpg')

        composition = [
            CropBorder()
        ]
        # if from_ndarray:
        #     composition.append(torchvision.transforms.ToTensor())
        # composition.append(torchvision.transforms.RandomRotation(360,
        #                                                          expand=True))
        # composition.append(
        #     iaa.Sequential([
        #         iaa.Affine(rotate=(-360, 360), mode='constant')
        #     ]).augment_image
        # )

        # composition.append(torchvision.transforms.RandomHorizontalFlip())
        # composition.append(torchvision.transforms.RandomVerticalFlip())
        # composition.append(torchvision.transforms.RandomRotation(30,
        #                                                          fill=(245,
        #                                                                244,
        #                                                                238)))


        composition.append(torchvision.transforms.Resize((448, 314)))

        transforms = torchvision.transforms.Compose(composition)
        pic = transforms(pic)

        end = time.time()
        print(end - start)
        ax[cell].imshow(np.array(pic))
    plt.show()


if __name__ == '__main__':
    main(from_ndarray=False)