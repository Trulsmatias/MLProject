import numpy as np


def downscale(img: np.ndarray, tilesize=(12, 12)):
    return img[tilesize[0]//2::tilesize[0], tilesize[1]//2::tilesize[1]]


def naive_downscale(img: np.ndarray, tilesize=(12, 12)):
    h_scaled = img.shape[0] // tilesize[0]
    w_scaled = img.shape[1] // tilesize[1]
    scaled = np.empty((h_scaled, w_scaled, img.shape[2]), dtype=int)

    y_offset = tilesize[0] // 2
    x_offset = tilesize[1] // 2

    for x in range(w_scaled):
        for y in range(h_scaled):
            """for i in range(0):
                print('scaled[{}, {}, {}] is set to img[{}, {}, {}]: {}'.format(
                    y, x, i,
                    y_offset + y * tilesize[0],
                    x_offset + x * tilesize[1],
                    i,
                    img[y_offset + y * tilesize[0], x_offset + x * tilesize[1], i]
                ))"""
            scaled[y, x] = img[y_offset + y * tilesize[0], x_offset + x * tilesize[1]]
#            scaled[y, x, 0] = img[y_offset + y * tilesize[0], x_offset + x * tilesize[1], 0]
#            scaled[y, x, 1] = img[y_offset + y * tilesize[0], x_offset + x * tilesize[1], 1]
#            scaled[y, x, 2] = img[y_offset + y * tilesize[0], x_offset + x * tilesize[1], 2]

    return scaled


if __name__ == '__main__':
    from scipy.ndimage import imread
    import matplotlib.pyplot as plt

    image = imread('bild.png')
    #plt.imshow(image)
    #plt.show()

    ts = (4, 4)
    dscaled = downscale(image, tilesize=ts)
    ndscaled = naive_downscale(image, tilesize=ts)
    #print('eq:', np.array_equal(dscaled, ndscaled))
    #print(dscaled - ndscaled)

    print(type(dscaled), dscaled.shape)
    #print(dscaled)
    plt.imshow(dscaled)
    plt.show()

    #print('image size: {}, ndscale size: {}'.format(image.shape, ndscaled.shape))
    #print('highest value in ndscale:', np.max(ndscaled))
    plt.imshow(ndscaled)
    plt.show()

    import time
    tstart = time.time()
    for i in range(1000):
        downscale(image, ts)
    print('downscale time:', time.time() - tstart)

    tstart = time.time()
    for i in range(1000):
        naive_downscale(image, ts)
    print('naive_downscale time:', time.time() - tstart)
