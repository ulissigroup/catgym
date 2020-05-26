import numpy as np
import imageio
import os
import cv2
from matplotlib import pyplot as plt

from surface_env import *

pos = np.load('./new_pos/pos_2_3000.npy')
print(pos.shape)
lattice = Surface()
print("max pos:", lattice.max_positions())
print("min pos:", lattice.min_positions())
img_dir = './trajectory'
energy_path = []

for i in range(pos.shape[0]):
    lattice.set_free_atoms(pos[i,:])
    e = lattice.calculate_energy()
    energy_path.append(e)
    if (i % 100 == 0) or (i % 20 == 0 and i < 99):
        print("On step {} energy {}".format(i, e))
    lattice.save_fig(os.path.join(img_dir, str(i) + ".png"))

images = []
img_fn = [fn for fn in os.listdir(img_dir) if fn.endswith('.png')]
img_fn = sorted(img_fn, key=lambda f: int(f.split('.')[0]))
# for fn in img_fn:
#     # print(fn)
#     images.append(imageio.imread(os.path.join(img_dir, fn)))
# imageio.mimsave('./simple_env.gif', images, duration=0.02)

# img_array = []
# for filename in img_fn:
#     img = cv2.imread(os.path.join(img_dir, filename))
#     height, width, layers = img.shape
#     size = (width,height)
#     img_array.append(img)

# out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

def get_padding_size(image, height, width):
    h, w, _ = image.shape
    # print(w)
    # longest_edge = max(h, w)
    top, bottom, left, right = (0, 0, 0, 0)
    if h < height:
        dh = height - h
        top = dh // 2
        bottom = dh - top
    if w < width:
        dw = width - w
        left = dw // 2
        right = dw - left
    return top, bottom, left, right

for filename in img_fn:
    img = cv2.imread(os.path.join(img_dir, filename))
    top, bottom, left, right = get_padding_size(img, 512, 512)
    # print(top, bottom, left, right)
    WHITE = [255, 255, 255]
    pad_img = cv2.copyMakeBorder(img, top , bottom, left, right, cv2.BORDER_CONSTANT, value=WHITE)
    cv2.imwrite(os.path.join('./pad_traj', filename), pad_img)

images = []
img_fn = [fn for fn in os.listdir('./pad_traj') if fn.endswith('.png')]
img_fn = sorted(img_fn, key=lambda f: int(f.split('.')[0]))
for fn in img_fn:
    # print(fn)
    images.append(imageio.imread(os.path.join('./pad_traj', fn)))
imageio.mimsave('./simple_env.gif', images, duration=0.02)
print("gif saved!")

plt.figure()
plt.xlabel('timestep')
plt.ylabel('surface energy')
plt.plot(energy_path)
# plt.show()
plt.savefig('energy_path.png')
print("figure saved!")