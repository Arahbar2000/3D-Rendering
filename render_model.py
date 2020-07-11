import os
import sys
import numpy as np
import cv2
import open3d as o3d
import skimage.measure
import skimage.segmentation
import skimage.color
import argparse


def parse_arguments():
    describe = "Obtain patch directories and modeling method"
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("required arguments")

    help_i = "patch directory containing all patches"
    required.add_argument("-i", "--patch_dir", help=help_i, type=str, required=True)

    help_l = "label method"
    required.add_argument("-l", "--label", help=help_l, type=int, required=False)

    help_n = "number of layers"
    required.add_argument("-n", "--num_layers", help=help_n, type=int, required=False)

    help_t = "layer thickness"
    required.add_argument("-t", "--thickness", help=help_t, type=int, required=False, default=15)

    args = parser.parse_args()
    return {"patch_dir": args.patch_dir,
            "label": args.label,
            "num_layers": args.num_layers,
            "thickness": args.thickness}




def read_patch(dir_path, label=0):
    if not os.path.isdir(dir_path):
        sys.exit("Unable to find patch directory")
    if label == 3:
        dir_path = os.path.join(dir_path, 'OutMasks')
    elif label:
        dir_path = os.path.join(dir_path, 'OutLabels')
    image_tiles = []
    tiles = os.listdir(dir_path)
    for x in range(14):
        new_tiles = sorted([tile for tile in tiles if int(tile.split('_')[1]) == x])
        if not label:
            new_tiles = [cv2.imread(os.path.join(dir_path, tile)) for tile in new_tiles]
        else:
            new_tiles = [cv2.imread(os.path.join(dir_path, tile), flags=cv2.IMREAD_GRAYSCALE) for tile in new_tiles]
            if label < 3:
                new_tiles = transform_labels_to_rgb(new_tiles)
        image_tiles.append(new_tiles)
    temp = [cv2.vconcat(tiles) for tiles in image_tiles]
    patch = cv2.hconcat(([cv2.vconcat(tiles) for tiles in image_tiles]))
    if not label:
        layer = int(dir_path.split('/')[-1].split('_')[2][-4:])
    else:
        layer = int(dir_path.split('/')[-2].split('_')[2][-4:])


    return patch, layer


def read_all_patches(user_options):
    if not os.path.isdir(user_options["patch_dir"]):
        sys.exit("Unable to find Tiles Directory")

    coords = None
    rgb_values = None
    flag = False
    counter = 0
    for patch_file in sorted(os.listdir(user_options["patch_dir"])):
        path = os.path.join(user_options["patch_dir"], patch_file)
        if counter == user_options["num_layers"]:
            break
        if path.split('/')[-1][0] != '.':
            print(path)
            patch, layer = read_patch(os.path.join(user_options["patch_dir"], patch_file), label=user_options["label"])
            if user_options["label"] != 3:
                y_layer_coords, x_layer_coords = np.meshgrid(np.arange(patch.shape[0]), np.arange(patch.shape[1]))
            else:
                y_layer_coords, x_layer_coords = np.where(patch)
            y_layer_coords = (patch.shape[0] - 1) - y_layer_coords
            y_layer_coords = y_layer_coords.flatten()
            x_layer_coords = x_layer_coords.flatten()
            z_coords = np.repeat(layer*user_options["thickness"], y_layer_coords.shape[0])
            layer_coords = list(zip(x_layer_coords, y_layer_coords, z_coords))
            patch = patch / 255
            if user_options["label"] < 3:
                rgb_layer_values = list(zip(patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten()))
                rgb_layer_values = np.asarray(rgb_layer_values)
            layer_coords = np.asarray(layer_coords)
            if user_options["label"] == 2:
                indices = np.where((rgb_layer_values == (0, 0, 0)).all(axis=1))
                rgb_layer_values = np.delete(rgb_layer_values, indices, axis=0)
                layer_coords = np.delete(layer_coords, indices, axis=0)
            if flag:
                coords = np.append(coords, layer_coords, axis=0)
                if user_options["label"] < 3:
                    rgb_values = np.append(rgb_values, rgb_layer_values, axis=0)
            else:
                coords = layer_coords
                if user_options["label"] < 3:
                    rgb_values = rgb_layer_values
            flag = True
            counter += 1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    if user_options["label"] == 3:
        pcd = pcd.paint_uniform_color([0, 0, 0])
    else:
        pcd.colors = o3d.utility.Vector3dVector(rgb_values)
    o3d.visualization.draw_geometries([pcd])


def read_all_masks(dir_path):
    if not os.path.isdir(dir_path):
        sys.exit("Unable to find Tiles Directory")

    coords = None
    rgb_values = None
    flag = False
    counter = 0
    for patch_file in sorted(os.listdir(dir_path)):
        path = os.path.join(dir_path, patch_file)
        if counter == 4:
            break
        if path.split('/')[-1][0] != '.':
            print(path)
            patch, layer = read_patch(os.path.join(dir_path, patch_file), masks=True)
            y_layer_coords, x_layer_coords = np.where(patch)
            y_layer_coords = (patch.shape[0] - 1) - y_layer_coords
            y_layer_coords = y_layer_coords.flatten()
            x_layer_coords = x_layer_coords.flatten()
            z_coords = np.repeat(layer*5, y_layer_coords.shape[0])
            layer_coords = list(zip(y_layer_coords, x_layer_coords, z_coords))
            patch = patch / 255
            layer_coords = np.asarray(layer_coords)
            if flag:
                coords = np.append(coords, layer_coords, axis=0)
            else:
                coords = layer_coords
            flag = True
            counter += 1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd = pcd.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([pcd])

def transform_labels_to_rgb(images):
    new_images = []
    colors = {  0: [255, 0, 255],
                1: [0, 0, 255],
                2: [0, 255, 0],
                3: [255, 0, 0],
                4: [255, 255, 0],
                5: [0, 255, 255],
                }
    for image in images:
        image = skimage.measure.label(image, background=0)
        rgb_image = np.zeros((image.shape[0], image.shape[1], 3))
        for key in colors.keys():
            rgb_image[image % len(colors) == key] = colors[key]
            rgb_image[image == 0] = [0, 0, 0]
        new_images.append(rgb_image)
    return new_images



if __name__ == '__main__':
    read_all_patches(parse_arguments())