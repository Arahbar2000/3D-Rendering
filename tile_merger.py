import os
import sys
import numpy as np
import cv2
import open3d as o3d


def read_patch(dir_path, masks=False):
    if not os.path.isdir(dir_path):
        sys.exit("Unable to find patch directory")
    if masks:
        dir_path = os.path.join(dir_path, 'OutMasks')
    image_tiles = []
    tiles = os.listdir(dir_path)
    for x in range(14):
        new_tiles = sorted([tile for tile in tiles if int(tile.split('_')[1]) == x])
        new_tiles = [cv2.imread(os.path.join(dir_path, tile)) for tile in new_tiles]
        image_tiles.append(new_tiles)
    patch = cv2.hconcat(([cv2.vconcat(tiles) for tiles in image_tiles]))
    if not masks:
        layer = int(dir_path.split('/')[-1].split('_')[2][-4:])
    else:
        layer = int(dir_path.split('/')[-2].split('_')[2][-4:])
    return patch, layer


def read_all_patches(dir_path, masks=False):
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
            patch, layer = read_patch(os.path.join(dir_path, patch_file))
            if not masks:
                x_layer_coords, y_layer_coords = np.meshgrid(np.arange(patch.shape[0]), np.arange(patch.shape[1]))
            else:
                y_layer_coords, x_layer_coords, _ = np.where(patch)
            y_layer_coords = (patch.shape[0] - 1) - y_layer_coords
            y_layer_coords = y_layer_coords.flatten()
            x_layer_coords = x_layer_coords.flatten()
            z_coords = np.repeat(layer*15, y_layer_coords.shape[0])
            layer_coords = list(zip(x_layer_coords, y_layer_coords, z_coords))
            patch = patch / 255
            rgb_layer_values = list(zip(patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten()))
            layer_coords = np.asarray(layer_coords)
            rgb_layer_values = np.asarray(rgb_layer_values)
            if flag:
                coords = np.append(coords, layer_coords, axis=0)
                rgb_values = np.append(rgb_values, rgb_layer_values, axis=0)
            else:
                coords = layer_coords
                rgb_values = rgb_layer_values
            flag = True
            counter += 1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
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
        if counter == 3:
            break
        if path.split('/')[-1][0] != '.':
            print(path)
            patch, layer = read_patch(os.path.join(dir_path, patch_file), masks=True)
            y_layer_coords, x_layer_coords, _ = np.where(patch)
            y_layer_coords = (patch.shape[0] - 1) - y_layer_coords
            y_layer_coords = y_layer_coords.flatten()
            x_layer_coords = x_layer_coords.flatten()
            z_coords = np.repeat(layer*5, y_layer_coords.shape[0])
            layer_coords = list(zip(x_layer_coords, y_layer_coords, z_coords))
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
    print(pcd.points)
    o3d.visualization.draw_geometries([pcd])
if __name__ == '__main__':
    # read_all_masks('TilesAndMasksExample1/Res1_47layers/SegmentedTiles')
    read_all_patches('TilesAndMasksExample1/20190514_PC4_patch_0065/Tiles')
