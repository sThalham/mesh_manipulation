import numpy as np
import open3d as o3d
import csv
import cv2
import os

cad_short = 'soup'
mesh_short = 'obj_000001'

path_cad = os.path.join('/home/stefan/data/datasets/EvalMesh/Meshes/CAD/HOPE_raw', cad_short + '.ply')
path_ply = os.path.join("/home/stefan/data/datasets/EvalMesh/Meshes/BOP/HOPE_sub", mesh_short + '.ply')
path_img = os.path.join("/home/stefan/data/datasets/EvalMesh/Meshes/BOP/HOPE_sub", mesh_short + '.png')
path_tar = os.path.join("/home/stefan/data/datasets/EvalMesh/Meshes/CAD/HOPE", mesh_short + '.ply')

#path_cad = "/home/stefan/data/datasets/EvalMesh/Meshes/CAD/HOPE_raw/re_bbq_sauce.ply"
#path_ply = "/home/stefan/data/datasets/EvalMesh/Meshes/BOP/HOPE_sub/obj_000002.ply"
#path_img = "/home/stefan/data/datasets/EvalMesh/Meshes/BOP/HOPE_sub/obj_000002.png"
#path_tar = "/home/stefan/data/datasets/EvalMesh/Meshes/CAD/HOPE/obj_000002.ply"


texture = cv2.imread(path_img)
source_mesh = []

recon = o3d.io.read_triangle_mesh(path_ply)
recon_uv = recon.triangle_uvs
recon.paint_uniform_color([1, 0.706, 0])
cad = o3d.io.read_triangle_mesh(path_cad)
cad.paint_uniform_color([1, 0.706, 0])


uvs = []
points = []
colors = []
with open(path_ply, newline='') as uvsrc:
    bopreader = csv.reader(uvsrc, delimiter=' ')
    for pdx, pt in enumerate(bopreader):
        if pdx > 13 and len(pt) > 4:
            uv = np.array(pt[6:8], dtype=np.float32)
            uvs.append(uv)
            point = np.array(pt[:3], dtype=np.float32)
            points.append(point)
            color = np.array(pt[:3], dtype=np.float32)
            colors.append(color)

uvs = np.asarray(uvs)
points = np.asarray(points)

for idx, poi in enumerate(recon.vertices):
    uv_coord = uvs[idx]
    x_coord = int(texture.shape[1] * uv_coord[0])
    y_coord = int(texture.shape[0] - (texture.shape[0] * uv_coord[1]))
    # x_coord = int(uv_coord[0] * center * texture.shape[1])
    # y_coord = int(uv_coord[1] * center * texture.shape[0])

    color = texture[y_coord, x_coord, :]

    np.asarray(recon.vertex_colors)[idx, 0] = color[0] * (1 / 255)
    np.asarray(recon.vertex_colors)[idx, 1] = color[1] * (1 / 255)
    np.asarray(recon.vertex_colors)[idx, 2] = color[2] * (1 / 255)
    #texture[y_coord, x_coord, :] = [255, 255, 255]

    np.asarray(recon.vertices)[idx, :] = np.asarray(recon.vertices)[idx, :] * 0.001

#o3d.visualization.draw_geometries([cad])

color_scale = 1.1
#if path_ply.endswith('obj_000013.ply'):
#    color_scale = np.nanmax(np.linalg.norm(np.asarray(recon.vertex_colors), axis=1))
#    #print(np.linalg.norm(np.asarray(recon.vertex_colors), axis=1).shape)
#    print('Scaling brightness for obj_000013.ply with ', color_scale)

#cv2.imwrite('/home/stefan/texture_man.png', texture)
#o3d.visualization.draw_geometries([recon])

cad_tree = o3d.geometry.KDTreeFlann(cad)
recon_tree = o3d.geometry.KDTreeFlann(recon)

#for idx, poi in enumerate(cad.vertices):
for idx, vert in enumerate(cad.vertices):
    center = np.asarray(vert)
    #print(np.asarray(recon.triangle_uvs)[idx])
    #uv = np.asarray(recon.triangle_uvs)[idx, :]

    #[k, ldx, _] = recon_tree.search_knn_vector_3d(center, 3)
    [k, ldx, _] = recon_tree.search_hybrid_vector_3d(center, 0.01, 3)
    #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

    #orth = np.argmin(np.var(np.asarray(recon.vertices)[ldx, :], axis=0))
    #if center[orth] < 0:
    #    continue

    ldx = np.asarray(ldx)
    print('1: ', ldx)

    #center = np.linalg.norm(np.asarray(recon.vertices)[ldx, :][0])
    äuv_coord = uvs[ldx]
    uv_coord = np.median(uvs[ldx], axis=0)
    print('2: ', uv_coord.shape)

    #print(uv_coord)

    x_coord = int(texture.shape[1] * uv_coord[0])
    y_coord = int(texture.shape[0] - (texture.shape[0] * uv_coord[1]))
    #x_coord = int(uv_coord[0] * center * texture.shape[1])
    #y_coord = int(uv_coord[1] * center * texture.shape[0])

    #color = texture[y_coord, x_coord, :]
    #color = np.asarray(recon.vertex_colors)[ldx, :][0] * color_scale
    color = np.mean(np.asarray(recon.vertex_colors)[ldx, :], axis=0) * color_scale

    #texture[x_coord, y_coord, :] = 0

    #print(x_coord, y_coord, color)

    np.asarray(cad.vertex_colors)[idx, 0] = color[2]# * (1/255)
    np.asarray(cad.vertex_colors)[idx, 1] = color[1]# * (1 / 255)
    np.asarray(cad.vertex_colors)[idx, 2] = color[0]# * (1 / 255)
    #np.asarray(cad.vertex_colors)[ldx, :] = poi * 1000.0 * 100.0
    #print('point: ', k, ldx, poi, np.asarray(recon.vertices)[ldx])

#v_uv = np.random.rand(len(recon.triangles) * 3, 2)
#cad.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
#cv2.imwrite('/home/stefan/texture_man.png', texture)
o3d.visualization.draw_geometries([cad])
o3d.io.write_triangle_mesh(path_tar, cad)

'''
with open(path_ply, newline='') as uvsrc:
    bopreader = csv.reader(uvsrc, delimiter=' ')
    for ptx, pt in enumerate(bopreader):
        source_mesh.append(pt)

with open(path_tar, 'w', newline='') as target:
    plywriter = csv.writer(target, delimiter=' ')
    with open(path_cad, newline='') as cad:
        cadreader = csv.reader(cad, delimiter=' ')
        for idx, row in enumerate(cadreader):
            newitem = []
            #newitem = ''.join(map(str, my_lst))
            for obj in row:
                newitem.append(obj)
            if idx == 10:
                plywriter.writerow(['property', 'uchar', 'blue'])
                plywriter.writerow(['property', 'uchar', 'green'])
                plywriter.writerow(['property', 'uchar', 'red'])
            if idx > 12 and len(newitem) > 4:
                distance = 1000.0
                color = None
                uv = None
                pt_tar = np.array(row[:3], dtype=np.float32)

                loops = 0
                for ptx, pt in enumerate(source_mesh):
                    if ptx < 15:
                        continue
                    pt_src = np.array(pt[:3], dtype=np.float32)
                    d_query = np.sqrt(np.sum(np.power(pt_tar - pt_src, 2)))
                    if d_query < distance:
                        distance = d_query
                        uv = np.array(pt[6:8], dtype=np.float32)
                    loops += 1

                x_coord = int(texture.shape[0] * uv[0])
                y_coord = int(texture.shape[1] * uv[1])

                color = texture[x_coord, y_coord, :]
                newitem.append(str(color[0]))
                newitem.append(str(color[1]))
                newitem.append(str(color[2]))
                plywriter.writerow(newitem)
            else:
                plywriter.writerow(newitem)
'''

