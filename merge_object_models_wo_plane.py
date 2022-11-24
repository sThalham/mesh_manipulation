import numpy as np
import open3d as o3d
import csv
import cv2
import os
import sys
import transforms3d as tf3d
import copy


def main(argv):
    mesh1 = argv[1]
    mesh2 = argv[2]

    mesh1 = o3d.io.read_triangle_mesh(mesh1)
    mesh2 = o3d.io.read_triangle_mesh(mesh2)
    #mesh1 = o3d.io.read_point_cloud(mesh1)
    #mesh2 = o3d.io.read_point_cloud(mesh2)

    # outlier removal
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters1, cluster_n_triangles1, cluster_area1 = (
            mesh1.cluster_connected_triangles())
    triangle_clusters1 = np.asarray(triangle_clusters1)
    cluster_n_triangles1 = np.asarray(cluster_n_triangles1)
    mesh_0 = copy.deepcopy(mesh1)
    triangles_to_remove1 = cluster_n_triangles1[triangle_clusters1] < 200
    #largest_cluster_idx1 = cluster_n_triangles1.argmax()
    #triangles_to_remove1 = triangle_clusters1 != largest_cluster_idx1
    mesh_0.remove_triangles_by_mask(triangles_to_remove1)
    mesh1 = copy.deepcopy(mesh_0)

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters2, cluster_n_triangles2, cluster_area2 = (
            mesh2.cluster_connected_triangles())
    triangle_clusters2 = np.asarray(triangle_clusters2)
    cluster_n_triangles2 = np.asarray(cluster_n_triangles2)
    mesh_02 = copy.deepcopy(mesh2)
    triangles_to_remove2 = cluster_n_triangles2[triangle_clusters2] < 200
    #largest_cluster_idx2 = cluster_n_triangles2.argmax()
    #triangles_to_remove2 = triangle_clusters2 != largest_cluster_idx2
    mesh_02.remove_triangles_by_mask(triangles_to_remove2)
    mesh2 = copy.deepcopy(mesh_02)

    vertices1 = np.asarray(mesh1.vertices)
    #colors1 = np.asarray(mesh1.colors)
    #print(mesh1.has_color())
    offset = np.array([(np.max(vertices1[:, 0]) - np.min(vertices1[:, 0])) * 0.5 + np.min(vertices1[:, 0]), (np.max(vertices1[:, 1]) - np.min(vertices1[:, 1])) * 0.5 + np.min(vertices1[:, 1]), (np.max(vertices1[:, 2]) - np.min(vertices1[:, 2])) * 0.5 + np.min(vertices1[:, 2])])
    for i in range(vertices1.shape[0]):
        vertices1[i, :] -= offset

    vertices2 = np.asarray(mesh2.vertices)
    offset = np.array([(np.max(vertices2[:, 0]) - np.min(vertices2[:, 0])) * 0.5 + np.min(vertices2[:, 0]),
                       (np.max(vertices2[:, 1]) - np.min(vertices2[:, 1])) * 0.5 + np.min(vertices2[:, 1]),
                       (np.max(vertices2[:, 2]) - np.min(vertices2[:, 2])) * 0.5 + np.min(vertices2[:, 2])])
    for i in range(vertices2.shape[0]):
        vertices2[i, :] -= offset

    transform2 = np.eye(4)
    #transform2[:3, :3] = tf3d.euler.euler2mat(np.pi, 0.0, np.pi*1.5, 'sxyz') # minispray ocean
    transform2[:3, :3] = tf3d.euler.euler2mat(np.pi, 0.0, -np.pi*0.6, 'sxyz')
    mesh2.transform(transform2)

    o3d.visualization.draw_geometries([mesh1, mesh2])

    #mesh1 = mesh1.merge_close_vertices(2.0)
    #mesh2 = mesh2.merge_close_vertices(2.0)
    pc1 = o3d.geometry.PointCloud()
    pc1.points = mesh1.vertices
    pc1.colors = mesh1.vertex_colors
    pc1.normals = mesh1.vertex_normals
    pc2 = o3d.geometry.PointCloud()
    pc2.points = mesh2.vertices
    pc2.colors = mesh2.vertex_colors
    pc2.normals = mesh2.vertex_normals

    pc1 = pc1.voxel_down_sample(voxel_size=1.0)
    pc2 = pc2.voxel_down_sample(voxel_size=1.0)

    current_transformation = np.identity(4)
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        pc1, pc2, 1.0, current_transformation, o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.5000),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                relative_rmse=1e-6,
                                                max_iteration=100))
    current_transformation = result_icp.transformation
    print(current_transformation)
    mesh1.transform(current_transformation)

    o3d.visualization.draw_geometries([mesh1, mesh2])

    final_mesh = mesh1 + mesh2
    final_name = os.path.split(argv[1])[-1]
    print(final_name)
    print(argv[1])
    out_name = final_name[:-5] + '_final.ply'
    out_name = argv[1].replace(final_name, out_name)
    print(out_name)
    o3d.io.write_triangle_mesh(out_name,
                               final_mesh,
                               write_triangle_uvs=True)


if __name__ == "__main__":
    main(sys.argv)



'''
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

