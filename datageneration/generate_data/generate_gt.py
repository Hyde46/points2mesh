#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 ComputerGraphics Tuebingen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch

# parallel python generate_gt.py --label 1 --dir /graphics/scratch/datasets/ModelNet40/advanced/100 --num_parts 10 --samples 100 --part_id ::: {0..9}

# gpu04
# parallel python generate_gt.py --label 9 --dir /graphics/scratch/datasets/ModelNet40/advanced/10k --num_parts 20 --samples 10000 --part_id ::: {0..19}

# gpu02
# parallel python generate_gt.py --label 8 --dir /graphics/scratch/datasets/ModelNet40/advanced/10k --num_parts 20 --samples 10000 --part_id ::: {0..19}

# # pulsatilla [0 -> train]

# # gloriosa [2 -> train]


'''
conda activate gflex
unalias python
cd ~/git/cgtuebingen/Flex-Convolution-Dev/data_generation
'''

import numpy as np
import trimesh
import time
import json
import csv
from open3d import *

from tensorpack.dataflow import DataFlow, LMDBSerializer, PrintData, ConcatData
from tensorpack.utils import fs
import os
import argparse


shape_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
               'bottle', 'bowl', 'car', 'chair', 'cone',
               'cup', 'curtain', 'desk', 'door', 'dresser',
               'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
               'laptop', 'mantel', 'monitor', 'night_stand', 'person',
               'piano', 'plant', 'radio', 'range_hood', 'sink',
               'sofa', 'stairs', 'stool', 'table', 'tent',
               'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']


class ShapeNetCore55ClassDataFlow(DataFlow):
    def __init__(self, dirname, mode, label, start=0, filtered_files=[], objects=None, samples=1000, sphere_samples=200):

        # work_dir = os.path.join(dirname, shape_names[label], mode)
        #assert os.path.isdir(work_dir), work_dir
        assert os.path.isdir(dirname), dirname
        self.work_dir = dirname

        self.start = start
        self.objects = objects
        self.label = label
        self.filelist = sorted([k for k in fs.recursive_walk(
            self.work_dir) if k.endswith('.obj')])[start:]
        # self.filelist = filtered_files[start:]

        if objects is not None:
            self.filelist = self.filelist[:objects]

        self.samples = samples
        self.sphere_samples = sphere_samples

    def __len__(self):
        return len(self.filelist)

    def __iter__(self):
        for f in self.filelist:
            # obj_id = int(f[-8:-4])
            obj_id = 0
            print('generate points for: ', f)
            m = trimesh.load_mesh(f, validate=True, use_embree=True)
            bbnormalize(m)
            P, N, F = sampleGT(m, self.samples, self.sphere_samples)
            V = combinedVertices(m, F)
            yield [P, N, V.base, self.label, obj_id]


def sampleGT(mesh, N, ray_samples=1000, assertion=False, max_hits=5, junks=15):

    result_points = np.zeros([0, 3], dtype=np.float32)
    result_normals = np.zeros([0, 3], dtype=np.float32)
    result_faces = np.zeros([0], dtype=np.int32)

    t = time.time()
    while(result_points.shape[0] < N):

        # print('N_current:', result_points.shape[0])

        S, Sidx = trimesh.sample.sample_surface(mesh, junks)

        sphere = trimesh.sample.sample_surface_sphere(ray_samples)
        o = np.repeat(S, sphere.shape[0], 0)
        d = np.repeat(sphere, S.shape[0], 0)

        Itri, Iray = mesh.ray.intersects_id(o, d, max_hits=max_hits)

        kidx = Itri != Sidx[Iray // ray_samples]
        Irayk = Iray[kidx]
        Itrik = Itri[kidx]

        ele = np.arange(junks * ray_samples)
        Iele = np.isin(ele, Irayk, invert=True)
        ray_survivor_candidate = np.unique(ele[Iele])

        remove_rays = []
        for r, i in enumerate(ray_survivor_candidate):
            if len(Itri[Iray == i]) == max_hits:
                remove_rays.append(r)

        ray_survivor = np.delete(ray_survivor_candidate, remove_rays)
        point_survivor = np.unique(ray_survivor // ray_samples)

        point_survived = S[point_survivor]
        faces_survived = Sidx[point_survivor]
        normals_survived = mesh.face_normals[faces_survived]

        result_points = np.concatenate([result_points, point_survived], 0)
        result_faces = np.concatenate([result_faces, faces_survived], 0)
        result_normals = np.concatenate(
            [result_normals, normals_survived], 0)

    elapsed = time.time() - t
    print('{} points in {} seconds.'.format(result_points.shape[0], elapsed))

    # print(result_points.shape)
    # print(result_faces.shape)
    return result_points[:N], result_normals[:N], result_faces[:N]


def combinedVertices(mesh, faces_list):
    class FaceMap:
        def __init__(self, face_adjacency):
            self.dic = dict()
            for f1, f2 in mesh.face_adjacency:
                self.addFacePair(f1, f2)

        def addKV(self, key, value):
            if key in self.dic.keys():
                self.dic[key].append(value)
            else:
                self.dic[key] = [value]

        def addFacePair(self, face1, face2):
            self.addKV(face1, face2)
            self.addKV(face2, face1)

    def convertNBFaces(T):
        for x in T:
            if len(x) < 3:
                for _ in range(3 - len(x)):
                    x.append(x[0])
        return np.asarray(T)

    fm = FaceMap(mesh.face_adjacency)

    faces_list_all = faces_list.copy()
    faces_list = faces_list.tolist()

    # find faces w/o NB
    # this should just happen on some special model, thus we use Try-Catch
    cond = True
    removed_keys = []
    while cond:
        try:
            T = [fm.dic[x] for x in faces_list]
        except KeyError as err:
            i = err.args[0]
            faces_list.remove(i)
            removed_keys.append(i)
        else:
            cond = False

    if len(removed_keys) > 0:
        print(removed_keys)
    NBfaces = convertNBFaces(T)
    faces = np.concatenate([np.expand_dims(faces_list, -1), NBfaces], -1)

    tri = np.sort(mesh.faces[faces][:, 0, :], -1)[:, np.newaxis, :, np.newaxis]
    nbs = np.sort(mesh.faces[faces][:, 1:], -1)[:, :, np.newaxis]

    tri_nbs = tri == nbs

    idx_list = np.tile(
        np.arange(3) + 1, np.prod(tri_nbs.shape[:-1])).reshape(tri_nbs.shape)

    nbidx = np.sum(np.where(tri_nbs is True, idx_list,
                            np.zeros_like(tri_nbs)), axis=-1) - 1

    nbtidx = np.sort(nbidx)[:, :, 1:]

    i = np.arange(nbs.shape[0])[:, np.newaxis, np.newaxis]
    j = np.arange(nbs.shape[1])[np.newaxis, :, np.newaxis]

    edge_vert = nbs[i, j, 0, nbtidx]

    vert_idx = (np.sum(tri_nbs, axis=-2, keepdims=True) == 0)
    nb_vert = nbs[vert_idx].reshape(nbs.shape[:2])

    comb_vert = mesh.vertices[np.concatenate([tri[:, 0, :, 0], nb_vert], 1)]

    if len(removed_keys) > 0:
        # fixing faces w/o NB
        urk, urkc = np.unique(removed_keys, return_counts=True)

        indcies = []
        for ik, k in enumerate(urk):
            ind = np.where(faces_list_all == k)[0]
        #     print(k,ind)
            for c in range(urkc[ik]):
                indcies.append(ind[c])

        comb_vert = np.insert(comb_vert, np.sort(indcies) - np.arange(len(indcies)),
                              np.tile(mesh.vertices[mesh.faces[faces_list_all[np.sort(indcies)]]], [1, 2, 1]), axis=0)

    return comb_vert


def bbnormalize(mesh):
    bbmin = mesh.vertices.min(0)
    bbmax = mesh.vertices.max(0)

    bbscale = 2. / np.linalg.norm(bbmax - bbmin)
    bbmid = bbmin + (bbmax - bbmin) / 2.

    mesh.apply_translation(-bbmid)
    mesh.apply_scale(bbscale)

# parallel python generate_gt.py --label 1 --num_parts 10 --samples 100 --part_id ::: {0..9}


def get_synsetid(filter_class):
    synsetId = 0
    filter_class = filter_class.encode('ascii', 'replace')
    with open('../taxonomy.json') as json_file:
        data = json.load(json_file)

        # Find synsetId for given model class name
        for d in data:
            class_name = d['name'].encode('ascii', 'replace')
            if len(d['children']) == 0:
                continue
            if filter_class in class_name:
                synsetId = d['synsetId']
                break
    return synsetId


def filter_files(filter_class, mode, dirname):
    synsetId = get_synsetid(filter_class)
    # Setup model hashmap
    # Load train/val.csv
    file_name = mode + ".csv"
    with open('../'+file_name, 'rb') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        # Filter train/val.csv for models with correct synthId
        class_object_lst = []
        for row in data:
            if row[1] == synsetId:
                class_object_lst.append(os.path.join(dirname, row[0]+".obj"))
    return class_object_lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--merge', help='merging together all the parts to the final lmdb', action='store_true')
    parser.add_argument(
        '--label', help='label_id of ShapeNetCorev2 [0,54] (default: 0)', type=int, default=0)
    parser.add_argument(
        '--samples', help='number of points to generate', type=int)
    parser.add_argument('--sphere_samples',
                        help='number of sphere_samples to test per points (default: 200)', default=200, type=int)
    parser.add_argument('--dir', help='output directory')
    # parser.add_argument('--start', help='start object for ModelNet40 class (default:0)', default=0, type=int)
    # parser.add_argument('--objects', help='number of objects to process. If not set all objects are processed.')

    parser.add_argument(
        '--mode', help='process train or test data [\'train\',\'test\'] (default:\'train\')', default='train')
    parser.add_argument(
        '--part_id', help='id of part to process', default=0, type=int)
    parser.add_argument(
        '--num_parts', help='number of part to process (default: 10)', default=10, type=int)

    args = parser.parse_args()
    class_syntid = get_synsetid(shape_names[args.label])
    snc55dir = '/graphics/scratch/datasets/ShapeNetCorev2/ShapeNetCore.v2/'
    snc55dir = os.path.join(snc55dir, str(class_syntid))

    def getFN_parts(part_id):
        return os.path.join(args.dir, '{}_{}_N{}_S{}__{}_of_{}.parts'.format(
            args.mode, shape_names[args.label], args.samples, args.sphere_samples, part_id, args.num_parts))

    def getFN_final():
        return os.path.join(args.dir, '{}_{}_N{}_S{}.lmdb'.format(
            args.mode, shape_names[args.label], args.samples, args.sphere_samples))

    if args.merge:
        print('merging...')
        df_list = []
        for pi in range(args.num_parts):
            df = LMDBSerializer.load(getFN_parts(pi), shuffle=False)
            df_list.append(df)
        ds = ConcatData(df_list)
        LMDBSerializer.save(ds, getFN_final())
    else:
        print('generating data..')
        # work_dir = os.path.join(snc55dir, str(class_syntid))
        # filtered_files = filter_files(
        #    shape_names[args.label], args.mode, snc55dir)
        num_models = len(os.listdir(snc55dir))
        #num_models = 100
        num_objects = (num_models - 1) // args.num_parts + 1

        start_object = args.part_id * num_objects
        rest_objects = num_models - start_object

        if rest_objects < num_objects:
            num_objects = rest_objects
        ds = ShapeNetCore55ClassDataFlow(snc55dir, label=args.label, start=start_object,
                                         objects=num_objects, mode=args.mode, samples=args.samples, sphere_samples=args.sphere_samples)

        out_file = getFN_parts(args.part_id)
        out_file = getFN_final()
        out_file = "/graphics/scratch/datasets/ShapeNetCorev2/data/10k/train_airplane_N10000_S200.lmdb"
        LMDBSerializer.save(ds, out_file)
