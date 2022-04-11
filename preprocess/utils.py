import os
import h5py
from enum import Enum
from multiprocessing import Pool
from tqdm import tqdm
import scipy.io as sio
import pandas as pd
import numpy as np
from types import SimpleNamespace

from tools.utils import io
from tools.visualizations import Visualizer

class DatasetName(Enum):
    SHAPE2MOTION = 1
    MULTISCAN = 2

class Mat2Hdf5Impl:
    def __init__(self, cfg):
        self.set = cfg.set
        self.tmp_dir = cfg.tmp_dir
        self.ext = os.path.splitext(cfg.output_path)[-1]

    def __call__(self, idx, filepath):
        output_filepath = os.path.join(self.tmp_dir, self.set + f'_{idx}' + self.ext)
        h5file = h5py.File(output_filepath, 'w')
        if self.set == 'train':
            object_cat = f'any_{idx}'
        else:
            filename = os.path.splitext(os.path.basename(filepath))[0]
            object_cat = os.path.basename(filename).split('_')[2]
            object_id_base = os.path.basename(filename).split('_')[-1]
        
        data = sio.loadmat(filepath)['Training_data']
        data_info_list = []

        tqdm_text = "#" + "{}".format(idx).zfill(3)
        with tqdm(total=len(data), desc=tqdm_text) as pbar:
            for i in range(len(data)):
                instance_data = data[i][0]

                input_pts = instance_data['inputs_all'][0,0]

                anchor_pts = instance_data['core_position'][0,0].reshape(-1)
                joint_direction_cat = instance_data['motion_direction_class'][0,0].reshape(-1)
                joint_direction_reg = instance_data['motion_direction_delta'][0,0]
                joint_origin_reg = instance_data['motion_position_param'][0,0]
                joint_type = instance_data['motion_dof_type'][0,0].reshape(-1)
                joint_all_directions = instance_data['all_direction_kmeans'][0,0]
                gt_joints = instance_data['dof_matrix'][0,0]
                gt_proposals = instance_data['proposal'][0,0]
                simmat = instance_data['similar_matrix'][0,0]

                if self.set == 'train':
                    object_id = str(i)
                else:
                    object_id = object_id_base + str(i)
                articulation_id = '0'

                row = pd.DataFrame([[object_cat, object_id, articulation_id]],
                            columns=['objectCat', 'objectId', 'articulationId'])
                data_info_list.append(row)

                instance_name = f'{object_cat}_{object_id}_{articulation_id}'
                h5instance = h5file.require_group(instance_name)
                h5instance.create_dataset('input_pts', shape=input_pts.shape, data=input_pts, compression='gzip')
                h5instance.create_dataset('anchor_pts', shape=anchor_pts.shape, data=anchor_pts, compression='gzip')
                h5instance.create_dataset('joint_direction_cat', shape=joint_direction_cat.shape, data=joint_direction_cat, compression='gzip')
                h5instance.create_dataset('joint_direction_reg', shape=joint_direction_reg.shape, data=joint_direction_reg, compression='gzip')
                h5instance.create_dataset('joint_origin_reg', shape=joint_origin_reg.shape, data=joint_origin_reg, compression='gzip')
                h5instance.create_dataset('joint_type', shape=joint_type.shape, data=joint_type, compression='gzip')
                h5instance.create_dataset('joint_all_directions', shape=joint_all_directions.shape, data=joint_all_directions, compression='gzip')
                h5instance.create_dataset('gt_joints', shape=gt_joints.shape, data=gt_joints, compression='gzip')
                h5instance.create_dataset('gt_proposals', shape=gt_proposals.shape, data=gt_proposals, compression='gzip')
                h5instance.create_dataset('simmat', shape=simmat.shape, data=simmat, compression='gzip')
                pbar.update(1)
            
        h5file.close()
        data_info = pd.concat(data_info_list, ignore_index=True)
        data_info.to_csv(os.path.splitext(output_filepath)[0] + '.csv')
        return output_filepath

class Mat2Hdf5:
    def __init__(self, cfg):
        cfg = SimpleNamespace(**cfg)
        self.cfg = cfg
        self.data_path = cfg.path
        self.output_path = cfg.output_path
        self.num_processes = cfg.num_processes
        self.debug = cfg.debug
        self.log = cfg.log

    def convert(self, input_file_indices=[], num_output_instances=-1):
        files = io.get_file_list(self.data_path, join_path=True)
        if len(input_file_indices) > 0:
            files = np.asarray(files)[input_file_indices]
        
        pool = Pool(processes=self.num_processes)
        proc_impl = Mat2Hdf5Impl(self.cfg)
        jobs = [pool.apply_async(proc_impl, args=(i,file,)) for i, file in enumerate(files)]
        pool.close()
        pool.join()
        output_filepath_list = [job.get() for job in jobs]

        # output_filepath_list = []
        # for i, file in enumerate(files):
        #     output_filepath_list.append(proc_impl(i, file))

        h5file = h5py.File(self.output_path, 'w')
        data_info_list = []
        count_instances = 0
        stop = False
        for filepath in output_filepath_list:
            if filepath is None:
                continue
            with h5py.File(filepath, 'r') as h5f:
                for key in h5f.keys():
                    if count_instances == num_output_instances:
                        stop = True
                        break
                    h5f.copy(key, h5file)
                    count_instances += 1
            data_info_list.append(pd.read_csv(os.path.splitext(filepath)[0] + '.csv'))
            if stop:
                break

        if self.debug:
            self.visualize()

        h5file.close()

        self.log.info(f'{count_instances} object instances saved in {self.output_path}')

        data_info = pd.concat(data_info_list, ignore_index=True)
        input_info = pd.DataFrame({'file': files})
        return input_info, data_info

    def visualize(self):
        h5file = h5py.File(self.output_path, 'r')

        for key in h5file.keys():
            instance_data = h5file[key]

            viz = Visualizer()
            viz.view_stage1_input(instance_data)

