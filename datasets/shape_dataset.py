import os
import numpy as np
from itertools import product, combinations, permutations
from glob import glob

import torch
from torch.utils.data import Dataset

from utils.shape_util import read_shape
from utils.registry import DATASET_REGISTRY
from utils.shape_dataset_util import sort_list, get_shape_operators_and_data
from tqdm import tqdm
import igl
import pickle
import trimesh



class SingleShapeDataset(Dataset):
    def __init__(self,
                 data_root, return_faces=True,
                 return_corr=True, **config):
        """
        Single Shape Dataset

        Args:
            data_root (str): Data root.
            return_evecs (bool, optional): Indicate whether return eigenfunctions and eigenvalues. Default True.
            return_faces (bool, optional): Indicate whether return faces. Default True.
            num_evecs (int, optional): Number of eigenfunctions and eigenvalues to return. Default 120.
            return_corr (bool, optional): Indicate whether return the correspondences to reference shape. Default True.
            return_dist (bool, optional): Indicate whether return the geodesic distance of the shape. Default False.
        """
        self.config = config
        # sanity check
        assert os.path.isdir(data_root), f'Invalid data root: {data_root}.'

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_corr = return_corr

        self.off_files = []
        self.corr_files = [] if self.return_corr else None

        self._init_data()

        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0

        if self.return_corr:
            assert self._size == len(self.corr_files)
        
        # treat phase attribute
        if not hasattr(self, 'phase'):
            self.phase = 'no_phase'
            # warn message print
            print("WARNING: no phase specified for dataset, using no_phase")


    def _init_data(self):
        # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))


    def __getitem__(self, index):
        item = dict()

        # get shape name
        off_file = self.off_files[index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        item['name'] = basename

        # get vertices and faces
        verts, faces = read_shape(off_file)
        item['verts'] = torch.from_numpy(verts).float()
        if self.return_faces:
            item['faces'] = torch.from_numpy(faces).long()

        # get shape operators and data
        item = get_shape_operators_and_data(item, cache_dir=os.path.join(self.data_root), config={**self.config, "return_dist": False})

        # get correspondences
        if self.return_corr:
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['corr'] = torch.from_numpy(corr).long()

        return item

    def __len__(self):
        return self._size


@DATASET_REGISTRY.register()
class SingleFaustDataset(SingleShapeDataset):
    def __init__(self, data_root,
                 phase, return_faces=True,
                 return_corr=True, **config):
        self.config = config
        self.phase = phase
        super(SingleFaustDataset, self).__init__(data_root, return_faces,
                                                 return_corr, **config)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'

    def _init_data(self):
         # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))
        
        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0 

        assert len(self) == 100, f'FAUST dataset should contain 100 human body shapes, but get {len(self)}.'
        if self.phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:80]
            if self.corr_files:
                self.corr_files = self.corr_files[:80]
            self._size = 80
        elif self.phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[80:]
            if self.corr_files:
                self.corr_files = self.corr_files[80:]
            self._size = 20


@DATASET_REGISTRY.register()
class SingleScapeDataset(SingleShapeDataset):
    def __init__(self, data_root,
                 phase, return_faces=True,
                 return_corr=True, **config):
        self.config = config
        self.phase = phase
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        super(SingleScapeDataset, self).__init__(data_root, return_faces,
                                                 return_corr, **config)
        
    def _init_data(self):
        # check the data path contains .off files
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} not containing .off files'
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        # check the data path contains .vts files
        if self.return_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))
        
        # sanity check
        self._size = len(self.off_files)
        assert self._size != 0

        assert len(self) == 71, f'FAUST dataset should contain 71 human body shapes, but get {len(self)}.'
        if self.phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:51]
            if self.corr_files:
                self.corr_files = self.corr_files[:51]
            self._size = 51
        elif self.phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[51:]
            if self.corr_files:
                self.corr_files = self.corr_files[51:]
            self._size = 20


@DATASET_REGISTRY.register()
class SingleShrec19Dataset(SingleShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 **config):
        super(SingleShrec19Dataset, self).__init__(data_root, return_faces, False, **config)


@DATASET_REGISTRY.register()
class SingleSmalDataset(SingleShapeDataset):
    def __init__(self, data_root, phase='train', category=True,
                 return_faces=True,
                 return_corr=True, **config):
        self.config = config
        self.phase = phase
        self.category = category
        super(SingleSmalDataset, self).__init__(data_root, return_faces,
                                                return_corr, **config)

    def _init_data(self):
        if self.category:
            txt_file = os.path.join(self.data_root, f'{self.phase}_cat.txt')
        else:
            txt_file = os.path.join(self.data_root, f'{self.phase}.txt')
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.off_files += [os.path.join(self.data_root, 'off', f'{line}.off')]
                if self.return_corr:
                    self.corr_files += [os.path.join(self.data_root, 'corres', f'{line}.vts')]
                

@DATASET_REGISTRY.register()
class SingleDT4DDataset(SingleShapeDataset):
    def __init__(self, data_root, phase='train',
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, **config):
        self.phase = phase
        self.ignored_categories = ['pumpkinhulk']
        super(SingleDT4DDataset, self).__init__(data_root, return_faces,
                                                return_corr, **config)

    def _init_data(self):
        with open(os.path.join(self.data_root, f'{self.phase}.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.split('/')[0] not in self.ignored_categories:
                    self.off_files += [os.path.join(self.data_root, 'off', f'{line}.off')]
                    if self.return_corr:
                        self.corr_files += [os.path.join(self.data_root, 'corres', f'{line}.vts')]
                    # if self.return_dist:
                    #     self.dist_files += [os.path.join(self.data_root, 'dist', f'{line}.mat')]


@DATASET_REGISTRY.register()
class SingleShrec20Dataset(SingleShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=200):
        super(SingleShrec20Dataset, self).__init__(data_root, return_faces,
                                                   return_evecs, num_evecs, False, False)


@DATASET_REGISTRY.register()
class SingleTopKidsDataset(SingleShapeDataset):
    def __init__(self, data_root, phase='train',
                 return_faces=True,
                 return_evecs=True, num_evecs=200, **config):
        self.phase = phase
        super(SingleTopKidsDataset, self).__init__(data_root, return_faces,
                                                   False, **config)


class PairShapeDataset(Dataset):
    def __init__(self, dataset):
        """
        Pair Shape Dataset

        Args:
            dataset (SingleShapeDataset): single shape dataset
        """
        assert isinstance(dataset, SingleShapeDataset), f'Invalid input data type of dataset: {type(dataset)}'
        self.dataset = dataset
        # if dataset.include_same_pair exists, use it, otherwise, set it to True
        include_same_pair = getattr(dataset, 'include_same_pair', True)
        if include_same_pair:
            self.combinations = list(product(range(len(dataset)), repeat=2))
        else:
            self.combinations = list(permutations(range(len(dataset)), 2))


    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        return item

    def __len__(self):
        return len(self.combinations)


@DATASET_REGISTRY.register()
class PairDataset(PairShapeDataset):
    def __init__(self, data_root, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, **config):
        dataset = SingleShapeDataset(data_root, return_faces, return_evecs, num_evecs,
                                     return_corr, **config)
        super(PairDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class PairFaustDataset(PairShapeDataset):
    def __init__(self, data_root,
                 phase, include_same_pair=True, return_faces=True,
                 return_corr=True, **config):
        self.config = config
        self.data_root = data_root
        dataset = SingleFaustDataset(data_root, phase, return_faces,
                                     return_corr, **config)
        dataset.include_same_pair = include_same_pair
        super(PairFaustDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class PairScapeDataset(PairShapeDataset):
    def __init__(self, data_root,
                 phase, include_same_pair=True, return_faces=True,
                 return_corr=True, **config):
        self.config = config
        self.data_root = data_root
        dataset = SingleScapeDataset(data_root, phase, return_faces,
                                     return_corr, **config)
        dataset.include_same_pair = include_same_pair
        super(PairScapeDataset, self).__init__(dataset)


@DATASET_REGISTRY.register()
class PairShrec19Dataset(Dataset):
    def __init__(self, data_root, phase='test',
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=False, **config):
        assert phase in ['train', 'test'], f'Invalid phase: {phase}'
        self.dataset = SingleShrec19Dataset(data_root, return_faces, return_evecs, num_evecs, **config)
        self.phase = phase
        if phase == 'test':
            corr_path = os.path.join(data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            # ignore the shape 40, since it is a partial shape
            self.corr_files = list(filter(lambda x: '40' not in x, sort_list(glob(f'{corr_path}/*.map'))))
            self._size = len(self.corr_files)
        else:
            self.combinations = list(product(range(len(self.dataset)), repeat=2))
            self._size = len(self.combinations)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if self.phase == 'train':
            # get index
            first_index, second_index = self.combinations[index]
        else:
            # extract pair index
            basename = os.path.basename(self.corr_files[index])
            indices = os.path.splitext(basename)[0].split('_')
            first_index = int(indices[0]) - 1
            second_index = int(indices[1]) - 1

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        if self.phase == 'test':
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['first']['corr'] = torch.arange(0, len(corr)).long()
            item['second']['corr'] = torch.from_numpy(corr).long()
        return item


@DATASET_REGISTRY.register()
class PairSmalDataset(PairShapeDataset):
    def __init__(self, data_root, phase='train', include_same_pair=True,
                 category=True, return_faces=True,
                 return_corr=True, **config):
        self.config = config
        dataset = SingleSmalDataset(data_root, phase, category, return_faces,
                                    return_corr, **config)
        dataset.include_same_pair = include_same_pair
        super(PairSmalDataset, self).__init__(dataset=dataset)


@DATASET_REGISTRY.register()
class PairDT4DDataset(PairShapeDataset):
    def __init__(self, data_root, phase='train',  include_same_pair=True,
                 inter_class=False, return_faces=True,
                 return_evecs=True, num_evecs=200,
                 return_corr=True, **config):
        dataset = SingleDT4DDataset(data_root, phase, return_faces,
                                    return_evecs, num_evecs,
                                    return_corr, **config)
        super(PairDT4DDataset, self).__init__(dataset=dataset)
        self.inter_class = inter_class
        self.combinations = []
        if self.inter_class:
            self.inter_cats = set()
            files = os.listdir(os.path.join(self.dataset.data_root, 'corres', 'cross_category_corres'))
            for file in files:
                cat1, cat2 = os.path.splitext(file)[0].split('_')
                self.inter_cats.add((cat1, cat2))
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset)):
                # same category
                cat1, cat2 = self.dataset.off_files[i].split('/')[-2], self.dataset.off_files[j].split('/')[-2]
                if cat1 == cat2:
                    if not self.inter_class:
                        # whether include same pair
                        if include_same_pair:
                            self.combinations.append((i, j))
                        else: # exclude same pair
                            if i != j:
                                self.combinations.append((i, j))
                else:
                    if self.inter_class and (cat1, cat2) in self.inter_cats:
                        self.combinations.append((i, j))

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]
        if self.dataset.return_corr and self.inter_class:
            # read inter-class correspondence
            first_cat = self.dataset.off_files[first_index].split('/')[-2]
            second_cat = self.dataset.off_files[second_index].split('/')[-2]
            corr = np.loadtxt(os.path.join(self.dataset.data_root, 'corres', 'cross_category_corres',
                                           f'{first_cat}_{second_cat}.vts'), dtype=np.int32) - 1
            
            # fix the cache problem because the corr entry is modified on the fly


            item['second']['corr'] = item['second']['corr'][corr]

        return item


@DATASET_REGISTRY.register()
class PairShrec20Dataset(PairShapeDataset):
    def __init__(self, data_root,
                 return_faces=True,
                 return_evecs=True, num_evecs=120):
        dataset = SingleShrec20Dataset(data_root, return_faces, return_evecs, num_evecs)
        super(PairShrec20Dataset, self).__init__(dataset=dataset)


@DATASET_REGISTRY.register()
class PairShrec16Dataset(Dataset):
    """
    Pair SHREC16 Dataset
    """
    categories = [
        'cat', 'centaur', 'david', 'dog', 'horse', 'michael',
        'victoria', 'wolf'
    ]

    def __init__(self,
                 data_root,
                 categories=None,
                 cut_type='cuts', return_faces=True,
                 return_corr=False, **config):
        self.config = config
        assert cut_type in ['cuts', 'holes', 'cuts24'], f'Unrecognized cut type: {cut_type}'

        categories = self.categories if categories is None else categories
        # sanity check
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = sorted(categories)
        self.cut_type = cut_type

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_corr = return_corr

        # full shape files
        self.full_off_files = dict()

        # partial shape files
        self.partial_off_files = dict()
        self.partial_corr_files = dict()

        # load full shape files
        off_path = os.path.join(data_root, 'null', 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} without .off files'
        for cat in self.categories:
            off_file = os.path.join(off_path, f'{cat}.off')
            assert os.path.isfile(off_file)
            self.full_off_files[cat] = off_file

        # load partial shape files
        self._size = 0
        off_path = os.path.join(data_root, cut_type, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} without .off files.'
        for cat in self.categories:
            partial_off_files = sorted(glob(os.path.join(off_path, f'*{cat}*.off')))
            assert len(partial_off_files) != 0
            self.partial_off_files[cat] = partial_off_files
            self._size += len(partial_off_files)

        if self.return_corr:
            # check the data path contains .vts files
            corr_path = os.path.join(data_root, cut_type, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} without .vts files.'
            for cat in self.categories:
                partial_corr_files = sorted(glob(os.path.join(corr_path, f'*{cat}*.vts')))
                assert len(partial_corr_files) == len(self.partial_off_files[cat])
                self.partial_corr_files[cat] = partial_corr_files

    def _get_category(self, index):
        assert index < len(self)
        size = 0
        for cat in self.categories:
            if index < size + len(self.partial_off_files[cat]):
                return cat, index - size
            else:
                size += len(self.partial_off_files[cat])

    def __getitem__(self, index):
        # get category
        cat, index = self._get_category(index)

        # get full shape
        full_data = dict()
        # get vertices
        off_file = self.full_off_files[cat]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        full_data['name'] = basename
        verts_full, faces_full = read_shape(off_file)
        full_data['verts'] = torch.from_numpy(verts_full).float().cpu()
        if self.return_faces:
            full_data['faces'] = torch.from_numpy(faces_full).long().cpu()

        # get shape operators and data
        full_data = get_shape_operators_and_data(full_data, cache_dir=os.path.join(self.data_root), config=self.config)

        # get partial shape
        partial_data = dict()
        # get vertices
        off_file = self.partial_off_files[cat][index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        partial_data['name'] = basename
        verts, faces = read_shape(off_file)
        partial_data['verts'] = torch.from_numpy(verts).float().cpu()
        if self.return_faces:
            partial_data['faces'] = torch.from_numpy(faces).long().cpu()

        # get shape operators and data
        partial_data = get_shape_operators_and_data(partial_data, cache_dir=os.path.join(self.data_root), config={**self.config, "return_dist": False})

        # get correspondences
        if self.return_corr:
            corr = np.loadtxt(self.partial_corr_files[cat][index], dtype=np.int32) - 1
            full_data['corr'] = torch.from_numpy(corr).long()
            partial_data['corr'] = torch.arange(0, len(corr)).long()

            # add partiality mask
            squared_distances, I, C = igl.point_mesh_squared_distance(verts_full, verts_full[corr], faces)
            full_data['partiality_mask'] = torch.from_numpy(squared_distances < 1e-5).float().cpu()
            # partial always has full correspondence
            partial_data['partiality_mask'] = torch.ones(len(verts)).float().cpu()

        return {'first': full_data, 'second': partial_data}

    def __len__(self):
        return self._size


@DATASET_REGISTRY.register()
class PairCP2PDataset(Dataset):
    """
    Pair CP2P Dataset
    """
    categories = [
        'cat', 'centaur', 'david', 'dog', 'horse', 'michael',
        'victoria', 'wolf'
    ]

    def __init__(self,
                 data_root,
                 categories=None,
                 return_faces=True,
                 return_corr=False, **config):
        # Store any additional kwargs as instance attributes
        # self.__dict__.update(config)
        self.config = config

        categories = self.categories if categories is None else categories
        # sanity check
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = sorted(categories)

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_corr = return_corr

        # partial shape files
        self.partial_off_files = dict()
        self.partial_corr_files = dict()

        # load partial shape files
        self._size = 0
        off_path = os.path.join(data_root, 'off')
        assert os.path.isdir(off_path), f'Invalid path {off_path} without .off files.'
        for cat in self.categories:
            partial_off_files = sorted(glob(os.path.join(off_path, f'*{cat}*.off')))
            # assert len(partial_off_files) != 0
            self.partial_off_files[cat] = partial_off_files

        if self.return_corr:
            # check the data path contains .vts files
            corr_path = os.path.join(data_root, 'maps')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} without .map files.'
            for cat in self.categories:
                partial_corr_files = sorted(glob(os.path.join(corr_path, f'*{cat}*.map')))
                self.partial_corr_files[cat] = partial_corr_files
                self._size += len(partial_corr_files)

    def _get_category(self, index):
        assert index < len(self)
        size = 0
        for cat in self.categories:
            if index < size + len(self.partial_corr_files[cat]):
                return cat, index - size
            else:
                size += len(self.partial_corr_files[cat])

    def __getitem__(self, index):
        # get category
        cat, index = self._get_category(index)

        partial_shape_y, partial_shape_x = os.path.basename(self.partial_corr_files[cat][index])[:-4].split('_')
        # eg. cat-01_cat-02.map, contain a map of size 5110, cat-01 has 5110 vertices
        # eg. in SHREC16, cat-xx.vts contain a map of size cat-xx.verts, this is the partial shape, put in position 2

        # get partial shape_x
        partial_data_x = dict()
        # get vertices
        off_file = os.path.join(self.data_root, 'off', f'{partial_shape_x}.off')
        basename = os.path.splitext(os.path.basename(off_file))[0]
        partial_data_x['name'] = basename
        verts, faces = read_shape(off_file)
        partial_data_x['verts'] = torch.from_numpy(verts).float().cpu()
        if self.return_faces:
            partial_data_x['faces'] = torch.from_numpy(faces).long().cpu()
        partial_data_x = get_shape_operators_and_data(partial_data_x, cache_dir=os.path.join(self.data_root), config=self.config)

        partial_data_y = dict()
        # get vertices
        off_file = os.path.join(self.data_root, 'off', f'{partial_shape_y}.off')
        basename = os.path.splitext(os.path.basename(off_file))[0]
        partial_data_y['name'] = basename
        verts, faces = read_shape(off_file)
        partial_data_y['verts'] = torch.from_numpy(verts).float().cpu()
        if self.return_faces:
            partial_data_y['faces'] = torch.from_numpy(faces).long().cpu()
        
        partial_data_y = get_shape_operators_and_data(partial_data_y, cache_dir=os.path.join(self.data_root), config={**self.config, "return_dist": False})
        # get correspondences
        if self.return_corr: # the .map files from cp2p has quite different structures and contains more information eg. partiality mask
            # ------corr--------
            map = np.loadtxt(self.partial_corr_files[cat][index], dtype=np.int32) # no need to minus 1 because this is .map file
            size_y = len(partial_data_y['verts'])
            corr = map[:size_y]
            corr_x = torch.from_numpy(corr).long()
            corr_y = torch.arange(0, len(corr)).long()
            # clean up the -1 entries
            valid = corr != -1
            corr_x = corr_x[valid]
            corr_y = corr_y[valid]
            partial_data_x['corr'] = corr_x
            partial_data_y['corr'] = corr_y

            # --------partiality mask--------
            gt_partiality_mask12 = map[size_y:]
            partial_data_x['partiality_mask'] = torch.from_numpy(gt_partiality_mask12).float()

            # try to get the gt partiality mask21 from the other direction: this will be full covered partiality mask than if use corrs to generate
            partial_corr_file_other_direction = os.path.join(self.data_root, 'maps', f'{partial_shape_x}_{partial_shape_y}.map')
            # if it exists, use it
            if os.path.exists(partial_corr_file_other_direction):
                map = np.loadtxt(partial_corr_file_other_direction, dtype=np.int32)
                size_x = len(partial_data_x['verts'])
                gt_partiality_mask21 = map[size_x:]
                partial_data_y['partiality_mask'] = torch.from_numpy(gt_partiality_mask21).float()
            else: # create the mask from corr_y; this will have some gaps but it's better than nothing
                gt_partiality_mask21 = np.zeros(len(partial_data_y['verts']))
                gt_partiality_mask21[corr_y] = 1
                partial_data_y['partiality_mask'] = torch.from_numpy(gt_partiality_mask21).float()
            # --------------------------
        return {'first': partial_data_x, 'second': partial_data_y}

    def __len__(self):
        return self._size



@DATASET_REGISTRY.register()
class PairTopKidsDataset(Dataset):
    def __init__(self, data_root, phase='train',
                 return_faces=True,
                 return_evecs=True, num_evecs=200,
                 **config):
        assert phase in ['train', 'test'], f'Invalid phase: {phase}'
        self.dataset = SingleTopKidsDataset(data_root, phase, return_faces, return_evecs, num_evecs, **config)
        self.phase = phase
        if phase == 'test':
            corr_path = os.path.join(data_root, 'corres')
            assert os.path.isdir(corr_path), f'Invalid path {corr_path} not containing .vts files'
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))
            self._size = len(self.corr_files)
        else:
            self.combinations = list(product(range(len(self.dataset)), repeat=2))
            self._size = len(self.combinations)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        if self.phase == 'train':
            # get index
            first_index, second_index = self.combinations[index]
        else:
            # extract pair index
            first_index, second_index = 0, index + 1

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        if self.phase == 'test':
            corr = np.loadtxt(self.corr_files[index], dtype=np.int32) - 1  # minus 1 to start from 0
            item['first']['corr'] = torch.from_numpy(corr).long()
            item['second']['corr'] = torch.arange(0, len(corr)).long()

        return item


@DATASET_REGISTRY.register()
class PARTIALSMALDataset(Dataset):
    """Dataset class for loading PARTIALSMAL dataset."""

    categories = ["cougar", "cow", "dog", "fox", "hippo", "horse", "lion", "wolf"]

    def __init__(self, data_root, categories=None, return_faces=True,
                 return_corr=False, **config):
        self.config = config
        categories = self.categories if categories is None else categories
        # sanity check
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.categories = sorted(categories)

        # initialize
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_corr = return_corr

        # partial shape files
        self.partial_off_files = dict()
        self.partial_corr_files = dict()
        self.partial_dist_files = dict()

        # load partial shape files
        self._size = 0
        off_path = os.path.join(data_root, "shapes")
        assert os.path.isdir(off_path), f"Invalid path {off_path} without .off files."
        for cat in self.categories:
            partial_off_files = sorted(glob(os.path.join(off_path, f"*{cat}*.off")))
            # assert len(partial_off_files) != 0
            self.partial_off_files[cat] = partial_off_files

        if self.return_corr:
            # check the data path contains .vts files
            corr_path = os.path.join(data_root, "maps")
            assert os.path.isdir(corr_path), f"Invalid path {corr_path} without .map files."
            partial_corr_files = sorted(glob(os.path.join(corr_path, "*.vts")))
            self.partial_corr_files = partial_corr_files
            self._size = len(partial_corr_files)

    def __getitem__(self, index):
        file = os.path.basename(self.partial_corr_files[index])[:-4] # eg. PARTIALSMAL/test/maps/cuts_0_cougar_04_cuts_14_horse_02.vts
        parts = file.split("_") 
        partial_shape_y = "_".join(parts[:4]) # cuts_0_cougar_04
        partial_shape_x = "_".join(parts[4:8]) # cuts_14_horse_02

        # get partial shape_x
        partial_data_x = dict()
        # get vertices
        off_file = os.path.join(self.data_root, "shapes", f"{partial_shape_x}.off")
        basename = os.path.splitext(os.path.basename(off_file))[0]
        partial_data_x["name"] = basename
        verts, faces = read_shape(off_file)
        partial_data_x["verts"] = torch.from_numpy(verts).float().cpu()
        if self.return_faces:
            partial_data_x["faces"] = torch.from_numpy(faces).long().cpu()

        partial_data_x = get_shape_operators_and_data(partial_data_x, cache_dir=os.path.join(self.data_root), config=self.config)

        # get partial shape_y
        partial_data_y = dict()
        # get vertices
        off_file = os.path.join(self.data_root, "shapes", f"{partial_shape_y}.off")
        basename = os.path.splitext(os.path.basename(off_file))[0]
        partial_data_y["name"] = basename
        verts, faces = read_shape(off_file)
        partial_data_y["verts"] = torch.from_numpy(verts).float().cpu()
        if self.return_faces:
            partial_data_y["faces"] = torch.from_numpy(faces).long().cpu()

        partial_data_y = get_shape_operators_and_data(partial_data_y, cache_dir=os.path.join(self.data_root), config={**self.config, "return_dist": False})

        # get correspondences
        if self.return_corr:  # the .map files from cp2p has quite different structures and contains more information eg. partiality mask
            # ------corr--------
            map = np.loadtxt(self.partial_corr_files[index], dtype=np.int32)  # no need to minus 1 because this is .map file
            corr = map
            corr_x = torch.from_numpy(corr).long()
            corr_y = torch.arange(0, len(corr)).long()
            # clean up the -1 entries
            valid = corr != -1
            corr_x = corr_x[valid]
            corr_y = corr_y[valid]
            partial_data_x["corr"] = corr_x
            partial_data_y["corr"] = corr_y

            # --------partiality mask--------
            gt_partiality_mask21 = valid
            partial_data_y["partiality_mask"] = torch.from_numpy(gt_partiality_mask21).float()

            # get the gt partiality mask12 from the other direction
            partial_corr_file_other_direction = os.path.join(self.data_root, "maps", f"{partial_shape_x}_{partial_shape_y}.vts")
            map_reverse = np.loadtxt(partial_corr_file_other_direction, dtype=np.int32)
            gt_partiality_mask12 = map_reverse != -1
            partial_data_x["partiality_mask"] = torch.from_numpy(gt_partiality_mask12).float()
            # --------------------------
        return {"first": partial_data_x, "second": partial_data_y}

    def __len__(self):
        return self._size

@DATASET_REGISTRY.register()
class BeCoSDataset(Dataset):
    """Pair BeCoS Dataset"""

    def __init__(self, data_root="../data/BeCoS/partial_partial/test", num_instances='all', shape_scale="default", 
                 return_faces=True, return_corr=False, **config):
        self.data_root = data_root
        self.return_faces = return_faces
        self.return_corr = return_corr
        self.config = config
        self.shape_scale = shape_scale
        self.is_bidirectional = "partial_partial" in data_root or "full_full" in data_root

        if num_instances == 'all': # if not specified, use all instances
            instance_folders = sorted(glob(os.path.join(data_root, "*")))
            highest_instance_num = max(int(os.path.basename(f)) for f in instance_folders if os.path.basename(f).isdigit())
            num_instances = highest_instance_num

        self.corr_files = []
        self.pair_off_files = []

        for i in range(num_instances + 1):  # from 0 to N
            instance_folder_path = os.path.join(data_root, str(i))
            
            off_file_0 = glob(os.path.join(instance_folder_path, "0_*.off"))[0]
            off_file_1 = glob(os.path.join(instance_folder_path, "1_*.off"))[0]
            
            self.pair_off_files.append((off_file_0, off_file_1))
            self.corr_files.append(os.path.join(instance_folder_path, "corres_10.npy"))

            if self.is_bidirectional:
                self.pair_off_files.append((off_file_1, off_file_0))
                self.corr_files.append(os.path.join(instance_folder_path, "corres_01.npy"))

        # sanity check
        expected_pairs = (num_instances + 1) * (2 if self.is_bidirectional else 1)
        actual_pairs = len(self.pair_off_files)
        if actual_pairs != expected_pairs:
            raise ValueError(f"Expected {expected_pairs} shape pairs but found {actual_pairs} pairs")

        self._size = len(self.pair_off_files)

    def __getitem__(self, index):
        data_x = dict()
        data_y = dict()

        corr_file = self.corr_files[index]
        off_file_x, off_file_y = self.pair_off_files[index]
        name_shape_x = os.path.splitext(os.path.basename(off_file_x))[0]
        name_shape_y = os.path.splitext(os.path.basename(off_file_y))[0]

        info_file_x = off_file_x.replace(".off", "_info.pkl")
        info_file_y = off_file_y.replace(".off", "_info.pkl")
        info_x = pickle.load(open(info_file_x, "rb"))
        info_y = pickle.load(open(info_file_y, "rb"))

        data_x["random_rotation"] = torch.from_numpy(info_x["random_rotation"]).float()
        data_y["random_rotation"] = torch.from_numpy(info_y["random_rotation"]).float()
        data_x["surface_area_after_scaling"] = info_x["surface_area_after_scaling"]
        data_y["surface_area_after_scaling"] = info_y["surface_area_after_scaling"]

        # load meshes
        mesh_x = trimesh.load(off_file_x, process=False, maintain_order=True)
        mesh_y = trimesh.load(off_file_y, process=False, maintain_order=True)

        if self.shape_scale == "default":
            verts_x = mesh_x.vertices
            verts_y = mesh_y.vertices
        elif self.shape_scale == "normalized":  # normalize by surface area
            verts_x = mesh_x.vertices / data_x["surface_area_after_scaling"]
            verts_y = mesh_y.vertices / data_y["surface_area_after_scaling"]
        else:
            raise ValueError(f"Invalid shape_scale option: {self.shape_scale}. Must be 'default' or 'normalize'")

        faces_x = mesh_x.faces
        faces_y = mesh_y.faces

        data_x["name"] = name_shape_x
        data_y["name"] = name_shape_y
        data_x["verts"] = torch.from_numpy(verts_x).float().cpu()
        data_y["verts"] = torch.from_numpy(verts_y).float().cpu()
        if self.return_faces:
            data_x["faces"] = torch.from_numpy(faces_x).long().cpu()
            data_y["faces"] = torch.from_numpy(faces_y).long().cpu()

        data_x = get_shape_operators_and_data(data_x, cache_dir=os.path.join(self.data_root), config=self.config)
        if self.config.get("return_dist", False) and self.shape_scale == "default":  # dist correction for evaluations when using default scale
            data_x["dist"] = data_x["dist"] / data_x["surface_area_after_scaling"]
        data_y = get_shape_operators_and_data(data_y, cache_dir=os.path.join(self.data_root), config={**self.config, "return_dist": False})

        # get correspondences and partiality mask
        if self.return_corr:
            corr = np.load(corr_file)
            corr_x = torch.from_numpy(corr).long()
            corr_y = torch.arange(0, len(corr)).long()
            # clean up the -1 entries
            valid = corr != -1
            corr_x = corr_x[valid]
            corr_y = corr_y[valid]
            data_x["corr"] = corr_x
            data_y["corr"] = corr_y

            # gt partiality mask
            if "partial_partial" in self.data_root:
                if "found_faces_mask_yx" in info_x:# inspect the keys it's either 'found_faces_mask_yx' or 'found_faces_mask', why?
                    partiality_mask_x = info_x["found_faces_mask_yx"].astype(int) # bool to int
                    partiality_mask_y = info_y["found_faces_mask"].astype(int)
                else:
                    partiality_mask_x = info_x["found_faces_mask"].astype(int)
                    partiality_mask_y = info_y["found_faces_mask_yx"].astype(int)
                partiality_mask_x, partiality_mask_y = partiality_mask_y, partiality_mask_x # somehow the mask live on the other side, we need to swap them, why?
            elif "partial_full" in self.data_root:
                partiality_mask_x = info_x["found_faces_mask"].astype(int)
                partiality_mask_y = np.ones(len(verts_y))
            elif "full_full" in self.data_root:
                partiality_mask_x = np.ones(len(verts_x))
                partiality_mask_y = np.ones(len(verts_y))
            data_x["partiality_mask"] = torch.from_numpy(partiality_mask_x).float()
            data_y["partiality_mask"] = torch.from_numpy(partiality_mask_y).float()

        return {"first": data_x, "second": data_y}

    def __len__(self):
        return self._size