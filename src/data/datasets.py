import os
import pathlib

import joblib
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from functools import partial
from .. import paths
from ..log import logger
from ..utils import load_json, save_json
from .datasources import DataSource, available_datasources

__all__ = [
    'Dataset',
    'cached_datasets',
    'available_datasets',
]

def cached_datasets(dataset_path=None, keys_only=True, check_hashes=False):
    """Get the set of datasets currently cached to disk.

    Parameters
    ----------
    dataset_path: path
        location of saved dataset files
    keys_only: Boolean
        if True, return a set of dataset names
        if False, return dictionary mapping dataset names to their stored metadata
    check_hashes: Boolean
        if True, hashes will  be checked against `dataset_cache`.
        If they differ, an exception will be raised

    """
    if dataset_path is None:
        dataset_path = paths['processed_data_path']
    else:
        dataset_path = pathlib.Path(dataset_path)

    ds_dict = {}
    for dsfile in dataset_path.glob("*.metadata"):
        ds_stem = str(dsfile.stem)
        ds_meta = Dataset.load(ds_stem, data_path=dataset_path, metadata_only=True, check_hashes=check_hashes)
        ds_dict[ds_stem] = ds_meta

    if keys_only:
        return set(ds_dict.keys())
    return ds_dict

def available_datasets(catalog_path=None, catalog_file='datasets.json', keys_only=True):
    """Get the set of available datasets from the catalog.

    Parameters
    ----------
    catalog_path: path or None
    catalog_file: str
        File containing dataset metadata. By default, 'datasets.json'
    keys_only: Boolean
        if True, return a set of dataset names
        if False, return dictionary mapping dataset names to their stored metadata
    """
    if catalog_path is None:
        catalog_path = paths['catalog_path']
    else:
        catalog_path = pathlib.Path(catalog_path)

    catalog_file_fq = catalog_path / catalog_file

    if catalog_file_fq.exists():
        ds_dict = load_json(catalog_file_fq)
    else:
        logger.warning(f"Dataset catalog '{catalog_file}' does not exist. Writing new dataset catalog")
        ds_dict = {}
        save_json(catalog_file_fq, ds_dict)

    if keys_only:
        return set(ds_dict.keys())
    return ds_dict, catalog_file_fq



class Dataset(Bunch):
    def __init__(self,
                 dataset_name=None,
                 data=None,
                 target=None,
                 metadata=None,
                 update_hashes=True,
                 catalog_path=None,
                 catalog_file='datasets.json',
                 **kwargs):
        """
        Object representing a dataset object.
        Notionally compatible with scikit-learn's Bunch object

        dataset_name: string (required)
            key to use for this dataset
        data:
            Data: (usually np.array or np.ndarray)
        target: np.array
            Either classification target or label to be used. for each of the points
            in `data`
        metadata: dict
            Data about the object. Key fields include `license_txt`, `descr`, and `hashes`
        update_hashes: Boolean
            If True, recompute the data/target hashes in the Metadata
        """
        super().__init__(**kwargs)

        if dataset_name is None:
            if metadata is not None and metadata.get("dataset_name", None) is not None:
                dataset_name = metadata['dataset_name']
            else:
                raise Exception('dataset_name is required')

        if metadata is not None:
            self['metadata'] = metadata
        else:
            self['metadata'] = {}
        self['metadata']['dataset_name'] = dataset_name
        self['data'] = data
        self['target'] = target
        data_hashes = self.get_data_hashes()

        if update_hashes:
            self['metadata'] = {**self['metadata'], **data_hashes}

    def update_catalog(self, catalog_path=None, catalog_file='datasets.json'):
        """Update the dataset catalog

        Parameters
        ----------
        catalog_path: path or None
            Location of catalog file. default paths['catalog_path']
        catalog_file: str
            dataset catalog file. relative to `catalog_path`. Default 'datasets.json'
        """
        dataset_name = self["metadata"]["dataset_name"]
        dataset_catalog, catalog_file_fq = available_datasets(catalog_path=catalog_path, catalog_file=catalog_file, keys_only=False)
        dataset_catalog[dataset_name] = self['metadata']
        logger.debug(f"Updating dataset catalog with '{dataset_name}' metadata")
        save_json(catalog_file_fq, dataset_catalog)


    def __getattribute__(self, key):
        if key.isupper():
            try:
                return self['metadata'][key.lower()]
            except:
                raise AttributeError(key)
        else:
            return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key.isupper():
            self['metadata'][key.lower()] = value
        elif key == 'name':
            self['metadata']['dataset_name'] = value
        else:
            super().__setattr__(key, value)

    def __str__(self):
        s = f"<Dataset: {self.name}"
        if self.get('data', None) is not None:
            shape = getattr(self.data, 'shape', 'Unknown')
            s += f", data.shape={shape}"
        if self.get('target', None) is not None:
            shape = getattr(self.target, 'shape', 'Unknown')
            s += f", target.shape={shape}"
        meta = self.get('metadata', {})
        if meta:
            s += f", metadata={list(meta.keys())}"

        s += ">"
        return s

    @property
    def name(self):
        return self['metadata'].get('dataset_name', None)

    @name.setter
    def name(self, val):
        self['metadata']['dataset_name'] = val

    @property
    def has_target(self):
        return self['target'] is not None

    @classmethod
    def load(cls, dataset_name, data_path=None, metadata_only=False, errors=True,
             catalog_path=None, dataset_cache='datasets.json', check_hashes=True):
        """Load a dataset by name (or its metadata)

        errors: Boolean
            if True, raise exception if dataset is not available on disk
            if False, returns None if not found
        metadata_only: Boolean
            if True, return only metadata. Otherwise, return the entire dataset
        dataset_name: str
            name of dataset_dir
        data_path: str
            path containing `dataset_name`
        catalog_path: str or None:
            path to data catalog. default paths['catalog_path']
        dataset_cache: str. default 'datasets.json'
            name of dataset cache file. Relative to `catalog_path`.
        check_hashes: Boolean
            if True, hashes will  be checked against `dataset_cache`.
            If they differ, an exception will be raised
        """
        if catalog_path is None:
            catalog_path = paths['catalog_path']
        else:
            catalog_path = pathlib.Path(catalog_path)

        if data_path is None:
            data_path = paths['processed_data_path']
        else:
            data_path = pathlib.Path(data_path)

        metadata_fq = data_path / f'{dataset_name}.metadata'
        dataset_fq = data_path / f'{dataset_name}.dataset'
        dataset_cache_fq = catalog_path / dataset_cache

        if not metadata_fq.exists() and not dataset_fq.exists():
            if errors:
                raise FileNotFoundError(f"No dataset {dataset_name} in {data_path}.")
            else:
                return None

        if check_hashes:
            if not dataset_cache_fq.exists():
                raise FileNotFoundError(f"No '{dataset_cache}' in catalog, but `check_hashes` is True")
            else:
                dataset_cache = load_json(dataset_cache_fq)

        with open(metadata_fq, 'rb') as fd:
            meta = joblib.load(fd)

        if metadata_only:
            return meta

        with open(dataset_fq, 'rb') as fd:
            ds = joblib.load(fd)
        return ds

    @classmethod
    def from_datasource(cls, dataset_name,
                        cache_path=None,
                        fetch_path=None,
                        force=False,
                        unpack_path=None,
                        **kwargs):
        '''Creates Dataset object from a named DataSource.

        Dataset will be cached after creation. Subsequent calls with matching call
        signature will return this cached object.

        Parameters
        ----------
        dataset_name:
            Name of dataset to load. see `available_datasources()` for the current list
        cache_path: path
            Directory to search for Dataset cache files
        fetch_path: path
            Directory to download raw files into
        force: boolean
            If True, always regenerate the dataset. If false, a cached result can be returned
        unpack_path: path
            Directory to unpack raw files into
        **kwargs:
            Remaining keywords arguments are passed directly to DataSource.process().
            See that docstring for details.

        Remaining keywords arguments are passed to the DataSource's `process()` method
        '''
        dataset_list, _ = available_datasources(keys_only=False)
        if dataset_name not in dataset_list:
            raise Exception(f'Unknown Dataset: {dataset_name}')
        dsrc = DataSource.from_dict(dataset_list[dataset_name])
        dsrc.fetch(fetch_path=fetch_path, force=force)
        dsrc.unpack(unpack_path=unpack_path, force=force)
        ds = dsrc.process(cache_path=cache_path, force=force, **kwargs)

        return ds

    def get_data_hashes(self, exclude_list=None, hash_type='sha1'):
        """Compute a the hash of data items

        exclude_list: list or None
            List of attributes to skip.
            if None, skips ['metadata']

        hash_type: {'sha1', 'md5', 'sha256'}
            Algorithm to use for hashing. Must be valid joblib hash type
        """
        if exclude_list is None:
            exclude_list = ['metadata']

        ret = {}
        #ret = {'hash_type': hash_type}
        hashes = {}
        for key, value in self.items():
            if key in exclude_list:
                continue
            data_hash = joblib.hash(value, hash_name=hash_type)
            #ret[f"{key}_hash"] = data_hash
            hashes[key] = (hash_type, data_hash)
        ret["hashes"] = hashes
        return ret

    def dump(self, file_base=None, dump_path=None, hash_type='sha1',
             force=False, create_dirs=True, dump_metadata=True, update_catalog=True,
             catalog_path=None, catalog_file='datasets.json'):
        """Dump a dataset.

        Note, this dumps a separate copy of the metadata structure,
        so that metadata can be looked up without loading the entire dataset,
        which could be large

        dump_metadata: boolean
            If True, also dump a standalone copy of the metadata.
            Useful for checking metadata without reading
            in the (potentially large) dataset itself
        file_base: string
            Filename stem. By default, just the dataset name
        hash_type: {'sha1', 'md5'}
            Hash function to use for hashing data/labels
        dump_path: path. (default: `paths['processed_data_path']`)
            Directory where data will be dumped.
        force: boolean
            If False, raise an exception if the file already exists
            If True, overwrite any existing files
        create_dirs: boolean
            If True, `dump_path` will be created (if necessary)
        update_catalog: Boolean
            if True, new metadata will be written to catalog
        catalog_path: path or None
            Location of catalog file. default paths['catalog_path']
        catalog_file: str
            dataset catalog file. relative to `catalog_path`. Default 'datasets.json'

        """
        if dump_path is None:
            dump_path = paths['processed_data_path']
        dump_path = pathlib.Path(dump_path)

        if file_base is None:
            file_base = self.name

        metadata = self['metadata']

        metadata_filename = file_base + '.metadata'
        dataset_filename = file_base + '.dataset'
        metadata_fq = dump_path / metadata_filename

        data_hashes = self.get_data_hashes(hash_type=hash_type)
        self['metadata'] = {**self['metadata'], **data_hashes}

        # check for a cached version
        if metadata_fq.exists() and force is not True:
            logger.warning(f"Existing metatdata file found: {metadata_fq}")
            cached_metadata = joblib.load(metadata_fq)
            # are we a subset of the cached metadata? (Py3+ only)
            if metadata.items() <= cached_metadata.items():
                raise Exception(f'Dataset with matching metadata exists already. '
                                'Use `force=True` to overwrite, or change one of '
                                '`dataset.metadata` or `file_base`')
            else:
                raise Exception(f'Metadata file {metadata_filename} exists '
                                'but metadata has changed. '
                                'Use `force=True` to overwrite, or change '
                                '`file_base`')

        if create_dirs:
            os.makedirs(metadata_fq.parent, exist_ok=True)

        if dump_metadata:
            with open(metadata_fq, 'wb') as fo:
                joblib.dump(metadata, fo)
            logger.debug(f'Wrote Dataset Metadata: {metadata_filename}')

        if update_catalog:
            self.update_catalog(catalog_path=catalog_path, catalog_file=catalog_file)

        dataset_fq = dump_path / dataset_filename
        with open(dataset_fq, 'wb') as fo:
            joblib.dump(self, fo)
        logger.debug(f'Wrote Dataset: {dataset_filename}')
