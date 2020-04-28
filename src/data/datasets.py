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
    'get_dataset_catalog',
    'available_datasets',
    'check_dataset_hashes',
    'TransformerGraph',
]

def default_transformer(dsdict, **kwargs):
    """Placeholder for transformerdata processing function"""
    transformer_name = kwargs.get('transformer_name', 'unknown-transformer')
    logger.error(f"'{transformer_name}()' function not found. Define it add it to the `user` namespace for correct behavior")
    return dsdict

def check_dataset_hashes(subset_hashdict, hashdict):
    """Verify that one hash dictionary is a subset of another

    Verifies that all keys and values of `subset_hashdict` are present (and equal) in `hashdict`

    Hashes are a dict of attributes mapped to their hash value
    Hash values are strings f"{hash_type}:{hash_value}"; e.g.
    {
        'data': 'sha1:38f65f3b11da4851aaaccc19b1f0cf4d3806f83b',
        'target': 'sha1:38f65f3b11da4851aaaccc19b1f0cf4d3806f83b'
    }

    """
    return subset_hashdict.items() <= hashdict.items()

def available_datasets(dataset_path=None, keys_only=True, check_hashes=False):
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

def get_dataset_catalog(catalog_path=None, catalog_file='datasets.json', include_filename=False):
    """Get the set of available datasets from the catalog (nodes in the transformer graph).

    Parameters
    ----------
    include_filename: boolean
        if True, returns a tuple: (list, filename)
    catalog_path: path. (default: paths['catalog_dir'])
        Location of `catalog_file`
    catalog_file: str, default 'datasets.json'
        Name of json file that contains the dataset metadata

    Returns
    -------
    If include_filename is True:
        A tuple: (catalog_dict, catalog_file_fq)
    else:
        catalog_dict
    """
    if catalog_path is None:
        catalog_path = paths['catalog_path']
    else:
        catalog_path = pathlib.Path(catalog_path)

    catalog_file_fq = catalog_path / catalog_file

    if catalog_file_fq.exists():
        catalog_dict = load_json(catalog_file_fq)
    else:
        logger.warning(f"Dataset catalog '{catalog_file}' does not exist.")
        catalog_dict = {}

    if include_filename:
        return catalog_dict, catalog_file_fq
    return catalog_dict



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
        dataset_catalog, catalog_file_fq = get_dataset_catalog(catalog_path=catalog_path, catalog_file=catalog_file, include_filename=True)
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
    def from_disk(cls, dataset_name, data_path=None, metadata_only=False, errors=True,
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

    load = from_disk

    @classmethod
    def from_catalog(dataset_name,
         metadata_only=False,
         dataset_cache_path=None,
         catalog_path=None,
         dataset_catalog='datasets.json',
         transformer_catalog='transformers.json',
        ):
        """Load a dataset (or its metadata) from the dataset catalog.

        The named dataset must exist in the `dataset_catalog`.

        If a cached copy of the dataset is present on disk, (and its hashes match those in the dataset catalog),
        the cached copy will be returned. Otherwise, the dataset will be regenerated by traversing the
        transformer graph.

        Parameters
        ----------
        dataset_name: str
            name of dataset in the `dataset_catalog`
        metadata_only: Boolean
            if True, return only metadata. Otherwise, return the entire dataset
        check_hashes: Boolean
            if True, hashes will  be checked against `dataset_cache`.
            If they differ, an exception will be raised
        dataset_cache_path: str
            path containing cachec copy of `dataset_name`.
            Default `paths['processed_data_path']`
        catalog_path: str or None:
            path to data catalog (containing dataset_catalog and transformer_catalog)
            Default `paths['catalog_path']`
        dataset_catalog: str. default 'datasets.json'
            name of dataset catalog file. Relative to `catalog_path`.
        transformer_catalog: str. default 'transformers.json'
            name of dataset cache file. Relative to `catalog_path`.
        """
        if dataset_cache_path is None:
            dataset_cache_path = paths['processed_data_path']
        else:
            dataset_cache_path = pathlib.Path(dataset_cache_path)

        xform_graph = TransformerGraph(catalog_path=catalog_path,
                                       transformer_catalog=transformer_catalog,
                                       dataset_catalog=dataset_catalog)
        if dataset_name not in xform_graph.datasets:
            raise AttributeError(f"'{dataset_name}' not found in datset catalog.")
        meta = xform_graph.datasets[dataset_name]
        catalog_hashes = meta['hashes']

        if metadata_only:
            return meta

        ds = generate(xform_graph, dataset_name, dataset_cache_path=dataset_cache_path) ## fix

        generated_hashes = ds.metadata['hashes']
        if not check_hashes(catalog_hashes, generated_hashes):
            raise Exception(f"Dataset '{dataset_name}' hashes {generated_hashes} do not match catalog: {catalog_hashes}")


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
        hashes = {}
        for key, value in self.items():
            if key in exclude_list:
                continue
            data_hash = joblib.hash(value, hash_name=hash_type)
            hashes[key] = f"{hash_type}:{data_hash}"
        ret["hashes"] = hashes
        return ret

    def verify_hashes(self, hashdict):
        """Verify the supplied hash dictionary is a subset of my hash dictionary

        Hashes are a dict of attributes mapped to their hash value
        Hash values are strings f"{hash_type}:{hash_value}"; e.g.
        {
            'data': 'sha1:38f65f3b11da4851aaaccc19b1f0cf4d3806f83b',
            'target': 'sha1:38f65f3b11da4851aaaccc19b1f0cf4d3806f83b'
        }

        This test is order independent; e.g.
        >>> ds = Dataset("test")
        >>> ds.verify_hashes(reversed(ds.metadata['hashes']))
        True
        """
        return hashdict.items() <= self.metadata['hashes'].items()

    def dump(self, file_base=None, dump_path=None, hash_type='sha1',
             force=False, create_dirs=True, dump_metadata=True, update_catalog=True,
             catalog_path=None, catalog_file='datasets.json'):
        """Dump a dataset to disk.

        Note, this dumps a separate copy of the metadata structure,
        so that metadata can be looked up without loading the entire dataset,
        which could be large. It also (optionally, but by default) adds this
        metadata to the dataset catalog.

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

class TransformerGraph:
    """Transformer side of the bipartite Dataset Dependency Graph

    A "transformer" is a function that:

    * takes in zero or more `Dataset` objects (the `input_datasets`),
    * produces one or more `Dataset` objects (the `output_datasets`).

    Edges in this graph are directed, indicating the direction the direction of data dependencies.
    e.g. `output_datasets` depend on `input_datasets`.


    While the functions themselves are stores in the source module (default `src/user/transformers.py`),
    metadata describing these functions and which `Dataset` objects are generated are
    serialzed to `paths['catalog_path']/transformers.json`.

    Properties
    ----------
    nodes: set of dataset nodes (nodes in the hypergraph)
    edges: set of transformer nodes (edges in the hypergraph)
    """

    def __init__(self, catalog_path=None, transformer_catalog='transformers.json',  dataset_catalog='datasets.json'):
        """Create the Transformer (Dataset Dependency) Graph

        This can be thought of as a bipartite graph (node sets are datasets and transformers respectively), or a hypergraph,
        (nodes=datasets, edges=transformers) depending on your preference.

        catalog_path:
            Location of catalog files. Default paths['catalog_path']
        transformer_catalog:
            Catalog file. default 'transformers.json'. Relative to catalog_path
        dataset_catalog:
            Default 'transformers.json'. Relative to `catalog_path`
        """
        if catalog_path is None:
            catalog_path = paths['catalog_path']
        else:
            catalog_path = pathlib.Path(catalog_path)

        self.transformers = get_transformer_catalog(catalog_path=catalog_path, catalog_file=transformer_catalog)
        self.datasets = get_dataset_catalog(catalog_path=catalog_path, catalog_file=dataset_catalog)

        self._validate_hypergraph()

        self.edges_out = {}
        self.edges_in = {}
        for n in self.nodes:
            self.edges_in[n] = 0
            self.edges_out[n] = 0
        for he_name, he in self.transformers.items():
            for node in he['output_datasets']:
                self.edges_in[node] += 1
            for node in he.get('input_datasets', []):
                self.edges_out[node] += 1
            else:
                if self.is_source(he_name):
                    self.edges_in[node] = 0

    def _validate_hypergraph(self):
        """Check the basic structure of the hypergraph is valid"""

        valid = True
        for node in self.nodes:
            if node not in self.datasets:
                logger.warning(f"Node '{node}' not found in Dataset catalog.")
                valid = False

        return valid

    @property
    def nodes(self):
        ret = set()
        for he in self.transformers.values():
            for node in he['output_datasets']:
                ret.add(node)
        return ret

    @property
    def edges(self):
        return set(self.transformers)

    @property
    def sources(self):
        return [n for (n, count) in self.edges_in.items() if count < 1]

    @property
    def sinks(self):
        return [n for (n, count) in self.edges_out.items() if count < 1]

    def find_child(self, node):
        """Find its parents, siblings and the edge that produced a given child node.
        Parameters
        ----------
        node: String
            name of an output node

        Returns
        -------
        (parents, edge, siblings) where

        parents: Set(str)
            parents needed to generate this child node
        edge: str
            name of the edge that generated this node
        siblings: Set(str)
            set of all the output nodes generated by this edge

        """
        for hename, he in self.transformers.items():
            if node in he['output_datasets']:
                return set(he.get('input_datasets', [])), hename, set(he['output_datasets'])

    def is_source(self, edge):
        """Is this a source?

        Source edges terminate at a DataSource, and are identified
        by the an empty (or missing) input_datasets field
        """
        return not self.transformers[edge].get('input_datasets', False)

    def traverse(self, node, kind="breadth-first", force=False):
        """Find the path needed to regenerate the given node

        Traverse the graph as far as necessary to regenerate `node`.

        This will stop at the first upstream node whose parents are fully satisfied,
        (i.e. cached on disk, and whose hashes match the datset catalog)
        or all the way to source nodes, depending on the setting of `force`.

        Parameters
        ----------
        start: string
            Name of start node. Dendencies will be traced form this node back to sources

        kind: {'depth-first', 'breadth-first'}. Default 'breadth-first'
        force: Boolean
            if True, stop when all upstream dependencies are satisfied
            if False, always traverse all the way to source nodes.

        Returns
        -------
        (nodes , edges)
        where:
            nodes: List(str)
                list of node names traversed in the dependency graph
            edges: List(str)
                list of edge names traversed in the dependcy graph
        """
        if kind == 'breadth-first':
            pop_loc = 0
        elif kind == 'depth-first':
            pop_loc = -1
        else:
            raise Exception(f"Unknown kind: {kind}")
        visited = []
        edges = []
        queue = [node]
        while queue:
            vertex = queue.pop(pop_loc)
            if vertex not in visited:
                visited += [vertex]
                parents, edge, children = self.find_child(vertex)
                satisfied = self.fully_satisfied(edge)
                if satisfied:
                    if force:
                        logger.debug(f"All dependencies {parents} satisfied for edge: '{edge}' but force=True specified.")
                    else:
                        logger.debug(f"All dependencies {parents} satisfied for edge: '{edge}'")
                else:
                    logger.debug(f"Parent dependencies {parents} not satisfied for edge '{edge}'.")
                if not satisfied or force:
                    queue.extend(parents - set(visited))
                edges += [edge]
        return list(reversed(visited)), list(reversed(edges))

    def generate_outputs(self, edge_name):
        """Generate the outputs for a given edge

        This assumes all dependencies are on-disk and have valid caches.

        """
        if not self.fully_satisfied(edge_name):
            raise Exception(f"Edge '{edge_name}' has unsatisfied dependencies.")
        edge = self.transformers[edge_name]
        dsdict = {}
        logger.debug(f"Processing input datasets for Edge '{edge_name}'")
        for in_ds in edge.get('input_datasets', []):  # sources have no inputs
            logger.debug(f"Loading Input Dataset '{in_ds}'")
            if in_ds not in self.datasets:
                raise Exception(f"Edge '{edge_name}' specifies an input dataset, '{in_ds}' that is not in the dataset catalog")
            ds = Dataset.from_disk(in_ds)
            cached_hashes, catalog_hashes = ds.HASHES, self.datasets[in_ds]['hashes']
            if not check_dataset_hashes(cached_hashes, catalog_hashes):
                raise Exception(f"Cached Dataset '{in_ds}' hashes {cached_hashes} do not match catalog {catalog_hashes}")
            dsdict[in_ds] = ds

        for xform, xform_opts in edge.get('transformations', ()):
            transformer = deserialize_partial({'load_function_name':xform, 'load_function_kwargs': xform_opts}, fail_func=partial(default_transformer, transformer_name=xform))
            _, xform_func_str = partial_call_signature(transformer)
            logger.debug(f"Applying transformer: {xform_func_str}")
            dsdict = transformer(dsdict)


    def fully_satisfied(self, edge):
        """Determine whether all dependencies of the given edge (transformer) are satisfied

        Satisfied here means all input datasets are present (cached) on disk with valid hashes.
        Sources are always considered satisfied
        """
        if self.is_source(edge):
            return True

        input_datasets = self.transformers[edge].get('input_datasets', [])

        for ds_name in input_datasets:
            ds_meta = Dataset.load(ds_name, metadata_only=True, errors=False)
            if not ds_meta:  # does not exist
                logger.debug(f"No cached dataset found for dataset '{ds_name}'.")
                return False
            if ds_name not in self.datasets:
                raise Exception(f"Missing '{ds_name}' in dataset catalog")
            cached_hashes, catalog_hashes = ds_meta['hashes'], self.datasets[ds_name]['hashes']
            if not check_dataset_hashes(cached_hashes, catalog_hashes):
                logger.debug(f"Cached dataset '{ds_name}' hash {cached_hashes} != catalog hash {catalog_hashes}")
                return False
        return True

    def generate(dataset_name):
        pass
