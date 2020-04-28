import os
import pathlib
import sys

from ..log import logger
from ..utils import load_json, save_json
from .datasets import Dataset, DataSource, available_datasources
from .transformer_functions import *
from .. import paths

__all__ = [
    'add_transformer_to_catalog',
    'apply_transforms',
    'del_transformer_from_catalog',
    'get_transformer_catalog',
    'dataset_from_datasource',
]


def create_transformer_pipeline(func_list, ignore_module=False):
    """Create a serialize transformer pipeline.

    Output is suitable for passing to `add_transformer`

    ignore_module: Boolean
        if True, remove the module from the serialized function call
        (i.e. rely on it being in the namespace when called)
        This makes it easy to eliminate some common errors when building
        transformer graphs in Notebooks
    """
    ret = []

    for f in func_list:
        serialized = serialize_partial(f, key_base='transformer')
        if ignore_module:
            del(serialized['transformer_module'])
        ret.append(serialized)

    return ret

def dataset_from_datasource(dsdict, *, dataset_name, **dsrc_args):
    """Transformer: Create a Dataset from a DataSource object

    This is just a thin wrapper around Dataset.from_datasource in order to
    conform to the transformer API

    Parameters
    ----------
    dsdict: dict, ignored.
        Because this is a source, this argument is unnecessary (except to
        conform to the transformer function API) and is ignored
    dataset_name: str, required
        Name of datasource in DataSource catalog
    dsrc_args: dict
        Arguments are the same as the `Dataset.from_datasource()` constructor

    Returns
    -------
    dict: {dataset_name: Dataset}
    """
    ds = Dataset.from_datasource(dataset_name, **dsrc_args)
    return {dataset_name: ds}

def normalize_to_list(str_or_iterable):
    """Convert strings to lists. Pass lists (or None) unchanged.
    """
    if isinstance(str_or_iterable, str):
        return [str_or_iterable]
    if str_or_iterable is None:
        return []
    return str_or_iterable

def get_transformer_catalog(
        catalog_path=None,
        catalog_file=None,
        include_filename=False,
    ):
    """Get the dictionary of transformers (edges in the transformer graph)

    Parameters
    ----------
    include_filename: boolean
        if True, returns a tuple: (list, filename)
    catalog_path: path. (default: paths['catalog_dir'])
        Location of `catalog_file`
    catalog_file: str, default 'transformers.json'
        Name of json file that contains the transformer pipeline

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
    if catalog_file is None:
        catalog_file = 'transformers.json'

    catalog_file_fq = catalog_path / catalog_file

    if catalog_file_fq.exists():
        catalog_dict = load_json(catalog_file_fq)
    else:
        logger.warning(f"Catalog '{catalog_file}' does not exist.")
        catalog_dict = {}

    if not isinstance(catalog_dict, dict):
        raise Exception(f"Obsolete file format: {transformer_file} must contain a dict.")

    if include_filename:
        return catalog_dict, catalog_file_fq
    return catalog_dict

def del_transformer_from_catalog(transformer, transformer_path=None, transformer_file=None):
    """Delete an entry in the transformer catalog

    transformer: name (unique key) of transformer
    transformer_path: path. (default: MODULE_DIR)
        Location of `transformer_file`
    transformer_file: string, default 'transformer_list.json'
        Name of json file that contains the transformer pipeline
    """
    transformer_dict, transformer_file_fq = get_transformer_catalog(catalog_path=transformer_path,
                                                                    catalog_file=transformer_file,
                                                                    include_filename=True)

    del(transformer_dict[transformer])
    save_json(transformer_file_fq, transformer_dict)

def add_transformer_to_catalog(
        name=None,
        datasource_name=None,
        datasource_opts=None,
        input_datasets=None,
        output_datasets=None,
        transformations=None,
        dag_path=None,
        edge_file=None,
        node_file=None,
        write_to_catalog=True,
        force=False,
):
    """Create and add a dataset transformation pipeline to the workflow.

    Transformer pipelines apply a sequence of transformer functions to a Dataset (or DataSource),
    to produce new Dataset objects.

    Parameters
    ----------
    name: string
        Name for this transformer instance (must be unique).
        By default, one will be created from the input and output dataset names; e.g.
        _input_ds1_input_ds2_to_output_ds1
    input_datasets: string or iterable
        Upstream data dependencies. These must be present
    output_datasets: string or Iterable
        These datasets will be generated
    datasource_name: string
        Name of a DataSource to use to generate the output
        Setting this option will create a source node in the dataset flow graph
        (or a sink node in the data dependency graph).
        Transformers of this type must specify at most one entry in `output_datasets`
    datasource_opts: dict
        Options to use when generating a Dataset from this DataSource
    transformations: list of tuples
        Squence of transformer functions to apply. tuples consist of:
        (transformer_name, transformer_opts)
    dag_path: path. (default: paths['catalog_path'])
        Location of `dag_file`
    edge_file: string, default 'transformers.json'
        Name of json file that contains the transformer pipeline
    node_file: string, default 'datasets.json'
        Name of json file that contains the dataset metadata
    write_to_catalog: Boolean, Default True
        If False, don't actually write this entry to the catalog.
    force: Boolean
        If True, overwrite entries in transformer catalog
        If False, raise an exception on duplicate transformer catalog entries

    Returns
    -------
    dict: {name: catalog_entry} where catalog_entry is the serialized transformer operation

    Examples
    --------

    If you only have one input or output, it may be specified simply as a string;
    i.e. these are identical
    >>> add_transformer(input_datasets='other', output_datasets='p_other', write_to_catalog=False)
    {'_p_other': {'input_datasets': ['other'], 'output_datasets': ['p_other']}}
    >>> add_transformer(input_datasets=['other'], output_datasets='p_other', write_to_catalog=False)
    {'_p_other': {'input_datasets': ['other'], 'output_datasets': ['p_other']}}

    >>> add_transformer(input_datasets=['cc-by', 'cc-by-nc'], output_datasets='cc', write_to_catalog=False)
    {'_cc': {'input_datasets': ['cc-by', 'cc-by-nc'], 'output_datasets': ['cc']}}
    >>> add_transformer(input_datasets=['cc-by', 'cc-by-nc'], output_datasets='cc', write_to_catalog=False)
    {'_cc': {'input_datasets': ['cc-by', 'cc-by-nc'], 'output_datasets': ['cc']}}

    Names can be given explicitly:

    >>> add_transformer(input_datasets=['cc'], output_datasets=['cc_train','cc_test'], write_to_catalog=False)
    {'_cc_train_cc_test': {'input_datasets': ['cc'], 'output_datasets': ['cc_train', 'cc_test']}}
    >>> add_transformer(input_datasets=['cc'], output_datasets=['cc_train','cc_test'], name='tts', write_to_catalog=False)
    {'tts': {'input_datasets': ['cc'], 'output_datasets': ['cc_train', 'cc_test']}}


    Invalid use cases:

    >>> add_transformer(datasource_name="foo", output_datasets=['bar', 'baz'])
    Traceback (most recent call last):
    ...
    Exception: Edges from data sources must have only one output_dataset.

    >>> add_transformer(datasource_name="foo", input_datasets='bar')
    Traceback (most recent call last):
    ...
    Exception: Cannot set both `datasource_name` and `input_datasets`

    >>> add_transformer(datasource_opts={'foo':'bar'})
    Traceback (most recent call last):
    ...
    Exception: Must specify `datasource_name` when using `datasource_opts`

    >>> add_transformer(output_datasets="foo")
    Traceback (most recent call last):
    ...
    Exception: Must specify one of from `datasource_name` or `input_datasets`

    >>> add_transformer(input_datasets="foo")
    Traceback (most recent call last):
    ...
    Exception: Must specify `output_dataset`
    """
    input_datasets = normalize_to_list(input_datasets)
    output_datasets = normalize_to_list(output_datasets)

    if datasource_name is not None:
        if input_datasets:
            raise Exception('Cannot set both `datasource_name` and `input_datasets`')
        if output_datasets is not None and len(output_datasets) > 1:
            raise Exception("Edges from data sources must have only one output_dataset.")
    if datasource_name is None and datasource_opts is not None:
        raise Exception('Must specify `datasource_name` when using `datasource_opts`')

    if write_to_catalog:
        ds_dag, ds_dag_fq = get_transformer_catalog(catalog_path=dag_path,
                                                    catalog_file=edge_file,
                                                    include_filename=True)
    catalog_entry = {}

    if transformations is None:
        transformations = []
    if datasource_name:  # special case. Convert this to a transformer call
        if not output_datasets:  # Default output_datasets
            output_datasets = [datasource_name]
        if datasource_opts is None:
            datasource_opts = {}
        transformations.insert(0, (datasource_name, datasource_opts))
    elif input_datasets:
        catalog_entry['input_datasets'] = input_datasets

    if transformations:
        catalog_entry['transformations'] = transformations

    if not output_datasets:
        raise Exception("Must specify `output_dataset`")
    else:
        catalog_entry['output_datasets'] = output_datasets

    if name is None:
        name = f"_{'_'.join([ids for ids in output_datasets])}"

    if write_to_catalog:
        if name in ds_dag and not force:
            raise Exception(f"Transformer '{name}' already in catalog. Use force=True to overwrite")
        ds_dag[name] = catalog_entry
        save_json(ds_dag_fq, ds_dag)
    return {name:catalog_entry}

def apply_transforms(datasets=None, transformer_path=None, transformer_file='transformer_list.json', output_dir=None):
    """Apply all data transforms to generate the specified datasets.

    transformer_file: string, default "transformer_list.json"
        Name of transformer file.
    transformer_path: path
        Path containing `transformer_file`. Default paths['catalog_path']
    output_dir: path
        Path to write the generated datasets. Default paths['processed_data_path']

    """

    if output_dir is None:
        output_dir = paths['processed_data_path']
    else:
        output_dir = pathlib.Path(output_dir)

    if transformer_path is None:
        transformer_path = paths['catalog_path']
    else:
        transformer_path = pathlib.Path(transformer_path)

    transformer_list = get_transformer_list(transformer_path=transformer_path,
                                            transformer_file=transformer_file)
    datasources = available_datasources()
    transformers = available_transformers(keys_only=False)

    for tdict in transformer_list:
        datasource_opts = tdict.get('datasource_opts', {})
        datasource_name = tdict.get('datasource_name', None)
        output_dataset = tdict.get('output_dataset', None)
        input_dataset = tdict.get('input_dataset', None)
        transformations = tdict.get('transformations', [])
        if datasource_name is not None:
            if datasource_name not in datasources:
                raise Exception(f"Unknown DataSource: {datasource_name}")
            logger.debug(f"Creating Dataset from Raw: {datasource_name} with opts {datasource_opts}")
            rds = DataSource.from_name(datasource_name)
            ds = rds.process(**datasource_opts)
        else:
            logger.debug("Loading Dataset: {input_dataset}")
            ds = Dataset.load(input_dataset)

        for tname, topts in transformations:
            tfunc = transformers.get(tname, None)
            if tfunc is None:
                raise Exception(f"Unknwon transformer: {tname}")
            logger.debug(f"Applying {tname} to {ds.name} with opts {topts}")
            ds = tfunc(ds, **topts)

        if output_dataset is not None:
            logger.info(f"Writing transformed Dataset: {output_dataset}")
            ds.name = output_dataset
            ds.dump(dump_path=output_dir)
