import json
import functools
from collections import defaultdict
import random
import os

import seqio
import t5
import t5.data
from t5.data import preprocessors as t5_preprocessors
import tensorflow as tf
import numpy as np
from praxis import py_utils
from google.cloud import storage

from absl import logging
from paxml import checkpoint_paths
import jax


def get_feature(key_map, vocabulary):
    feature_desc, output_features = {}, {}
    for k, v in key_map.items():
        if v is None:
            continue
        feature_desc[v] = tf.io.VarLenFeature(tf.int64)
        output_features[k] = seqio.Feature(vocabulary=vocabulary, dtype=tf.int32)
    return feature_desc, output_features


def tfids_registry(task, mode):
    @seqio.map_over_dataset
    def convert_datatype(ex):
        return {k: tf.cast(tf.sparse.to_dense(v, default_value=0), dtype=tf.int32) for k, v in ex.items()}

    preprocessors = [
        convert_datatype,
        functools.partial(t5_preprocessors.rekey, key_map=task.KEY_MAP),
    ]
    feature_desc, output_features = get_feature(task.KEY_MAP, task.VOCABULARY)
    shuffle_buffer_size = task.SHUFFLE_SIZE if task.SHUFFLE[mode] else None
    if 'pythia' in task.TASK_NAME.lower():
        file_path_list = extract_pythia_datapath(task, mode)
    else:
        raise ValueError('Unknow dataset.....')
    source = seqio.TFExampleDataSource(
        split_to_filepattern={mode: file_path_list[mode]},
        feature_description=feature_desc,
    )
    print(f"mode: {mode} shuffle_size: {shuffle_buffer_size} task.SHUFFLE[mode]: {task.SHUFFLE[mode]}")
    name = f"{task.TASK_NAME}.{mode}"
    if check_registry_name(name):
        seqio.TaskRegistry.add(
            name,
            source,
            preprocessors=preprocessors,
            output_features=output_features,
            shuffle_buffer_size=shuffle_buffer_size,
        )


def check_registry_name(name):
    return False if name in t5.data.TaskRegistry._REGISTRY else True


def c4_registry(task, mode):
    preprocessors = [
        functools.partial(t5_preprocessors.rekey, key_map=task.KEY_MAP),
        seqio.preprocessors.tokenize,
        functools.partial(t5_preprocessors.reduce_concat_tokens, batch_size=4096),
        t5_preprocessors.split_tokens_to_targets_length,
    ]
    feature_desc, output_features = get_feature(task.KEY_MAP, task.VOCABULARY)
    shuffle_buffer_size = task.SHUFFLE_SIZE if task.SHUFFLE[mode] else None
    # data_path = "gs://common_datasets"
    bucket_name = task.DATA_PATH[mode]
    if 'gs:' not in bucket_name:
        bucket_name = 'gs://' + bucket_name
    print(f'c4 bucket_name: {bucket_name}')
    source = seqio.TfdsDataSource(tfds_name="c4/en:3.0.1", tfds_data_dir=bucket_name)
    name = f"c4.{mode}"
    if check_registry_name(name):
        t5.data.TaskRegistry.add(
            name,
            seqio.Task,
            source=source,
            preprocessors=preprocessors,
            output_features=output_features,
            metric_fns=[],
            shuffle_buffer_size=shuffle_buffer_size,
        )


def extract_pythia_datapath(task, mode):
    client = storage.Client()
    #v3: us-east1-d -> common_datasets, v4: us-central2-b -> common_datasets_us-central2-b
    path = task.DATA_PATH[mode].replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path + '/'
    logging.info(f'bucket_name = {bucket_name}, directory_path = {directory_path}')
    step_map_path = {}
    rerank = 0
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        # logging.info(f"filename: {blob.name}=====")
        if ".tfrecord" not in blob.name: continue
        try:
            step = int(blob.name.rsplit("pile.tfrecord.b", maxsplit=1)[-1])
        except:
            step = rerank
            rerank += 1
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        step_map_path[step] = path
    sorted_step_path = sorted(step_map_path.items(), key=lambda x: x[0])
    steps, pathes = zip(*sorted_step_path)
    if not isinstance(pathes, list):
        pathes = list(pathes)
    # 目前只是为了测试，训练的时候可以选择是否需要test
    train_test_dataset = {"test": pathes[:1], "train": pathes}
    logging.info(f'Train file: {len(train_test_dataset["train"])},  test file: {len(train_test_dataset["test"])}')
    return train_test_dataset


def extract_zh_en_novel_datapath(task, mode):
    random.seed(task.TRAINING_SEED)
    dataset = defaultdict(list)
    client = storage.Client()
    bucket_name = os.path.dirname(task.DATA_PATH[mode])
    if 'gs:' in bucket_name:
        bucket_name = bucket_name[5: ]
    directory_path = os.path.basename(task.DATA_PATH[mode]) + '/'
    for lang in ["zh", "en"]:
        # directory_path = f'xiaomeng/processed_{lang}_data_split'
        prefix = directory_path.format(lang=lang)
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            logging.info(f"filename: {blob.name}=====")
            if not blob.name or "_R" not in blob.name:
                continue
            if len(dataset[lang]) > 5:
                break
            index = int(blob.name.rsplit("_", maxsplit=1)[-1])
            # 每本书的前多少个4096
            if index < task.SPLIT_BSZ[lang]:
                path = os.path.join(f"gs://{bucket_name}", blob.name)
                dataset[lang].append(path)
    total = dataset["zh"] + dataset["en"]
    random.shuffle(total)
    test_num = int(len(total) * task.TEST_RATIO)
    test_num = max(test_num, 1)

    train_test_dataset = {"test": total[:test_num], "train": total[test_num:]}
    logging.info(f'Train file: {len(train_test_dataset["train"])},  test file: {len(train_test_dataset["test"])}')
    return train_test_dataset


def extract_train_skip_step(job_log_dir, step, only_eval=False):
    if job_log_dir is None:
        return {}
    model_dir = job_log_dir / "checkpoints"
    if step is not None:
        fill_step = checkpoint_paths.CHECKPOINT_PREFIX + str(step).zfill(checkpoint_paths._STEP_FORMAT_FIXED_LENGTH)
        skip_file_and_step_path = model_dir / fill_step / checkpoint_paths.SKIP_STEP_NAME
    else:
        skip_file_and_step_path = model_dir / checkpoint_paths.SKIP_STEP_NAME
    logging.info(f"model_dir: {model_dir}")
    try:
        with skip_file_and_step_path.open('r') as f:
            meta_dict = json.load(f)
        logging.info(f"Load skip_file_and_step_path: ’{skip_file_and_step_path}‘ Finished.......")
    except:
        logging.info(f"skip_file_and_step_path: ’{skip_file_and_step_path}‘ is not existed.......")
        meta_dict = {}

    if jax.process_index() == 0:
        mode = 'train_break_steps' if not only_eval else 'eval_metric_steps'
        back_meta_dict_path = job_log_dir / mode /f'{meta_dict.get("checkpoint_step", None)}.json'
        with back_meta_dict_path.open('w') as f1:
            json.dump(meta_dict, f1)
    return meta_dict
