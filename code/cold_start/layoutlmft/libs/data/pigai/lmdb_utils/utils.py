import os
import io
import sys
import json
import lmdb
import logging
import struct
import numpy as np
import sentencepiece as spm
import re
import random
from tqdm import tqdm

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path)
from lmdb_pb2 import ASRProto

log_fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    format=log_fmt, datefmt=datefmt,
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

logger = logging.getLogger("db_convert")
logger.setLevel(logging.INFO)

MAX_DB_SIZE = 1024 ** 3 * 500  # 500GB
UINT16_MAX_VAL = 65535
UINT32_MAX_VAL = 4294967295

PROTO_DTYPE_MAPPING = {
    # 0:               #"UNDEFINED",
    1: np.float32,     # "FLOAT",
    2: np.uint8,       # "UINT8",
    3: np.int8,        # "INT8",
    4: np.uint16,      # "UINT16",
    5: np.int16,       # "INT16",
    6: np.int32,       # "INT32",
    7: np.int64,       # "INT64",
    8: np.string_,     # "STRING",
    9: np.bool_,       # "BOOL",
    10: np.float16,    # "FLOAT16",
    11: np.float64,    # "DOUBLE",
    12: np.uint32,     # "UINT32",
    13: np.uint64,     # "UINT64",
    14: np.complex64,  # "COMPLEX64",
    15: np.complex128, # "COMPLEX128",
    # 16:              # "BFLOAT16",
}

def make_filepath_dir(file_path):
    if os.path.exists(file_path):
        return
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)


def open_readable_lmdb(lmdb_path, sub_dir=False):
    env = lmdb.Environment(lmdb_path, subdir=sub_dir, readonly=True, lock=False)
    return env


def open_writeable_lmdb(lmdb_path, sub_dir=False, map_size=MAX_DB_SIZE):
    env = lmdb.Environment(lmdb_path, subdir=sub_dir, map_size=map_size)
    return env


def close_lmdb(env):
    env.close()


class LmdbMix(object):
    def __init__(self, lmdb_path, is_subdir, mode='r', map_size=None) -> None:
        self.lmdb_path = lmdb_path
        self.sub_dir = is_subdir
        self.map_size = map_size
        self.mode = mode
        self._env = None

    @property
    def env(self):
        return self._env

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_env'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = None

    def open_lmdb(self):
        if self._env is not None:
            return
        if self.mode == 'r':
            self._env = open_readable_lmdb(
                self.lmdb_path,
                self.sub_dir
            )
        elif self.mode == 'w':
            self._env = open_writeable_lmdb(
                self.lmdb_path,
                self.sub_dir,
                self.map_size,
            )
        else:
            raise ValueError(f"Unvalid mode={self.mode}")

    def close_lmdb(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __del__(self):
        self.close_lmdb()

    def __len__(self):
        self.open_lmdb()
        size = int(self._env.stat()['entries'])
        return size

    @property
    def size(self):
        return len(self)


class LmdbReader(LmdbMix):
    def __init__(self, lmdb_path, sub_dir, key_format=None) -> None:
        super(LmdbReader, self).__init__(lmdb_path, sub_dir, 'r', None)
        self.key_format = key_format if key_format is not None else "{}"

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key):
        self.open_lmdb()
        bytes_key = key
        if not isinstance(key, bytes):
            bytes_key = str(key).encode()
        with self.env.begin() as txn:
            value = txn.get(bytes_key)
            if value is None:
                raise KeyError(f"Key={key} not found!")
            value = ASRProto.FromString(value)
            dtype = PROTO_DTYPE_MAPPING[value.data_type]
            data = np.frombuffer(value.data, dtype=dtype).copy()            
        return data

    def __iter__(self):
        self.open_lmdb()
        with self.env.begin() as txn:
            for i in txn.cursor():
                yield i



class LmdbWriter(LmdbMix):
    def __init__(self, lmdb_path: str, is_subdir: bool, map_size=MAX_DB_SIZE, write_interval=1000, key_format=None,
                 shuffle=False):
        super(LmdbWriter, self).__init__(lmdb_path, is_subdir, 'w', map_size)
        make_filepath_dir(lmdb_path)
        self.write_interval = write_interval
        self.key_format = key_format if key_format is not None else "{}"
        self._cache = list()
        self._count = None
        self.shuffle = shuffle
        self.meta = None
        self._load_meta()
        self.close_lmdb()  # avoid lmdb mp error

    def _load_meta(self):
        desc_path = self.lmdb_path + ".json"
        if not os.path.exists(desc_path):
            self._count = 0
            self.meta = {
                "lmdb_file": self.lmdb_path,
                "lengths": [],
                "length_file": self.lmdb_path + "_length.bin",
                "size": 0,
                "sample_rate": None,
            }
            return
        with open(desc_path, 'r') as fp:
            meta = json.load(fp)
        self.meta = meta
        if "sample_rate" not in self.meta:
            self.meta['sample_rate'] = None
        assert self.size == meta['size'], \
            f"Dataset desc size({meta['size']}) != lmdb size({self.size})."
        logger.info(f"Load meta from {desc_path}, restore count={self._count}")

    @property
    def count(self):
        if self._count is None:
            self._count = self.size
        return self._count

    @count.setter
    def count(self, i):
        self._count = i

    def __del__(self):
        self.flush()
        super().__del__()

    def write(self, item):
        self.open_lmdb()
        self._cache.append(item)
        if len(self._cache) >= self.write_interval:
            self.flush()

    def flush(self):
        if len(self._cache) <= 0:
            return
        if self.shuffle:
            np.random.shuffle(self._cache)
        txn = self.env.begin(write=True, buffers=True)
        for data, length in self._cache:
            key = self.key_format.format(self.count)
            value = data
            if not isinstance(data, bytes):
                value = data.SerializeToString()
            txn.put(key.encode(), value)
            self.meta["lengths"].append(int(length))
            self.count += 1
        txn.commit()
        self._cache.clear()

    def write_desc(self):
        self.meta.update(size=self.count)
        assert len(self.meta['lengths']) == self.count

        with open(self.meta['length_file'], 'wb') as fp:
            bin_format = f">{len(self.meta['lengths'])}I"
            fp.write(
                struct.pack(
                    bin_format,
                    *self.meta['lengths']
                )
            )

        desc_path = self.lmdb_path + ".json"
        with open(desc_path, 'w') as desc_file:
            json.dump(self.meta, desc_file)




class TxtReader(object):
    def __init__(self, input_tokens):
        self.input_tokens = input_tokens
        self.sample_tokens = list()
        self.sample_sizes = list()
        self.max_token_val = 0
        self._init()
        if self.max_token_val > UINT32_MAX_VAL:
            self.dtype = np.uint64
        elif self.max_token_val > UINT16_MAX_VAL:
            self.dtype = np.uint32
        else:
            self.dtype = np.uint16

    def _init(self):
        for cur_token in self.input_tokens:
            self.max_token_val = max(self.max_token_val, max(cur_token))
            self.sample_tokens.append(cur_token)
            self.sample_sizes.append(len(cur_token))

   
    def __len__(self):
        return len(self.sample_tokens)

    @property
    def size(self):
        return len(self)

    def __getitem__(self, key):
        return self.get(key)

    def get_txt(self, key):
        try:
            key = int(key)
        except:
            KeyError(f"key only support int type, but got {type(key)}")
        sentence = np.array(self.sample_tokens[int(key)], dtype=self.dtype)
        length = self.sample_sizes[int(key)]
        assert sentence.shape[0] == length
        return {'sentence': sentence,
                'length': length}

    def get(self, key):
        value = self.get_txt(key)
        datum = ASRProto()
        datum.name = str(key).encode()
        if self.dtype == np.uint64:
            datum.data_type = ASRProto.UINT64
        elif self.dtype == np.uint32:
            datum.data_type = ASRProto.UINT32
        else:
            datum.data_type = ASRProto.UINT16
        datum.dim = 1
        datum.data = value['sentence'].tobytes()
        value.update(datum=datum)
        return value

    def __iter__(self):
        for i in range(len(self.sample_tokens)):
            yield i, self.get(i)

    @staticmethod
    def convert_handle(name, item):
        return item['datum'], item['length']


def convert(reader: LmdbReader,
            writer: LmdbWriter,
            convert_handle,
            max_size=None):
    save_dir = os.path.dirname(writer.lmdb_path)
    handler = logging.FileHandler(os.path.join(save_dir, "convert.log"))
    formatter = logging.Formatter(fmt=log_fmt, datefmt=datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Start convert ...")
    success = 0
    ignore = 0
    if max_size is None:
        max_size = len(reader)

    for name, item in reader:
        new_item, length = convert_handle(name, item)
        try:
            writer.write((new_item, length))
        except Exception as e:
            logger.error(f"writer error: {e}")
            raise
        success += 1
        if success > 0 and success % writer.write_interval == 0:
            logger.info(f"Process {success}/{len(reader)} ...")
        if success >= max_size:
            break

    writer.flush()
    logger.info(f"Process {success}/{len(reader)} ...")

    if writer.count != success:
        logger.error(f"Convert dataset failed, reader size={reader.size}, " \
                     f"success={success}, ignore={ignore}, writer size={writer.count}!")
    else:
        writer.write_desc()
        logger.info("Convert done!")

    logger.removeHandler(handler)

