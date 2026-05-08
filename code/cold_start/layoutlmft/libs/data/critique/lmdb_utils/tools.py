import lmdb
import torch
import io
import tqdm
import os
import json
import numpy as np
import pickle

class LmdbWriter:
    def __init__(self, db_path, map_size=1024 ** 3 * 500):
        assert not os.path.exists(db_path), f'目标地址{db_path}已存在, 覆盖可能会发生错误, 请手动删除'
        self.db_path = db_path
        self.map_size = map_size
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.env = lmdb.open(
            db_path,
            map_size=map_size,
            create=True,
            subdir=False,
            readonly=False
        )
        self.meta_info = {'lmdb_path':db_path, 'size':0}
        self._cache = []
        self._cache_size = 1000

    def write(self, data):
        """data为字典组成的列表, 字典的value为numpy的array或字符串"""
        for record in data:
            index = self.meta_info['size']
            self.meta_info['size'] += 1
            if 'key' not in data:
                key = str(index).encode()
            else:
                key = data['key'].encode()

            serialized_dict = {}
            for sub_key, arr in record.items():
                if isinstance(arr, np.ndarray):
                    serialized_dict[sub_key] = arr.tobytes()
                    serialized_dict[f'{sub_key}_shape'] = arr.shape
                    serialized_dict[f'{sub_key}_dtype'] = arr.dtype
                else:
                    serialized_dict[sub_key] = arr # string
            bytes_data = pickle.dumps(serialized_dict)
            self._cache.append((key, bytes_data))
            if len(self._cache) == self._cache_size:
                self.flush()
                cur_size = self.meta_info['size']
                print(f'writing lmdb {cur_size}')
        if len(self._cache) > 0:
            self.flush()
            
    def flush(self):
        txn = self.env.begin(write=True, buffers=True)
        while len(self._cache) > 0:
            key, bytes_data = self._cache.pop()
            txn.put(key, bytes_data)
        txn.commit()


    def close(self):
        """关闭数据库连接"""
        self.env.close()
        meta_path = self.db_path + '.meta.json'
        with open(meta_path, 'w') as f:
            json.dump(self.meta_info, f)        


class LmdbReader:
    def __init__(self, db_path):
        """初始化 LmdbReader，专注于读取效率"""
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=False,
            readonly=True,
            lock=False
        )

    def __getitem__(self, index):
        return self.read(index)        

    def read(self, index):
        """
        根据 index 读取数据
        index: 要读取的记录索引
        return: 返回对应的字典（如果存在）
        """
        key = str(index).encode()
        with self.env.begin(write=False) as txn:
            value = txn.get(key)
        if value is None:
            raise IndexError(f"Index {index} not found in the database.")
        
        serialized_dict = pickle.loads(value)
        data_dict = {}
        for k in serialized_dict.keys():
            if not k.endswith("_shape") and not k.endswith('_dtype') and not k.endswith('_string') and k != 'key':
                shape_key = f"{k}_shape"
                dtype_key = f"{k}_dtype"
                shape = serialized_dict[shape_key]
                dtype = serialized_dict[dtype_key]
                arr = np.frombuffer(serialized_dict[k], dtype=dtype).reshape(shape)
                data_dict[k] = arr
            elif k.endswith('_string') or k == 'key':
                data_dict[k] = serialized_dict[k]
        return data_dict

    def close(self):
        """关闭数据库连接"""
        self.env.close()


# 示例使用
if __name__ == "__main__":
    # 准备一些测试数据
    data = [
        {"a": torch.tensor([1, 2, 3], dtype=torch.float32), 
         "b": torch.tensor([1.1, 2.2, 3.3])},
        {"a": torch.tensor([4, 5, 6]), 
         "b": torch.tensor([4.4, 5.5, 6.6])},
    ]

    # 写入数据
    writer = LmdbWriter(db_path='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/prm/v1/dataset/debug/debug.lmdb')
    writer.write(data)
    data = [
        {"a": torch.tensor([7, 8, 9]), 
         "b": torch.tensor([7.7, 8.8, 9.9])},
    ]
    writer.write(data)
    writer.close()

    # 读取数据
    reader = LmdbReader(db_path='/train34/mmu/permanent/cxqin/zrzhang6/ChartQA/pfhu6/dataset/prm/v1/dataset/debug/debug.lmdb')

    for i in range(len(data)):
        a = reader.read(i)
        print(a)

    reader.close()
