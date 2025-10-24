#!/usr/bin/env python3
import os
import sys

lmdb_dir = 'F:/DATASATES/UBB_train.lmdb'

if os.path.exists(lmdb_dir):
    files = os.listdir(lmdb_dir)
    total_size = sum(os.path.getsize(os.path.join(lmdb_dir, f)) 
                    for f in files if os.path.isfile(os.path.join(lmdb_dir, f)))
    
    print(f"LMDB转换进度:")
    print(f"  大小: {total_size/(1024**3):.2f} GB")
    print(f"  文件: {files}")
    
    # 尝试读取样本数
    try:
        import lmdb
        env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            stats = txn.stat()
            print(f"  条目数: {stats['entries']}")
            
            # 尝试读取元数据
            meta_bytes = txn.get(b'__meta__')
            if meta_bytes:
                import pickle
                meta = pickle.loads(meta_bytes)
                print(f"  样本数: {meta.get('num_samples', 'unknown')}")
        env.close()
    except Exception as e:
        print(f"  无法读取详情: {e}")
else:
    print("LMDB尚未创建")




