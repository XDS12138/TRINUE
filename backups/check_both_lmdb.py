#!/usr/bin/env python3
import os
import lmdb
import pickle

print("="*60)
print("检查LMDB转换状态")
print("="*60)

# 检查多输入LMDB
multi_lmdb = 'F:/DATASATES/UBB_train.lmdb'
print(f"\n多输入LMDB: {multi_lmdb}")
if os.path.exists(multi_lmdb):
    files = os.listdir(multi_lmdb)
    total_size = sum(os.path.getsize(os.path.join(multi_lmdb, f)) 
                    for f in files if os.path.isfile(os.path.join(multi_lmdb, f)))
    
    print(f"  大小: {total_size/(1024**3):.2f} GB")
    
    try:
        env = lmdb.open(multi_lmdb, readonly=True, lock=False)
        with env.begin() as txn:
            stats = txn.stat()
            print(f"  条目数: {stats['entries']}")
            
            meta_bytes = txn.get(b'__meta__')
            if meta_bytes:
                meta = pickle.loads(meta_bytes)
                print(f"  样本数: {meta.get('num_samples', 'unknown')}")
                print(f"  退化数: {meta.get('num_degradations', 'unknown')}")
                print(f"  分辨率: {meta.get('resolution', 'unknown')}")
                
                expected = 11098
                actual = meta.get('num_samples', 0)
                if actual == expected:
                    print(f"  ✅ 完成! ({actual}/{expected})")
                else:
                    print(f"  🔄 进行中: {actual}/{expected} ({actual/expected*100:.1f}%)")
        env.close()
    except Exception as e:
        print(f"  ⚠️ 无法读取: {e}")
else:
    print(f"  ❌ 不存在")

# 检查单输入LMDB
single_lmdb = 'E:/DATASATES/UBB_train_single_input.lmdb'
print(f"\n单输入LMDB: {single_lmdb}")
if os.path.exists(single_lmdb):
    files = os.listdir(single_lmdb)
    total_size = sum(os.path.getsize(os.path.join(single_lmdb, f)) 
                    for f in files if os.path.isfile(os.path.join(single_lmdb, f)))
    
    print(f"  大小: {total_size/(1024**3):.2f} GB")
    
    try:
        env = lmdb.open(single_lmdb, readonly=True, lock=False)
        with env.begin() as txn:
            stats = txn.stat()
            print(f"  条目数: {stats['entries']}")
            
            meta_bytes = txn.get(b'__meta__')
            if meta_bytes:
                meta = pickle.loads(meta_bytes)
                print(f"  样本数: {meta.get('num_samples', 'unknown')}")
                print(f"  分辨率: {meta.get('resolution', 'unknown')}")
                
                expected = 166470
                actual = meta.get('num_samples', 0)
                if actual == expected:
                    print(f"  ✅ 完成! ({actual}/{expected})")
                else:
                    print(f"  🔄 进行中: {actual}/{expected} ({actual/expected*100:.1f}%)")
        env.close()
    except Exception as e:
        print(f"  ⚠️ 无法读取: {e}")
else:
    print(f"  ❌ 不存在")

print("\n" + "="*60)


