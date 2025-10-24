import os

print("="*60)
print("验证D盘数据完整性")
print("="*60)

# 检查多输入训练集
d_train = 'D:/UBB_train'
if os.path.exists(d_train):
    folders = ['gt', 'depth', 'color_B_1', 'color_B_2', 'color_B_3', 
               'color_BG_1', 'color_G_1', 'color_Y_1', 'color_YG_1']
    
    print("\n多输入训练集（D:/UBB_train）:")
    for f in folders:
        path = os.path.join(d_train, f)
        if os.path.exists(path):
            count = len([x for x in os.listdir(path) if x.endswith('.png')])
            print(f"  {f:15s}: {count:6d}/11098")
else:
    print("\n多输入训练集: 不存在")

# 检查单输入训练集
d_single = 'D:/UBB_train_single_input'
if os.path.exists(d_single):
    input_path = os.path.join(d_single, 'input')
    gt_path = os.path.join(d_single, 'gt')
    
    input_count = len([f for f in os.listdir(input_path) if f.endswith('.png')]) if os.path.exists(input_path) else 0
    gt_count = len([f for f in os.listdir(gt_path) if f.endswith('.png')]) if os.path.exists(gt_path) else 0
    
    print(f"\n单输入训练集（D:/UBB_train_single_input）:")
    print(f"  input: {input_count:6d}/166470 ({input_count/166470*100:.1f}%)")
    print(f"  gt:    {gt_count:6d}/166470 ({gt_count/166470*100:.1f}%)")
else:
    print("\n单输入训练集: 不存在")

print("\n" + "="*60)




