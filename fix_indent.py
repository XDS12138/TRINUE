#!/usr/bin/env python3
with open("/media/xxx/233-3/PycharmProjects/TRINUE/scripts/train.py", "r") as file:
    lines = file.readlines()

# 修复缩进问题的行号
problematic_section = {
    1815: "            if ((epoch + 1) % save_interval == 0) or (is_best and save_best):",
    1816: "                checkpoint_save_dir = os.path.join(exp_dir, 'checkpoints') # Renamed",
    1817: "                os.makedirs(checkpoint_save_dir, exist_ok=True)",
    1818: "",
    1819: "                # 准备保存的状态",
    1820: "                checkpoint_data = {",
    1821: "                    'epoch': epoch + 1,",
    1822: "                    'state_dict': model.state_dict(),",
    1823: "                    'best_metric': best_psnr, # Save best_psnr metric",
    1824: "                    'optimizer': optimizer.state_dict(),",
    1825: "                }",
    1826: "                if scheduler is not None:",
    1827: "                    checkpoint_data['scheduler'] = scheduler.state_dict()",
    1828: "",
    1829: "                # 添加混合精度状态 (如果使用)",
    1830: "                if mixed_precision and scaler is not None:",
    1831: "                    checkpoint_data['scaler'] = scaler.state_dict()",
    1832: "",
    1833: "                # 保存检查点",
    1834: "                save_checkpoint(checkpoint_data, is_best, checkpoint_save_dir, epoch=(epoch+1))",
    1835: "",
    1836: "                # 如果是最佳模型，额外保存一个独立的模型权重文件，便于部署",
    1837: "                if is_best and save_best:",
    1838: "                    model_to_save_weights = model.module if hasattr(model, 'module') else model # Renamed",
    1839: "                    torch.save(model_to_save_weights.state_dict(), ",
    1840: "                             os.path.join(checkpoint_save_dir, 'best_model_weights.pth'))"
}

# 替换相应行
for line_num, new_content in problematic_section.items():
    if line_num - 1 < len(lines):
        lines[line_num - 1] = new_content + "\n"

# 写回文件
with open("/media/xxx/233-3/PycharmProjects/TRINUE/scripts/train.py", "w") as file:
    file.writelines(lines)

print("缩进修复完成") 