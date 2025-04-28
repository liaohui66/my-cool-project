# train.py
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # 导入 functional
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
import os
import argparse
from pathlib import Path
import time
import traceback # 用于打印详细错误

from sparsity_losses import sparsity_loss_pytorch # Import for the test

# 导入我们自己实现的模块
try:
    from data_utils import SENPyTorchDataset
    # --- 修改：导入顶层模型 ---
    from model import SEN_FullModel
    # --- 修改结束 ---
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure train.py is run from the project root directory (e.g., TuNN)"
          " and the modules directory exists with __init__.py.")
    exit(1)

# --- 导入稀疏性损失计算函数 ---
try:
    from sparsity_losses import sparsity_loss_pytorch
except ImportError:
    print("Warning: sparsity_losses_pytorch not found. Sparsity losses will not be calculated.")
    sparsity_loss_pytorch = None # 设置为 None 以便后续检查
print(f"PyTorch version: {torch.__version__}")
try:
    import torch_geometric
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError:
    print("Warning: PyTorch Geometric not found.")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train SEN Model (PyTorch Version)')

    # --- 路径相关参数 ---
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing data.json, eij_data.json, matscholar-embedding.json')
    parser.add_argument('--job_dir', type=str, default='./Results/pytorch_train', help='Directory for saving models, logs, and results.')
    
    # --- 设备参数 ---
    parser.add_argument('--device', type=str, default='auto', help="Device to use for training (e.g., 'cuda', 'cpu', or 'auto').")

    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=6e-5, help='Learning rate.')
    parser.add_argument('--use_lr_schedule', action='store_true', default=False, help='Use learning rate scheduling (Exponential Decay).')
    parser.add_argument('--lr_decay_steps', type=float, default=10000, help='Decay steps for learning rate schedule.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.96, help='Decay rate for learning rate schedule.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for DataLoader.')

    parser.add_argument('--target_transform', type=str, default='none',
                       choices=['log1p', 'none'],
                       help="Transformation to apply to target values before normalization (e.g., log(1+x)).")
    parser.add_argument('--target_norm_method', type=str, default='none',
                        choices=['mean_std', 'min_max', 'max', 'none'],
                        help='Normalization/Standardization method applied AFTER transformation.')

    # --- 损失权重 ---
    parser.add_argument('--mae_weight', type=float, default=1, help='Weight for MAE prediction loss.')
    parser.add_argument('--likelihood_weight', type=float, default=0, help='Weight for capsule likelihood loss (-log_prob).')
    parser.add_argument('--dynamic_l2_weight', type=float, default=0, help='Weight for dynamic weights L2 loss.')

    # 对象胶囊后验稀疏性
    parser.add_argument('--post_within_sparsity_weight', type=float, default=0, help='Weight for posterior within-example sparsity loss.')
    parser.add_argument('--post_between_sparsity_weight', type=float, default=0, help='Weight for posterior between-example sparsity loss.')
    parser.add_argument('--posterior_loss_type', type=str, default='kl', choices=['entropy', 'kl', 'l2'], help='Type of posterior sparsity loss.')

    # 对象胶囊先验稀疏性
    parser.add_argument('--prior_within_sparsity_weight', type=float, default=0, help='Weight for prior within-example sparsity loss.')
    parser.add_argument('--prior_between_sparsity_weight', type=float, default=0, help='Weight for prior between-example sparsity loss.')
    parser.add_argument('--prior_loss_type', type=str, default='kl', choices=['entropy', 'kl', 'l2'], help='Type of prior sparsity loss.')

    # 主胶囊稀疏性
    parser.add_argument('--primary_caps_sparsity_weight', type=float, default=0, help='Weight for primary capsule sparsity loss.')
    parser.add_argument('--primary_loss_type', type=str, default='l1', help='Type of primary capsule sparsity loss (e.g., l1).')

    # L2 稀疏性相关 (仅当 loss_type='l2' 时使用)
    parser.add_argument('--num_classes', type=float, default=10.0, help='Number of classes used for calculating L2 sparsity constants.')
    parser.add_argument('--within_example_constant', type=float, default=0.0, help='Target constant for L2 within-example sparsity (default: num_caps/num_classes).')

    # 权重衰减
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty) coefficient for the optimizer (AdamW recommended).')

    # --- 数据处理参数 ---
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff radius.')
    parser.add_argument('--gaussian_width', type=float, default=0.5, help='Gaussian expansion width.')
    parser.add_argument('--n_gaussian_centers', type=int, default=100, help='Number of Gaussian centers.')
    parser.add_argument('--max_gaussian_dist', type=float, default=5.0, help='Max distance for Gaussians.')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Train/Validation/Test split ratios.')
    parser.add_argument('--random_seed', type=int, default=66, help='Random seed.')
    parser.add_argument('--target_norm', type=str, default='max', choices=['max', 'mean_std', 'none'],
                        help='Target normalization.')

    # --- 模型通用参数 ---
    parser.add_argument('--atom_vocab_size', type=int, default=103, help='Size of atom vocabulary (for one-hot).') # 从 StoiRep 获取
    parser.add_argument('--num_bond_features', type=int, default=100, help='Dimension of Gaussian bond features.') # 匹配 n_gaussian_centers
    parser.add_argument('--atom_emb_dim', type=int, default=80, help='Dimension of atom embedding.')
    parser.add_argument('--state_emb_dim', type=int, default=64, help='Dimension of state embedding.')
    parser.add_argument('--activation_name', type=str, default='selu', help='Activation function name.')

    # --- MatCheCon 参数 ---
    parser.add_argument('--mc_initial_mlp', type=int, nargs='+', default=[64, 192], help='MatCheCon initial MLP hidden dims.')
    parser.add_argument('--mc_n_loop', type=int, default=3, help='MatCheCon number of loops.')
    parser.add_argument('--mc_tf_hidden_v', type=int, nargs='+', default=[128, 128], help='MatCheCon TransformerLayer V MLP hidden dims.')
    parser.add_argument('--mc_tf_hidden_e', type=int, nargs='+', default=[128, 128], help='MatCheCon TransformerLayer E MLP hidden dims.')
    parser.add_argument('--mc_tf_hidden_u', type=int, nargs='+', default=[128, 128], help='MatCheCon TransformerLayer U MLP hidden dims.')
    # (可以添加 StoiRep 的所有参数作为命令行参数)
    parser.add_argument('--mc_pool_method', type=str, default='mean', choices=['mean', 'sum'], help='MatCheCon aggregation method.')
    parser.add_argument('--mc_residual_alpha', type=float, default=0.5, help='MatCheCon residual connection alpha.')
    parser.add_argument('--mc_set_tf_hidden', type=int, default=128, help='MatCheCon SetTransformer hidden dim.')
    parser.add_argument('--mc_final_output_dim', type=int, default=256, help='MatCheCon final node feature output dim.')

    # --- Capsule/Autoencoder 参数 ---
    # (可以添加 CNNEncoder, PrimaryCapsule, CapsuleLayer, CapsuleLikelihood 的所有参数)
    parser.add_argument('--cap_n_part_caps', type=int, default=16, help='Number of primary capsules.')
    parser.add_argument('--cap_n_part_caps_dims', type=int, default=6, help='Primary capsule pose dimension (must be 6).')
    parser.add_argument('--cap_n_part_special_features', type=int, default=16, help='Primary capsule feature dimension.')
    # ... 添加更多 capsule 参数 ...

    args = parser.parse_args()

    # --- 参数后处理和检查 (保持不变) ---
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    if abs(sum(args.split_ratios) - 1.0) > 1e-6 or any(r < 0 for r in args.split_ratios):
        raise ValueError(f"Split ratios must sum to 1 and be non-negative: {args.split_ratios}")
    # if args.split_ratios[1] < 1e-6:
        # print("Warning: Validation split ratio is zero. Validation metrics will not be calculated.")

    args.job_dir = Path(args.job_dir)
    args.job_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {args.job_dir}")

    args.data_dir = Path(args.data_dir)
    req_files = ['data.json', 'eij_data.json', 'matscholar-embedding.json']
    for fname in req_files:
        fpath = args.data_dir / fname
        if not fpath.is_file():
             raise FileNotFoundError(f"Required data file not found: {fpath}")
    args.embedding_path = args.data_dir / 'matscholar-embedding.json'
    args.data_json_path = args.data_dir / 'data.json'
    args.target_json_path = args.data_dir / 'eij_data.json'

    return args

def main(args):
    """主训练和评估函数"""

    start_time = time.time()

    # --- 1. 设置设备 ---
    device = torch.device(args.device)
    print(f"Training on device: {device}")

    # --- 2. 加载数据  ---
    print("\n--- Loading Data ---")
    train_loader = None
    val_loader = None
    train_dataset = None
    val_dataset = None

    try:
        # --- 创建训练集 ---
        train_dataset = SENPyTorchDataset(
            data_json_path=str(args.data_json_path), target_json_path=str(args.target_json_path),
            embedding_path=str(args.embedding_path), cutoff=args.cutoff, gaussian_width=args.gaussian_width,
            n_gaussian_centers=args.n_gaussian_centers, max_gaussian_dist=args.max_gaussian_dist,
            split_ratios=args.split_ratios, random_seed=args.random_seed,
            phase='train',
            target_transform=args.target_transform, # <--- 传递变换方法
            target_norm=args.target_norm_method      # <--- 传递标准化方法
        )
        # 检查训练集是否成功创建
        if train_dataset is None or len(train_dataset) == 0:
            raise ValueError("Training dataset creation failed or is empty.")

        # --- 创建验证集 (如果比例 > 0) ---
        if args.split_ratios[1] > 1e-6:
            val_dataset = SENPyTorchDataset(
                data_json_path=str(args.data_json_path), target_json_path=str(args.target_json_path),
                embedding_path=str(args.embedding_path), cutoff=args.cutoff, gaussian_width=args.gaussian_width,
                n_gaussian_centers=args.n_gaussian_centers, max_gaussian_dist=args.max_gaussian_dist,
                split_ratios=args.split_ratios, random_seed=args.random_seed,
                phase='val',
                target_transform=args.target_transform, # <--- 传递变换方法
                target_norm=args.target_norm_method      # <--- 传递标准化方法
            )
            if val_dataset is None or len(val_dataset) == 0:
                print("Warning: Validation dataset creation failed or is empty, though ratio > 0.")
                val_dataset = None # 重置为 None
            else:
                print("Passing normalization stats (post-transformation) from train to val dataset...")
                val_dataset.set_normalization_stats(
            mean=train_dataset.target_mean, # <-- 使用 target_mean
            std=train_dataset.target_std,   # <-- 使用 target_std
            min_val=train_dataset.target_min, # <-- 使用 target_min
            max_val=train_dataset.target_max  # <-- 使用 target_max (移除 original_max)
        )
                print(f"Validation dataset created with {len(val_dataset)} samples.")
        else:
            print("Validation dataset not created due to split ratio being zero.")

        # --- 创建 DataLoader ---
        train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

        # val_loader 在 try 块外已初始化为 None
        if val_dataset: # 直接检查 val_dataset 是否成功创建
            val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
            print(f"Validation loader created with {len(val_loader)} batches.")

        # 打印训练加载器信息
        print(f"Train loader: {len(train_loader)} batches.")

    except Exception as e:
        print(f"ERROR during data loading or DataLoader creation: {e}")
        traceback.print_exc(); exit(1)

    # --- 确保 train_loader 已创建 ---
    if train_loader is None:
        print("ERROR: Train loader was not created successfully. Exiting.")
        exit(1)

    # --- 3. 创建模型 ---
    print("\n--- Creating Model ---")
    # 构建配置字典
    matchecon_config = {
        'raw_atom_emb_dim': args.atom_emb_dim,
        'raw_bond_emb_dim': args.num_bond_features,
        'raw_state_emb_dim': args.state_emb_dim,
        'initial_mlp_units': args.mc_initial_mlp,
        'n_loop': args.mc_n_loop,
        'tf_hidden_units_v': args.mc_tf_hidden_v,
        'tf_hidden_units_e': args.mc_tf_hidden_e,
        'tf_hidden_units_u': args.mc_tf_hidden_u,
        'stoi_config': {
             "n_target": 1, # StoiRepTorch 需要 n_target，即使可能不用
             "elem_emb_len": args.stoi_elem_emb_len if hasattr(args, 'stoi_elem_emb_len') else 128, # 线性嵌入输出维度，需要从 args 获取或设默认值
             "vocab_size": args.atom_vocab_size, # 需要词汇大小 (one-hot维度)
             # --- 添加 DescriptorNetworkTorch 需要的参数 ---
             "elem_fea_len": args.stoi_elem_fea_len if hasattr(args, 'stoi_elem_fea_len') else 64, # 从 args 或默认
             "n_graph": args.stoi_n_graph if hasattr(args, 'stoi_n_graph') else 3,
             "elem_heads": args.stoi_elem_heads if hasattr(args, 'stoi_elem_heads') else 3,
             "elem_gate_hidden": args.stoi_elem_gate_hidden if hasattr(args, 'stoi_elem_gate_hidden') else [128],
             "elem_msg_hidden": args.stoi_elem_msg_hidden if hasattr(args, 'stoi_elem_msg_hidden') else [128],
             "cry_heads": args.stoi_cry_heads if hasattr(args, 'stoi_cry_heads') else 3,
             "cry_gate_hidden": args.stoi_cry_gate_hidden if hasattr(args, 'stoi_cry_gate_hidden') else [128],
             "cry_msg_hidden": args.stoi_cry_msg_hidden if hasattr(args, 'stoi_cry_msg_hidden') else [128],
        },
        'pool_method': args.mc_pool_method,
        'residual_alpha': args.mc_residual_alpha,
        'set_transformer_hidden': args.mc_set_tf_hidden,
        'set_transformer_loop': args.mc_n_loop, # 假设 loop 次数一致
        'final_output_dim': args.mc_final_output_dim,
        'activation_name': args.activation_name,
    }

    # --- [修改] 分别构建 pred_capsule_config 和 likelihood_capsule_config ---
    pred_capsule_config = {
        'cnn_output_channels': args.cnn_output_channels if hasattr(args, 'cnn_output_channels') else [64, 64], # 从 args 获取或默认
        'cnn_kernel_size': args.cnn_kernel_size if hasattr(args, 'cnn_kernel_size') else 3,
        'cnn_strides': args.cnn_strides if hasattr(args, 'cnn_strides') else [1, 1],
        'cnn_padding': args.cnn_padding if hasattr(args, 'cnn_padding') else 'same',
        'cnn_activate_final': args.cnn_activate_final if hasattr(args, 'cnn_activate_final') else True,
        'n_part_caps': args.cap_n_part_caps,
        'n_part_caps_dims': args.cap_n_part_caps_dims,
        'n_part_special_features': args.cap_n_part_special_features,
        'primary_encoder_mlp_hiddens': args.primary_encoder_mlp_hiddens if hasattr(args, 'primary_encoder_mlp_hiddens') else [64], # 从 args 获取或默认
        # ... 添加 MaterialAutoencoderTorch 的其他参数 (pose conv, final mlp) ...
        'final_mlp_hidden': args.cap_final_mlp_hidden if hasattr(args, 'cap_final_mlp_hidden') else 64, # 从 args 获取或默认
        # 确保 pred_capsule_config 包含 SEN_FullModel 内部实例化 CNNEncoderTorch,
        # PrimaryCapsuleTorch, MaterialAutoencoderTorch 所需的所有键
    }

    likelihood_capsule_config = {
         'obj_encoder_hidden': args.cap_obj_encoder_hidden if hasattr(args, 'cap_obj_encoder_hidden') else 64, # 从 args 获取或默认
         'obj_encoder_loop': args.cap_obj_encoder_loop if hasattr(args, 'cap_obj_encoder_loop') else 2,
         'n_obj_caps': args.cap_n_obj_caps if hasattr(args, 'cap_n_obj_caps') else 8,
         'n_obj_caps_dims': 6, # 强制为 6
         'n_obj_votes': args.cap_n_part_caps, # 假设与 n_part_caps 相同
         'n_obj_caps_params_dim': args.cap_n_obj_caps_params_dim if hasattr(args, 'cap_n_obj_caps_params_dim') else 32,
         'pdf': args.cap_pdf if hasattr(args, 'cap_pdf') else 'normal',
         # ... 添加 CapsuleLayerTorch 和 CapsuleLikelihoodTorch 需要的其他参数 ...
         # 例如: 'caps_layer_n_hiddens': [...], 'learn_vote_scale': True, ...
    }

    # 实例化顶层模型
    try:
        model = SEN_FullModel(
            atom_vocab_size=args.atom_vocab_size,
            num_bond_features=args.num_bond_features,
            atom_emb_dim=args.atom_emb_dim,
            state_emb_dim=args.state_emb_dim,
            activation_name=args.activation_name,
            matchecon_config=matchecon_config,
            pred_capsule_config=pred_capsule_config,
            likelihood_capsule_config=likelihood_capsule_config
        ).to(device)
        print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
        # print(model) # 可以取消注释打印模型结构
    except Exception as e:
        print(f"ERROR during model instantiation: {e}")
        traceback.print_exc()
        exit(1)

    # --- [修改] 获取命令行解析出的稀疏性损失权重和类型 ---
    mae_weight = args.mae_weight
    likelihood_weight = args.likelihood_weight
    dynamic_l2_weight = args.dynamic_l2_weight
    post_within_sparsity_weight = args.post_within_sparsity_weight
    post_between_sparsity_weight = args.post_between_sparsity_weight
    posterior_loss_type = args.posterior_loss_type
    prior_within_sparsity_weight = args.prior_within_sparsity_weight
    prior_between_sparsity_weight = args.prior_between_sparsity_weight
    prior_loss_type = args.prior_loss_type
    primary_caps_sparsity_weight = args.primary_caps_sparsity_weight
    primary_loss_type = args.primary_loss_type
    num_classes_for_l2 = args.num_classes # 重命名以便区分
    within_const_for_l2 = args.within_example_constant

    # --- 4. 定义优化器 (正常训练使用) ---
    print("\n--- Setting up Optimizer and Loss for Full Training ---")
    # --- [修改] 固定使用 AdamW 优化器 ---
    print(f"Using AdamW optimizer with lr={args.lr} and weight_decay={args.weight_decay}") # 更新打印信息
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)

    # --- 5. 定义学习率调度器 (保持不变) ---
    scheduler = None
    if args.use_lr_schedule:
        lr_lambda = lambda step: args.lr_decay_rate**(step / args.lr_decay_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print("Using learning rate schedule.")


    # --- 6. 训练和验证循环 (修正) ---
    print("\n--- Starting Training ---")
    best_val_mae = float('inf')

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_total_loss_accum = 0.0
        train_mae_accum = 0.0
        train_like_loss_accum = 0.0
        train_dyn_l2_accum = 0.0
        num_train_samples = 0

        # --- 使用 enumerate 获取 batch_idx ---
        for batch_idx, batch in enumerate(train_loader):

            # --- 添加空批次检查 ---
            if batch.num_graphs == 0:
                print(f"Warning: Skipping empty batch at epoch {epoch+1}, batch_idx {batch_idx}.")
                continue # 跳过这个批次

            batch = batch.to(device)
            optimizer.zero_grad()

            # --- 前向传播 (修正) ---
            # 假设 model.forward 返回字典: {'prediction': ..., 'log_prob': ..., 'dynamic_l2': ...}
            try:
                model_output = model(batch) # 直接传递 PyG Data 对象
                predictions = model_output['prediction'] # [B, 1]
                log_prob = model_output['log_prob'] # scalar (batch avg log likelihood)
                dynamic_l2 = model_output['dynamic_l2'] # scalar
                object_caps_presence_prob = model_output.get('object_caps_presence_prob')
                primary_caps_presence_agg = model_output.get('primary_caps_presence_agg')
                primary_caps_feature_agg_norm = model_output.get('primary_caps_feature_agg_norm')
            except Exception as e:
                print(f"\nERROR during model forward pass in training loop: {e}")
                print("Skipping this batch.")
                traceback.print_exc()
                continue

            targets = batch.y.view(-1, 1)

            # --- 检查必要的输出是否存在 ---
            if predictions is None or log_prob is None or dynamic_l2 is None:
                print(f"Warning: Missing essential outputs. Skipping loss for this batch.")
                continue
            if predictions.shape[0] != targets.shape[0] or predictions.dim() != 2 or targets.dim() != 2:
                 print(f"Warning: Shape mismatch. Skipping loss calculation.")
                 continue

            # --- 计算完整损失 ---
            # --- 1. 主损失 (MAE) ---
            prediction_loss = F.l1_loss(predictions, targets)
            # print(f"    DEBUG (Train Loop): prediction_loss requires_grad = {prediction_loss.requires_grad}, grad_fn = {prediction_loss.grad_fn is not None}")

            # --- 2. 基础正则化 ---
            # 如果 log_prob 是标量(批次平均), 直接用; 如果是 [B], 需要取 mean
            if log_prob.dim() == 0:
                 likelihood_loss = -log_prob
            else:
                 likelihood_loss = -torch.mean(log_prob) # 如果模型返回的是 [B]
            # 同样检查 dynamic_l2
            dynamic_l2 = model_output.get('dynamic_l2')
            if dynamic_l2 is not None:
                # print(f"    DEBUG (Train Loop): dynamic_l2 from model_output = {dynamic_l2.item():.4e} (requires_grad={dynamic_l2.requires_grad})")
                dynamic_l2_loss = dynamic_l2.mean() if dynamic_l2.dim() > 0 else dynamic_l2
            else:
                # print("    DEBUG (Train Loop): dynamic_l2 from model_output is None")
                dynamic_l2_loss = torch.tensor(0.0, device=device)
            
            # --- 3. 计算稀疏性损失 (初始化为 0) ---
            post_within_loss = torch.tensor(0.0, device=device)
            post_between_loss = torch.tensor(0.0, device=device)
            prior_within_loss = torch.tensor(0.0, device=device)
            prior_between_loss = torch.tensor(0.0, device=device)
            primary_sparsity_loss = torch.tensor(0.0, device=device) # 重命名

                        # --- 检查稀疏性损失函数和所需数据是否存在 ---
            if sparsity_loss_pytorch is not None:
                # 计算对象胶囊后验稀疏性
                if object_caps_presence_prob is not None and object_caps_presence_prob.numel() > 0:
                    try:
                        # 传递从 args 获取的参数
                        post_within_loss, post_between_loss = sparsity_loss_pytorch(
                            posterior_loss_type, object_caps_presence_prob,
                            num_classes=num_classes_for_l2, # 使用获取的参数
                            within_example_constant=within_const_for_l2
                        )
                        # L2 between loss 需要取反才是惩罚项
                        if posterior_loss_type == 'l2': post_between_loss = -post_between_loss
                    except Exception as e: print(f"Error calculating posterior sparsity: {e}")

                # 计算先验稀疏性 (示例: 作用于对象胶囊概率)
                # !! 确认先验稀疏性作用的对象 !!
                prior_input_prob = object_caps_presence_prob # 或 primary_caps_presence_agg
                if prior_input_prob is not None and prior_input_prob.numel() > 0:
                    try:
                        prior_within_loss, prior_between_loss = sparsity_loss_pytorch(
                            prior_loss_type, prior_input_prob,
                            num_classes=num_classes_for_l2,
                            within_example_constant=within_const_for_l2
                        )
                        if prior_loss_type == 'l2': prior_between_loss = -prior_between_loss
                    except Exception as e: print(f"Error calculating prior sparsity: {e}")

                # 计算主胶囊稀疏性
                if primary_loss_type == 'l1':
                    if primary_caps_feature_agg_norm is not None and primary_caps_feature_agg_norm.numel() > 0:
                        try:
                            # 计算批次平均 L1 范数和
                            primary_sparsity_loss = torch.mean(torch.sum(torch.abs(primary_caps_feature_agg_norm), dim=-1))
                        except Exception as e: print(f"Error calculating primary L1 sparsity: {e}")
                # else: # 可以添加对其他 primary_loss_type 的处理，例如使用 sparsity_loss_pytorch

            # print(f"DEBUG: dynamic_l2_weight = {dynamic_l2_weight}")
            # --- 4. 加权求和得到总损失 ---
            total_loss = (mae_weight * prediction_loss +
                          likelihood_weight * likelihood_loss +
                          dynamic_l2_weight * dynamic_l2_loss +
                          post_within_sparsity_weight * post_within_loss +
                          post_between_sparsity_weight * post_between_loss +
                          prior_within_sparsity_weight * prior_within_loss +
                          prior_between_sparsity_weight * prior_between_loss +
                          primary_caps_sparsity_weight * primary_sparsity_loss
                         )

            # print(f"    DEBUG (Train Loop): Calculated total_loss = {total_loss.item():.4e} (requires_grad={total_loss.requires_grad})") # 打印总损失信息

            # --- 记录损失 (根据需要添加新的损失项的累加) ---
            current_batch_size = batch.num_graphs # 或 predictions.shape[0]
            train_total_loss_accum += total_loss.item() * current_batch_size
            train_mae_accum += prediction_loss.item() * current_batch_size
            train_like_loss_accum += likelihood_loss.item() * current_batch_size
            train_dyn_l2_accum += dynamic_l2_loss.item() * current_batch_size
            # 例如: train_post_within_accum += post_within_loss.item() * current_batch_size
            num_train_samples += current_batch_size

      
# --- 反向传播和优化 ---
        try:
            total_loss.backward()  # 计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 保留梯度裁剪

            if hasattr(model, 'matchecon_module') and \
                hasattr(model.matchecon_module, 'transformer_layer') and \
                hasattr(model.matchecon_module.transformer_layer, 'phi_v_mlp'):
                    if hasattr(model.matchecon_module.transformer_layer.phi_v_mlp, 'layers') and \
                    isinstance(model.matchecon_module.transformer_layer.phi_v_mlp.layers, nn.ModuleList) and \
                    len(model.matchecon_module.transformer_layer.phi_v_mlp.layers) > 0:
                        first_linear_layer = model.matchecon_module.transformer_layer.phi_v_mlp.layers[0]
                        if isinstance(first_linear_layer, nn.Linear) and hasattr(first_linear_layer, 'weight'):
                            grad_phi_v = first_linear_layer.weight.grad

                            pass

            final_node_vec_check = model_output.get('final_node_vec')
            if final_node_vec_check is not None and hasattr(final_node_vec_check, 'grad'):
                grad_fnv = final_node_vec_check.grad
                if grad_fnv is not None:

                    if torch.isnan(grad_fnv).any() or torch.isinf(grad_fnv).any():
                        print("  WARNING: NaN/Inf detected in final_node_vec gradient!")

            if hasattr(model, 'prediction_module') and hasattr(model.prediction_module, 'final_mlp'):
                final_mlp_layer1 = model.prediction_module.final_mlp[0]
                final_mlp_layer3 = model.prediction_module.final_mlp[2]

                if hasattr(final_mlp_layer1, 'weight'):
                    grad1 = final_mlp_layer1.weight.grad

                    pass 

                if hasattr(final_mlp_layer3, 'weight'):
                    grad3 = final_mlp_layer3.weight.grad

                    pass

            optimizer.step()  # 使用计算出的梯度更新参数

        except Exception as e:
            print(f"\nERROR during backward/optimizer step: {e}")
            print("Skipping this batch.")
            traceback.print_exc()
            continue

        # --- 计算并打印 Epoch 平均损失 ---
        if num_train_samples > 0:
            avg_train_total_loss = train_total_loss_accum / num_train_samples
            avg_train_mae = train_mae_accum / num_train_samples
            avg_train_like_loss = train_like_loss_accum / num_train_samples
            avg_train_dyn_l2 = train_dyn_l2_accum / num_train_samples
            print(f"Epoch {epoch+1}/{args.epochs} | Train Total Loss: {avg_train_total_loss:.6f} | MAE: {avg_train_mae:.6f} | NLL: {avg_train_like_loss:.6f} | DynL2: {avg_train_dyn_l2:.6f}", end='')
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | No training samples processed.", end='')


        # --- (可选) 更新学习率 (按 epoch, 如果用 StepLR 等) ---
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.LambdaLR):
             scheduler.step()

        avg_val_mae_orig_scale = float('inf') # 变量名明确表示原始尺度
        if val_loader:
            model.eval()
            val_mae_accum_original_scale = 0.0 # 累加原始尺度的绝对误差和
            num_val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    try:
                        model_output = model(batch)
                        predictions_final = model_output.get('prediction') # 模型输出是最终处理后尺度
                        if predictions_final is None: continue

                        # 获取原始目标值用于评估
                        targets_raw = batch.y_raw.view(-1, 1) # <--- 使用 batch.y_raw

                        # --- [关键修改] 调用 denormalize 将预测反算回原始尺度 ---
                        # 使用 train_dataset 或 val_dataset 都可以，它们现在有相同的统计量
                        predictions_original_scale = train_dataset.denormalize(predictions_final)
                        # --- 反算结束 ---

                        if predictions_original_scale.shape[0] != targets_raw.shape[0]: continue

                        # 在原始尺度上计算 MAE
                        mae_original_scale = F.l1_loss(predictions_original_scale, targets_raw)
                        val_mae_accum_original_scale += mae_original_scale.item() * batch.num_graphs # 累加原始尺度 MAE * batch大小
                        num_val_samples += batch.num_graphs
                    except Exception as e:
                        print(f"\nERROR during validation forward pass: {e}")
                        continue # 跳过这个验证批次

            if num_val_samples > 0:
                # 计算原始尺度上的平均验证 MAE
                avg_val_mae_orig_scale = val_mae_accum_original_scale / num_val_samples
                print(f" | Val MAE (Original Scale): {avg_val_mae_orig_scale:.6f}", end='') # 明确是原始尺度

                # 保存最佳模型 (基于原始尺度的验证 MAE)
                if avg_val_mae_orig_scale < best_val_mae:
                    best_val_mae = avg_val_mae_orig_scale
                    save_path = args.job_dir / 'best_model.pth'
                    # --- [推荐] 保存模型和变换/归一化参数 ---
                    save_content = {
                    'model_state_dict': model.state_dict(),
                    'target_transform': train_dataset.target_transform_method,
                    'target_norm_method': train_dataset.target_norm_method,
                    'target_mean': train_dataset.target_mean, # 保存原始均值
                    'target_std': train_dataset.target_std,   # 保存原始标准差
                    'target_min': train_dataset.target_min,   # 保存原始最小值
                    'target_max': train_dataset.target_max,   # 保存原始最大值
                    'best_val_mae': best_val_mae,
                    'epoch': epoch + 1
                }
                    torch.save(save_content, save_path)
                    # --- 保存结束 ---
                    print(" | Model & Stats Saved.", end='')
            else:
                 print(" | No validation samples processed.", end='')

        epoch_time = time.time() - epoch_start_time
        print(f" | Time: {epoch_time:.2f}s")

    # --- 训练结束 ---
    print("\n--- Training Finished ---")
    print(f"Best Validation MAE: {best_val_mae:.6f}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")


if __name__ == '__main__':
    args = parse_args() 
    main(args)
