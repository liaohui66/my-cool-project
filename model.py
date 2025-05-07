# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as PyGData
from typing import List, Optional, Tuple, Dict # <--- 添加 Dict
import traceback
from torch_geometric.utils import scatter

# 导入我们已经创建的子模块
try:
    from repr.matchecon import SE3InvariantGraphEncoder
    from repr.cnn import CNNEncoderTorch
    from geometry.primary_caps import PrimaryCapsuleTorchMLP
    # from repr.matchecon import MatCheConTorch
    from repr.mlp import MLP
    from geometry.geo_torch import geometric_transform_torch, safe_log_torch
    from repr.batch_mlp_torch import BatchMLPTorch
    # --- 新增导入 ---
    from repr.set_transformer import SetTransformerTorch # 用于 obj_encoder
    from geometry.capsule_layer import CapsuleLayerTorch # 用于 obj_capsule_layer
    from geometry.capsule_likelihood import CapsuleLikelihoodTorch, CapsuleLikelihoodOutputTuple # 用于 obj_capsule_likelihood
    # --- 新增结束 ---
except ImportError as e:
    print(f"Error importing modules for model.py: {e}")
    raise


# [--- 第一部分：定义 MaterialAutoencoderTorch (模型的第二阶段) ---]
class MaterialAutoencoderTorch(nn.Module):
    """PyTorch 版 MaterialAutoencoder"""
    def __init__(self,
                 primary_encoder: PrimaryCapsuleTorchMLP, # 接收实例化的 Primary Encoder
                 n_caps: int,
                 n_features: int,
                 encoder_output_dim: int,
                 # Pose 处理 Conv1D 参数
                 # pose_conv1_out_channels: int = 9,
                 # pose_conv1_kernel_size: int = 2,
                 # pose_conv1_stride: int = 1,
                 # pose_conv2_out_channels: int = 1,
                 # pose_conv2_kernel_size: int = 2,
                 # pose_conv2_stride: int = 1,
                 # 最终 MLP 参数
                 final_mlp_hidden_dim: int = 64,
                 final_mlp_output_dim: int = 1,
                 # final_activation: nn.Module = nn.ReLU(),
                 gnn_output_dim: int = 256 # 需要知道 GNN 输出的维度
                 ):
        super().__init__()
        self.primary_encoder = primary_encoder
        self.n_caps = n_caps
        self.n_features = n_features
        # self._pose_dim = 6 # 假设 Primary Encoder 输出的 pose 维度固定为 6

        # 创建本模块独有的层
        # self.pose_conv1 = nn.Conv1d(in_channels=self._pose_dim,
        #                             out_channels=pose_conv1_out_channels,
        #                             kernel_size=pose_conv1_kernel_size, stride=pose_conv1_stride, padding='valid')
        # self.pose_conv2 = nn.Conv1d(in_channels=pose_conv1_out_channels,
        #                             out_channels=pose_conv2_out_channels,
        #                             kernel_size=pose_conv2_kernel_size, stride=pose_conv2_stride, padding='valid')
        # pose_fea_input_dim = pose_conv2_out_channels + self.n_features
        # self.linear_pose_pre = nn.Linear(pose_fea_input_dim, 1)
        # final_mlp_input_dim = 2 # 假设聚合后是 2 维
        # self.final_mlp = nn.Sequential(
        #     nn.Linear(final_mlp_input_dim, final_mlp_hidden_dim, bias=True), # 使用明确维度
        #     nn.SELU(),
        #     nn.Linear(final_mlp_hidden_dim, final_mlp_output_dim, bias=True),
        #     final_activation
        # )
        if self.n_features > 0:
            # 输入维度现在只是 n_features
            self.linear_pose_pre = nn.Linear(self.n_features, 1)
        else:
            # 如果没有 feature，这个线性层可能不需要，或者需要特殊处理
            self.linear_pose_pre = None

        final_mlp_input_dim = encoder_output_dim # 直接使用 GNN 的图级表示
        print(f"  MaterialAutoencoderTorch (Simplified): final_mlp input dim = {final_mlp_input_dim}")

        self.final_mlp = nn.Sequential(
            nn.Linear(final_mlp_input_dim, final_mlp_hidden_dim),
            nn.SELU(), # 保持和之前一致
            nn.Linear(final_mlp_hidden_dim, final_mlp_output_dim),
            # 最后一层通常不加激活，或者用与目标范围匹配的激活
            # final_activation # 暂时移除最后的 ReLU/激活
        )
        self._final_output_dim_pred = final_mlp_output_dim

    def forward(self, node_invariant_features: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_invariant_features (torch.Tensor): 来自 SE3InvariantGraphEncoder 的节点不变特征 [N, encoder_output_dim]。
            batch (Optional[torch.Tensor]): 节点到图的映射 [N]。
        Returns:
            torch.Tensor: 最终预测值，形状 [B, 1]。
        """
        # 1. 调用 Primary Encoder (现在输入是 node_invariant_features)
        try:
             _ = self.primary_encoder(node_invariant_features)
        except Exception as e:
             print(f"   Warning: Calling primary_encoder in MaterialAutoencoderTorch failed: {e}. Skipping prediction.")
             num_graphs = int(batch.max().item() + 1) if batch is not None and batch.numel() > 0 else 1
             # 返回形状匹配的零张量
             return torch.zeros((num_graphs, self._final_output_dim_pred), device=node_invariant_features.device, dtype=node_invariant_features.dtype)

        # 2. 图级别聚合 (聚合不变特征)
        if batch is None:
            # Handle case with no batch index (single graph)
            if node_invariant_features.shape[0] > 0:
                graph_level_features = torch.mean(node_invariant_features, dim=0, keepdim=True)
            else: # Handle empty graph case
                return torch.zeros((1, self._final_output_dim_pred), device=node_invariant_features.device, dtype=node_invariant_features.dtype)
        else:
            # Ensure batch tensor is long type for scatter
            graph_level_features = scatter(node_invariant_features, batch.long(), dim=0, reduce='mean')
            # Handle case where scatter might return empty tensor if input is empty
            if graph_level_features.shape[0] == 0 and node_invariant_features.shape[0] == 0:
                 # If input was empty, return empty tensor with correct last dim
                 num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
                 return torch.zeros((num_graphs, self._final_output_dim_pred), device=node_invariant_features.device, dtype=node_invariant_features.dtype)

        # 3. 最终预测 MLP (使用聚合后的不变特征)
        prediction = self.final_mlp(graph_level_features)

        return prediction



# [--- 第二部分：定义 SEN_FullModel (顶层模型) ---]
class SEN_FullModel(nn.Module):
    """
    完整的 SEN 模型 PyTorch 实现。
    整合了 SE3InvariantGraphEncoder 和 Capsule 网络部分。
    """
    def __init__(self,
                 # --- 新增：SE3InvariantGraphEncoder 参数字典 ---
                 equivariant_gnn_config: dict, # 不再有默认值，必须从 train.py 传入
                 # --- 预测路径 Primary Capsule (MLP 版本) 参数字典 ---
                 pred_primary_capsule_config: dict = {}, # 保持原样
                 # --- 似然路径 Capsule 参数字典 ---
                 likelihood_capsule_config: dict = {}, # 保持原样
                 # --- 其他通用参数 ---
                 activation_name: str = 'selu',
                 # --- 移除旧 GNN 和 Embedding 相关参数 ---
                 # atom_vocab_size: int = 100,       # 移除 (在 gnn_config 中处理)
                 # num_bond_features: int = 100,    # 移除 (与旧 GNN 相关)
                 # atom_emb_dim: int = 80,          # 移除 (在 gnn_config 中处理)
                 # state_emb_dim: int = 64,         # 移除 (旧 GNN 状态)
                 # matchecon_config: dict = {},     # 移除
                 ):

        super().__init__()

        # 1. 创建激活函数 (保持不变)
        if activation_name.lower() == 'selu':
            self.activation = nn.SELU()
        elif activation_name.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            try:
                self.activation = getattr(nn, activation_name)()
            except AttributeError:
                raise ValueError(f"Unknown activation function: {activation_name}")
        print(f"SEN_FullModel using activation: {self.activation}")

        # 2. 移除旧的 Embedding 层实例化
        # self.atom_embedding = nn.Embedding(atom_vocab_size, atom_emb_dim) # 移除
        # self.state_embedding = nn.Embedding(1, state_emb_dim)             # 移除

        # 3. 实例化 SE3InvariantGraphEncoder
        #    直接使用传入的配置字典 equivariant_gnn_config
        print("Initializing SE3InvariantGraphEncoder with config:", equivariant_gnn_config)
        try:
            self.encoder_module = SE3InvariantGraphEncoder(**equivariant_gnn_config)
        except Exception as e:
             print("ERROR: Failed to initialize SE3InvariantGraphEncoder.")
             print("Ensure 'equivariant_gnn_config' contains all required arguments:")
             print("num_atom_types, embedding_dim_scalar, irreps_node_hidden, irreps_node_output, "
                   "irreps_edge_attr, irreps_sh, max_radius, num_basis_radial, radial_mlp_hidden, "
                   "num_interaction_layers, num_attn_heads, use_attention, activation_gate")
             raise e
        #    获取编码器输出的节点不变特征维度
        self.encoder_output_dim = self.encoder_module.final_scalar_dim
        print(f"SE3InvariantGraphEncoder initialized. Output node feature dim: {self.encoder_output_dim}")

        # 4. 移除旧的 MatCheConTorch 实例化
        # self.matchecon_module = MatCheConTorch(...) # 移除

        # 5. 实例化预测路径模块
        #    a. 准备 PrimaryCapsuleTorchMLP 配置
        pred_caps_cfg = { # 提供合理的默认值
            'n_caps': 16, 'n_caps_dims': 6, 'n_features': 16,
            'mlp_hiddens': [128], 'noise_scale': 0.0, # 默认关闭噪声，可在 config 中开启
            'similarity_transform': False,
        }
        pred_caps_cfg.update(pred_primary_capsule_config) # 用传入的覆盖默认值

        #    b. 实例化 PrimaryCapsuleTorchMLP
        #    **关键**: 必须传入 input_dim=self.encoder_output_dim
        try:
            self.pred_primary_encoder = PrimaryCapsuleTorchMLP(
                input_dim=self.encoder_output_dim, # <--- 传递编码器输出维度
                n_caps=pred_caps_cfg['n_caps'],
                n_caps_dims=pred_caps_cfg['n_caps_dims'],
                n_features=pred_caps_cfg['n_features'],
                mlp_hiddens=pred_caps_cfg['mlp_hiddens'],
                noise_scale=pred_caps_cfg['noise_scale'],
                similarity_transform=pred_caps_cfg['similarity_transform'],
                activation=self.activation
            )
            print("PrimaryCapsuleTorchMLP initialized.")
        except TypeError as e:
             # 捕获错误，提示用户修改 PrimaryCapsuleTorchMLP
             if 'input_dim' in str(e):
                 print("\nCRITICAL ERROR: PrimaryCapsuleTorchMLP.__init__() does not accept 'input_dim'.")
                 print("Please modify the 'PrimaryCapsuleTorchMLP' class in 'geometry/primary_caps.py':")
                 print("  1. Add 'input_dim: int' to the __init__ arguments.")
                 print("  2. Use this 'input_dim' when defining the first nn.Linear layer in 'self.mlp_replacement'.")
                 raise e
             else: raise e

        #    c. 实例化 MaterialAutoencoderTorch (预测头)
        self.prediction_module = MaterialAutoencoderTorch(
            primary_encoder=self.pred_primary_encoder,
            n_caps=pred_caps_cfg['n_caps'],
            n_features=pred_caps_cfg['n_features'],
            encoder_output_dim=self.encoder_output_dim, # <-- 传递编码器输出维度
            # 从配置获取 MLP 参数
            final_mlp_hidden_dim=pred_caps_cfg.get('final_mlp_hidden_dim', 64),
            final_mlp_output_dim=pred_caps_cfg.get('final_mlp_output_dim', 1)
        )
        print("MaterialAutoencoderTorch (prediction module) initialized.")

        # 6. 实例化似然路径模块 (逻辑基本不变)
        primary_feature_dim = pred_caps_cfg['n_features']
        like_cfg = likelihood_capsule_config

        # a. 对象编码器 (Set Transformer)
        self.obj_encoder = None # Initialize as None
        obj_enc_output_dim = 0
        if primary_feature_dim > 0:
             obj_enc_hidden = like_cfg.get('obj_encoder_hidden', 128)
             obj_enc_loop = like_cfg.get('obj_encoder_loop', 3)
             try: # Add try-except for robustness
                 self.obj_encoder = SetTransformerTorch(input_dim=primary_feature_dim, n_hidden=obj_enc_hidden, loop=obj_enc_loop)
                 obj_enc_output_dim = 2 * obj_enc_hidden
                 print("Likelihood Path: obj_encoder initialized.")
             except Exception as e:
                  print(f"Error initializing SetTransformerTorch: {e}")
                  self.obj_encoder = None
        else:
             print("Warning: Likelihood Path: n_features=0, obj_encoder disabled.")

        # b. 对象胶囊层 (CapsuleLayerTorch)
        self.obj_capsule_layer = None # Initialize as None
        if self.obj_encoder is not None:
             n_obj_caps = like_cfg.get('n_obj_caps', 8)
             n_obj_caps_dims = like_cfg.get('n_obj_caps_dims', 6)
             n_obj_votes = pred_caps_cfg['n_caps']
             n_obj_caps_params_dim = like_cfg.get('n_obj_caps_params_dim', 32)
             # 尝试实例化，如果 CapsuleLayerTorch 接受 input_dim
             try:
                 self.obj_capsule_layer = CapsuleLayerTorch(
                     n_caps=n_obj_caps, n_caps_dims=n_obj_caps_dims, n_votes=n_obj_votes,
                     n_caps_params=n_obj_caps_params_dim,
                     input_dim=obj_enc_output_dim, # 尝试传递 input_dim
                     activation=self.activation
                     # ... other CapsuleLayerTorch params ...
                 )
                 print("Likelihood Path: obj_capsule_layer initialized (with input_dim).")
             except TypeError as e:
                 if 'input_dim' in str(e):
                     print("Warning: CapsuleLayerTorch does not accept 'input_dim'. Assuming BatchMLP handles lazy init.")
                     # Try without input_dim if BatchMLP is lazy
                     try:
                         self.obj_capsule_layer = CapsuleLayerTorch(
                             n_caps=n_obj_caps, n_caps_dims=n_obj_caps_dims, n_votes=n_obj_votes,
                             n_caps_params=n_obj_caps_params_dim, activation=self.activation)
                         print("Likelihood Path: obj_capsule_layer initialized (without input_dim).")
                     except Exception as e2:
                         print(f"Error initializing CapsuleLayerTorch even without input_dim: {e2}")
                         self.obj_capsule_layer = None
                 else:
                     print(f"Error initializing CapsuleLayerTorch: {e}")
                     self.obj_capsule_layer = None
             except Exception as e: # Catch other potential errors
                 print(f"Error initializing CapsuleLayerTorch: {e}")
                 self.obj_capsule_layer = None

        # c. 胶囊似然层 (CapsuleLikelihoodTorch)
        self.obj_capsule_likelihood = None # Initialize as None
        if self.obj_capsule_layer is not None:
             try: # Add try-except
                 n_obj_caps = like_cfg.get('n_obj_caps', 8) # Re-get n_obj_caps
                 n_obj_caps_params_dim = like_cfg.get('n_obj_caps_params_dim', 32) # Re-get params_dim
                 self.obj_capsule_likelihood = CapsuleLikelihoodTorch(
                     raw_caps_params_dim=n_obj_caps_params_dim,
                     n_caps=n_obj_caps,
                     pdf=like_cfg.get('pdf', 'normal'),
                     use_internal_prediction=like_cfg.get('use_internal_prediction', False)
                     # ... other CapsuleLikelihoodTorch params ...
                 )
                 print("Likelihood Path: obj_capsule_likelihood initialized.")
             except Exception as e:
                  print(f"Error initializing CapsuleLikelihoodTorch: {e}")
                  self.obj_capsule_likelihood = None


# model.py -> SEN_FullModel 类内部 (修改 forward 方法)

    def forward(self, data: PyGData) -> Dict[str, Optional[torch.Tensor]]:
        """ 完整的模型前向传播，使用 SE3InvariantGraphEncoder """

        # 1. 通过 SE3InvariantGraphEncoder 获取节点不变特征
        #    Encoder 内部处理了初始嵌入
        #    输出: [N, final_scalar_dim]
        try:
            # 直接将 PyG Data 对象传递给编码器
            node_invariant_features = self.encoder_module(data)
        except Exception as e:
            print(f"\nERROR during SE3InvariantGraphEncoder forward pass: {e}")
            traceback.print_exc()
            # 返回包含错误信息的字典或空字典
            # (可以考虑返回一个包含 None 或零值的完整字典结构，以便后续代码处理)
            # return {'error': str(e)}
            # 或者返回一个空的预测等
            num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else 0
            batch_size_fallback = int(data.batch.max().item() + 1) if hasattr(data, 'batch') and data.batch is not None and data.batch.numel() > 0 else (1 if num_nodes > 0 else 0)
            device_fallback = data.pos.device if hasattr(data, 'pos') else torch.device('cpu')
            return {
                'prediction': torch.zeros((batch_size_fallback, 1), device=device_fallback), # 假设预测输出是 [B, 1]
                'log_prob': torch.tensor(0.0, device=device_fallback),
                'dynamic_l2': torch.tensor(0.0, device=device_fallback),
                'object_caps_presence_prob': None,
                'primary_caps_presence_agg': None,
                'primary_caps_feature_agg_norm': None,
                'final_node_features': None,
                'error': str(e) # 添加错误信息
            }


        # --- 后续代码现在使用 node_invariant_features ---

        # 获取 batch 索引 (如果 Encoder 没有返回，从 data 获取)
        # 需要确保 batch 索引的正确性
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(node_invariant_features.shape[0], dtype=torch.long, device=node_invariant_features.device)
        batch_size = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        current_device = node_invariant_features.device

        # (可选) 保留梯度检查点
        node_invariant_features_grad_check = None
        if node_invariant_features.requires_grad:
            node_invariant_features.retain_grad()
            node_invariant_features_grad_check = node_invariant_features

        # 2. 预测路径 (现在输入 node_invariant_features)
        try:
            prediction = self.prediction_module(node_invariant_features, batch=batch)
        except Exception as e:
             print(f"\nERROR during prediction_module forward pass: {e}")
             traceback.print_exc()
             # 预测失败时也应返回错误或零值
             prediction = torch.zeros((batch_size, 1), device=current_device) # 假设预测输出 [B, 1]

        # 3. 似然路径 和 主胶囊聚合
        log_prob = torch.tensor(0.0, device=current_device)
        dynamic_l2 = torch.tensor(0.0, device=current_device)
        object_caps_presence_prob = None
        primary_caps_presence_agg = None
        primary_caps_feature_agg_norm = None

        try:
            # --- 获取初级胶囊输出 (现在输入是 node_invariant_features) ---
            primary_caps_output = None
            primary_features = None
            primary_presence = None
            primary_pose = None
            if hasattr(self, 'pred_primary_encoder'):
                # **再次确认**: PrimaryCapsuleTorchMLP 是否已修改以接受正确的 input_dim
                primary_caps_output = self.pred_primary_encoder(node_invariant_features) # <-- 输入改变
                primary_features = primary_caps_output.feature
                primary_presence = primary_caps_output.presence
                primary_pose = primary_caps_output.pose
            else:
                print("Warning: pred_primary_encoder not found.")

            # --- 聚合主胶囊信息 (逻辑不变) ---
            part_caps_num = 0 # Calculate part_caps_num as before
            if primary_features is not None: part_caps_num = primary_features.shape[1]
            elif primary_presence is not None: part_caps_num = primary_presence.shape[1]
            elif hasattr(self, 'pred_primary_encoder'): part_caps_num = self.pred_primary_encoder._n_caps

            if batch_size > 0:
                 if primary_presence is not None:
                     # Ensure presence shape matches part_caps_num
                     if primary_presence.shape[1] != part_caps_num: print(f"Warning: presence shape[1] {primary_presence.shape[1]} != caps# {part_caps_num}")
                     primary_caps_presence_agg = scatter(primary_presence, batch, dim=0, reduce='mean', dim_size=batch_size)
                 else: primary_caps_presence_agg = torch.zeros((batch_size, part_caps_num), device=current_device)

                 if primary_features is not None:
                     if primary_features.shape[1] != part_caps_num: print(f"Warning: feature shape[1] {primary_features.shape[1]} != caps# {part_caps_num}")
                     feature_norms = torch.linalg.norm(primary_features, ord=2, dim=-1)
                     primary_caps_feature_agg_norm = scatter(feature_norms, batch, dim=0, reduce='mean', dim_size=batch_size)
                 else: primary_caps_feature_agg_norm = torch.zeros((batch_size, part_caps_num), device=current_device)
            else: # Empty batch
                 primary_caps_presence_agg = torch.empty((0, part_caps_num), device=current_device)
                 primary_caps_feature_agg_norm = torch.empty((0, part_caps_num), device=current_device)

            # --- 似然路径计算 (逻辑不变) ---
            if self.obj_encoder is not None and self.obj_capsule_layer is not None and self.obj_capsule_likelihood is not None:
                if primary_features is not None:
                    aggregated_primary_features = torch.mean(primary_features, dim=1)
                    object_encodings = self.obj_encoder(aggregated_primary_features, batch)
                    obj_caps_output = self.obj_capsule_layer(object_encodings)
                    # ... (获取 presence_prob, dynamic_l2) ...
                    object_caps_presence_prob_calc = obj_caps_output.get('presence_prob')
                    if object_caps_presence_prob_calc is not None: object_caps_presence_prob = object_caps_presence_prob_calc
                    dynamic_l2_calc = obj_caps_output.get('dynamic_weights_l2', torch.tensor(0.0, device=current_device))
                    if dynamic_l2_calc is not None : dynamic_l2 = dynamic_l2_calc

                    if primary_pose is not None:
                        x_target_for_likelihood = torch.mean(primary_pose, dim=1)
                        likelihood_output = self.obj_capsule_likelihood(
                            x=x_target_for_likelihood, caps_layer_output=obj_caps_output, batch=batch
                        )
                        log_prob_calc = likelihood_output.log_prob
                        if log_prob_calc is not None: log_prob = log_prob_calc # Use calculated value
                    else: print("Warning: Primary pose is None, skipping likelihood calculation.")
                else: print("Warning: Primary features are None, skipping likelihood path.")
            # ... (Fallback logic for likelihood path remains) ...
            else: # Fallback if likelihood path modules are None
                 # Need likelihood_capsule_config accessible here, maybe store on self?
                 like_cfg = getattr(self, 'likelihood_capsule_config', {}) # Get if exists
                 n_obj_caps_fallback = like_cfg.get('n_obj_caps', 0)
                 object_caps_presence_prob = torch.zeros((batch_size, n_obj_caps_fallback), device=current_device) if batch_size > 0 else None


        except Exception as e:
             print(f"\nERROR during capsule/likelihood path: {e}")
             traceback.print_exc()
             # Fallback values on error (can reuse logic from above)
             b_size_fallback = prediction.shape[0] if prediction is not None else batch_size
             # Need pred_primary_capsule_config and likelihood_capsule_config accessible
             pred_cfg = getattr(self, 'pred_primary_capsule_config', {})
             like_cfg = getattr(self, 'likelihood_capsule_config', {})
             part_caps_num_fallback = pred_cfg.get('n_caps', 16)
             n_obj_caps_fallback = like_cfg.get('n_obj_caps', 0)
             object_caps_presence_prob = torch.zeros((b_size_fallback, n_obj_caps_fallback), device=current_device) if b_size_fallback > 0 else None
             primary_caps_presence_agg = torch.zeros((b_size_fallback, part_caps_num_fallback), device=current_device) if b_size_fallback > 0 else None
             primary_caps_feature_agg_norm = torch.zeros((b_size_fallback, part_caps_num_fallback), device=current_device) if b_size_fallback > 0 else None


        # 4. 返回字典
        return {
            'prediction': prediction,
            'log_prob': log_prob,
            'dynamic_l2': dynamic_l2,
            'object_caps_presence_prob': object_caps_presence_prob,
            'primary_caps_presence_agg': primary_caps_presence_agg,
            'primary_caps_feature_agg_norm': primary_caps_feature_agg_norm,
            'final_node_features': node_invariant_features_grad_check # 返回节点不变特征供调试
        }
    
# --- 测试代码 (在 model.py 末尾) ---
if __name__ == '__main__':
     print("\n" + "="*30)
     print("--- Testing SEN_FullModel with Equivariant Encoder ---") # 更新标题
     print("="*30)

     # --- 1. 定义 GNN, Capsule 配置 ---
     # 定义 SE3InvariantGraphEncoder 配置
     equivariant_gnn_config_example = {
         'num_atom_types': 103,       # 替换 atom_vocab_size
         'embedding_dim_scalar': 16, # 替换 atom_emb_dim (部分)
         'irreps_node_hidden': "32x0e + 8x1o", # 简化版 Hidden Irreps
         'irreps_node_output': "64x0e",     # 最终输出标量
         'irreps_edge_attr': "0x0e",        # 假设无额外边属性
         'irreps_sh': "1x0e + 1x1o",        # lmax=1 球谐函数
         'max_radius': 5.0,
         'num_basis_radial': 16,
         'radial_mlp_hidden': [64],
         'num_interaction_layers': 2, # 减少层数以加速测试
         'num_attn_heads': 2,         # 减少头数
         'use_attention': True,       # 测试 SE(3)T 模式
         'activation_gate': True
     }

     # 保持 Capsule 配置不变 (如果它们的参数没变)
     pred_capsule_config_example = {
         'n_part_caps': 8, 'n_part_caps_dims': 6, 'n_part_special_features': 16,
         'mlp_hiddens': [64], # 简化 MLP
         'noise_scale': 0.0, # 关闭噪声
         'final_mlp_hidden_dim': 32, # 匹配 MaterialAutoencoderTorch
         'final_mlp_output_dim': 1
     }
     likelihood_capsule_config_example = {
         'obj_encoder_hidden': 64, 'obj_encoder_loop': 2,
         'n_obj_caps': 4, 'n_obj_caps_dims': 6, #'n_obj_votes': 8, # n_votes = n_part_caps
         'n_obj_caps_params_dim': 16,
         'pdf': 'normal'
     }

     # --- 2. 模拟输入数据 (保持不变) ---
     device = torch.device("cpu"); N=7; B=2; VOCAB_SIZE=103
     # 注意: Encoder 现在需要 data.x 是原子类型索引 (Long)
     mock_data_batch = PyGData(
         x=torch.randint(0, VOCAB_SIZE, (N,), device=device, dtype=torch.long), # <-- 原子类型索引
         edge_index=torch.tensor([[0, 1, 3, 4, 5, 6], [1, 0, 4, 3, 6, 5]], device=device, dtype=torch.long),
         # edge_attr 不再直接需要，除非 irreps_edge_attr 不是 "0x0e"
         # atom_fea 不再需要
         # comp_w 不再直接需要
         pos=torch.randn(N, 3, device=device), # <-- 需要位置信息
         batch=torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device, dtype=torch.long),
         y_raw=torch.randn(B, device=device) # y_raw 仍然可能用于某些旧逻辑或调试
     ).to(device)
     print("\nMock DataBatch created:"); print(mock_data_batch)


     # --- 3. 实例化 SEN_FullModel (使用新签名) ---
     print("\n--- Instantiating SEN_FullModel ---")
     try:
         model = SEN_FullModel(
             # --- 传递新的配置字典 ---
             equivariant_gnn_config=equivariant_gnn_config_example,
             pred_primary_capsule_config=pred_capsule_config_example,
             likelihood_capsule_config=likelihood_capsule_config_example,
             # --- 通用参数 ---
             activation_name='selu'
             # --- 移除旧参数 ---
         ).to(device)
         print("\nModel Instance Instantiated.")
         # print(model) # 可以取消注释查看模型结构
     except Exception as e:
         print(f"\n--- ERROR during model instantiation ---"); traceback.print_exc(); exit(1)

     # --- 4. 运行前向传播 (保持不变) ---
     print("\n--- Running Forward Pass ---")
     try:
         model.train() # 设为训练模式以测试所有路径
         output_dict = model(mock_data_batch)

         # --- 5. 打印输出形状 (保持不变) ---
         print("\n--- Forward Pass Successful ---")
         print(f"Output dictionary keys: {output_dict.keys()}")
         pred = output_dict.get('prediction')
         logp = output_dict.get('log_prob')
         dynl2 = output_dict.get('dynamic_l2')
         node_features = output_dict.get('final_node_features') # 检查节点特征

         # 基本检查
         assert 'prediction' in output_dict, "Missing 'prediction' in output"
         assert pred is not None, "'prediction' is None"
         assert pred.shape[0] == B, f"Prediction batch size mismatch: expected {B}, got {pred.shape[0]}"
         # prediction 输出维度可能由 MaterialAutoencoderTorch 决定，检查是否为 1
         assert pred.shape[1] == 1, f"Prediction output dim mismatch: expected 1, got {pred.shape[1]}"

         if node_features is not None:
             print(f"  Node features shape: {node_features.shape}") # 预期 [N, encoder_output_dim]
             assert node_features.shape[0] == N, "Node features N mismatch"
             # 维度检查 (需要知道 encoder_output_dim)
             # encoder_output_dim_expected = model.encoder_module.final_scalar_dim
             # assert node_features.shape[1] == encoder_output_dim_expected, "Node features dim mismatch"

         print(f"  Prediction shape: {pred.shape}") # 预期 [B, 1]
         # ... (其他输出检查保持不变) ...

         print("\nTest finished successfully!")

     except Exception as e:
         print(f"\n--- ERROR during forward pass ---"); traceback.print_exc()