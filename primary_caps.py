# geometry.primary_caps.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import numpy as np
from typing import Optional, List
import sys
import traceback

# Define _original_geometric_transform as None initially
_original_geometric_transform = None

# 使用绝对导入 (假设这些模块存在且可导入)
try:
    from repr.cnn import CNNEncoderTorch
    from repr.batch_mlp_torch import BatchMLPTorch
    from geometry.geo_torch import geometric_transform_torch, safe_log_torch
    _original_geometric_transform = geometric_transform_torch # 保存原始引用
except ImportError as e:
    print(f"Warning: Error importing dependencies in primary_caps_mlp.py: {e}. Tests might fail or use mocks.")
    geometric_transform_torch = None
    _original_geometric_transform = None

AttrDict = dict # 使用普通字典

class PrimaryCapsuleTorch(nn.Module):
    """PyTorch 版本的 CapsuleMaterialEncoder (Primary Capsules)"""

    OutputTuple = collections.namedtuple(
        'PrimaryCapsuleTorchTuple',
        ['pose', 'feature', 'presence', 'presence_logit']
    )
    _n_transform_params = 6

    def __init__(self,
                 encoder: nn.Module,
                 n_caps: int,
                 n_caps_dims: int,
                 n_features: int = 0,
                 encoder_type: str = 'conv_att',
                 mlp_hiddens: List[int] = [128],
                 noise_scale: float = 0.0,
                 similarity_transform: bool = False,
                 activation: nn.Module = nn.SELU()
                 ):
        super().__init__()

        if n_caps_dims != self._n_transform_params:
            raise ValueError(f"n_caps_dims must be {self._n_transform_params} for geometric_transform, "
                             f"but got {n_caps_dims}. Check FLAGS.n_part_caps_dims.")

        self.encoder = encoder
        # --- 获取 CNN 输出通道 ---
        self.cnn_final_channels = None
        if hasattr(encoder, 'get_output_channels') and callable(encoder.get_output_channels):
            self.cnn_final_channels = encoder.get_output_channels()
        elif hasattr(encoder, 'out_channels'):
             self.cnn_final_channels = encoder.out_channels
        elif isinstance(encoder, nn.Sequential): # 尝试获取 Sequential 最后一个 Conv2d
             for layer in reversed(encoder):
                  if isinstance(layer, nn.Conv2d):
                      self.cnn_final_channels = layer.out_channels
                      break
        if self.cnn_final_channels is None:
             print("Warning: Could not determine CNN output channels for PrimaryCapsuleTorch post_process_layer.")
        # --- 获取结束 ---

        self._n_caps = n_caps
        self._n_caps_dims = n_caps_dims
        self._n_features = n_features
        self._noise_scale = noise_scale
        self._similarity_transform = similarity_transform
        self._encoder_type = encoder_type.lower()
        self.activation = activation
        self._splits = [self._n_caps_dims, self._n_features, 1]
        self._n_dims = sum(self._splits)

        # --- 后续处理层 (根据 encoder_type) ---
        if self._encoder_type == 'linear':
            self.post_process_layer = nn.LazyLinear(self._n_caps * self._n_dims)
        elif self._encoder_type == 'conv' or self._encoder_type == 'conv_att':
            out_channels_conv = self._n_dims * self._n_caps
            if self._encoder_type == 'conv_att': out_channels_conv += self._n_caps
            if self.cnn_final_channels:
                self.post_process_layer = nn.Conv2d(self.cnn_final_channels, out_channels_conv, kernel_size=1)
            else: # Fallback to LazyConv2d
                print("Warning: Using LazyConv2d for post_process_layer in PrimaryCapsuleTorch.")
                self.post_process_layer = nn.LazyConv2d(out_channels_conv, kernel_size=1)
        else: raise ValueError(f"Invalid encoder_type: {self._encoder_type}")
        # --- 后续处理层结束 ---

    def forward(self, final_vec: torch.Tensor) -> OutputTuple:
        """
        Args:
            final_vec (torch.Tensor): 输入特征，形状 [N, F=256]。
        Returns:
            OutputTuple: 包含 pose, feature, presence, presence_logit。
        """
        if final_vec.dim() != 2 or final_vec.shape[-1] != 256:
             raise ValueError(f"Expected final_vec shape [N, 256], but got {final_vec.shape}")

        N = final_vec.shape[0]
        device = final_vec.device
        dtype = final_vec.dtype
        # print(f"    DEBUG (PrimaryCaps Entry): final_vec requires_grad = {final_vec.requires_grad}") # 检查点 0

        # 1. Reshape 输入
        x_reshaped = final_vec.view(N, 1, 16, 16).to(device=device, dtype=dtype)

        # 2. 通过 CNN Encoder
        img_embedding = self.encoder(x_reshaped)
        # print(f"    DEBUG (PrimaryCaps): img_embedding req_grad={img_embedding.requires_grad}") # 检查点 1

        batch_size_eff = N
        h = None

        # 3. 后续处理
        if self._encoder_type == 'linear':
            h_flat = torch.flatten(img_embedding, start_dim=1)
            h = self.post_process_layer(h_flat)
            # print(f"    DEBUG (PrimaryCaps Linear): final h req_grad={h.requires_grad}") # 检查点 L

        elif self._encoder_type == 'conv':
            h_conv = self.post_process_layer(img_embedding)
            # print(f"    DEBUG (PrimaryCaps Conv): h_conv req_grad={h_conv.requires_grad}") # 检查点 C1
            h = torch.mean(h_conv, dim=(2, 3))
            # print(f"    DEBUG (PrimaryCaps Conv): final h req_grad={h.requires_grad}") # 检查点 C2

        elif self._encoder_type == 'conv_att':
            h_conv_att = self.post_process_layer(img_embedding)
            # print(f"    DEBUG (PrimaryCaps ConvAtt): h_conv_att req_grad={h_conv_att.requires_grad}") # 检查点 CA1
            C_att = h_conv_att.shape[1]; L = h_conv_att.shape[2] * h_conv_att.shape[3]
            h_flat = h_conv_att.view(batch_size_eff, C_att, L).permute(0, 2, 1)

            split_dims = [self._n_dims * self._n_caps, self._n_caps]
            h_features_flat, a_logits_flat = torch.split(h_flat, split_dims, dim=-1)
            # print(f"    DEBUG (PrimaryCaps ConvAtt): h_features_flat req_grad={h_features_flat.requires_grad}") # 检查点 CA2
            # print(f"    DEBUG (PrimaryCaps ConvAtt): a_logits_flat req_grad={a_logits_flat.requires_grad}") # 检查点 CA3

            a_weights = F.softmax(a_logits_flat, dim=1)
            # print(f"    DEBUG (PrimaryCaps ConvAtt): a_weights req_grad={a_weights.requires_grad}") # 检查点 CA4

            a_weights_reshaped = a_weights.unsqueeze(2)
            h_features_reshaped = h_features_flat.view(batch_size_eff, L, self._n_dims, self._n_caps)

            # --- [修改] 检查乘法输入 ---
            # print(f"    DEBUG (PrimaryCaps ConvAtt): h_features_reshaped req_grad={h_features_reshaped.requires_grad}")
            # print(f"    DEBUG (PrimaryCaps ConvAtt): a_weights_reshaped req_grad={a_weights_reshaped.requires_grad}")
            # --- 修改结束 ---
            weighted_sum_input = h_features_reshaped * a_weights_reshaped
            # print(f"    DEBUG (PrimaryCaps ConvAtt): weighted_sum_input req_grad={weighted_sum_input.requires_grad}") # 检查点 CA5

            h_aggregated = torch.sum(weighted_sum_input, dim=1)
            # print(f"    DEBUG (PrimaryCaps ConvAtt): h_aggregated req_grad={h_aggregated.requires_grad}") # 检查点 CA6

            h = h_aggregated.permute(0, 2, 1)
            # print(f"    DEBUG (PrimaryCaps ConvAtt): final h req_grad={h.requires_grad}") # 检查点 CA7

        else: raise ValueError(f"Invalid encoder_type: {self._encoder_type}")

        # 4. 最终 Reshape
        if h is None: raise RuntimeError("Variable 'h' was not assigned.")
        try: 
            h_before_view = h
            h = h.view(batch_size_eff, self._n_caps, self._n_dims)
            # print(f"    DEBUG (PrimaryCaps): h after view req_grad={h.requires_grad}")
        except RuntimeError as e: raise RuntimeError(f"Failed to reshape final features 'h' with shape {h.shape} "
                               f"to [B={batch_size_eff}, n_caps={self._n_caps}, n_dims={self._n_dims}]. Error: {e}")

        # 5. 分裂输出
        pose_params, feature, pres_logit = torch.split(h, self._splits, dim=-1)
        if feature is not None and (torch.isnan(feature).any() or torch.isinf(feature).any()): print("WARNING: NaN/Inf in primary feature!") # 添加 NaN/Inf 检查
        if self._n_features == 0: feature = None
        # print(f"    DEBUG (PrimaryCaps Split): pose_params req_grad={pose_params.requires_grad}") # 检查点 S1
        # if feature is not None: print(f"    DEBUG (PrimaryCaps Split): feature req_grad={feature.requires_grad}") # 检查点 S2
        # else: print("    DEBUG (PrimaryCaps Split): feature is None")
        # print(f"    DEBUG (PrimaryCaps Split): pres_logit req_grad={pres_logit.requires_grad}") # 检查点 S3

        # 6. 处理存在概率
        pres_logit = pres_logit.squeeze(-1)
        pres = F.softmax(pres_logit, dim=-1)
        # print(f"    DEBUG (PrimaryCaps Presence): pres req_grad={pres.requires_grad}") # 检查点 P

        # 7. 几何变换
        if pose_params.shape[-1] != self._n_transform_params: 
                        raise ValueError(f"Pose params dimension mismatch: expected {self._n_transform_params}, got {pose_params.shape[-1]}")
        pose_transformed = geometric_transform_torch(
            pose_params,
            n_caps=self._n_caps,
            n_votes=1, # Primary caps 没有内部投票
            similarity=self._similarity_transform,
            nonlinear=False, # 假设 Primary pose 不做非线性变换
            as_matrix=False  # 返回 6D 向量
        )
        # print(f"    DEBUG (PrimaryCaps GeoTF): pose_transformed req_grad={pose_transformed.requires_grad}") # 检查点 G

        return self.OutputTuple(pose=pose_transformed, feature=feature,
                                presence=pres, presence_logit=pres_logit)

class PrimaryCapsuleTorchMLP(nn.Module): # <--- 新类名
    """使用 MLP 直接处理输入向量的 Primary Capsules 版本"""

    # 可以重命名元组以匹配类名 (可选)
    OutputTuple = collections.namedtuple(
        'PrimaryCapsuleTorchMLPTuple',
        ['pose', 'feature', 'presence', 'presence_logit']
    )
    _n_transform_params = 6
    _input_feature_dim = 256 # 定义期望的输入维度

    def __init__(self,
                 # encoder: nn.Module,        # <-- 移除 encoder
                 n_caps: int,
                 n_caps_dims: int,
                 n_features: int = 0,
                 # encoder_type: str = 'conv_att', # <-- 移除 encoder_type
                 mlp_hiddens: List[int] = [128], # <-- 保留 MLP 隐藏层设置
                 noise_scale: float = 4.0,      # <-- 默认值改为 4.0 (匹配 TF)
                 similarity_transform: bool = False,
                 activation: nn.Module = nn.SELU() # <-- 保留激活函数
                 ):
        super().__init__()
        # print(f"Initializing PrimaryCapsuleTorchMLP with mlp_hiddens={mlp_hiddens}") # 调试信息 (可注释掉)

        if n_caps_dims != self._n_transform_params:
            raise ValueError(f"n_caps_dims must be {self._n_transform_params} for geometric_transform, "
                             f"but got {n_caps_dims}. Check config.")

        self._n_caps = n_caps
        self._n_caps_dims = n_caps_dims
        self._n_features = n_features
        self._noise_scale = noise_scale
        self._similarity_transform = similarity_transform
        self.activation = activation
        self._splits = [self._n_caps_dims, self._n_features, 1]
        self._n_dims = sum(self._splits)

        # +++ 添加 MLP 定义 +++
        self._mlp_output_dim = self._n_caps * self._n_dims # MLP 输出维度
        mlp_layers = []
        in_dim = self._input_feature_dim
        for hidden_dim in mlp_hiddens:
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(self.activation)
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(in_dim, self._mlp_output_dim))
        self.mlp_replacement = nn.Sequential(*mlp_layers)
        # print(f"  MLP Replacement Structure: {self.mlp_replacement}") # 调试信息 (可注释掉)
        # +++ MLP 定义结束 +++

    def forward(self, final_vec: torch.Tensor) -> OutputTuple:
        """
        Args:
            final_vec (torch.Tensor): 输入特征，形状 [N, F=256]。
        Returns:
            OutputTuple: 包含 pose, feature, presence, presence_logit。
        """
        if final_vec.dim() != 2 or final_vec.shape[-1] != self._input_feature_dim:
             raise ValueError(f"Expected final_vec shape [N, {self._input_feature_dim}], but got {final_vec.shape}")

        N = final_vec.shape[0]
        device = final_vec.device
        dtype = final_vec.dtype
        batch_size_eff = N

        # 1. 执行 MLP 替换
        h_flat = self.mlp_replacement(final_vec)

        # 2. 执行关键的 Reshape
        try:
            h = h_flat.view(batch_size_eff, self._n_caps, self._n_dims)
        except RuntimeError as e: raise RuntimeError(f"Failed to reshape MLP output 'h_flat' shape {h_flat.shape} "
                                   f"to [B={batch_size_eff}, n_caps={self._n_caps}, n_dims={self._n_dims}]. Error: {e}")

        # 3. 分裂输出
        pose_params, feature, pres_logit = torch.split(h, self._splits, dim=-1)
        if feature is not None and (torch.isnan(feature).any() or torch.isinf(feature).any()): print("WARNING: NaN/Inf in MLP-generated feature!")
        if self._n_features == 0: feature = None

        # 4. 处理存在概率
        pres_logit = pres_logit.squeeze(-1)

        # --- 加入噪声注入 ---
        if self._noise_scale > 0. and self.training:
             if not pres_logit.is_floating_point(): pres_logit = pres_logit.float()
             noise = (torch.rand_like(pres_logit) - 0.5) * self._noise_scale
             pres_logit = pres_logit + noise
        # --- 噪声注入结束 ---

        pres = F.softmax(pres_logit, dim=-1)

        # 5. 几何变换
        if pose_params.shape[-1] != self._n_transform_params:
                        raise ValueError(f"Pose params dimension mismatch: expected {self._n_transform_params}, got {pose_params.shape[-1]}")
        use_nonlinear_transform = True # 假设默认使用非线性变换以匹配 TF

        # 确保 geometric_transform_torch 可用
        if geometric_transform_torch is None:
             raise RuntimeError("geometric_transform_torch function is not available (import failed or using mock). Cannot proceed.")

        pose_transformed = geometric_transform_torch(
            pose_params,
            n_caps=self._n_caps,
            n_votes=1,
            similarity=self._similarity_transform,
            nonlinear=use_nonlinear_transform,
            as_matrix=False
        )

        return self.OutputTuple(pose=pose_transformed, feature=feature,
                                presence=pres, presence_logit=pres_logit)


# 测试部分
if __name__ == '__main__':
    print("\n" + "="*30)
    print("--- Running PrimaryCapsuleTorchMLP Self-Test ---") # <--- 修改标题
    print("="*30)

    # --- 0. 设置测试参数 ---
    _test_batch_size = 2
    _test_n_caps = 8
    _test_n_caps_dims = 6 # 必须是 6
    _test_n_features = 16
    _test_input_dim = 256 # 必须是 256
    _test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _test_mlp_hiddens = [128, 64] # 测试用的 MLP 隐藏层
    _test_noise_scale = 4.0      # 测试用的噪声尺度
    print(f"Using device: {_test_device}")
    print(f"Testing with MLP hidden layers: {_test_mlp_hiddens}")
    print(f"Testing with noise_scale: {_test_noise_scale}")

    # --- 1. Mock 依赖 (主要是 geometric_transform_torch) ---
    # Mock geometric_transform_torch (如果原始导入失败或想隔离测试)
    def mock_geometric_transform_torch(pose_params, n_caps, n_votes, similarity, nonlinear, as_matrix):
        print("--- Using MOCK geometric_transform_torch ---")
        if pose_params.shape[-1] != 6:
             raise ValueError(f"Mock GeoTF expects dim 6, got {pose_params.shape[-1]}")
        # 简单返回输入本身，形状不变
        return pose_params

    # 如果原始导入失败，用 mock 替换；否则，可以选择性地使用 mock
    _use_mock_geotf = False
    if _original_geometric_transform is None:
         print("Original geometric_transform_torch not found, using mock.")
         geometric_transform_torch = mock_geometric_transform_torch # 注意：这会全局替换
         _use_mock_geotf = True
    else:
         print("Original geometric_transform_torch found.")
         # 你可以选择强制使用 mock 进行测试:
         # geometric_transform_torch = mock_geometric_transform_torch
         # _use_mock_geotf = True

    # --- 2. 创建模拟输入 ---
    _mock_input_vec = torch.randn(_test_batch_size, _test_input_dim, device=_test_device, dtype=torch.float32)
    print(f"Mock input vector shape: {_mock_input_vec.shape}")

    # --- 3. 测试实例化和前向传播 ---
    print("\n" + "-"*20)
    print(f"Testing PrimaryCapsuleTorchMLP Instantiation & Forward Pass")
    print("-"*20)

    _test_instance = None # 重置实例

    try:
        # --- 3.1 实例化 PrimaryCapsuleTorchMLP ---
        print("Instantiating PrimaryCapsuleTorchMLP...")
        _test_instance = PrimaryCapsuleTorchMLP(
            n_caps=_test_n_caps,
            n_caps_dims=_test_n_caps_dims,
            n_features=_test_n_features,
            mlp_hiddens=_test_mlp_hiddens,
            noise_scale=_test_noise_scale, # 使用测试值
            activation=nn.SELU() # 使用 SELU 测试
        ).to(_test_device)
        print("Instantiation successful.")
        # print(_test_instance) # 打印模型结构

    except Exception as e:
        print(f"ERROR during instantiation:")
        traceback.print_exc()
        exit(1) # 实例化失败则退出

    try:
        # --- 3.2 运行前向传播 (测试训练模式以激活噪声) ---
        print("Running forward pass (in training mode)...")
        _test_instance.train() # 设置为训练模式来测试噪声注入
        output_train = _test_instance(_mock_input_vec)
        print("Forward pass (train mode) successful.")

        # --- 3.3 检查输出形状和基本属性 (训练模式) ---
        print("Checking outputs (train mode)...")
        # (检查代码与之前类似，这里省略重复代码，仅作示意)
        assert isinstance(output_train, tuple), "Output (train) is not a tuple"
        assert output_train.pose.shape == (_test_batch_size, _test_n_caps, _test_n_caps_dims)
        if _test_n_features > 0: assert output_train.feature.shape == (_test_batch_size, _test_n_caps, _test_n_features)
        assert output_train.presence.shape == (_test_batch_size, _test_n_caps)
        assert output_train.presence_logit.shape == (_test_batch_size, _test_n_caps) # Logit 在返回前没有 squeeze
        # 检查 NaN
        assert not torch.isnan(output_train.pose).any(), "NaN found in output pose (train)"
        if output_train.feature is not None: assert not torch.isnan(output_train.feature).any(), "NaN found in output feature (train)"
        assert not torch.isnan(output_train.presence).any(), "NaN found in output presence (train)"
        assert not torch.isnan(output_train.presence_logit).any(), "NaN found in output presence_logit (train)"
        # 检查 presence 和
        assert torch.allclose(output_train.presence.sum(dim=-1), torch.tensor(1.0, device=_test_device), atol=1e-6), "Presence (train) probabilities do not sum to 1"
        print("Output checks passed (train mode).")

        # --- 3.4 运行前向传播 (测试评估模式) ---
        print("\nRunning forward pass (in eval mode)...")
        _test_instance.eval() # 设置为评估模式
        with torch.no_grad():
             output_eval = _test_instance(_mock_input_vec)
        print("Forward pass (eval mode) successful.")

        # --- 3.5 检查输出形状和基本属性 (评估模式) ---
        print("Checking outputs (eval mode)...")
        # (检查代码与之前类似)
        assert isinstance(output_eval, tuple), "Output (eval) is not a tuple"
        assert output_eval.pose.shape == (_test_batch_size, _test_n_caps, _test_n_caps_dims)
        # ... 其他检查 ...
        assert torch.allclose(output_eval.presence.sum(dim=-1), torch.tensor(1.0, device=_test_device), atol=1e-6), "Presence (eval) probabilities do not sum to 1"
        print("Output checks passed (eval mode).")

        # --- 3.6 (可选) 比较训练和评估模式输出 ---
        # 在评估模式下，由于没有噪声，presence_logit 和 presence 应该与训练模式不同
        if _test_noise_scale > 0.:
            assert not torch.allclose(output_train.presence_logit, output_eval.presence_logit), "Presence logit should differ between train and eval mode due to noise"
            assert not torch.allclose(output_train.presence, output_eval.presence), "Presence should differ between train and eval mode due to noise"
            print("Train/Eval output difference check passed (due to noise).")


    except Exception as e:
        print(f"ERROR during forward pass or output check:")
        traceback.print_exc()


    print("\n" + "="*30)
    print("--- PrimaryCapsuleTorchMLP Self-Test Finished ---")
    print("="*30)

    # 恢复原始的 geometric_transform_torch (如果被 mock 替换了)
    if _use_mock_geotf and _original_geometric_transform is not None:
        geometric_transform_torch = _original_geometric_transform # 恢复全局引用
        print("\nRestored original geometric_transform_torch.")