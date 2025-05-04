# repr.matchecon.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.data import Data as PyGData
from typing import List, Tuple, Optional, Dict

try:
    from repr.mlp import MLP
    from repr.transformer_layer import TransformerLayerTorch
    from repr.set_transformer import SetTransformerTorch
    from repr.chemical_transform import ChemicalTransformTorch
except ImportError as e: print(f"Error importing MatCheConTorch dependencies: {e}"); raise

import traceback

class MatCheConTorch(nn.Module):
    def __init__(self,
                 raw_atom_emb_dim: int = 80, raw_bond_emb_dim: int = 100, raw_state_emb_dim: int = 64,
                 initial_mlp_units: List[int] = [64, 192], # 初始MLP输出维度在这里确定
                 n_loop: int = 3,
                 tf_hidden_units_v: List[int] = [128, 128],
                 tf_hidden_units_e: List[int] = [128, 128],
                 tf_hidden_units_u: List[int] = [128, 128],
                 stoi_config: dict = {},
                 pool_method: str = "mean",
                 residual_alpha: float = 0.5,
                 set_transformer_hidden: int = 128, # SetTF 隐藏层维度
                 set_transformer_loop: Optional[int] = None,
                 final_output_dim: int = 256, # 最终期望输出维度 (LSTM hidden size)
                 activation_name: str = 'selu',
                 # --- [新增] chemical_transform 参数 ---
                 chemical_transform_dropout: float = 0.2,
                 attention_dropout: float = 0.0 # ChemicalTransform 内部 Attention 的 dropout
                 # --- 新增结束 ---
                 ):
        super().__init__()
        self.n_loop = n_loop
        self.residual_alpha = residual_alpha
        # --- 在方法体内部创建激活函数实例 ---
        if activation_name.lower() == 'selu':
            self.activation = nn.SELU()
        elif activation_name.lower() == 'relu':
            self.activation = nn.ReLU()
        # 添加其他需要的激活函数
        else:
            # 尝试从 torch.nn 获取，如果失败则报错
            try:
                self.activation = getattr(nn, activation_name)()
            except AttributeError:
                raise ValueError(f"Unsupported activation function name: {activation_name}")
        # print(f"Using activation: {self.activation}")

        # --- 存储 pool_method 到 self.reduce_op ---
        if pool_method not in ["mean", "sum"]: raise ValueError(...)
        self.pool_method = pool_method # 可以保留，如果别处用到
        self.reduce_op = pool_method # <-- 添加这一行
        
        self.final_output_dim = final_output_dim

        # 1. 初始 MLP 层
        mlp_output_dim = initial_mlp_units[-1] # 获取 MLP 的输出维度
        self.atom_mlp_initial = MLP(raw_atom_emb_dim, initial_mlp_units, activation=self.activation, activate_last=True)
        # --- [修改] 添加 bond_mlp_initial ---
        self.bond_mlp_initial = MLP(raw_bond_emb_dim, initial_mlp_units, activation=self.activation, activate_last=True)
        # --- 修改结束 ---
        self.state_mlp_initial= MLP(raw_state_emb_dim, initial_mlp_units, activation=self.activation, activate_last=True)

        v_dim_init = mlp_output_dim
        e_dim_init = mlp_output_dim # 边特征经过 MLP 后维度也变为 mlp_output_dim
        u_dim_init = mlp_output_dim

        # --- 再打印调试信息 ---
        # print(f"\n--- Debug MatCheConTorch Init ---")
        # print(f"  Input dimensions for TransformerLayerTorch:")
        # print(f"    v_dim: {v_dim_init}")
        # print(f"    e_dim: {e_dim_init}")
        # print(f"    u_dim: {u_dim_init}")
        # print(f"  Hidden units for TransformerLayerTorch MLPs:")
        # print(f"    hidden_units_v: {tf_hidden_units_v}")
        # print(f"    hidden_units_e: {tf_hidden_units_e}")
        # print(f"    hidden_units_u: {tf_hidden_units_u}")
        # print(f"  Stoi Config: {stoi_config}")
        # print(f"  Pool Method: {pool_method}")


        # 2. chemical_env 核心 (TransformerLayerTorch)
        self.transformer_layer = TransformerLayerTorch(
            v_dim=v_dim_init, e_dim=e_dim_init, u_dim=u_dim_init,
            hidden_units_v=tf_hidden_units_v, hidden_units_e=tf_hidden_units_e,
            hidden_units_u=tf_hidden_units_u, stoi_config=stoi_config,
            pool_method=pool_method, activation=self.activation
        )
        self.tf_out_v_dim = v_dim_init
        self.tf_out_e_dim = e_dim_init # 输出维度与输入一致 (由 TransformerLayerTorch 保证)
        self.tf_out_u_dim = u_dim_init
        self.comp_fea_dim = stoi_config.get('elem_fea_len', 64)


        # 3. Set Transformer 层
        set_tf_loop = set_transformer_loop if set_transformer_loop is not None else n_loop
        self.atom_set_transformer = SetTransformerTorch(self.tf_out_v_dim, set_transformer_hidden, set_tf_loop)
        # --- [修改] SetTransformer 输入维度使用更新后的 e_dim_init ---
        self.bond_set_transformer = SetTransformerTorch(self.tf_out_e_dim, set_transformer_hidden, set_tf_loop)
        # --- 修改结束 ---
        self.set_tf_output_dim = 2 * set_transformer_hidden

        # 5. 用于组合特征的投影层
        self.comp_accum_to_logit = nn.Linear(self.comp_fea_dim, 1)

        # --- [修改] 6. 最终输出 MLP ---
        # 计算拼接后的特征维度
        chemical_transform_input_dim = self.set_tf_output_dim + self.set_tf_output_dim + self.tf_out_u_dim
        print(f"DEBUG (MatCheCon Init): ChemicalTransformTorch input dimension: {chemical_transform_input_dim}")

        self.chemical_transform = ChemicalTransformTorch(
            input_dim=chemical_transform_input_dim, # 拼接后的维度
            lstm_hidden_size=final_output_dim, # LSTM 输出维度即最终输出维度
            attention_dropout=attention_dropout,
            dropout_rate=chemical_transform_dropout
        )

    def forward(self, data: PyGData,
                      atom_vec_embedded: torch.Tensor,
                      bond_vec_embedded: torch.Tensor, # 传入原始边特征
                      state_vec_embedded: torch.Tensor
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # --- [修改] 1. 获取必需的属性并检查 ---
        # 优先从 data.num_nodes 获取 N
        if hasattr(data, 'num_nodes') and data.num_nodes is not None:
            N = data.num_nodes
        elif hasattr(data, 'x') and data.x is not None:
            N = data.x.shape[0]
        else: raise ValueError("Cannot determine N")
        if N == 0: return torch.empty((0, self.final_output_dim), device=atom_vec_embedded.device), None # 处理空图

        # 获取 batch 向量
        if not hasattr(data, 'batch') or data.batch is None:
             B = 1; batch = torch.zeros(N, dtype=torch.long, device=atom_vec_embedded.device)
        else: batch = data.batch.long(); B = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        if B == 0: return torch.empty((N, self.final_output_dim), device=atom_vec_embedded.device), None # 处理空批次

        # 获取 edge_index
        edge_index = data.edge_index if hasattr(data, 'edge_index') else None
        if edge_index is None: raise ValueError("Missing edge_index")
        N_edges = edge_index.shape[1]

        # 获取 atom_fea (one-hot)
        atom_fea = data.atom_fea if hasattr(data, 'atom_fea') else None
        if atom_fea is None: raise ValueError("Missing atom_fea")

        # 获取 comp_w (每个原子的权重 [N, 1])
        comp_w = data.comp_w if hasattr(data, 'comp_w') else None
        if comp_w is None: raise ValueError("Missing comp_w")
        # 确保 comp_w 是 [N, 1]
        comp_w = comp_w.view(-1, 1)

        # 使用传入的 bond_vec_embedded 作为 edge_attr
        edge_attr = bond_vec_embedded

        # 进行维度检查
        if atom_vec_embedded.shape[0] != N: raise ValueError(f"Shape mismatch: atom_vec_embedded ({atom_vec_embedded.shape[0]}) != N ({N})")
        if edge_attr.shape[0] != N_edges: raise ValueError(f"Shape mismatch: edge_attr ({edge_attr.shape[0]}) != N_edges ({N_edges})")
        if state_vec_embedded.shape[0] != N: raise ValueError(f"Shape mismatch: state_vec_embedded ({state_vec_embedded.shape[0]}) != N ({N})")
        if atom_fea.shape[0] != N: raise ValueError(f"Shape mismatch: atom_fea ({atom_fea.shape[0]}) != N ({N})")
        if comp_w.shape[0] != N: raise ValueError(f"Shape mismatch: comp_w ({comp_w.shape[0]}) != N ({N})")
        # --- [修改结束] ---


        # print(f"  DEBUG (MatCheCon Entry): atom_vec_embedded req_grad={atom_vec_embedded.requires_grad}")
        # print(f"  DEBUG (MatCheCon Entry): edge_attr(bond_vec_embedded) req_grad={edge_attr.requires_grad}")
        # print(f"  DEBUG (MatCheCon Entry): state_vec_embedded req_grad={state_vec_embedded.requires_grad}")

        # --- 2. 初始 MLP 处理 ---
        atom_vec_ = self.atom_mlp_initial(atom_vec_embedded)
        bond_vec_ = self.bond_mlp_initial(edge_attr) # 应用 bond MLP
        state_vec_ = self.state_mlp_initial(state_vec_embedded)
        # print(f"  DEBUG (MatCheCon After Initial MLP): atom_vec_ req_grad={atom_vec_.requires_grad}")
        # print(f"  DEBUG (MatCheCon After Initial MLP): bond_vec_ req_grad={bond_vec_.requires_grad}")
        # print(f"  DEBUG (MatCheCon After Initial MLP): state_vec_ req_grad={state_vec_.requires_grad}")

        # --- 3. 准备 chemical_env/transformer 输入 ---
        atom_index, nei_index = edge_index[0], edge_index[1]
        atom_sou = batch
        bond_sou = batch[atom_index]

        x_comp_accum = None

        # --- 4. chemical_env 循环 ---
        for i in range(self.n_loop):
            # print(f"  --- MatCheCon Loop {i} Start ---")
            atom_vec_prev, bond_vec_prev, state_vec_prev = atom_vec_, bond_vec_, state_vec_
            # print(f"    Input to TF Layer: atom req_grad={atom_vec_prev.requires_grad}, bond req_grad={bond_vec_prev.requires_grad}, state req_grad={state_vec_prev.requires_grad}")

            transformer_inputs = {
                 'atom_vec_': atom_vec_prev, 'bond_vec_': bond_vec_prev, 'state_vec_': state_vec_prev,
                 'com_w': comp_w, 'atom_fea': atom_fea, 'edge_index': edge_index, 'batch': batch,
                 'loop_num': i, 'batch_size': B
            }
            atom_che, bond_che, state_che, comp_che = self.transformer_layer(**transformer_inputs)
            # print(f"    Output from TF Layer: atom_che req_grad={atom_che.requires_grad}, bond_che req_grad={bond_che.requires_grad}, state_che req_grad={state_che.requires_grad}, comp_che req_grad={comp_che.requires_grad}")

            atom_vec_ = atom_vec_prev + self.residual_alpha * atom_che
            bond_vec_ = bond_vec_prev + self.residual_alpha * bond_che
            state_vec_ = state_vec_prev + self.residual_alpha * state_che
            # print(f"    After Residual: atom req_grad={atom_vec_.requires_grad}, bond req_grad={bond_vec_.requires_grad}, state req_grad={state_vec_.requires_grad}")
            # print(f"  --- MatCheCon Loop {i} End ---")

            if comp_che is not None:
                 if x_comp_accum is None: x_comp_accum = comp_che
                 else:
                      if x_comp_accum.shape == comp_che.shape: x_comp_accum = x_comp_accum + comp_che
                      else: print(f"Warning: Shape mismatch in x_comp_accum...")


        # 5. Set Transformer 聚合
        # print(f"  DEBUG (MatCheCon Before Agg): atom_vec_ req_grad={atom_vec_.requires_grad}, bond_vec_ req_grad={bond_vec_.requires_grad}")
        atom_vec_graph = self.atom_set_transformer(atom_vec_, atom_sou)
        bond_vec_graph = self.bond_set_transformer(bond_vec_, bond_sou)
        # print(f"  DEBUG (MatCheCon After Agg): atom_vec_graph req_grad={atom_vec_graph.requires_grad}, bond_vec_graph req_grad={bond_vec_graph.requires_grad}")

        # --- 6. 特征组合 ---
        if x_comp_accum is None: x_comp_accum = torch.zeros(B, self.comp_fea_dim, device=atom_vec_.device)
        comp_logits = self.comp_accum_to_logit(x_comp_accum)
        comps = F.softmax(comp_logits, dim=0) # Softmax across batch
        # print(f"  DEBUG (MatCheCon Combine): comps req_grad={comps.requires_grad}")
        atom_inp = comps * atom_vec_graph
        # print(f"  DEBUG (MatCheCon Combine): atom_inp req_grad={atom_inp.requires_grad}")
        state_vec_graph = scatter(state_vec_, batch, dim=0, reduce=self.reduce_op, dim_size=B)
        # print(f"  DEBUG (MatCheCon Combine): state_vec_graph req_grad={state_vec_graph.requires_grad}")

        # --- [修改] 7. 准备 ChemicalTransformTorch 的输入 (广播和拼接) ---
        atom_inp_node = atom_inp[batch]             # [N, set_tf_output_dim]
        bond_vec_graph_node = bond_vec_graph[batch] # [N, set_tf_output_dim]
        # 使用循环结束后的节点级 state_vec_
        state_vec_node = state_vec_                   # [N, tf_out_u_dim]

        # 拼接得到 chemical_transform 的输入
        final_vec_before_transform = torch.cat([atom_inp_node, bond_vec_graph_node, state_vec_node], dim=-1)
        # print(f"  DEBUG (MatCheCon Before ChemTF): input shape={final_vec_before_transform.shape}, req_grad={final_vec_before_transform.requires_grad}")
        # --- 修改结束 ---

       # --- [修改] 8. 调用 ChemicalTransformTorch ---
        final_node_vec = self.chemical_transform(final_vec_before_transform, batch) # 假设 batch 用于内部处理，或者不需要
        # print(f"  DEBUG (MatCheCon End): final_node_vec (after ChemTF) req_grad={final_node_vec.requires_grad}") # 最终检查

        # 9. 提取目标值
        real_bp_out = data.y_raw if hasattr(data, 'y_raw') else None
        if real_bp_out is not None: real_bp_out = real_bp_out.view(-1)

        return final_node_vec, real_bp_out


# --- 测试代码 (需要更新 __init__ 调用) ---
if __name__ == '__main__':
    print("--- Testing MatCheConTorch ---")
    # ... (模拟参数定义，确保包含 stoi_config) ...
    stoi_config_example = {
         "n_target": 1, "elem_emb_len": 128, "elem_fea_len": 64,
         "n_graph": 3, "elem_heads": 3, "elem_gate_hidden": [128], # 简化维度
         "elem_msg_hidden": [128], "cry_heads": 3, "cry_gate_hidden": [128],
         "cry_msg_hidden": [128],
     }
    RAW_ATOM_EMB_DIM = 80; RAW_BOND_EMB_DIM = 100; RAW_STATE_EMB_DIM = 64
    INITIAL_MLP_UNITS = [64, 192]
    TF_UNITS_V = [128, 80]; TF_UNITS_E = [128, 64]; TF_UNITS_U = [128, 64] # 示例维度
    FINAL_OUTPUT_DIM = 256
    N_LOOP = 3; RESIDUAL_ALPHA = 0.5; SET_TRANSFORMER_HIDDEN = 128

        # --- 定义 TransformerLayerTorch 的 MLP 隐藏层维度 ---
    TF_HIDDEN_UNITS_V = [128, 128] # 示例: 两个隐藏层，每层 128 维
    TF_HIDDEN_UNITS_E = [128, 128]
    TF_HIDDEN_UNITS_U = [128, 128]

    # ... (模拟输入数据 mock_data_batch, mock_atom_emb 等) ...
    device = torch.device("cpu"); N=7; N_edges=6; B=2; N_GAUSSIAN_CENTERS=100; VOCAB_SIZE=103
    mock_atom_emb = torch.randn(N, RAW_ATOM_EMB_DIM, device=device)
    mock_bond_emb = torch.randn(N_edges, RAW_BOND_EMB_DIM, device=device)
    mock_state_emb = torch.randn(N, RAW_STATE_EMB_DIM, device=device)
    mock_data_batch = PyGData(
        x=torch.randint(1, 80, (N,), device=device, dtype=torch.long), # 原子序数 Z (long)
        edge_index=torch.tensor([[0, 1, 3, 4, 5, 6],  # 边索引 (long)
                                 [1, 0, 4, 3, 6, 5]], device=device, dtype=torch.long),
        edge_attr=mock_bond_emb, # 使用上面创建的 mock_bond_emb (float) [N_edges, 100]
        atom_fea=F.one_hot(torch.randint(0, VOCAB_SIZE, (N,)), num_classes=VOCAB_SIZE).float().to(device), # One-hot (float) [N, 103]
        comp_w=torch.rand(N, 1, device=device), # 组成权重 (float) [N, 1]
        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device, dtype=torch.long), # Batch 索引 (long) [N]
        y_raw=torch.randn(B, device=device) # 目标值 (float) [B]
        # state 属性可以不加，forward 内部会处理
        # num_nodes 属性 PyG 会自动推断（如果需要显式设置也可以）
    )

    print("\n--- Instantiating MatCheConTorch ---")
    try:
        model = MatCheConTorch(
            raw_atom_emb_dim=RAW_ATOM_EMB_DIM,
            raw_bond_emb_dim=RAW_BOND_EMB_DIM,
            raw_state_emb_dim=RAW_STATE_EMB_DIM,
            initial_mlp_units=INITIAL_MLP_UNITS,
            n_loop=N_LOOP,
            tf_hidden_units_v=TF_HIDDEN_UNITS_V,
            tf_hidden_units_e=TF_HIDDEN_UNITS_E,
            tf_hidden_units_u=TF_HIDDEN_UNITS_U,
            stoi_config=stoi_config_example,
            pool_method="mean",
            residual_alpha=RESIDUAL_ALPHA,
            set_transformer_hidden=SET_TRANSFORMER_HIDDEN,
            final_output_dim=FINAL_OUTPUT_DIM
            # --- !! 检查这里是否意外添加了 activation=... !! ---
            # activation = ...  <--- 如果有这一行且值为 Ellipsis，删除它
        ).to(device)
        print("\nModel Instance Instantiated.")

    except Exception as e:
        print(f"\n--- ERROR during model instantiation ---"); traceback.print_exc(); exit(1)

    # --- 运行前向传播 ---
    print("\n--- Running Forward Pass ---")
    try:
        model.train()
        final_node_vec, real_bp = model(mock_data_batch, mock_atom_emb, mock_bond_emb, mock_state_emb)

        # --- 打印输出形状 ---
        print("\n--- Forward Pass Successful ---")
        print(f"Output final_node_vec shape: {final_node_vec.shape}") # 预期 [N=7, 256]
        print(f"Output real_bp shape: {real_bp.shape if real_bp is not None else None}") # 预期 [B=2]
        assert final_node_vec.shape == (N, FINAL_OUTPUT_DIM), "final_node_vec shape mismatch!"
        if real_bp is not None: assert real_bp.shape == (B,), "real_bp shape mismatch!"
        print("\nTest finished successfully!")

    except Exception as e:
        print(f"\n--- ERROR during forward pass ---"); traceback.print_exc()