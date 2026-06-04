"""PR2: Qwen2 text-only block-mask patch.

复用:
- starVLA.training.trainer_utils.monkey_patch.return_mask

说明:
本文件只做运行期 patch，不修改 transformers 或 starVLA 现有源码。
"""

######### // code // ##########
# 中文注释：把 transformers 的 Qwen2(text) 因果掩码工厂替换为“透传”。
# 输入：framework 传入的 4D additive attention mask，shape [B,1,T,T]。
# 输出：Qwen2Model 内部继续使用同一个 mask，不再用默认 causal mask 覆盖。
from starVLA.training.trainer_utils.monkey_patch import return_mask


def apply_qwen2_text_block_mask_patch() -> None:
    import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling

    qwen2_modeling.create_causal_mask = return_mask
    if hasattr(qwen2_modeling, "create_sliding_window_causal_mask"):
        qwen2_modeling.create_sliding_window_causal_mask = return_mask
######### // code // ##########

