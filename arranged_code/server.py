import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from typing import Union, List

# ================= 配置区域 =================
# 
MODEL_PATH = "./Qwen2.5-7B-Instruct"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PORT = 8088

# DEBUG = True
DEBUG = False

app = FastAPI()

print(f"正在加载模型自: {MODEL_PATH} ...")
print(f"运行设备: {DEVICE}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH, 
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
   
    model.eval()
    print("模型加载完成！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 定义请求的数据格式
class InputRequest(BaseModel):
    text: str
    top_k: int = 5 # 获取 top_k 的attention权重
    aim_word: str = None
    layer_idx: Union[int, list[int]] = list(range(10,20))  # 默认为10-19层
    vector_mode:str = "mean" # 聚合方法，默认为 mean


def find_token_positions_by_char(text: str, aim_word: str, offset_mapping: list, find_last: bool = False):
    """
    通过字符偏移(offset)来匹配目标词对应的 token 位置
    """
    # 要求找最后一个匹配的 token 位置
    if find_last:
        start_index = text.rfind(aim_word)
        end_index = start_index + len(aim_word)
        # print(f"目标词 '{aim_word}' 在输入文本中的字符位置: ({start_index}, {end_index}), 目标词文本: '{text[start_index:end_index]}'")
        positions = []
        for idx, (start, end) in enumerate(offset_mapping):
            # print(f"Token index: {idx}, Token offset: ({start}, {end}), Token text: '{text[start:end]}'")
            if end > start_index and start < end_index:
                positions.append(idx)
        return positions
    else:
    # 找到所有的匹配的 token 位置
        positions = []
        start_index = 0
        while True:
            start_index = text.find(aim_word, start_index) # 从上一个匹配的结束位置继续查找下一个匹配
            if start_index == -1:
                break
            end_index = start_index + len(aim_word)
            # print(f"目标词 '{aim_word}' 在输入文本中的字符位置: ({start_index}, {end_index}), 目标词文本: '{text[start_index:end_index]}'")
            for idx, (start, end) in enumerate(offset_mapping):
                # print(f"Token index: {idx}, Token offset: ({start}, {end}), Token text: '{text[start:end]}'")
                if end > start_index and start < end_index:
                    positions.append(idx)
            start_index += len(aim_word)  # 继续查找下一个匹配
        return positions


def aggregate_vectors(vectors, method="mean")->torch.Tensor:
    # vectors: list[tensor], 每个元素是一个层的向量列表
    # mean: 先对同一层的向量求平均，再对不同层的结果求平均
    if method == "mean":
        # 对每一层的向量求平均
        mean_vectors = []
        # vectors : list[tensor], 每个元素是一个层的目标词向量列表, shape: [num_tokens, hidden_size]
        for layer_vectors in vectors: 
            if layer_vectors.shape[0] > 0: # 如果该层有向量
                mean_vector = layer_vectors.mean(dim=0) # shape: [hidden_size]
                mean_vectors.append(mean_vector)
            else:
                mean_vectors.append(torch.zeros(vectors[0].shape[1])) # shape: [hidden_size]
        # 对所有层的平均向量求平均
        if len(mean_vectors) > 0:
            return torch.stack(mean_vectors).mean(dim=0) # shape: [hidden_size]
        else:
            return torch.zeros(vectors[0].shape[1])
    # concat: 先对同一层的向量求平均，再对不同层的结果进行拼接
    elif method == "concat":
        mean_vectors = []
        for layer_vectors in vectors:
            if layer_vectors.shape[0] > 0:
                mean_vector = layer_vectors.mean(dim=0)
                mean_vectors.append(mean_vector)
            else:
                mean_vectors.append(torch.zeros(vectors[0].shape[1]))
        return torch.cat(mean_vectors, dim=0) # shape: [num_layers * hidden_size]
    else:
        raise ValueError(f"不支持的聚合方法: {method}")
    

# 获取最后一层最后一个 token 的向量，以及目标层中目标词在句子中最后一个位置对应的向量
@app.post("/get_last_word_and_aim_word_vector")
async def get_last_word_and_aim_word_vector(request: InputRequest):
    try:
        input_device = model.get_input_embeddings().weight.device
        input = tokenizer(request.text, return_tensors="pt", return_offsets_mapping=True).to(input_device)
        offset_mapping = input["offset_mapping"][0].tolist() # list of (start, end) tuples
        full_tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][0].tolist())
        positions = find_token_positions_by_char(request.text, request.aim_word, offset_mapping, find_last=True)
        

        if DEBUG:
            print(f"输入文本的 tokens: {full_tokens}")
            print(f"目标词 '{request.aim_word}'")
            print(f"目标词 '{request.aim_word}' 在输入文本中的最后位置索引: {positions}")
        if(not positions):
                raise HTTPException(status_code=422, detail=f"无法在输入文本中找到目标词 '{request.aim_word}' 的位置，可能是因为分词结果无法匹配到输入文本的 token 中")
        with torch.no_grad():
            outputs = model(**input, output_hidden_states=True)
        
        if DEBUG:
            print(f"模型输出的 hidden_states 层数: {len(outputs.hidden_states)}")
        hidden_states = outputs.hidden_states
        last_layer_last_token_vector = hidden_states[-1][0, -1, :].cpu().tolist() # 最后一层最后一个 token 的向量
        
        if isinstance(request.layer_idx, int):
            layer_indices = [request.layer_idx]
        elif isinstance(request.layer_idx, list):
            layer_indices = request.layer_idx
        else:
            layer_indices = range(len(hidden_states))  # 默认返回所有层
        
        hidden_states_list = list(hidden_states)
        aim_word_vectors = []
        for i in layer_indices:
            if i >= len(hidden_states):
                raise HTTPException(status_code=422, detail=f"请求的层索引 {i} 超出模型的层数范围")
            
            layer_tensor = hidden_states_list[i]
            layer_tensor = layer_tensor.squeeze(0) # 去掉 batch 维度
            positions_tensor = torch.tensor(positions, dtype=torch.long).to(layer_tensor.device)
            layer_data = layer_tensor[positions_tensor, :].cpu() #[num_tokens, hidden_size]
            aim_word_vectors.append(layer_data) # list[tensor], 每个元素是一个层的目标词向量列表, shape: [num_tokens, hidden_size]
        
        aim_word_vector_mean = aggregate_vectors(aim_word_vectors, method=request.vector_mode) # shape: [hidden_size]
        return {
            "status": "success",
            "last_token_vector": last_layer_last_token_vector,
            "aim_word_vector": aim_word_vector_mean.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_all_hidden_states")
async def get_all_hidden_states(request: InputRequest):
    try:
        input_device = model.get_input_embeddings().weight.device
        input = tokenizer(request.text, return_tensors="pt", return_offsets_mapping=True).to(input_device)
        offset_mapping = input["offset_mapping"][0].tolist() # list of (start, end) tuples
        full_tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][0].tolist())
        positions = find_token_positions_by_char(request.text, request.aim_word, offset_mapping, find_last=True)

        with torch.no_grad():
            outputs = model(**input, output_hidden_states=True)
        hidden_states =outputs.hidden_states
        stacked = torch.cat(hidden_states, dim=0) # shape: [num_layers, batch_size, seq_len, hidden_size]
        stacked = stacked.squeeze(1) # shape: [num_layers, seq_len, hidden_size]
        aim_word_vectors = stacked[:, positions, :]
        aim_word_vectors = aim_word_vectors.mean(dim=1) # shape: [num_layers, hidden_size]
        aim_word_vectors = aim_word_vectors.cpu().tolist() # 转移到 CPU 并转换为列表
        
        return {
            "status": "success",
            "hidden_states": aim_word_vectors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'outputs' in locals():
            del outputs
        if 'hidden_states' in locals():
            del hidden_states
        if 'input' in locals():
            del input
            
        import gc
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

@app.post("/get_all_k_vectors")
async def get_all_k_vectors(request: InputRequest):
    hooks = []
    try:
        input_device = model.get_input_embeddings().weight.device
        input = tokenizer(request.text, return_tensors="pt", return_offsets_mapping=True).to(input_device)
        offset_mapping = input["offset_mapping"][0].tolist() 
        full_tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][0].tolist())
        positions = find_token_positions_by_char(request.text, request.aim_word, offset_mapping, find_last=True)

        if not positions:
            raise HTTPException(status_code=422, detail=f"无法找到目标词 '{request.aim_word}'")

        k_states = {}
        
        # 定义获取 K 向量的 Hook 函数
        def get_k_hook(layer_idx):
            def hook(module, input, output):
                # Qwen2ForCausalLM 的 k_proj 输出 shape: [batch_size, seq_len, hidden_size]
                # 保存到字典中，去掉 batch 维度并保留在 GPU 上以便后续并行处理
                k_states[layer_idx] = output.squeeze(0).detach() 
            return hook

        num_layers = len(model.model.layers)
        
        # 为每一层的 k_proj 挂载 Hook
        for i in range(num_layers):
            layer = model.model.layers[i]
            hook_k = layer.self_attn.k_proj.register_forward_hook(get_k_hook(i))
            hooks.append(hook_k)

        with torch.no_grad():
            model(**input)
            
        # 卸载所有 Hooks
        for h in hooks:
            h.remove()
        hooks.clear()

        # k_states 包含 num_layers 个 tensor，按层索引排序并堆叠
        # 把各层的结果组合成一个 shape 为 [num_layers, seq_len, hidden_size] 的张量
        k_tensors = [k_states[i] for i in range(num_layers)]
        stacked_k = torch.stack(k_tensors, dim=0) 
        
        # 提取目标词位置的向量: [num_layers, num_tokens, hidden_size]
        aim_word_k_vectors = stacked_k[:, positions, :]
        
        # 对 token 维度进行 mean 聚合: [num_layers, hidden_size]
        aim_word_k_vectors = aim_word_k_vectors.mean(dim=1)
        
        # 一次性转移到 CPU 并转化为列表返回
        aim_word_k_vectors_list = aim_word_k_vectors.cpu().tolist()
        
        return {
            "status": "success",
            "k_vectors": aim_word_k_vectors_list
        }
    except Exception as e:
        for h in hooks:
            h.remove()
        raise HTTPException(status_code=500, detail=str(e))

# 定义top_k_attentioned_v的输入格式
class InputRequest_attention(BaseModel):
    text: List[str]
    aim_word: List[str] = None
    top_k: List[int] = None # 获取 top_k 的attention权重
    layer_idx: Union[int, list[int]] = list(range(10,20))  # 默认为10-19层
    vector_mode:str = "mean" # 聚合方法，默认为 mean

def get_top_k_attned_v(v_states:dict, attn:dict, target_layers:list, positions:list, k=5)->dict:
    results = []
    num_heads = model.config.num_attention_heads
    seq_length = list(attn.values())[0].shape[1]
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads # 每个head的维度
    num_kv_groups = num_heads // num_kv_heads
    hidden_size = model.config.hidden_size

    for layer in target_layers:
        v = v_states[layer]  # [seq_length, hidden_size]
        attn_weights = attn[layer]  # [num_heads, seq_length, seq_length]
        k = min(k, seq_length)  # 确保k不超过序列长度
        v = v.view(seq_length, num_kv_heads, head_dim).permute(1, 0, 2)  # [num_kv_heads, seq_length, head_dim]
        # GQA对齐
        v_expend = v[:, None, :, :].expand(-1, num_kv_groups, -1, -1).contiguous().view(num_heads, seq_length, head_dim)
        target_attn = attn_weights[:,positions,:] # [num_heads, len(positions), seq_length]
        top_weights, top_indices = torch.topk(target_attn, k=k, dim=-1)  # [num_heads, len(positions), k]
        masked_target_attn = torch.zeros_like(target_attn)
        masked_target_attn.scatter_(-1, top_indices, top_weights) # [num_heads, len(positions), seq_length]

        # 归一化
        masked_target_attn = masked_target_attn / (masked_target_attn.sum(dim=-1, keepdim=True) + 1e-8)  # [num_heads, len(positions), seq_length]

        attned_v = torch.einsum("hps,hsl->hpl", masked_target_attn, v_expend)  # [num_heads, len(positions), head_dim], 这里只对positions位置的token进行加权求和
        attned_v = attned_v.permute(1, 0, 2).contiguous().view(len(positions), hidden_size)  # [len(positions), hidden_size]
        results.append(attned_v.cpu().detach())
    return results

@app.post("/get_top_k_attentioned_v")
def get_top_k_attentioned_v(request: InputRequest_attention):
    hooks = []
    try:
        texts = request.text
        aim_words = request.aim_word
        if len(texts) != len(aim_words):
            raise HTTPException(status_code=422, detail=f"输入文本列表和目标词列表的长度必须相同")
        batch_size = len(texts)

        input_device = model.get_input_embeddings().weight.device
        input = tokenizer(texts, padding=True, return_tensors="pt", return_offsets_mapping=True).to(input_device)

        batch_positions = []
        for b in range(batch_size):
            offset_mapping = input["offset_mapping"][b].tolist()
            full_tokens = tokenizer.convert_ids_to_tokens(input["input_ids"][b].tolist())
            pos = find_token_positions_by_char(texts[b], aim_words[b], offset_mapping, find_last=True)
            batch_positions.append(pos)
            if DEBUG:
                print(f"输入文本的 tokens: {full_tokens}")
                print(f"目标词 '{aim_words[b]}'")
                print(f"目标词 '{aim_words[b]}' 在输入文本中的最后位置索引: {pos}")
            if not pos:
                raise HTTPException(status_code=422, detail=f"无法在输入文本中找到目标词 '{aim_words[b]}' 的位置，可能是因为分词结果无法匹配到输入文本的 token 中")
        
        
        v_states = {}
        def get_v_hook(layer_idx):
            def hook(module, input, output):
                v_states[layer_idx] = output.detach().cpu()  # [batch_size, seq_length, hidden_size]
            return hook
        
        num_layers = len(model.model.layers)
        if isinstance(request.layer_idx, int):
            layer_indices = [request.layer_idx]
        elif isinstance(request.layer_idx, list):
            layer_indices = request.layer_idx
        else:
            layer_indices = range(num_layers)
        
        for i in layer_indices:
            if i >= num_layers:
                raise HTTPException(status_code=422, detail=f"层索引 {i} 超出范围")
            layer = model.model.layers[i]
            hook_v = layer.self_attn.v_proj.register_forward_hook(get_v_hook(i))
            hooks.append(hook_v)

        with torch.no_grad():
            outputs = model(**input, output_attentions=True)
        for h in hooks:
            h.remove()
        
        res_attned_v = []
        attns_batch = {
            i:outputs["attentions"][i].cpu().detach() for i in layer_indices
        }

        for b in range(batch_size):
            positions = batch_positions[b]
            if DEBUG:
                print(f"处理第 {b} 个输入文本，目标词位置索引: {positions}")
            if not positions:
                raise HTTPException(status_code=422, detail=f"无法在输入文本中找到目标词 '{aim_words[b]}' 的位置，可能是因为分词结果无法匹配到输入文本的 token 中")
            single_v_states = {
                layer:v_states[layer][b] for layer in layer_indices # [seq_length, hidden_size]
            }
            single_attns = {
                layer:attns_batch[layer][b] for layer in layer_indices # [num_heads, seq_length, seq_length]
            }
            if DEBUG:
                # print(f"第 {b} 个输入文本的 v_states 层数: {len(single_v_states)}, 每层 v_states 的 shape: {single_v_states[layer_indices[0]].shape if single_v_states else 'N/A'}")
                print(f"第 {b} 个输入文本的 attns 层数: {len(single_attns)}, 每层 attns 的 shape: {single_attns[layer_indices[0]].shape if single_attns else 'N/A'}")
            attned_v = get_top_k_attned_v(single_v_states, single_attns, layer_indices, positions, k=request.top_k[b] if request.top_k else 5) # list[tensor], 每个元素是一个层的目标词向量列表, shape: [num_tokens, hidden_size]
            if DEBUG:
                print(f"第 {b} 个输入文本获取到的 attned_v 层数: {len(attned_v)}, 每层 attned_v 的 shape: {[v.shape for v in attned_v]}")

            aggregated_attned_v = aggregate_vectors(attned_v, method=request.vector_mode) # shape: [hidden_size]
            res_attned_v.append(aggregated_attned_v.cpu().tolist())

        return {
            "status": "success",
            "attned_v": res_attned_v
        }
    except Exception as e:
        # 如果发生异常，务必也要卸载 Hooks 以免污染后续模型调用
        for h in hooks:
            h.remove()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" 允许外部访问
    import gc
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
