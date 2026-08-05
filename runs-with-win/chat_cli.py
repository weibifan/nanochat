"""
交互式聊天：加载本地 SFT 微调模型，对话测试
路径来自 config.yaml（相对本目录）
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
torch._dynamo.config.disable = True
from nanochat.tokenizer import RustBPETokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import load_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "config.yaml"), "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
def rel(p):
    return os.path.normpath(os.path.join(HERE, p))

TOKENIZER_DIR = rel(cfg["tokenizer"])
CKPT_DIR = rel(cfg.get("sft", {}).get("checkpoint_dir", "sft_checkpoint"))
CKPT_STEP = cfg.get("sft", {}).get("steps", 50)
MAX_NEW_TOKENS = 80

tokenizer = RustBPETokenizer.from_directory(TOKENIZER_DIR)
print(f"Tokenizer: 词表={tokenizer.get_vocab_size()}")

model_data, _, meta = load_checkpoint(CKPT_DIR, CKPT_STEP, device="cpu")
config = GPTConfig(**meta["model_config"])
with torch.device("meta"):
    model = GPT(config)
model = model.to_empty(device="cpu")
model.init_weights()
model.load_state_dict(model_data, strict=True, assign=True)
model.eval()
print(f"模型: {config.n_layer}层/{config.n_embd}维 | loss={meta['loss']:.4f}")
print()

ast_end = tokenizer.encode_special("<|assistant_end|>")
bos = tokenizer.get_bos_token_id()

def chat(user_input):
    ids = tokenizer.encode(user_input, prepend="<|bos|>", append="<|assistant_start|>")
    prompt = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(prompt)
            next_id = torch.argmax(logits[0, -1, :]).item()
            prompt = torch.cat([prompt, torch.tensor([[next_id]])], dim=1)
            if next_id == ast_end:
                break
    return tokenizer.decode(prompt[0].tolist())

print("输入问题直接回车（空行退出）")
print("-" * 40)
while True:
    try:
        q = input("You: ")
    except (EOFError, KeyboardInterrupt):
        break
    if not q.strip():
        break
    r = chat(q.strip())
    print(f"AI:  {r}")
    print()
