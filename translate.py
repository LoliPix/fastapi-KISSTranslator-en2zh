import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from model import TranslationItem, TranslationRequest, TranslationResponse

model_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="eng_Latn", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,
    dtype='auto',
    local_files_only=True
).to(device)

def translate(texts: list[str], target_lang="zho_Hans") -> list[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,      # 自动用 [PAD] 填充短句子
        truncation=True,   # 自动截断超长句子
        max_length=128     # 明确限制最大长度，防止显存爆炸
    ).to(device)
    try:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)
    except Exception:
        forced_bos_token_id = tokenizer.lang_code_to_id.get(target_lang)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=5,               # 维持在 5-8 之间
            # --- 核心改动 ---
            length_penalty=1.5,        # 增大惩罚，分值越高，模型越倾向于输出长句子（比如“苹果”而不是“果”）
            no_repeat_ngram_size=2,    # 更加严格地防止短词重复
            repetition_penalty=2.0,    # 强迫模型寻找更有意义的词
            early_stopping=True
        )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

# 运行
def run(req: TranslationRequest):
    try:
        result: list[str] = translate(req.texts)
        return result
    except Exception as e:
        raise RuntimeError(f"\n运行出错: {e}")

# 打包
def packup(result_list:list[str]) -> TranslationResponse:
    items = [TranslationItem(text=f'{item}') for item in result_list]
    return TranslationResponse(translations=items)
