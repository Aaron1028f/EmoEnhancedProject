#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
from funasr import AutoModel

SLICER_LIST = "/home/aaron/project/server/models/RAG/audio_prompt_selection/DATA/slicer.list"
OUT_JSON = "/home/aaron/project/server/models/RAG/audio_prompt_selection/DATA/emotion_map.json"

MAPPING_TABLE = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']

def iter_wavs_from_slicer(list_path: str):
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            try:
                audio_path, _, _, _ = line.split("|", 3)
                yield audio_path
            except Exception:
                continue

def main():
    model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)
    mapping: Dict[str, Dict] = {}
    wavs = list(dict.fromkeys(iter_wavs_from_slicer(SLICER_LIST)))  # 去重，保持順序
    if not wavs:
        raise SystemExit(f"slicer.list 無音檔：{SLICER_LIST}")

    for wav in tqdm(wavs, desc="情緒標記中"):
        try:
            res = model.generate(wav, output_dir=None, granularity="utterance", extract_embedding=True)
            scores = res[0]["scores"]
            # 取最高分的情緒與分數
            best: Tuple[str, float] = max(zip(MAPPING_TABLE, scores), key=lambda x: x[1])
            mapping[wav] = {"emotion": best[0], "emotion_score": float(best[1])}
        except Exception as e:
            mapping[wav] = {"emotion": "unknown", "emotion_score": 0.0}

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"已輸出：{OUT_JSON}，共 {len(mapping)} 筆")

if __name__ == "__main__":
    main()