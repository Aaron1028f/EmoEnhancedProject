#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Any, Dict
import requests


def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 請求失敗：{e}")
        sys.exit(1)


def cmd_reindex(base_url: str):
    url = f"{base_url.rstrip('/')}/reindex"
    print(f"[INFO] POST {url}")
    resp = post_json(url, {})
    print(json.dumps(resp, ensure_ascii=False, indent=2))


def cmd_retrieve(base_url: str, query: str, k: int, desired_emotion: str = None, emotion_mode: str = "off"):
    url = f"{base_url.rstrip('/')}/retrieve"
    payload = {"query": query, "k": k, "desired_emotion": desired_emotion, "emotion_mode": emotion_mode}
    print(f"[INFO] POST {url}")
    print(f"[INFO] payload: {json.dumps(payload, ensure_ascii=False)}")
    resp = post_json(url, payload)

    hits = resp.get("hits", [])
    if not hits:
        print("[WARN] 沒有找到相似結果。")
        return

    print("\n[RESULT] 相似結果：")
    for i, h in enumerate(hits, 1):
        transcript = h.get("transcript", "")
        audio_path = h.get("audio_path", "")
        filename = h.get("filename", "")
        score = h.get("score", None)
        emotion = h.get("emotion", None)
        emotion_score = h.get("emotion_score", None)
        print(f"- #{i}")
        print(f"  score    : {score}")
        print(f"  filename : {filename}")
        print(f"  audio    : {audio_path}")
        print(f"  emotion  : {emotion} ({emotion_score})")
        print(f"  text     : {transcript[:120]}{'...' if len(transcript) > 120 else ''}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="RAG audio prompt selection 測試用戶端",
        epilog="""
使用範例：
  1) 重新建立索引
     python test_client.py reindex

  2) 查詢（預設 k=3）
     python test_client.py retrieve -q "安樂死" -k 3

  3) 指定自訂伺服器位址
     python test_client.py retrieve -q "腿不寧症候群" --base-url http://127.0.0.1:45000
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:45000",
        help="伺服器根網址（預設：http://127.0.0.1:45000）",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reindex = sub.add_parser("reindex", help="重建索引")
    _ = p_reindex  # no args

    p_retrieve = sub.add_parser("retrieve", help="相似度查詢")
    p_retrieve.add_argument("-q", "--query", required=True, help="查詢文字")
    p_retrieve.add_argument("-k", "--k", type=int, default=3, help="回傳前 k 筆（1-20）")
    p_retrieve.add_argument("-e", "--emotion", default=None, help="期望情緒（例如：happy）")
    p_retrieve.add_argument("--mode", choices=["off", "soft", "strict"], default="off", help="情緒模式：off/soft/strict")

    args = parser.parse_args()

    if args.cmd == "reindex":
        cmd_reindex(args.base_url)
    elif args.cmd == "retrieve":
        k = max(1, min(20, int(args.k)))
        cmd_retrieve(args.base_url, args.query, k, args.emotion, args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    
# 測試查詢
# 純語義：python test_client.py retrieve -q "安樂死"
# 軟性重排（優先 happy）：python test_client.py retrieve -q "安樂死" -e happy --mode soft
# 嚴格只要 happy（找不到會回退軟性）：python test_client.py retrieve -q "安樂死" -e happy --mode strict