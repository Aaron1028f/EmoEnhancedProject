import requests

def main():
    while True:
        text = input("請輸入文字 (輸入 exit 結束): ")
        if text.lower() == "exit":
            break
        response = requests.post("http://localhost:8000/generate", json={"text": text})
        if response.ok:
            result = response.json()
            print("回答:", result.get("answer"))
            print("影片路徑:", result.get("video_url"))
        else:
            print("發生錯誤:", response.text)

if __name__ == "__main__":
    main()
