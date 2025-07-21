# pip install yt-dlp youtube-transcript-api pyannote.audio

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from pyannote.audio import Pipeline
import torch

# 替換成您的 Hugging Face access token
# HF_TOKEN = "YOUR_HUGGING_FACE_TOKEN"
HF_TOKEN = "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"

# 1. 下載音訊
def download_audio(youtube_url, output_path='audio.wav'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace('.wav', '')
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

# 2. 提取 CC 字幕
def get_transcript(video_id):
    try:
        # transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'zh-TW', 'ja']) # 可自行增加語言
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['zh-TW']) # 可自行增加語言
        return transcript_list
    except Exception as e:
        print(f"無法提取字幕: {e}")
        return None

# 3. 說話人分析
def speaker_diarization(audio_path, hf_token):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    # 將模型移至 GPU (如果可用)
    if torch.cuda.is_available():
      pipeline.to(torch.device("cuda"))

    diarization = pipeline(audio_path)
    return diarization

# --- 主程式 ---
if __name__ == '__main__':
    # # YOUTUBE_URL = 'https://www.youtube.com/watch?v=XXXXXXXXXXX' # 換成您要處理的 YouTube 影片網址
    # # YOUTUBE_URL = 'https://www.youtube.com/watch?v=BxExJ3cMAtY&list=PLDxCClQ3DiNsqKjdyCFj9qY102L4sHobe&index=34' # 換成您要處理的 YouTube 影片網址
    # YOUTUBE_URL = 'https://www.youtube.com/watch?v=BxExJ3cMAtY'
    
    # # VIDEO_ID = 'BxExJ3cMAtY' # 從網址中提取
    # VIDEO_ID = YOUTUBE_URL.split('v=')[-1].split('&')[0] if 'v=' in YOUTUBE_URL else YOUTUBE_URL.split('/')[-1]
    # print(f"處理 YouTube 影片: {YOUTUBE_URL} (ID: {VIDEO_ID})")

    # =======================================================================================================
    YOUTUBE_URL_LIST = [
        'https://www.youtube.com/watch?v=BxExJ3cMAtY', # live EP1
        'https://www.youtube.com/watch?v=eQqS85aH9lI', # live EP2
        'https://www.youtube.com/watch?v=eAEBnBENExw', # live EP3
    ]
    VIDEO_ID_LIST = [url.split('v=')[-1].split('&')[0] if 'v=' in url else url.split('/')[-1] for url in YOUTUBE_URL_LIST]
    # print(f"處理 YouTube 影片: {YOUTUBE_URL_LIST} (ID: {VIDEO_ID_LIST})")
    
    # =======================================================================================================
    
    for YOUTUBE_URL, VIDEO_ID in zip(YOUTUBE_URL_LIST, VIDEO_ID_LIST):
    
    
        output_transcript_file = f'output_transcript_{VIDEO_ID}.txt'
        output_transcript_list = []
        
        # 執行步驟
        print(f"\n處理 YouTube 影片: {YOUTUBE_URL} (ID: {VIDEO_ID})")
        print("步驟 1: 下載音訊中...")
        audio_file = download_audio(YOUTUBE_URL, output_path=f'{VIDEO_ID}.wav')

        print("步驟 2: 提取字幕中...")
        subtitles = get_transcript(VIDEO_ID)

        if subtitles:
            print("步驟 3: 進行說話人分析 (可能需要一些時間)...")
            diarization_result = speaker_diarization(audio_file, HF_TOKEN)

            # 4. 對齊字幕與說話人
            print("\n--- 加上說話人標記的字幕 ---")
            for segment in subtitles:
                start_time = segment['start']
                end_time = start_time + segment['duration']
                
                # 找出這段字幕時間內，最主要的說話者
                dominant_speaker = None
                max_overlap = 0

                for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                    overlap = max(0, min(end_time, turn.end) - max(start_time, turn.start))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        dominant_speaker = speaker
                
                # 5. 輸出結果
                # if dominant_speaker:
                #     print(f"[{dominant_speaker}] ({segment['start']:.2f}s): {segment['text']}")
                # else:
                #     print(f"[未知說話人] ({segment['start']:.2f}s): {segment['text']}")
                
                # 加入這次的說話人標記到輸出列表
                if dominant_speaker:
                    # output_transcript_list.append(f"[{dominant_speaker}] ({segment['start']:.2f}s): {segment['text']}")
                    output_transcript_list.append(f"[{dominant_speaker}] {segment['text']}")
                else:
                    output_transcript_list.append(f"[未知說話人] {segment['text']}")
                
                
            # # 將同一個說話人的連續字幕合併
            # merged_transcript = []
            # current_speaker = None
            # current_text = ""
            # for line in output_transcript_list:
            #     speaker = line.split(']')[0] + ']'
            #     text = line.split('] ')[1]
                
            #     if speaker == current_speaker:
            #         current_text += " " + text
            #     else:
            #         if current_speaker is not None:
            #             merged_transcript.append(f"{current_speaker} {current_text.strip()}")
            #         current_speaker = speaker
            #         current_text = text
            # # 添加最後一個說話人的文本
            # if current_speaker is not None:
            #     merged_transcript.append(f"{current_speaker} {current_text.strip()}")
                
            # print("\n--- 合併後的字幕 ---")
            # # 輸出結果到.txt檔案
            # with open(output_transcript_file, 'w', encoding='utf-8') as f:
            #     f.write("\n".join(merged_transcript))
            # print(f"\n結果已保存到 {output_transcript_file}")
            with open(output_transcript_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(output_transcript_list))
            print(f"\n結果已保存到 {output_transcript_file}")
        else:
            print("無法提取字幕，請檢查影片是否有可用的字幕。")
