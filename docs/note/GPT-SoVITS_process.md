
### steps

1. slice the audio
    - 0b: input: `/home/aaron/project/server/models/tts_model/GPT-SoVITS/DATA/Feng_EP32/raw/Feng_live_EP32.wav`
    - 0b: output: `/home/aaron/project/server/models/tts_model/GPT-SoVITS/DATA/Feng_EP32/slicer`
2. speech asr
    - 0c: input: `/home/aaron/project/server/models/tts_model/GPT-SoVITS/DATA/Feng_EP32/slicer` # 會自動出現
    - 0c: output: `/home/aaron/project/server/models/tts_model/GPT-SoVITS/DATA/Feng_EP32/list`

3. proofreading
    - 0d: input: `/home/aaron/project/server/models/tts_model/GPT-SoVITS/DATA/Feng_EP32/list/slicer.list` # 會自動出現
    - 使用方法
        - 每次切換頁面`next index`前點`submit text`保存修改
        - 可以`yes -> delete audio -> save file` 來刪除不要的文件
        - 可以`yes -> merge audio -> save file` 來合併文件

---

### inference

目前效果最好的模型:
GPT: `GPT_weights_v2ProPlus/Feng_EP32_01-e40.ckpt` or `GPT_weights_v2ProPlus/Feng_EP32_01-e45.ckpt`
SoVITS: `SoVITS_weights_v2ProPlus/Feng_EP32_01_e25_s150.pth` # 主要是這個影響

目前使用的資料集: https://www.youtube.com/watch?v=-xFxeotJLnQ&list=PLDxCClQ3DiNsqKjdyCFj9qY102L4sHobe&index=1
約只有16分鐘，所以效果普通

後續可能需要至少1小時的訓練資料來訓練，以獲得較好的結果
