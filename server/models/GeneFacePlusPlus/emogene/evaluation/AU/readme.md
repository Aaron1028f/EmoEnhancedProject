## AU evaluation

dataset: RAVDESS

compare: GeneFace++ and EmoGene

AU detection toolkit: py-feat

[py-feat AU explaination](https://py-feat.org/pages/au_reference.html)

### AU 重要觀察方向
[各種情緒所需要的AU組合](https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/)

- Happiness / Joy: 6(Cheek Raiser) + 12(Lip Corner Puller)
- Sadness: 1(Inner Brow Raiser) + 4(Brow Lowerer) + 15(Lip Corner Depressor)





## mean emotion scores
計算所有24個演員的相同影片
(編號一樣的部分，情緒相同，只採取激烈情緒、"Kids are talking by the door.")

每一個影片都會得到一組數值: emotion means，也就是該情緒在影片中的平均分數(由py-feat偵測到的)

測量方法一: 只計算目標情緒的占比，並比較genefacepp和emogene兩者在該情緒中的占比差距

```python
# dataset: RAVDESS
# 對於每個情緒，總共會有 (24 個演員) * (2次重複)，先只採用 "Kids are talking by the door."
VIDEO_DIR = '/Actor_{actor_id}/03-01-0{emo_id}-02-01-01-{actor_id}.mp4' # 第一次重複
VIDEO_DIR = '/Actor_{actor_id}/03-01-0{emo_id}-02-01-02-{actor_id}.mp4' # 第二次重複
# 隨後分別計算 來自 genefacepp以及emogene 的該情緒平均分數，以及每個情緒對應的AU
# plot 出表格，表格為以下 pseudo graph
# -------------------------------| happy     | sad            | angry                | fearful                            | disgust         | surprised           | AU25
# genefacepp avg emotion means   |           |
# emogene avg emotion means      |           |

# ----------------------------intensity-----------------------------
# genefacepp emotion key AUs means   | AU6, AU12 | AU1, AU4, AU15 | AU4, AU5, AU7, AU23  | AU1, AU2, AU4, AU5, AU7, AU20, AU26| AU9, AU15, AU16 | AU1, AU2, AU5, AU26 |
# emogene emotion AUs means          | AU6, AU12 | AU1, AU4, AU15 | AU4, AU5, AU7, AU23  | AU1, AU2, AU4, AU5, AU7, AU20, AU26| AU9, AU15, AU16 | AU1, AU2, AU5, AU26 |

# genefacepp emotion key AUs peak (max score)  |
# emogene emotion AUs peak (max score)    | 

# ----------------------------variability-----------------------------
# genefacepp emotion key AUs std     | AU6, AU12 | AU1, AU4, AU15 | AU4, AU5, AU7, AU23  | AU1, AU2, AU4, AU5, AU7, AU20, AU26| AU9, AU15, AU16 | AU1, AU2, AU5, AU26 |
# emogene emotion key AUs std        | AU6, AU12 | AU1, AU4, AU15 | AU4, AU5, AU7, AU23  | AU1, AU2, AU4, AU5, AU7, AU20, AU26| AU9, AU15, AU16 | AU1, AU2, AU5, AU26 |



```

## run the evaluation
```bash
conda activate pyfeat

# emotional video evaluation
cd server/models/GeneFacePlusPlus/emogene/evaluation/AU/
python pyfeat_eval_script.py

# neutral emotion video evaluation
cd server/models/GeneFacePlusPlus/emogene/evaluation/AU/
python pyfeat_eval_script_neutral.py

# !!! no longer use pyfeat_eval_plot.py !!!!!

# run sigle video evaluation
python eval_May_raw_vid.py

CUDA_VISIBLE_DEVICES=0 python eval_May_raw_vid.py

```


## py-feat support AUs

au_mean_AU01       | au_mean_AU02       | au_mean_AU04       | au_mean_AU05       | au_mean_AU06        | au_mean_AU07       | au_mean_AU09       | au_mean_AU10       | au_mean_AU11       | au_mean_AU12        | au_mean_AU14        | au_mean_AU15       | au_mean_AU17    | au_mean_AU20       | au_mean_AU23        | au_mean_AU24        | au_mean_AU25       | au_mean_AU26       | au_mean_AU28        | au_mean_AU43        