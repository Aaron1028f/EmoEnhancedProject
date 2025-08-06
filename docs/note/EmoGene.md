### TODO
- 加入one euro filter (已完成)
- 改善geneface lm3d 和 emotalk lm3d 的疊加方法


### 0725: one euro filter 測試
https://blog.csdn.net/gitblog_00431/article/details/144601848
https://github.com/jaantollander/OneEuroFilter

```bash
# pip install one-euro-filter
# pip install oneeurofilter
source: https://github.com/jaantollander/OneEuroFilter
```
主要參數：

- min_cutoff (f_c,min): 最小截止頻率。這是訊號緩慢移動時的基礎平滑程度。值越低，抖動抑制越強，但靜態延遲會略高。

- beta (beta): 截止頻率的反應靈敏度。這個參數控制截止頻率隨訊號速度變化的程度。值越高，濾波器在偵測到快速移動時，會更「激進」地提高截止頻率以減少延遲。

- d_cutoff (f_c,d): 導數濾波器的截止頻率。通常保持預設值即可（例如 1.0 Hz）。

### 0726: 動態疊加lm3d方法實驗


### 0729

使用method 6: 穩定May

使用method 7: 穩定方醫師(已有微笑和生氣的表情)，但會有部分偽影問題，待解決
```
eyebrow: emotalk
eye: emotalk
mouth: emotalk + geneface
other parts: use geneface

```

### 0730 

best combination

Feng: 
- method(BS Landmarks Area) 9 (通過減少下顎的變化量，解決偽影問題)
- bs52 level: 1~1.5

May: 
- method(BS Landmarks Area) 8 (極少出現偽影問題)(也可以使用9，穩定脖子以下的部分，但會增加臉部出現模糊的機率)
- bs52 level: 2

other problem: 不知為何，使用calm時嘴唇會模糊