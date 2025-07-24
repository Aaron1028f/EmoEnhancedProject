### TODO
- 加入one euro filter


### 0725: one euro filter 測試
https://blog.csdn.net/gitblog_00431/article/details/144601848
https://github.com/jaantollander/OneEuroFilter

```bash
# pip install one-euro-filter
pip install oneeurofilter
```
主要參數：

- min_cutoff (f_c,min): 最小截止頻率。這是訊號緩慢移動時的基礎平滑程度。值越低，抖動抑制越強，但靜態延遲會略高。

- beta (beta): 截止頻率的反應靈敏度。這個參數控制截止頻率隨訊號速度變化的程度。值越高，濾波器在偵測到快速移動時，會更「激進」地提高截止頻率以減少延遲。

- d_cutoff (f_c,d): 導數濾波器的截止頻率。通常保持預設值即可（例如 1.0 Hz）。
