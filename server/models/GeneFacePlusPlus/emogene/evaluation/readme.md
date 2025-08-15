# Evaluation of EmoGene

## LSE-D and LSE-C metric

source:  
[Evaluation Framework from Wav2Lip](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation)  
[syncnet source code](https://github.com/joonson/syncnet_python)


#### gemini explaination
- `LSE-D (Lip-Sync Error - Distance)`  
LSE-D 是最直接的指標，它衡量的是音訊嵌入和視覺嵌入之間的歐幾里得距離 (Euclidean Distance)。  
計算方式： `LSE-D = || Audio_Embedding - Visual_Embedding ||  `
解讀： 這個數值越低越好。一個較低的 LSE-D 值意味著在 SyncNet 的判斷中，您生成的嘴型與驅動音訊的內容高度匹配。
- `LSE-C (Lip-Sync Error - Confidence)`  
LSE-C 是一個更穩健、更嚴格的指標，它衡量的是模型對於「這個配對是正確的」這件事有多麼自信。  
計算方式:  
取一段短影片（例如5幀）和其對應的正確音訊。
同時，再從影片的其他地方隨機抽取 N 個錯誤的音訊片段（例如 N=100），我們稱之為「干擾項」。
計算這段短影片與「1個正確音訊 + N個錯誤音訊」中每一個音訊的 LSE-D，得到 N+1 個距離值。
理論上，與正確音訊的距離應該是最小的。LSE-C 就是衡量這個「最小距離」到底比其他距離小了多少，通常透過一個類似 Softmax 的函數來計算其置信度分數。
解讀： 這個數值越高越好。一個高的 LSE-C 值（例如 >90%）意味著 SyncNet 能在眾多干擾項中，非常有信心地指出只有那個正確的音訊才是與影片匹配的。這個指標能有效防止模型只學會生成一個「平均嘴型」來矇混過關。


---

## FID?



## AUE? (using openface)


## 