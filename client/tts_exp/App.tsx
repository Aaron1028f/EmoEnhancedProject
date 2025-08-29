import React, { useState, useRef, useEffect } from 'react';

const SendIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="currentColor"
    className="w-6 h-6"
  >
    <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
  </svg>
);

const LoadingSpinner: React.FC = () => (
  <svg
    className="animate-spin h-6 w-6 text-white"
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
  >
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
    ></circle>
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    ></path>
  </svg>
);

const App: React.FC = () => {
  const [text, setText] = useState<string>('先帝創業未半而中道崩殂，今天下三分，益州疲弊，此誠危急存亡之秋也。');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaSourceUrlRef = useRef<string | null>(null);

  // Effect for cleaning up the object URL on component unmount
  useEffect(() => {
    return () => {
      if (mediaSourceUrlRef.current) {
        URL.revokeObjectURL(mediaSourceUrlRef.current);
      }
    };
  }, []);


  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!text.trim()) {
      setError('請輸入一些文字。');
      return;
    }

    setIsLoading(true);
    setError(null);

    // Clean up any existing object URL to prevent memory leaks
    if (mediaSourceUrlRef.current) {
      URL.revokeObjectURL(mediaSourceUrlRef.current);
      mediaSourceUrlRef.current = null;
    }
    setAudioUrl(null);

    try {
      const mediaSource = new MediaSource();
      const url = URL.createObjectURL(mediaSource);
      mediaSourceUrlRef.current = url;
      setAudioUrl(url);

      mediaSource.addEventListener('sourceopen', async () => {
        // The URL is now attached to the audio element, we can revoke it from the global scope
        // The browser will keep the reference alive until the MediaSource is done.
        URL.revokeObjectURL(url);

        try {
          const mimeCodec = 'audio/webm; codecs=opus';
          if (!MediaSource.isTypeSupported(mimeCodec)) {
            throw new Error(`您的瀏覽器不支持必要的音訊格式: ${mimeCodec}。請嘗試更新您的瀏覽器。`);
          }
          const sourceBuffer = mediaSource.addSourceBuffer(mimeCodec);
          // Set mode to 'sequence' to append buffers sequentially, ignoring their internal timestamps.
          // This is crucial for stitching together streams that consist of multiple independent segments.
          sourceBuffer.mode = 'sequence';

          const params = new URLSearchParams({
            text: text,
            text_lang: 'zh',
            ref_audio_path: 'DATA/Feng_EP32/slicer/Feng_live_EP32.wav_0001674880_0001811840.wav',
            prompt_text: '生命是我的所有權，喔，我要去自殺。',
            prompt_lang: 'zh',
            text_split_method: 'cut5',
            media_type: 'webm',
            streaming_mode: 'true',
          });

          const API_BASE_URL = 'http://127.0.0.1:9880/tts';
          const response = await fetch(`${API_BASE_URL}?${params.toString()}`);

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || `API 錯誤 (${response.status})`);
          }

          if (!response.body) {
            throw new Error("API 回應沒有內容。");
          }

          const reader = response.body.getReader();
          let isFirstChunk = true;
          
          // Stream processing loop
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }

            // Wait for the previous append to complete before adding a new one
            await new Promise<void>((resolve, reject) => {
              const onUpdateEnd = () => {
                sourceBuffer.removeEventListener('updateend', onUpdateEnd);
                sourceBuffer.removeEventListener('error', onError);
                resolve();
              };
              const onError = (err: Event) => {
                sourceBuffer.removeEventListener('updateend', onUpdateEnd);
                sourceBuffer.removeEventListener('error', onError);
                reject(err);
              };

              sourceBuffer.addEventListener('updateend', onUpdateEnd);
              sourceBuffer.addEventListener('error', onError);
              sourceBuffer.appendBuffer(value);
            });
            
            if (isFirstChunk && audioRef.current) {
              isFirstChunk = false;
              audioRef.current.play().catch(e => {
                console.error("Autoplay failed:", e);
                setError("瀏覽器阻止了自動播放。請點擊播放按鈕。");
              });
            }
          }

          if (mediaSource.readyState === 'open') {
            mediaSource.endOfStream();
          }

        } catch (err: unknown) {
          console.error("Streaming Error:", err);
           if (err instanceof Error) {
              if (err.message.includes('Failed to fetch')) {
                  setError('網路請求失敗。請確認後端服務器正在運行，並且已正確設定 CORS。');
              } else {
                  setError(err.message || '無法獲取音頻。');
              }
            } else {
              setError('發生未知錯誤。');
            }
          // Ensure stream is properly closed on error
          if (mediaSource.readyState === 'open') {
            mediaSource.endOfStream();
          }
        } finally {
            setIsLoading(false);
        }
      }, { once: true });

    } catch (err) {
      console.error("Setup Error:", err);
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("建立音訊串流時發生未知錯誤。");
      }
      setIsLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4 font-sans">
      <div className="w-full max-w-3xl space-y-8">
        <header className="text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-cyan-400">
            語音合成工具
          </h1>
          <p className="text-lg text-gray-400 mt-2">輸入文字以生成語音</p>
        </header>

        <main className="w-full">
          <form onSubmit={handleSubmit}>
            <div className="flex items-center bg-gray-800 border-2 border-gray-700 rounded-full shadow-lg overflow-hidden focus-within:border-cyan-500 transition-colors">
              <input
                type="text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="在此輸入文字..."
                className="w-full bg-transparent p-4 text-lg text-white placeholder-gray-500 focus:outline-none"
                disabled={isLoading}
                aria-label="要轉換為語音的文字"
              />
              <button
                type="submit"
                disabled={isLoading}
                className="bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white p-3 sm:p-4 rounded-full m-2 transition-all duration-300 transform hover:scale-105"
                aria-label="生成語音"
              >
                {isLoading ? <LoadingSpinner /> : <SendIcon />}
              </button>
            </div>
          </form>
        </main>

        <footer className="mt-8 text-center h-24 flex items-center justify-center">
          {error && (
            <div role="alert" className="text-red-400 bg-red-900/50 p-3 rounded-lg w-full max-w-md">
              <p className="font-semibold">錯誤</p>
              <p>{error}</p>
            </div>
          )}
          {audioUrl && !error && (
            <div className="w-full">
                <audio controls src={audioUrl} ref={audioRef} className="w-full h-14 rounded-full">
                    您的瀏覽器不支持音頻元素。
                </audio>
            </div>
          )}
        </footer>
      </div>
    </div>
  );
};

export default App;
