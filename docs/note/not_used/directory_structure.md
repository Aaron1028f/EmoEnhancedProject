for reference: the dir should be like this(chatgpt recommendation)

project-root/
├── server/                     # 後端程式碼
│   ├── main.py                 # FastAPI 啟動檔
│   ├── routers/                # 路由定義
│   │   ├── rag_router.py
│   │   └── embed_router.py
│   ├── services/               # 核心邏輯（Embeddings / FAISS / RAG）
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   └── rag_service.py
│   ├── models/                 # Pydantic schema
│   │   └── schemas.py
│   ├── utils/                  # 共用工具、設定檔
│   │   └── config.py
│   ├── requirements.txt        # Python 套件清單
│   └── Dockerfile              # （可選）容器化設定
│
├── client/                     # 前端程式碼
│   ├── package.json            # NPM 套件清單
│   ├── public/                 # 靜態資源（index.html, favicon…）
│   └── src/                    # React/Vite/Next 原始碼
│       ├── index.jsx           # 入口
│       ├── App.jsx             # 主元件
│       ├── components/         # UI 元件
│       │   ├── ChatWindow.jsx
│       │   └── MessageList.jsx
│       ├── services/           # 對後端 API 的封裝
│       │   └── apiClient.js    # fetch/axios wrapper
│       └── styles/             # CSS / Sass / Tailwind 設定
│
├── docker-compose.yml          # （可選）前後端一鍵啟動
└── README.md                   # 專案說明
