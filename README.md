# Fusion Gateway

Fusion Gateway 是一個基於 FastAPI 的情緒融合服務，用於 MoodJournal 中協調文字與語音情緒模型的推論結果。系統會同時呼叫兩個上游模型、套用晚期 (late) 融合策略，並輸出統一的預測回應，方便前端或其他後端系統直接使用。

---

## 核心特色
- 將既有的文字與語音情緒分析 API 封裝為單一 `/predict-fusion` 端點
- 透過 `alpha` 權重調整兩個模型的貢獻比例，並自動正規化機率分佈
- 回傳文字、語音與融合三組預測結果及各自的 top-1 標籤，便於呈現
- 提供 `/healthz` 健康檢查端點，可快速檢視設定與上游服務狀態
- 支援以環境變數設定 CORS 白名單與驗證標頭，部署更彈性

---

## 專案結構
```
fusion-gateway/
|-- main.py          # FastAPI 應用程式入口
|-- requirements.txt # Python 套件需求
|-- .env.example     # 環境變數範例
|-- .gitignore       # Git 忽略規則
`-- README.md
```

---

## 快速開始
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
啟動後造訪 `http://localhost:8000/docs`，即可查看自動產生的 OpenAPI 互動文件。

---

## 環境變數
複製 `.env.example` 為 `.env`，並依部署環境調整下列設定：

| 變數 | 說明 |
| --- | --- |
| `TEXT_API_BASE` | 上游文字情緒分析 API 根網址 (需提供 `POST /predict-text`) |
| `AUDIO_API_BASE` | 上游語音情緒分析 API 根網址 (需提供 `POST /predict-audio`) |
| `TEXT_API_AUTH_HEADER` / `TEXT_API_AUTH_VALUE` | 若文字 API 需驗證，填入標頭名稱與值 |
| `AUDIO_API_AUTH_HEADER` / `AUDIO_API_AUTH_VALUE` | 若語音 API 需驗證，填入標頭名稱與值 |
| `DEFAULT_ALPHA` | 未指定 `alpha` 時的預設融合權重，範圍 0.0 ~ 1.0 |
| `CORS_ORIGINS` | 允許的跨來源網址，使用逗號分隔 |

請將 `.env` 放在專案根目錄，FastAPI 啟動時會透過 python-dotenv 自動載入設定。

---

## API 介面
### `GET /healthz`
回傳目前設定的上游端點，並指示是否皆已設定：
```json
{
  "ok": true,
  "text_api": "https://example-text",
  "audio_api": "https://example-audio"
}
```

### `POST /predict-fusion`
表單欄位：
- `text`：必填，待分析的文字內容
- `file`：必填，上傳的語音檔案 (multipart/form-data)
- `alpha`：選填，0 到 1 之間的浮點數，控制文字模型權重 (預設 `DEFAULT_ALPHA`)

範例請求：
```bash
curl -X POST http://localhost:8000/predict-fusion \
  -F "text=今天心情很好" \
  -F "file=@sample.wav" \
  -F "alpha=0.7"
```

範例回應：
```json
{
  "text_pred": {"pos": 0.62, "neu": 0.28, "neg": 0.10},
  "audio_pred": {"pos": 0.48, "neu": 0.41, "neg": 0.11},
  "fusion_pred": {"pos": 0.57, "neu": 0.34, "neg": 0.09},
  "text_top1": "pos",
  "audio_top1": "pos",
  "fusion_top1": "pos",
  "alpha": 0.7,
  "labels": ["pos", "neu", "neg"]
}
```

---

## 開發建議
- 開發期間可使用 `uvicorn main:app --reload` 取得熱重載體驗
- 建議使用 pytest 搭配 httpx 撰寫端對端測試，覆蓋融合邏輯與錯誤處理
- 若新增設定項目，記得同步更新 `.env.example` 與本文件
