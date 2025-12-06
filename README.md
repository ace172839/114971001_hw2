# NCCU Generative AI Assignment 2

本作業分為兩條路徑：

1. `traditional_methods.py`
   - 以演算法或人工定義規則，完成：
     - 字句相似度判斷（TF-IDF + cosine similarity）
     - 字句內容分類（情感與主題規則庫）
     - 文章選取摘要（統計式與進階統計式摘要器）
2. `modern_methods.py`
   - 透過呼叫 AI API（GPT-4o / Gemini 等生成式模型）完成同樣任務：
     - 語意相似度（模型評分）
     - 文本分類（情感 + 主題 JSON 輸出）
     - 生成式摘要（長度可控）

`comparison.py` 會同時呼叫兩套方法，將結果輸出至 `results/` 資料夾，包含：

- `tfidf_similarity_matrix.png`
- `classification_results.csv`
- `summarization_comparison.txt`
- `performance_metrics.json`

## 使用方式

1. 安裝依賴：`pip install -r requirements.txt`
2. 若需呼叫 AI API，請先設定 `OPENAI_API_KEY` 或對應環境變數。
3. 執行 `python comparison.py` 產出所有比較檔案。



### Advanced ###
另外，我設計了幾個更複雜、更有針對性的 testcases，期待能讓結果更明顯
執行檔為 comparison_advanced.py，輸出結果在 results_advanced

