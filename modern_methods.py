from logging import config
import os
import time
import json
import google.generativeai as genai
from google.api_core import exceptions

from tests.testcases import documents, test_texts, article
# from tests.testcases_advance import documents, test_texts, article


# Gemini API快取，避免重複呼叫
_API_CACHE = {}
MODEL_NAME = "gemini-2.0-flash" 

with open("../key.json", "r") as f:
    keys = json.load(f)
API_KEY = keys.get("gemini_api_key", None)
if not API_KEY:
    print("No api key found in key.json or environment variables")
    exit(1)


def _call_gemini_with_retry(contents, max_retries=3, json_mode=False):
    """
    封裝 Gemini API 呼叫，包含快取與指數退避重試機制
    這部分請 Gemini 幫忙產生的
    """
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    
    # 設定生成參數，若需要 JSON 模式則開啟
    if json_mode:
        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    else:
        generation_config = genai.types.GenerationConfig(temperature=0.1)

    # 1. 檢查快取
    # 加入 json_mode 到 cache key 以避免格式混淆
    cache_key = json.dumps({
        "model": MODEL_NAME, 
        "content": str(contents), 
        "json_mode": json_mode
    }, sort_keys=True)
    
    if cache_key in _API_CACHE:
        return _API_CACHE[cache_key]

    # 2. 執行 API 呼叫 (含重試邏輯)
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                contents,
                generation_config=generation_config
            )
            result = response.text
            
            # 3. 寫入快取
            _API_CACHE[cache_key] = result
            return result

        except exceptions.ResourceExhausted:
            if attempt == max_retries - 1:
                print(f"[Error] Quota exceeded after {max_retries} attempts.")
                return None
            sleep_time = 2 ** attempt
            # print(f"[Warning] Quota exceeded. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[Error] API Error: {e}")
                return None
            sleep_time = 2 ** attempt
            print(f"[Warning] Request failed ({e}). Retrying in {sleep_time}s...")
            time.sleep(sleep_time)


def ai_similarity(text1, text2, api_key=None):
    prompt = f"""
    請評估以下兩段文字的語意相似度。
    
    文字1: {text1}
    文字2: {text2}

    請「只回答一個 0 到 100 的整數」，代表相似度百分比。不要包含任何解釋。
    """
    
    response_text = _call_gemini_with_retry(prompt)
    
    if response_text:
        try:
            # 清理非數字字符
            cleaned_response = ''.join(filter(str.isdigit, response_text))
            return float(cleaned_response) if cleaned_response else 0.0
        except ValueError:
            return 0.0
    return 0.0


def ai_classify(text, api_key=None):
    prompt = f"""
    請分析以下文本的情感與主題。
    
    文本內容: "{text}"

    請輸出 JSON 格式:
    {{
        "sentiment": "positive", "negative", or "neutral",
        "topic": "科技", "運動", "美食", "旅遊", or "other",
        "confidence": 0.0 to 1.0,
        "keywords": {{ "keyword": weight }},
        "topic_keywords": ["keyword1", "keyword2"]
    }}
    注意：sentiment 請 mapping 到英文 (positive/negative/neutral) 以符合測試輸出格式。
    """
    
    response_text = _call_gemini_with_retry(prompt, json_mode=True)
    
    default_error_response = {
        "sentiment": "error", 
        "topic": "error", 
        "confidence": 0.0,
        "keywords": {},
        "topic_keywords": []
    }

    if response_text:
        try:
            parsed_data = json.loads(response_text)

            if isinstance(parsed_data, list):
                if len(parsed_data) > 0:
                    return parsed_data[0]
                else:
                    return default_error_response    
            elif isinstance(parsed_data, dict):
                return parsed_data
        except json.JSONDecodeError:
            return {**default_error_response, "sentiment": "json_error"}
    
    return default_error_response


def ai_summarize(text, max_length, api_key=None):
    prompt = f"""
    請將以下文章摘要，長度控制在 {max_length} 字以內。
    請直接輸出摘要內容。

    文章:
    {text}
    """
    
    response_text = _call_gemini_with_retry(prompt)
    return response_text.strip() if response_text else ""


if __name__ == "__main__":
    start_total = time.time()

    # A-1: TF-IDF Cosine Similarity Comparison
    print("=== AI Semantic Similarity Comparison ===")
    matrix_size = len(documents)
    similarity_matrix = [[0.0] * matrix_size for _ in range(matrix_size)]
    
    start_sim = time.time()
    for i in range(matrix_size):
        for j in range(matrix_size):
            if i == j:
                similarity_matrix[i][j] = 100.0 # 自己
            elif i > j:
                similarity_matrix[i][j] = similarity_matrix[j][i] # 對稱
            else:
                score = ai_similarity(documents[i], documents[j])
                similarity_matrix[i][j] = score / 100.0
                time.sleep(0.1)
    end_sim = time.time()
    
    print("AI Semantic Similarity Matrix:")
    print("[", end="")
    for i, row in enumerate(similarity_matrix):
        if i > 0: print(" ", end="")
        print("[", end="")
        print(" ".join(f"{val:.2f}".ljust(4) for val in row), end="")
        print("]" if i == len(similarity_matrix)-1 else "]")
    print("]")
    print(f"AI computation time: {end_sim - start_sim:.6f} seconds") # 這是假的時間，包含網路延遲
    print("============================================\n\n")

    # B-2: Rule-Based Sentiment and Topic Classification
    print("=== Rule-Based Sentiment and Topic Classification ===")
    start_total = time.time()
    for text in test_texts:
        res = ai_classify(text)
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {res.get('sentiment', 'unknown')} (Confidence: {res.get('confidence', 0)})")
        print(f"Predicted Topic: {res.get('topic', 'unknown')} (Confidence: {res.get('confidence', 0)})")
        print(f"Keywords: {res.get('keywords', {})}")
        print(f"Topic Keywords: {res.get('topic_keywords', [])}")
        print("")
        time.sleep(0.1)
    end_total = time.time()
    print("=====================================================\n\n")

    # B-3: Statistical Extractive Summarization
    print("=== Statistical Extractive Summarization ===")
    start_time = time.time()
    summary = ai_summarize(article, max_length=100)
    end_time = time.time()
    print("\nOriginal Article:")
    print(article)
    print("\nGenerated Summary:")
    print(summary)
    print(f"Summarization time: {end_time - start_time} seconds")
    print("=============================================\n\n")

    # B-3 Advanced: Advanced Statistical Extractive Summarization
    print("=== Advanced Statistical Extractive Summarization ===")
    start_time = time.time()
    adv_summary = ai_summarize(article, max_length=300)
    end_time = time.time()
    print("\nOriginal Article:")
    print(article)
    print("\nGenerated Advanced Summary:")
    print(adv_summary)
    print(f"Summarization time: {end_time - start_time} seconds")
    print("=====================================================\n")