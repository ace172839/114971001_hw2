import math
import jieba
import jieba.posseg as pseg  # 引入詞性標註
import re
import time
import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from tests.testcases import documents, test_texts, article
# from tests.testcases_advance import documents, test_texts, article


def chinese_corpus(documents):
    # 將中文斷句
    corpus = []
    for doc in documents:
        seg_list = jieba.cut(doc)
        # 像英文那樣，用空格分隔詞語
        corpus.append(" ".join(seg_list))
    return corpus


""" Task 1: Traditional TF-IDF Cosine Similarity Comparison """

def manual_tfidf(documents):
    def _calculate_tf(word_dict, total_words):
        tf_dict = {}
        for word, count in word_dict.items():
            tf_dict[word] = count / total_words
        return tf_dict

    def _calculate_idf(documents, word):
        doc_count = sum(1 for document in documents if word in document)
        numerator = len(documents) + 1
        denominator = doc_count + 1
        return math.log(numerator / denominator) + 1.0

    def _calculate_tfidf(tf_dict, idf_dict):
        tfidf_dict = {}
        for word, tf_value in tf_dict.items():
            tfidf_dict[word] = tf_value * idf_dict.get(word, 0.0)
        return tfidf_dict
    
    # 1. TF
    tf_list = []
    for doc in documents:
        words = jieba.lcut(doc)
        word_count = len(words)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        tf_list.append(_calculate_tf(word_freq, word_count))

    # 2. IDF
    idf_dict = {}
    all_words = set(word for tf_dict in tf_list for word in tf_dict)
    for word in all_words:
        idf_dict[word] = _calculate_idf(documents, word)

    # 3. TF-IDF
    tfidf_list = []
    for tf_dict in tf_list:
        tfidf_list.append(_calculate_tfidf(tf_dict, idf_dict))
    return tfidf_list

def sklearn_tfidf(documents):
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix.toarray()

def compare_tfidf_similarity(documents):
    def _compute_cosine_similarity(tfidf_matrix):
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix.round(2)

    manual_tfidf_matrix = manual_tfidf(documents)
    manual_df = pd.DataFrame(manual_tfidf_matrix).fillna(0)

    sklearn_tfidf_matrix = sklearn_tfidf(documents)
    sklearn_df = pd.DataFrame(sklearn_tfidf_matrix).fillna(0)

    manual_cosine_similarity = _compute_cosine_similarity(manual_df)
    sklearn_cosine_similarity = _compute_cosine_similarity(sklearn_df)

    return {
        "manual_matrix": manual_cosine_similarity,
        "sklearn_matrix": sklearn_cosine_similarity,
    }

""" Task 1: Traditional TF-IDF Cosine Similarity Comparison """


""" Task 2: Rule-Based Sentiment and Topic Classification """

class RuleBasedSentimentClassifier:
    def __init__(self):
        self.positive_words = {"好", "棒", "優秀", "喜歡", "推薦", "滿意", "開心", "值得", "精彩", "完美",
                               "不錯", "值得", "五星", "好評", "舒服", "讚", "美味", "好吃", "耐用", "厲害"}
        self.negative_words = {"差", "糟", "失望", "討厭", "不推薦", "浪費", "無聊", "爛", "糟糕", "差勁", 
                               "噁心", "難吃", "愚蠢", "智障", "笨蛋", "垃圾", "屎", "一星", "差評", "蟑螂", "老鼠"}
        self.negation_words = {"不", "沒", "無", "非", "別", "勿"}
        self.multiplier_words = {"非常", "超級", "特別", "很", "挺", "相當", "十分", "極", "太"}

    def classify(self, text):
        pos_count, neg_count = 0, 0
        keywords = {}

        for i, char in enumerate(text):
            if char in self.positive_words:
                multiplier = 1
                if i > 0 and text[i - 1] in self.multiplier_words:
                    # 程度副詞加成 50%
                    multiplier += 0.5
                if i > 0 and text[i - 1] in self.negation_words:
                    # 否定詞反轉情感
                    neg_count += multiplier
                else:
                    pos_count += multiplier
                keywords[char] = keywords.get(char, 0) + multiplier
            elif char in self.negative_words:
                multiplier = 1
                if i > 0 and text[i - 1] in self.multiplier_words:
                    # 程度副詞加成 50%
                    multiplier += 0.5

                if i > 0 and text[i - 1] in self.negation_words:
                    # 否定詞反轉情感
                    pos_count += multiplier
                else:
                    neg_count += multiplier
                keywords[char] = keywords.get(char, 0) + multiplier

        if pos_count > neg_count:
            sentiment = "positive"
            confidence = pos_count / (pos_count + neg_count)
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = neg_count / (pos_count + neg_count)
        else:
            sentiment = "neutral"
            confidence = 0.5

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "keywords": keywords
        }

class TopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            "科技": {"科技", "手機", "電腦", "軟體", "硬體", "程式", "應用", "網路", "AI", "人工智慧"},
            "運動": {"運動", "比賽", "球隊", "球員", "籃球", "足球", "跑步", "健身", "體育", "賽事"},
            "美食": {"美食", "餐廳", "料理", "食物", "菜單", "味道", "廚師", "飲料", "甜點", "小吃"},
            "旅遊": {"旅遊", "景點", "旅行", "飯店", "住宿", "行程", "導遊", "交通", "觀光", "度假"}
        }

    def predict(self, text):
        topic_counts = {topic: 0 for topic in self.topic_keywords}
        used_keywords = {keyword: 0 for keywords in self.topic_keywords.values() for keyword in keywords}


        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    # 關鍵字會隨著出現次數增加其權重
                    used_keywords[keyword] += 1
                    topic_counts[topic] += 1 + math.log10(used_keywords[keyword])

        predicted_topic = max(topic_counts, key=topic_counts.get)
        max_count = topic_counts[predicted_topic]
        total_count = sum(topic_counts.values())

        if total_count == 0:
            confidence = 0.0
            predicted_topic = "other"
        else:
            confidence = max_count / total_count

        return {
            "topic": predicted_topic,
            "keywords": [kw for kw, count in used_keywords.items() if count > 0],
            "confidence": round(confidence, 2)
        }

""" Task 2: Rule-Based Sentiment and Topic Classification """



""" Task 3: Statistical Extractive Summarization """

class StatisticalSummarizer:
    def __init__(self):
        self.stop_words = set(["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
                               "一", "一個", "上", "也", "很", "到", "說", "要", "去", "你",
                               "這", "那", "之", "與", "及"])

    def sentence_score(self, sentence, word_freq, index, total_sentences):
        if len(sentence) == 0:
            return 0
        
        words = list(jieba.cut(sentence))
        score = 0
        valid_word_count = 0
        
        for word in words:
            if word not in self.stop_words and word.strip():
                # 詞頻越高，分數越高
                score += word_freq.get(word, 0)
                valid_word_count += 1

        if index == 0 or index == total_sentences - 1:
            # 開頭或結尾句子加分
            score *= 1.2

        # 假設過短(<10)或過長(>80) 的句子通常不是摘要重點
        if len(sentence) < 10 or len(sentence) > 80:
            score *= 0.6
            
        return score

    def summarize(self, text, ratio=0.3):
        """
        生成摘要步驟 [cite: 130-138]
        """
        # 分句
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return ""

        # 分詞並計算詞頻
        all_words = []
        for s in sentences:
            words = jieba.cut(s)
            for w in words:
                if w not in self.stop_words and w.strip():
                    all_words.append(w)
        
        word_freq = Counter(all_words)

        # 每句話計分
        scored_sentences = []
        for i, sent in enumerate(sentences):
            score = self.sentence_score(sent, word_freq, i, len(sentences))
            scored_sentences.append((i, sent, score))

        # 計算要選幾句
        select_count = max(1, int(len(sentences) * ratio))
        
        # 根據分數選擇分數最高的幾句
        top_sentences = sorted(scored_sentences, key=lambda x: x[2], reverse=True)[:select_count]

        # 根據原文順序重新排序回來
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        return "。".join([item[1] for item in top_sentences]) + "。"

""" Task 3: Statistical Extractive Summarization """


""" Task 3 Advanced """
"""
1. stop_words 只有包含少部分詞語，所以很容易將語助詞或沒太多意義的常用詞，變成高頻率的字詞；
    所以我打算透過不同的「段落」(不是分句)，進行 idf 去除無意義的詞
2. 同一篇文章，雖然主題應該是不變的(像是科技類文章)，但每個段落之間重複的詞不一定一樣。
    像是一篇文章在探討 AI 如何影響人類生活，第一段討論 AI 發展，第二段討論 AI 取代人類工作，第三段討論 AI 的日常聊天...等。
    可能每個段落中 AI 出現的頻率不高，但每個段落他都佔一定比重，那肯定是這篇文章的重心主題。

因此，我讓 Gemini 幫我設計一個進階的統計式摘要器：
1. 透過 idf 計算詞在不同段落出現的頻率，並且加入詞性過濾，避免「的」這類語助詞被納入計算，但又避免「AI、跑步」這類的名詞或動詞被當成停用詞。
2. 根據詞在不同段落的分佈情況，給予主題詞更高的權重 (反向 idf)，讓這些詞在句子評分時能有更高的影響力。
"""
class AdvancedStatisticalSummarizer:
    def __init__(self):
        # 基礎停用詞 (主要過濾極端高頻的虛詞)
        self.stop_words = set(["的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
                               "一", "一個", "上", "也", "很", "到", "說", "要", "去", "你",
                               "這", "那", "之", "與", "及", "會", "著", "沒有", "看", "自己"])
        # 允許的詞性：n(名詞), v(動詞), eng(英文) - 這能幫您解決 "的" vs "AI" 的問題
        self.allowed_pos = {'n', 'v', 'vn', 'eng', 'l'} 

    def calculate_word_weights(self, text):
        """
        核心邏輯：結合段落分佈來計算詞權重
        """
        # 1. 依段落切分 (Paragraphs)
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        total_paragraphs = len(paragraphs)
        
        # 統計變數
        global_word_freq = Counter() # 全域詞頻 (TF)
        word_paragraph_appearance = Counter() # 該詞出現在幾個段落中 (DF)
        
        # 2. 遍歷每個段落進行統計
        for p in paragraphs:
            # 使用 pseg 取得詞性
            words = pseg.cut(p)
            seen_in_this_paragraph = set()
            
            for w, flag in words:
                # 過濾邏輯：必須不在停用詞表 AND (是允許的詞性 OR 是英文/數字)
                # 這樣可以把無意義的「語助詞」濾掉，保留「AI」(eng) 和「生活」(n)
                if w not in self.stop_words and (flag[0] in ['n', 'v'] or 'eng' in flag):
                    global_word_freq[w] += 1
                    seen_in_this_paragraph.add(w)
            
            # 更新 DF (Document Frequency)
            for w in seen_in_this_paragraph:
                word_paragraph_appearance[w] += 1
                
        # 3. 計算最終權重 (實作您的想法)
        word_weights = {}
        for word, tf in global_word_freq.items():
            df = word_paragraph_appearance[word]
            
            # --- 您的邏輯實作區 ---
            
            # A. 基礎重要性 (TF)
            score = tf
            
            # B. 主題加權 (Reverse IDF 概念)
            # 如果一個詞出現在超過 50% 的段落中，它極可能是「主題詞」(如 AI)
            # 我們給予它額外的乘數
            distribution_ratio = df / total_paragraphs
            
            if distribution_ratio > 0.5:
                # 這是貫穿全文的主題，加權 1.5 倍
                score *= 1.5
            elif distribution_ratio > 0.2:
                # 這是次要主題
                score *= 1.2
            
            # C. (選用) 傳統 IDF 懲罰
            # 如果您還是擔心某些無意義動詞(如 "進行") 出現在每一段
            # 可以加上 log 懲罰，但因為我們已經篩選過詞性，這裡影響較小
            
            word_weights[word] = score
            
        return word_weights

    def sentence_score(self, sentence, word_weights, index, total_sentences):
        # 簡單的分詞 (計算分數時不需要詞性了，只要對應權重)
        words = jieba.cut(sentence)
        score = 0
        valid_words = 0
        
        for w in words:
            if w in word_weights:
                score += word_weights[w]
                valid_words += 1
        
        if len(sentence) == 0: return 0
        
        # 正規化
        avg_score = score / (len(sentence) + 1)
        
        # 位置加權
        if index == 0: avg_score *= 1.3  # 第一句最重要
        if index == total_sentences - 1: avg_score *= 1.2
        
        return avg_score

    def summarize(self, text, ratio=0.3):
        # 1. 預先計算所有詞的權重 (包含段落分析)
        word_weights = self.calculate_word_weights(text)
        
        # 印出權重最高的幾個詞，驗證您的想法 (Debug用)
        # print("Top words:", sorted(word_weights.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # 2. 分句
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences: return ""

        # 3. 評分
        scored_sentences = []
        for i, sent in enumerate(sentences):
            s_score = self.sentence_score(sent, word_weights, i, len(sentences))
            scored_sentences.append((i, sent, s_score))
            
        # 4. 排序與選取
        select_count = max(1, int(len(sentences) * ratio))
        top_sentences = sorted(scored_sentences, key=lambda x: x[2], reverse=True)[:select_count]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        return "。".join([item[1] for item in top_sentences]) + "。"

""" Task 3 Advanced """


if __name__ == "__main__":
    # A-1: TF-IDF Cosine Similarity Comparison
    print("=== TF-IDF Cosine Similarity Comparison ===")
    start_time1 = time.time()
    corpus = chinese_corpus(documents)
    end_time1 = time.time()
    start_time2 = time.time()
    result = compare_tfidf_similarity(corpus)
    end_time2 = time.time()
    print("Manual TF-IDF Cosine Similarity Matrix:")
    print(result["manual_matrix"])
    print(f"Manual computation time: {end_time1 - start_time1}")
    print("\nSklearn TF-IDF Cosine Similarity Matrix:")
    print(result["sklearn_matrix"])
    print(f"Sklearn computation time: {end_time2 - start_time2}")
    print("============================================\n\n")


    # A-2: Rule-Based Sentiment and Topic Classification
    print("=== Rule-Based Sentiment and Topic Classification ===")
    start_time = time.time()
    sentiment_clf = RuleBasedSentimentClassifier()
    topic_clf = TopicClassifier()

    for text in test_texts:
        sentiment_result = sentiment_clf.classify(text)
        topic_result = topic_clf.predict(text)
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {sentiment_result['sentiment']} (Confidence: {sentiment_result['confidence']})")
        print(f"Predicted Topic: {topic_result['topic']} (Confidence: {topic_result['confidence']})")
        print(f"Keywords: {sentiment_result['keywords']}")
        print(f"Topic Keywords: {topic_result['keywords']}")
    end_time = time.time()
    print(f"\nTotal classification time: {end_time - start_time} seconds")
    print("=====================================================\n\n")


    # A-3: Statistical Extractive Summarization
    print("=== Statistical Extractive Summarization ===")
    start_time = time.time()
    summarizer = StatisticalSummarizer()
    summary = summarizer.summarize(article, ratio=0.3)
    end_time = time.time()
    print("\nOriginal Article:")
    print(article)
    print("\nGenerated Summary:")
    print(summary)
    print(f"Summarization time: {end_time - start_time} seconds")
    print("=============================================\n\n")


    # A-3 Advanced: Advanced Statistical Extractive Summarization
    print("=== Advanced Statistical Extractive Summarization ===")
    start_time = time.time()
    advanced_summarizer = AdvancedStatisticalSummarizer()
    advanced_summary = advanced_summarizer.summarize(article, ratio=0.3)
    end_time = time.time()
    print("\nOriginal Article:")
    print(article)
    print("\nGenerated Summary:")
    print(advanced_summary)
    print(f"Summarization time: {end_time - start_time} seconds")
    print("=====================================================\n")

