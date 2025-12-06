from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import modern_methods as modern
import traditional_methods as trad
from tests.testcases import article, documents, test_texts
import jieba


RESULTS_DIR = Path(__file__).resolve().parent / "results_advanced"
TFIDF_HEATMAP_PATH = RESULTS_DIR / "tfidf_similarity_matrix.png"
CLASSIFICATION_CSV_PATH = RESULTS_DIR / "classification_results.csv"
SUMMARY_PATH = RESULTS_DIR / "summarization_comparison.txt"

EXPECTED_LABELS = [
	{"sentiment": "positive", "topic": "美食"},
	{"sentiment": "positive", "topic": "科技"},
	{"sentiment": "negative", "topic": "other"},
	{"sentiment": "positive", "topic": "運動"},
]


def ensure_results_dir() -> None:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def format_seconds(seconds: float) -> str:
	if seconds >= 1:
		return f"{seconds:.2f} 秒"
	return f"{seconds * 1000:.2f} 毫秒"


def _plot_similarity_heatmaps(manual_matrix: np.ndarray, sklearn_matrix: np.ndarray, ai_matrix: np.ndarray) -> None:
	labels = [f"S{i+1}" for i in range(len(manual_matrix))]
	fig, axes = plt.subplots(1, 3, figsize=(15, 4))
	sns.heatmap(manual_matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0], vmin=0, vmax=1)
	axes[0].set_title("Manual TF-IDF")
	sns.heatmap(sklearn_matrix, annot=True, cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1], vmin=0, vmax=1)
	axes[1].set_title("Sklearn TF-IDF")
	sns.heatmap(ai_matrix, annot=True, cmap="Oranges", xticklabels=labels, yticklabels=labels, ax=axes[2], vmin=0, vmax=1)
	axes[2].set_title("Gemini Similarity")
	plt.tight_layout()
	plt.savefig(TFIDF_HEATMAP_PATH, dpi=200, bbox_inches="tight")
	plt.close(fig)


def _safe_ai_call(func, *args, **kwargs):
	start = time.perf_counter()
	try:
		result = func(*args, **kwargs)
	except Exception as exc:  # pragma: no cover
		print(f"[Warning] {func.__name__} failed: {exc}")
		return None, 0.0
	return result, time.perf_counter() - start


def evaluate_similarity() -> Dict[str, Any]:
	manual_start = time.perf_counter()
	manual_vectors = trad.manual_tfidf(documents)
	manual_matrix = cosine_similarity(pd.DataFrame(manual_vectors).fillna(0))
	manual_time = time.perf_counter() - manual_start

	sklearn_start = time.perf_counter()
	vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None, lowercase=False)
	sklearn_matrix = cosine_similarity(vectorizer.fit_transform(documents))
	sklearn_time = time.perf_counter() - sklearn_start

	ai_matrix = np.eye(len(documents))
	ai_time = 0.0
	for i in range(len(documents)):
		for j in range(i + 1, len(documents)):
			score, latency = _safe_ai_call(modern.ai_similarity, documents[i], documents[j])
			ai_time += latency
			if score is None:
				continue
			similarity = max(0.0, min(100.0, score)) / 100
			ai_matrix[i, j] = ai_matrix[j, i] = similarity

	_plot_similarity_heatmaps(manual_matrix, sklearn_matrix, ai_matrix)

	rmse_ai = float(np.sqrt(((manual_matrix - ai_matrix) ** 2).mean()))
	rmse_sklearn = float(np.sqrt(((manual_matrix - sklearn_matrix) ** 2).mean()))

	return {
		"manual_runtime_sec": manual_time,
		"sklearn_runtime_sec": sklearn_time,
		"ai_runtime_sec": ai_time,
		"rmse_ai_vs_manual": rmse_ai,
		"rmse_sklearn_vs_manual": rmse_sklearn,
	}


def evaluate_classification() -> Dict[str, Any]:
	sentiment_clf = trad.RuleBasedSentimentClassifier()
	topic_clf = trad.TopicClassifier()

	rows: List[Dict[str, Any]] = []
	trad_correct = 0
	ai_correct = 0
	trad_time = 0.0
	ai_time = 0.0

	for text, expected in zip(test_texts, EXPECTED_LABELS):
		start = time.perf_counter()
		sentiment_result = sentiment_clf.classify(text)
		topic_result = topic_clf.predict(text)
		trad_latency = time.perf_counter() - start
		trad_time += trad_latency
		rows.append(
			{
				"text": text,
				"method": "traditional",
				"sentiment": sentiment_result["sentiment"],
				"topic": topic_result["topic"],
				"confidence": round((sentiment_result["confidence"] + topic_result["confidence"]) / 2, 3),
				"latency_sec": round(trad_latency, 4),
				"expected_sentiment": expected["sentiment"],
				"expected_topic": expected["topic"],
			}
		)
		if (
			sentiment_result["sentiment"] == expected["sentiment"]
			and topic_result["topic"] == expected["topic"]
		):
			trad_correct += 1

		ai_result, latency = _safe_ai_call(modern.ai_classify, text)
		ai_time += latency
		ai_sentiment = ai_result.get("sentiment") if ai_result else "error"
		ai_topic = ai_result.get("topic") if ai_result else "error"
		ai_conf = ai_result.get("confidence", 0.0) if ai_result else 0.0
		rows.append(
			{
				"text": text,
				"method": "gemini",
				"sentiment": ai_sentiment,
				"topic": ai_topic,
				"confidence": ai_conf,
				"latency_sec": round(latency, 4),
				"expected_sentiment": expected["sentiment"],
				"expected_topic": expected["topic"],
			}
		)
		if ai_sentiment == expected["sentiment"] and ai_topic == expected["topic"]:
			ai_correct += 1

	pd.DataFrame(rows).to_csv(CLASSIFICATION_CSV_PATH, index=False)

	return {
		"traditional_accuracy": trad_correct / len(test_texts),
		"traditional_runtime_sec": trad_time,
		"modern_accuracy": ai_correct / len(test_texts) if ai_time else None,
		"modern_runtime_sec": ai_time,
	}


def _run_summarizer(summarizer_cls, ratio: float = 0.3) -> Tuple[Dict[str, Any], str]:
	summarizer = summarizer_cls()
	start = time.perf_counter()
	summary = summarizer.summarize(article, ratio=ratio)
	runtime = time.perf_counter() - start
	return {"runtime_sec": runtime, "summary_length": len(summary)}, summary


def evaluate_summarization() -> Dict[str, Any]:
	stat_metrics, stat_summary = _run_summarizer(trad.StatisticalSummarizer)
	adv_metrics, adv_summary = _run_summarizer(trad.AdvancedStatisticalSummarizer)
	ai_summary, ai_time = _safe_ai_call(modern.ai_summarize, article, max_length=220)
	ai_summary = ai_summary or "(AI summary unavailable)"

	SUMMARY_PATH.write_text(
		"=== 傳統統計式摘要 ===\n"
		f"{stat_summary}\n\n"
		"=== 進階統計式摘要 ===\n"
		f"{adv_summary}\n\n"
		"=== 生成式 AI 摘要 ===\n"
		f"{ai_summary}\n",
		encoding="utf-8",
	)

	return {
		"statistical": stat_metrics,
		"advanced": adv_metrics,
		"modern": {"runtime_sec": ai_time, "summary_length": len(ai_summary)},
	}


def main() -> None:
	ensure_results_dir()
	metrics = {
		"similarity": evaluate_similarity(),
		"classification": evaluate_classification(),
		"summarization": evaluate_summarization(),
	}
	print(f"Artifacts stored in {RESULTS_DIR}")


if __name__ == "__main__":
	main()

