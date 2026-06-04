import csv
import json
import threading

import pytest

from scripts import analyze_sentiment as sentiment_analysis
from scripts.analyze_content import record_dict_hash


BASE_URL = "https://example.test/v1"
API_KEY = "test-key"
MODEL = "test-model"
REASONING_EFFORT = "max"
RESPONSE_FORMAT_TYPE = "json_object"


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": self._content,
                    }
                }
            ]
        }


class FakeJSONResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def test_analyze_sentiment_respects_limit_and_writes_multilabel_jsonl(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(
        input_file,
        [
            "AI 会让画师没饭吃，太可怕了",
            "这帮人又开始阴阳怪气了",
            "第三条不应该处理",
        ],
    )
    posted_payloads = []
    posted_urls = []
    posted_headers = []

    def fake_post(url, headers, json, timeout):
        posted_urls.append(url)
        posted_headers.append(headers)
        posted_payloads.append(json)
        return FakeResponse(
            json_dumps(
                {
                    "summary": "评论表达对 AI 替代工作的担忧，并带有愤怒语气。",
                    "dominant_category": "焦虑",
                    "sentiment_polarity": "negative",
                    "labels": [
                        {"category": "焦虑", "confidence": 0.9, "rationale": "提到没饭吃和可怕"},
                        {"category": "愤怒", "confidence": 0.7, "rationale": "语气强烈不满"},
                    ],
                }
            )
        )

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    report = sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
        limit=2,
    )

    rows = read_jsonl(output_file)

    assert report.written_records == 2
    assert len(posted_payloads) == 2
    assert posted_urls == [f"{BASE_URL}/chat/completions", f"{BASE_URL}/chat/completions"]
    assert len(rows) == 2
    assert rows[0]["status"] == "ok"
    assert rows[0]["record_hash"]
    assert posted_headers[0]["User-Agent"] == "codex"
    assert posted_payloads[0]["reasoning_effort"] == REASONING_EFFORT
    assert posted_payloads[0]["response_format"] == {"type": RESPONSE_FORMAT_TYPE}
    assert len(posted_payloads[0]["messages"]) == 2
    assert posted_payloads[0]["messages"][0]["content"] == sentiment_analysis.SYSTEM_PROMPT
    assert posted_payloads[0]["messages"][1]["content"] == "评论内容：AI 会让画师没饭吃，太可怕了"
    assert "这帮人又开始阴阳怪气了" not in posted_payloads[0]["messages"][1]["content"]
    assert rows[0]["sentiment"]["dominant_category"] == "焦虑"
    assert rows[0]["sentiment"]["sentiment_polarity"] == "negative"
    assert [label["category"] for label in rows[0]["sentiment"]["labels"]] == ["焦虑", "愤怒"]


def test_analyze_sentiment_removes_fallback_when_substantive_labels_exist(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["这不就是拿别人作品训练吗？"])

    def fake_post(url, headers, json, timeout):
        return FakeResponse(
            json_dumps(
                {
                    "summary": "评论提出怀疑。",
                    "dominant_category": "中性/无法判断",
                    "sentiment_polarity": "neutral",
                    "labels": [
                        {"category": "中性/无法判断", "confidence": 0.4, "rationale": "误判为中性"},
                        {"category": "质疑", "confidence": 0.85, "rationale": "使用反问表达质疑"},
                    ],
                }
            )
        )

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert rows[0]["sentiment"]["dominant_category"] == "质疑"
    assert [label["category"] for label in rows[0]["sentiment"]["labels"]] == ["质疑"]


def test_analyze_sentiment_accepts_top_level_rationale_and_label_alias(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["反 AI 的又开始卖惨了"])

    def fake_post(url, headers, json, timeout):
        return FakeResponse(
            json_dumps(
                {
                    "labels": [
                        {"label": "嘲讽", "confidence": 0.9},
                    ],
                    "rationale": "评论用贬损表达嘲讽。",
                }
            )
        )

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert rows[0]["status"] == "ok"
    assert rows[0]["sentiment"]["summary"] == "评论用贬损表达嘲讽。"
    assert rows[0]["sentiment"]["dominant_category"] == "嘲讽"
    assert rows[0]["sentiment"]["sentiment_polarity"] == "negative"
    assert rows[0]["sentiment"]["labels"][0]["rationale"] == "评论用贬损表达嘲讽。"


def test_analyze_sentiment_save_tokens_omits_preamble_reasoning_and_reason_output(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(
        tmp_path,
        model="upstream-model",
        response_format_type="json_schema",
        save_tokens=True,
    )
    write_comment_csv(input_file, ["AI 这事挺可怕，但工具效率确实高"])
    posted_payloads = []

    def fake_post(url, headers, json, timeout):
        posted_payloads.append(json)
        return FakeResponse(
            json_dumps(
                {
                    "labels": [
                        {"category": sentiment_analysis.SENTIMENT_CATEGORIES[1], "confidence": 0.85},
                        {"category": sentiment_analysis.SENTIMENT_CATEGORIES[5], "confidence": 0.6},
                    ],
                }
            )
        )

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)
    payload = posted_payloads[0]
    messages = payload["messages"]
    response_schema = payload["response_format"]["json_schema"]["schema"]
    label_schema = response_schema["properties"]["labels"]["items"]

    assert "reasoning_effort" not in payload
    assert payload["thinking"] == {"type": "disabled"}
    assert all("cache_anchor_" not in message["content"] for message in messages)
    assert len(messages) == 2
    assert messages[0]["content"] != sentiment_analysis.SYSTEM_PROMPT
    assert "summary" not in messages[0]["content"]
    assert "rationale" not in messages[0]["content"]
    assert "dominant_category" not in messages[0]["content"]
    assert "sentiment_polarity" not in messages[0]["content"]
    assert "summary" not in response_schema["required"]
    assert "summary" not in response_schema["properties"]
    assert "dominant_category" not in response_schema["required"]
    assert "dominant_category" not in response_schema["properties"]
    assert "sentiment_polarity" not in response_schema["required"]
    assert "sentiment_polarity" not in response_schema["properties"]
    assert "rationale" not in label_schema["required"]
    assert "rationale" not in label_schema["properties"]
    assert rows[0]["llm"]["model"] == "gpt-5.4"
    assert rows[0]["sentiment"]["summary"] == ""
    assert rows[0]["sentiment"]["dominant_category"] == sentiment_analysis.SENTIMENT_CATEGORIES[1]
    assert rows[0]["sentiment"]["sentiment_polarity"] == "negative"
    assert [label["rationale"] for label in rows[0]["sentiment"]["labels"]] == ["", ""]


def test_analyze_sentiment_save_tokens_writes_error_when_labels_missing(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path, save_tokens=True)
    write_comment_csv(input_file, ["我们试试"])

    def fake_post(url, headers, json, timeout):
        return FakeResponse(json_dumps({"summary": "模型只返回了摘要"}))

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert rows[0]["status"] == "error"
    assert rows[0]["sentiment"] is None
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["error_message"] == "LLM sentiment is missing list field: labels"


def test_analyze_sentiment_infers_mixed_polarity_before_positive_dominant_category():
    sentiment = sentiment_analysis.normalize_sentiment(
        {
            "labels": [
                {"category": "乐观", "confidence": 0.9},
                {"category": "焦虑", "confidence": 0.6},
            ],
        },
        save_tokens=True,
    )

    assert sentiment["dominant_category"] == "乐观"
    assert sentiment["sentiment_polarity"] == "mixed"


def test_analyze_sentiment_writes_errors_and_continues(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["第一条", "第二条", "第三条", "第四条"])
    calls = 0

    def fake_post(url, headers, json, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("network down")
        if calls == 2:
            return FakeResponse("not json")
        if calls == 3:
            return FakeResponse(json_dumps({"labels": [{"category": "快乐", "confidence": 0.8}]}))
        return FakeResponse(success_response())

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    report = sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert report.written_records == 4
    assert report.error_records == 3
    assert [row["status"] for row in rows] == ["error", "error", "error", "ok"]
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[1]["error_type"] == "JSONDecodeError"
    assert rows[2]["error_type"] == "ValueError"


def test_analyze_sentiment_retries_existing_error_rows(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["第一次失败后应该重试"])
    record = record_dict("第一次失败后应该重试")
    output_file.write_text(
        json_dumps(
            {
                "record_index": 0,
                "record_hash": record_dict_hash(record),
                "record": record,
                "status": "error",
                "error_type": "RuntimeError",
                "error_message": "previous failure",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    posted_payloads = []

    def fake_post(url, headers, json, timeout):
        posted_payloads.append(json)
        return FakeResponse(success_response())

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    report = sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert report.skipped_records == 0
    assert report.written_records == 1
    assert len(posted_payloads) == 1
    assert [row["status"] for row in rows] == ["error", "ok"]
    assert rows[-1]["record_index"] == 0


def test_analyze_sentiment_skips_only_matching_success_rows(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["已经成功处理", "同一索引但内容变了"])
    old_record = record_dict("同一索引的旧内容")
    existing_rows = [
        {
            "record_index": 0,
            "record_hash": record_dict_hash(record_dict("已经成功处理")),
            "record": record_dict("已经成功处理"),
            "status": "ok",
        },
        {
            "record_index": 1,
            "record_hash": record_dict_hash(old_record),
            "record": old_record,
            "status": "ok",
        },
    ]
    output_file.write_text("\n".join(json_dumps(row) for row in existing_rows) + "\n", encoding="utf-8")
    posted_payloads = []

    def fake_post(url, headers, json, timeout):
        posted_payloads.append(json)
        return FakeResponse(success_response())

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    report = sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert report.skipped_records == 1
    assert report.written_records == 1
    assert len(posted_payloads) == 1
    assert posted_payloads[0]["messages"][1]["content"] == "评论内容：同一索引但内容变了"
    assert rows[-1]["record_index"] == 1
    assert rows[-1]["record"]["content"] == "同一索引但内容变了"


def test_analyze_sentiment_skips_matching_success_hash_after_index_shift(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["new sentiment row", "already ok moved"])
    moved_record = record_dict("already ok moved")
    output_file.write_text(
        json_dumps(
            {
                "record_index": 0,
                "record_hash": record_dict_hash(moved_record),
                "record": moved_record,
                "status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    posted_payloads = []

    def fake_post(url, headers, json, timeout):
        posted_payloads.append(json)
        return FakeResponse(success_response())

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    report = sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert report.skipped_records == 1
    assert report.written_records == 1
    assert len(posted_payloads) == 1
    assert posted_payloads[0]["messages"][1]["content"].endswith("new sentiment row")
    assert rows[-1]["record_index"] == 0
    assert rows[-1]["record"]["content"] == "new sentiment row"


def test_analyze_sentiment_requires_complete_config(tmp_path):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = tmp_path / "content_analysis_config.json"
    config_file.write_text(json_dumps({"base_url": "", "api_key": "", "model": ""}), encoding="utf-8")
    write_comment_csv(input_file, ["不会开始处理"])

    with pytest.raises(ValueError, match="base_url"):
        sentiment_analysis.analyze_sentiment(input_file=input_file, output_file=output_file, config_file=config_file)

    assert not output_file.exists()


def test_analyze_sentiment_supports_responses_api_output_text(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path, api_type="responses")
    write_comment_csv(input_file, ["AI sarcasm is obvious here"])
    posted_urls = []
    posted_payloads = []

    def fake_post(url, headers, json, timeout):
        posted_urls.append(url)
        posted_payloads.append(json)
        return FakeJSONResponse({"output_text": success_response()})

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    report = sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert report.written_records == 1
    assert posted_urls == [f"{BASE_URL}/responses"]
    assert "input" in posted_payloads[0]
    assert "messages" not in posted_payloads[0]
    assert len(posted_payloads[0]["input"]) == 2
    assert posted_payloads[0]["input"][0]["content"] == sentiment_analysis.SYSTEM_PROMPT
    assert posted_payloads[0]["input"][1]["content"] == (
        "Return only valid json. No Markdown.\nComment content:\nAI sarcasm is obvious here"
    )
    assert posted_payloads[0]["reasoning"] == {"effort": REASONING_EFFORT}
    assert posted_payloads[0]["text"]["format"] == {"type": RESPONSE_FORMAT_TYPE}
    assert rows[0]["status"] == "ok"
    assert rows[0]["sentiment"]["sentiment_polarity"] == "neutral"


def test_analyze_sentiment_uses_thread_count_alias(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path, thread_count=2)
    write_comment_csv(input_file, ["first sentiment row", "second sentiment row"])
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active_requests = 0
    max_active_requests = 0

    def fake_post(url, headers, json, timeout):
        nonlocal active_requests, max_active_requests
        with lock:
            active_requests += 1
            max_active_requests = max(max_active_requests, active_requests)
        try:
            barrier.wait(timeout=2)
        finally:
            with lock:
                active_requests -= 1
        return FakeResponse(success_response())

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    report = sentiment_analysis.analyze_sentiment(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert report.written_records == 2
    assert report.error_records == 0
    assert max_active_requests == 2
    assert sorted(row["record_index"] for row in rows) == [0, 1]
    assert all(row["status"] == "ok" for row in rows)


def test_analyze_sentiment_prints_progress_counts(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_sentiment.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["already ok", "will fail", "will pass"])
    ok_record = record_dict("already ok")
    output_file.write_text(
        json_dumps(
            {
                "record_index": 0,
                "record_hash": record_dict_hash(ok_record),
                "record": ok_record,
                "status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    def fake_post(url, headers, json, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        return FakeResponse(success_response())

    monkeypatch.setattr(sentiment_analysis.requests, "post", fake_post)

    sentiment_analysis.analyze_sentiment(input_file=input_file, output_file=output_file, config_file=config_file)

    progress_output = capsys.readouterr().err

    assert "Sentiment analysis progress" in progress_output
    assert "processed=2/2" in progress_output
    assert "ok=1" in progress_output
    assert "error=1" in progress_output
    assert "skipped=1" in progress_output
    assert "remaining=0" in progress_output


def write_config(tmp_path, **overrides):
    config_file = tmp_path / "content_analysis_config.json"
    config = {
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "response_format_type": RESPONSE_FORMAT_TYPE,
    }
    config.update(overrides)
    config_file.write_text(
        json_dumps(config),
        encoding="utf-8",
    )
    return config_file


def write_comment_csv(path, contents):
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["username", "gender", "content", "comment_time", "likes", "ip_location", "signature", "feature"],
        )
        writer.writeheader()
        for content in contents:
            writer.writerow(record_dict(content))


def record_dict(content):
    return {
        "username": f"user-{content}",
        "gender": "",
        "content": content,
        "comment_time": "2026-05-10 10:00:00",
        "likes": 1,
        "ip_location": "",
        "signature": "",
        "feature": "aigc",
    }


def success_response():
    return json_dumps(
        {
            "summary": "评论没有明显情绪。",
            "dominant_category": "中性/无法判断",
            "sentiment_polarity": "neutral",
            "labels": [
                {"category": "中性/无法判断", "confidence": 0.8, "rationale": "测试"},
            ],
        }
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)
