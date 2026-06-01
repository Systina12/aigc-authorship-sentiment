import csv
import json

import pytest

from scripts import analyze_content as content_analysis


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


class FakeHTTPErrorResponse:
    text = '{"error":{"message":"bad response"}}'

    def raise_for_status(self):
        raise content_analysis.requests.HTTPError("400 Client Error")

    def json(self):
        return {"error": {"message": "bad response"}}


def test_analyze_content_respects_limit_and_writes_multilabel_jsonl(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_labels.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(
        input_file,
        [
            "AI 工具能提高效率，但版权也要管",
            "这会让画师失业",
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
                    "summary": "认可工具效率并提到版权治理",
                    "labels": [
                        {"category": "工具化认知", "confidence": 0.9, "rationale": "提到工具和效率"},
                        {"category": "版权争议", "confidence": 0.8, "rationale": "提到版权"},
                    ],
                }
            )
        )

    monkeypatch.setattr(content_analysis.requests, "post", fake_post)

    report = content_analysis.analyze_content(
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
    assert [label["category"] for label in rows[0]["analysis"]["labels"]] == ["工具化认知", "版权争议"]
    assert posted_payloads[0]["messages"][1]["content"] == "评论内容：AI 工具能提高效率，但版权也要管"
    assert "这会让画师失业" not in posted_payloads[0]["messages"][1]["content"]


def test_analyze_content_removes_fallback_when_substantive_labels_exist(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_labels.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["马车车夫抵制汽车，时代趋势挡不住"])

    def fake_post(url, headers, json, timeout):
        return FakeResponse(
            json_dumps(
                {
                    "summary": "认为技术趋势不可阻挡",
                    "labels": [
                        {"category": "无法归类/无关讨论", "confidence": 0.4, "rationale": "误判为无关"},
                        {"category": "技术宿命论", "confidence": 0.9, "rationale": "认为趋势不可阻挡"},
                    ],
                }
            )
        )

    monkeypatch.setattr(content_analysis.requests, "post", fake_post)

    content_analysis.analyze_content(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert [label["category"] for label in rows[0]["analysis"]["labels"]] == ["技术宿命论"]


def test_analyze_content_accepts_top_level_rationale_response(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_labels.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["考古，26年ai裁员潮"])

    def fake_post(url, headers, json, timeout):
        return FakeResponse(
            json_dumps(
                {
                    "labels": [
                        {"label": "职业焦虑", "confidence": 0.9},
                    ],
                    "rationale": "评论提及 AI 裁员潮，表达对就业影响的关注。",
                }
            )
        )

    monkeypatch.setattr(content_analysis.requests, "post", fake_post)

    content_analysis.analyze_content(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert rows[0]["status"] == "ok"
    assert rows[0]["analysis"]["summary"] == "评论提及 AI 裁员潮，表达对就业影响的关注。"
    assert rows[0]["analysis"]["labels"][0]["rationale"] == "评论提及 AI 裁员潮，表达对就业影响的关注。"


def test_analyze_content_writes_errors_and_continues(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_labels.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["第一条", "第二条", "第三条"])
    calls = 0

    def fake_post(url, headers, json, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("network down")
        if calls == 2:
            return FakeResponse("not json")

        return FakeResponse(
            json_dumps(
                {
                    "summary": "无关讨论",
                    "labels": [
                        {"category": "无法归类/无关讨论", "confidence": 0.95, "rationale": "没有 AIGC 观点"},
                    ],
                }
            )
        )

    monkeypatch.setattr(content_analysis.requests, "post", fake_post)

    report = content_analysis.analyze_content(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert report.written_records == 3
    assert report.error_records == 2
    assert [row["status"] for row in rows] == ["error", "error", "ok"]
    assert rows[0]["record_hash"]
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[1]["error_type"] == "JSONDecodeError"


def test_analyze_content_writes_http_error_response_body(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_labels.jsonl"
    config_file = write_config(tmp_path, api_type="responses")
    write_comment_csv(input_file, ["bad provider request"])

    def fake_post(url, headers, json, timeout):
        return FakeHTTPErrorResponse()

    monkeypatch.setattr(content_analysis.requests, "post", fake_post)

    content_analysis.analyze_content(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert rows[0]["status"] == "error"
    assert rows[0]["error_type"] == "HTTPError"
    assert "response body" in rows[0]["error_message"]
    assert "bad response" in rows[0]["error_message"]


def test_analyze_content_retries_existing_error_rows(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_labels.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["第一次失败后应该重试"])
    record = record_dict("第一次失败后应该重试")
    output_file.write_text(
        json_dumps(
            {
                "record_index": 0,
                "record_hash": content_analysis.record_dict_hash(record),
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

    monkeypatch.setattr(content_analysis.requests, "post", fake_post)

    report = content_analysis.analyze_content(
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


def test_analyze_content_skips_only_matching_success_rows(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_labels.jsonl"
    config_file = write_config(tmp_path)
    write_comment_csv(input_file, ["已经成功处理", "同一索引但内容变了"])
    old_record = record_dict("同一索引的旧内容")
    existing_rows = [
        {
            "record_index": 0,
            "record_hash": content_analysis.record_dict_hash(record_dict("已经成功处理")),
            "record": record_dict("已经成功处理"),
            "status": "ok",
        },
        {
            "record_index": 1,
            "record_hash": content_analysis.record_dict_hash(old_record),
            "record": old_record,
            "status": "ok",
        },
    ]
    output_file.write_text("\n".join(json_dumps(row) for row in existing_rows) + "\n", encoding="utf-8")
    posted_payloads = []

    def fake_post(url, headers, json, timeout):
        posted_payloads.append(json)
        return FakeResponse(success_response())

    monkeypatch.setattr(content_analysis.requests, "post", fake_post)

    report = content_analysis.analyze_content(
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


def test_analyze_content_requires_complete_config(tmp_path):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_labels.jsonl"
    config_file = tmp_path / "content_analysis_config.json"
    config_file.write_text(json_dumps({"base_url": "", "api_key": "", "model": ""}), encoding="utf-8")
    write_comment_csv(input_file, ["不会开始处理"])

    with pytest.raises(ValueError, match="base_url"):
        content_analysis.analyze_content(input_file=input_file, output_file=output_file, config_file=config_file)

    assert not output_file.exists()


def test_analyze_content_supports_responses_api_output_text(tmp_path, monkeypatch):
    input_file = tmp_path / "comments_cleaned.csv"
    output_file = tmp_path / "comment_labels.jsonl"
    config_file = write_config(tmp_path, api_type="responses")
    write_comment_csv(input_file, ["AI helps with repetitive drawing work"])
    posted_urls = []
    posted_payloads = []

    def fake_post(url, headers, json, timeout):
        posted_urls.append(url)
        posted_payloads.append(json)
        return FakeJSONResponse({"output_text": success_response()})

    monkeypatch.setattr(content_analysis.requests, "post", fake_post)

    report = content_analysis.analyze_content(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
    )

    rows = read_jsonl(output_file)

    assert report.written_records == 1
    assert posted_urls == [f"{BASE_URL}/responses"]
    assert "input" in posted_payloads[0]
    assert "messages" not in posted_payloads[0]
    assert posted_payloads[0]["input"][1]["content"] == (
        "Return only valid json. No Markdown.\nComment content:\nAI helps with repetitive drawing work"
    )
    assert posted_payloads[0]["reasoning"] == {"effort": REASONING_EFFORT}
    assert posted_payloads[0]["text"]["format"] == {"type": RESPONSE_FORMAT_TYPE}
    assert rows[0]["status"] == "ok"


def test_extract_responses_content_supports_nested_output_text():
    response_body = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "{\"summary\":\""},
                    {"type": "output_text", "text": "ok\",\"labels\":[]}"},
                ],
            }
        ]
    }

    assert content_analysis.extract_response_content(response_body, "responses") == '{"summary":"ok","labels":[]}'


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
        ("localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
        ("127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1/chat/completions"),
        ("192.168.58.133:8000/v1", "http://192.168.58.133:8000/v1/chat/completions"),
        ("https://api.example.test/v1", "https://api.example.test/v1/chat/completions"),
        ("api.example.test/v1", "https://api.example.test/v1/chat/completions"),
        ("http://localhost:8000/v1/chat/completions", "http://localhost:8000/v1/chat/completions"),
    ],
)
def test_chat_completions_url_supports_http_and_bare_hosts(base_url, expected):
    assert content_analysis.chat_completions_url(base_url) == expected


@pytest.mark.parametrize(
    ("base_url", "api_type", "expected"),
    [
        ("http://localhost:8000/v1", "responses", "http://localhost:8000/v1/responses"),
        ("http://localhost:8000/v1/chat/completions", "responses", "http://localhost:8000/v1/responses"),
        ("http://localhost:8000/v1/responses", "responses", "http://localhost:8000/v1/responses"),
        ("http://localhost:8000/v1/responses", "chat_completions", "http://localhost:8000/v1/chat/completions"),
    ],
)
def test_llm_endpoint_url_supports_responses(base_url, api_type, expected):
    assert content_analysis.llm_endpoint_url(base_url, api_type) == expected


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
            "summary": "无关讨论",
            "labels": [
                {"category": "无法归类/无关讨论", "confidence": 0.8, "rationale": "测试"},
            ],
        }
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)
