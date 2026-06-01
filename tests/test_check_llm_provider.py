import json

from scripts import check_llm_provider


class FakeResponse:
    def __init__(self, *, status_code=200, body=None, text=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text if text is not None else json.dumps(self._body, ensure_ascii=False)

    def json(self):
        return self._body


def test_provider_check_reports_valid_minimal_json(tmp_path, monkeypatch):
    config_file = write_config(tmp_path)
    posted_payloads = []
    posted_headers = []
    posted_urls = []

    def fake_post(url, headers, json, timeout):
        posted_urls.append(url)
        posted_payloads.append(json)
        posted_headers.append(headers)
        return FakeResponse(
            body={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"ok": true, "note": "provider test"}'},
                    }
                ]
            }
        )

    monkeypatch.setattr(check_llm_provider.requests, "post", fake_post)

    report = check_llm_provider.run_provider_test(config_file=config_file, task="minimal")

    assert report.ok is True
    assert report.http_status == 200
    assert posted_urls == ["https://example.test/v1/chat/completions"]
    assert report.choice_count == 1
    assert report.finish_reason == "stop"
    assert report.content_json_ok is True
    assert report.normalized_ok is True
    assert posted_headers[0]["User-Agent"] == "codex"
    assert posted_payloads[0]["response_format"] == {"type": "json_object"}


def test_provider_check_supports_responses_api_minimal_payload(tmp_path, monkeypatch):
    config_file = write_config(tmp_path)
    posted_payloads = []
    posted_urls = []

    def fake_post(url, headers, json, timeout):
        posted_urls.append(url)
        posted_payloads.append(json)
        return FakeResponse(body={"output_text": '{"ok": true, "note": "provider test"}'})

    monkeypatch.setattr(check_llm_provider.requests, "post", fake_post)

    report = check_llm_provider.run_provider_test(
        config_file=config_file,
        task="minimal",
        api_type="responses",
    )

    assert report.ok is True
    assert report.http_status == 200
    assert posted_urls == ["https://example.test/v1/responses"]
    assert "input" in posted_payloads[0]
    assert "messages" not in posted_payloads[0]
    assert posted_payloads[0]["reasoning"] == {"effort": "low"}
    assert posted_payloads[0]["text"]["format"] == {"type": "json_object"}
    assert report.response_top_level_keys == ["output_text"]
    assert report.choice_count == 0


def test_provider_check_can_omit_response_format(tmp_path, monkeypatch):
    config_file = write_config(tmp_path)
    posted_payloads = []

    def fake_post(url, headers, json, timeout):
        posted_payloads.append(json)
        return FakeResponse(body={"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr(check_llm_provider.requests, "post", fake_post)

    report = check_llm_provider.run_provider_test(
        config_file=config_file,
        task="minimal",
        response_format_type="none",
    )

    assert report.ok is True
    assert "response_format" not in posted_payloads[0]


def test_provider_check_reports_empty_content(tmp_path, monkeypatch):
    config_file = write_config(tmp_path)

    def fake_post(url, headers, json, timeout):
        return FakeResponse(body={"choices": [{"message": {"content": ""}}]})

    monkeypatch.setattr(check_llm_provider.requests, "post", fake_post)

    report = check_llm_provider.run_provider_test(config_file=config_file, task="minimal")

    assert report.ok is False
    assert report.content_json_ok is False
    assert report.content_json_error == "empty message.content"
    assert report.normalized_error == "content is not JSON"


def test_provider_check_reports_http_error_without_raising(tmp_path, monkeypatch):
    config_file = write_config(tmp_path)

    def fake_post(url, headers, json, timeout):
        return FakeResponse(
            status_code=400,
            body={"error": {"message": "unsupported response_format"}},
        )

    monkeypatch.setattr(check_llm_provider.requests, "post", fake_post)

    report = check_llm_provider.run_provider_test(config_file=config_file, task="minimal", include_raw_response=True)

    assert report.ok is False
    assert report.http_status == 400
    assert report.http_error == "HTTP 400"
    assert report.raw_response == {"error": {"message": "unsupported response_format"}}


def write_config(tmp_path):
    config_file = tmp_path / "content_analysis_config.json"
    config_file.write_text(
        json.dumps(
            {
                "base_url": "https://example.test/v1",
                "api_key": "test-key",
                "model": "test-model",
                "reasoning_effort": "low",
                "response_format_type": "json_object",
            }
        ),
        encoding="utf-8",
    )
    return config_file
