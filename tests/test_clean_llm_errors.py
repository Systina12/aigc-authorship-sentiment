import json

from scripts.clean_llm_errors import clean_llm_error_rows


def test_clean_llm_error_rows_keeps_only_ok_rows_and_creates_backup(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    rows = [
        {"record_index": 0, "status": "error", "error_message": "failed"},
        {"record_index": 1, "status": "ok", "analysis": {"labels": []}},
        {"record_index": 2, "status": "error", "error_message": "failed again"},
        {"record_index": 3, "status": "ok", "analysis": {"labels": []}},
    ]
    input_file.write_text("\n".join(json.dumps(row) for row in rows) + "\nnot json\n", encoding="utf-8")

    report = clean_llm_error_rows(input_file)

    cleaned_rows = read_jsonl(input_file)
    backup_file = input_file.with_suffix(input_file.suffix + ".bak")

    assert report.total_rows == 5
    assert report.kept_ok_rows == 2
    assert report.removed_error_rows == 2
    assert report.removed_invalid_rows == 1
    assert report.backup_file == backup_file
    assert backup_file.exists()
    assert [row["record_index"] for row in cleaned_rows] == [1, 3]


def test_clean_llm_error_rows_dry_run_does_not_change_file(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    original = (
        json.dumps({"record_index": 0, "status": "error"})
        + "\n"
        + json.dumps({"record_index": 1, "status": "ok"})
        + "\n"
    )
    input_file.write_text(original, encoding="utf-8")

    report = clean_llm_error_rows(input_file, dry_run=True)

    assert report.total_rows == 2
    assert report.kept_ok_rows == 1
    assert report.removed_error_rows == 1
    assert report.backup_file is None
    assert input_file.read_text(encoding="utf-8") == original


def test_clean_llm_error_rows_can_write_to_separate_output(tmp_path):
    input_file = tmp_path / "comment_labels.jsonl"
    output_file = tmp_path / "cleaned.jsonl"
    input_file.write_text(
        json.dumps({"record_index": 0, "status": "error"}) + "\n" + json.dumps({"record_index": 1, "status": "ok"}) + "\n",
        encoding="utf-8",
    )

    report = clean_llm_error_rows(input_file, output_file=output_file)

    assert report.output_file == output_file
    assert report.backup_file is None
    assert read_jsonl(output_file) == [{"record_index": 1, "status": "ok"}]
    assert len(read_jsonl(input_file)) == 2


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
