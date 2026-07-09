from app.rag.citations import verify_answer_citations


def test_verify_answer_citations_supported():
    result = verify_answer_citations(
        question="普通本科四年制最长学习年限是多少？",
        answer="普通本科四年制一般学生学习年限为 3～6 年。",
        references=[
            {
                "snippet": "普通本科四年制专业学生学习年限调整为3～6年。"
            }
        ],
    )

    assert result["supported"] is True
    assert result["overlap_ratio"] >= 0.2


def test_verify_answer_citations_without_references():
    result = verify_answer_citations(
        question="普通本科四年制最长学习年限是多少？",
        answer="最长学习年限为 6 年。",
        references=[],
    )

    assert result["supported"] is False


def test_verify_answer_citations_empty_reference_text():
    result = verify_answer_citations(
        question="如何申请休学？",
        answer="需要提交休学申请。",
        references=[{"snippet": ""}],
    )

    assert result["supported"] is False


def test_verify_answer_citations_unsupported_low_overlap():
    result = verify_answer_citations(
        question="最长学习年限是多少？",
        answer="普通本科四年制最长学习年限为 6 年。",
        references=[
            {
                "snippet": "学生应当遵守课堂纪律，按时参加教学活动。"
            }
        ],
    )

    assert result["supported"] is False


def test_verify_answer_citations_refusal_answer():
    result = verify_answer_citations(
        question="某个不存在的问题",
        answer="我无法根据提供的文档确定答案",
        references=[],
    )

    assert result["supported"] is True
    assert result["reason"] == "refusal answer does not require citation verification"
