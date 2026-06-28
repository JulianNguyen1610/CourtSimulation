"""Tests for conservative debate answer post-processing."""

from __future__ import annotations

import unittest

from src.utils.answer_postprocess import shorten_legal_answer


class AnswerPostprocessTest(unittest.TestCase):
    def test_extracts_money_range_without_leading_preposition(self) -> None:
        answer = (
            "Người nào vô ý gây thiệt hại cho tài sản của người khác trị giá "
            "từ 100.000.000 đồng đến dưới 500.000.000 đồng, thì bị phạt cảnh cáo."
        )
        context = "trị giá từ 100.000.000 đồng đến dưới 500.000.000 đồng"

        self.assertEqual(
            shorten_legal_answer(answer, context),
            "100.000.000 đồng đến dưới 500.000.000 đồng",
        )

    def test_extracts_duration_compound(self) -> None:
        answer = (
            "Người sử dụng lao động có trách nhiệm bảo đảm cho người lao động "
            "được nghỉ tính bình quân 01 tháng ít nhất 04 ngày"
        )
        context = "người lao động được nghỉ tính bình quân 01 tháng ít nhất 04 ngày"

        self.assertEqual(shorten_legal_answer(answer, context), "01 tháng ít nhất 04 ngày")

    def test_extracts_context_duration_from_word_number(self) -> None:
        answer = "sau một tháng kể từ ngày thông báo công khai mà không có người đến nhận"
        context = "Sau 01 tháng, kể từ ngày thông báo công khai mà không có người đến nhận"

        self.assertEqual(shorten_legal_answer(answer, context), "01 tháng")

    def test_extracts_marker_phrases(self) -> None:
        self.assertEqual(
            shorten_legal_answer(
                "Người chưa thành niên là người chưa đủ mười tám tuổi",
                "Người chưa thành niên là người chưa đủ mười tám tuổi.",
            ),
            "người chưa đủ mười tám tuổi",
        )
        self.assertEqual(
            shorten_legal_answer(
                "Người lao động được tạm ứng tiền lương theo điều kiện "
                "do hai bên thỏa thuận và không bị tính lãi",
                "theo điều kiện do hai bên thỏa thuận",
            ),
            "do hai bên thỏa thuận",
        )

    def test_extracts_money_lower_bound(self) -> None:
        answer = (
            "Người làm giả tài liệu trong hồ sơ chào bán, niêm yết chứng khoán "
            "thu lợi bất chính từ 2.000.000.000 đồng trở lên sẽ bị phạt tù từ 02 năm đến 07 năm"
        )
        context = "thu lợi bất chính từ 2.000.000.000 đồng trở lên"

        self.assertEqual(shorten_legal_answer(answer, context), "2.000.000.000 đồng trở lên")

    def test_strips_redundant_sau_prefix_on_short_duration(self) -> None:
        context = (
            "Sau 01 tháng, kể từ ngày thông báo công khai mà không có người đến nhận "
            "thì quyền sở hữu vật nuôi dưới nước đó thuộc về người có ruộng, ao, hồ."
        )
        self.assertEqual(shorten_legal_answer("Sau 01 tháng", context), "01 tháng")
        self.assertEqual(shorten_legal_answer("Sau 06 tháng", "Sau 06 tháng, kể từ ngày"), "06 tháng")

    def test_does_not_cut_multiple_simple_spans(self) -> None:
        answer = "phạt tù từ 02 năm đến 07 năm theo quy định"
        context = "bị phạt tù từ 02 năm đến 07 năm"

        self.assertEqual(shorten_legal_answer(answer, context), answer)

    def test_marker_phrase_on_short_answer(self) -> None:
        self.assertEqual(
            shorten_legal_answer(
                "theo điều kiện do hai bên thỏa thuận",
                "theo điều kiện do hai bên thỏa thuận",
            ),
            "do hai bên thỏa thuận",
        )

    def test_strips_leading_tu_on_money_range(self) -> None:
        context = "trị giá từ 100.000.000 đồng đến dưới 500.000.000 đồng"
        self.assertEqual(
            shorten_legal_answer(
                "từ 100.000.000 đồng đến dưới 500.000.000 đồng",
                context,
            ),
            "100.000.000 đồng đến dưới 500.000.000 đồng",
        )

    def test_strips_bi_phat_prefix_when_grounded(self) -> None:
        context = "bị phạt tù từ 01 năm đến 05 năm"
        self.assertEqual(
            shorten_legal_answer("Bị phạt tù từ 01 năm đến 05 năm", context),
            "phạt tù từ 01 năm đến 05 năm",
        )

    def test_strips_phai_prefix_when_grounded(self) -> None:
        context = "thông báo bằng văn bản và phải được bên kia đồng ý"
        self.assertEqual(
            shorten_legal_answer(
                "phải thông báo bằng văn bản và phải được bên kia đồng ý",
                context,
            ),
            "thông báo bằng văn bản và phải được bên kia đồng ý",
        )


if __name__ == "__main__":
    unittest.main()
