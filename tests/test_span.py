import unittest
from src.span import parse_answer_type, generate_candidates, verify_span
class SpanTest(unittest.TestCase):
 def test_duration_offsets_and_verification(self):
  c='Người phạm tội bị phạt tù 02 năm.'; p=parse_answer_type('Bao nhiêu năm?'); xs=generate_candidates(c,p); self.assertTrue(xs); self.assertEqual(c[xs[0].start_offset:xs[0].end_offset],xs[0].text); self.assertTrue(verify_span(xs[0],c,p).passed)
 def test_definition_and_list_not_capped(self): self.assertEqual(parse_answer_type('Khái niệm này là gì?').answer_type,'definition'); self.assertTrue(parse_answer_type('Các điều kiện nào?').allows_multiple_spans)
