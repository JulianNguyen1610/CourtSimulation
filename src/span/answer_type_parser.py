from src.models import AnswerTypeProfile
def parse_answer_type(question: str) -> AnswerTypeProfile:
 q=question.lower()
 if any(x in q for x in ('bao nhiêu năm','thời hạn','bao lâu')): return AnswerTypeProfile(answer_type='duration',expected_unit='năm',confidence=.9)
 if any(x in q for x in ('bao nhiêu tiền','mức phạt tiền')): return AnswerTypeProfile(answer_type='money',expected_unit='đồng',confidence=.9)
 if q.startswith(('ai ','người nào')): return AnswerTypeProfile(answer_type='person',confidence=.8)
 if 'là gì' in q: return AnswerTypeProfile(answer_type='definition',expected_granularity='sentence',confidence=.8)
 if any(x in q for x in ('những','các ')): return AnswerTypeProfile(answer_type='list',expected_granularity='list',allows_multiple_spans=True,confidence=.7)
 return AnswerTypeProfile(answer_type='free_span',confidence=.3)
