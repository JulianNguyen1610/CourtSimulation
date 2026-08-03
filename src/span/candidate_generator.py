import re
from src.models import SpanCandidate, AnswerTypeProfile
def generate_candidates(context:str, profile:AnswerTypeProfile, source_doc_id='primary_context') -> list[SpanCandidate]:
 patterns = [r'\d{1,3}\s*(?:năm|tháng|ngày)', r'\d{1,3}(?:\.\d{3})+\s*đồng']
 found=[]
 for pat in patterns:
  for m in re.finditer(pat, context, re.I): found.append(SpanCandidate(span_id=f's{len(found)}',text=m.group(),source_doc_id=source_doc_id,start_offset=m.start(),end_offset=m.end(),source_sentence=context[max(0,context.rfind('.',0,m.start())+1):context.find('.',m.end()) if context.find('.',m.end())>=0 else len(context)].strip(),answer_type=profile.answer_type,generator='regex',scores={'extractive':1.0}))
 if not found:
  for m in re.finditer(r'[^.]{1,180}',context):
   text=m.group().strip(); start=m.start()+m.group().find(text)
   if text: found.append(SpanCandidate(span_id=f's{len(found)}',text=text,source_doc_id=source_doc_id,start_offset=start,end_offset=start+len(text),source_sentence=text,answer_type=profile.answer_type,generator='sentence_window',scores={'extractive':1.0}))
 return list({(x.text.lower(),x.start_offset):x for x in found}.values())
