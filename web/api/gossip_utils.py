import re

TAGGED_PATTERN_TEMPLATE = r"<{0}>(.*?)</{0}>"

def _extract_tagged(text, tag):
    matches = re.findall(TAGGED_PATTERN_TEMPLATE.format(tag), text)
    return matches[0]

def stage1_message(node_key: str, question:str, ts, outputs: dict ):
    answer = outputs['answer']
    return f"{question}...Answer: {answer}"

def stage2_message(node_key: str, question:str, ts, outputs: dict ):
    try:
        opinion = outputs["agent_opinion"][node_key]
        explain = _extract_tagged(opinion, "explain")
        identify = _extract_tagged(opinion, "identify")
        return f"{explain}...Identify: {identify}"
    except (ValueError, KeyError):
        return stage1_message(node_key, question, ts, outputs)

def stage3_message(node_key: str, question:str, ts, outputs: dict ):
    try:
        decision = outputs["final_agent_decision"][node_key]
        summarize_feedback = _extract_tagged(decision, "summarize_feedback")
        majority =_extract_tagged(decision, "majority")
        return f"{summarize_feedback}...Majority: {majority}"
    except (ValueError, KeyError):
        return stage1_message(node_key, question, ts, outputs)
