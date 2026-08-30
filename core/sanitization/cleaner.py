"""
SS Tutor BD - Output Sanitization & Protection Layer
Safely sanitizes raw model generation outputs:
  1. Strips leaked control tokens (<tool_call>, <|im_start|>, etc.)
  2. Detects and trims prompt echo
  3. Truncates pathological repetition loops
  4. Preserves authentic Bengali unicode characters and mathematical notations.
"""

import re
from typing import Dict, Any, Tuple


CONTROL_TOKEN_PATTERNS = [
    r"</?tool_call>",
    r"</?tool_response>",
    r"<\|im_start\|>(\w+)?",
    r"<\|im_end\|>",
    r"<\|endoftext\|>",
    r"\[assistant\]",
    r"\[user\]",
    r"\[system\]",
    r"</s>"
]

COMPILED_CONTROL_REGEX = re.compile("|".join(CONTROL_TOKEN_PATTERNS), re.IGNORECASE)


def strip_control_tokens(text: str) -> str:
    """Removes all known LLM internal control tokens and tool markup."""
    if not text:
        return ""
    cleaned = COMPILED_CONTROL_REGEX.sub("", text)
    # Clean dangling angle bracket remnants if corrupted
    cleaned = re.sub(r"<\s*/?\s*tool_[^>]*>", "", cleaned)
    return cleaned.strip()


def strip_prompt_echo(output: str, user_prompt: str) -> str:
    """Removes leading exact or fuzzy echo of the user prompt."""
    if not output or not user_prompt:
        return output

    cleaned_out = output.strip()
    cleaned_prompt = user_prompt.strip()

    # Exact prefix match
    if cleaned_out.startswith(cleaned_prompt):
        cleaned_out = cleaned_out[len(cleaned_prompt):].lstrip(":\n -।")

    # Check for "প্রশ্ন: <prompt>" prefix
    prefix_patterns = [
        rf"প্রশ্ন:\s*{re.escape(cleaned_prompt)}",
        rf"User:\s*{re.escape(cleaned_prompt)}",
        rf"Question:\s*{re.escape(cleaned_prompt)}"
    ]
    for pat in prefix_patterns:
        match = re.match(pat, cleaned_out)
        if match:
            cleaned_out = cleaned_out[match.end():].lstrip(":\n -।")

    return cleaned_out.strip()


def truncate_repetition_loops(text: str, min_repeat_len: int = 8, max_allowed_repeats: int = 2) -> Tuple[str, bool]:
    """Detects and truncates pathological autoregressive loop sequences."""
    if not text or len(text) < 30:
        return text, False

    # Check line-level repetition
    lines = text.split("\n")
    cleaned_lines = []
    seen_consecutive = {}
    loop_detected = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        
        seen_consecutive[stripped] = seen_consecutive.get(stripped, 0) + 1
        if seen_consecutive[stripped] <= max_allowed_repeats:
            cleaned_lines.append(line)
        else:
            loop_detected = True
            break

    result_text = "\n".join(cleaned_lines)

    # Check substring-level repetition (e.g. phrases repeating without newlines)
    sub_pattern = re.compile(rf"(.{{{min_repeat_len},}}?)\1{{{max_allowed_repeats},}}")
    match = sub_pattern.search(result_text)
    if match:
        loop_detected = True
        # Keep only one instance of the repeating chunk
        rep_phrase = match.group(1)
        start_idx = match.start()
        result_text = result_text[:start_idx + len(rep_phrase)].strip()

    return result_text, loop_detected


def sanitize_tutor_output(raw_output: str, user_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for output sanitization pipeline.
    Returns dictionary with cleaned text, flags, and diagnostic info.
    """
    if not raw_output:
        return {
            "cleaned_text": "",
            "had_control_tokens": False,
            "had_prompt_echo": False,
            "had_repetition_loop": False
        }

    # Step 1: Strip control tokens
    had_control_tokens = bool(COMPILED_CONTROL_REGEX.search(raw_output))
    step1 = strip_control_tokens(raw_output)

    # Step 2: Strip prompt echo
    step2 = strip_prompt_echo(step1, user_prompt) if user_prompt else step1
    had_prompt_echo = (len(step2) < len(step1))

    # Step 3: Truncate repetition loops
    step3, had_repetition = truncate_repetition_loops(step2)

    return {
        "cleaned_text": step3.strip(),
        "had_control_tokens": had_control_tokens,
        "had_prompt_echo": had_prompt_echo,
        "had_repetition_loop": had_repetition,
        "original_length": len(raw_output),
        "cleaned_length": len(step3)
    }
