"""
SS Tutor BD - Centralized Tutoring Prompt & Scaffolding Engine
Defines standard role definitions, Socratic hint modes, step-by-step scaffolds,
and strict textbook-grounded context templates for Bangladesh NCTB Class 6-10.
"""

from typing import Optional, Dict, Any, List


# Base System Instruction for SS Tutor BD
BASE_TUTOR_SYSTEM_PROMPT = """তুমি SS Tutor BD — বাংলাদেশ জাতীয় শিক্ষাক্রম ও পাঠ্যপুস্তক বোর্ড (NCTB) ষষ্ঠ থেকে দশম (Class 6-10) শ্রেণির জন্য একটি অফলাইন এআই শিক্ষক (Tutor)।

তোমার মূল নীতিমালা:
১. ভাষা: সর্বদা প্রমিত ও সহজ বাংলায় প্রাঞ্জল ভাষায় বুঝিয়ে বলবে।
২. শিখন পদ্ধতি (Pedagogy): শিক্ষার্থীকে সরাসরি উত্তর মুখস্থ না করিয়ে চিন্তাভাবনা করতে সাহায্য করবে।
৩. সত্যনিষ্ঠা ও বিশ্বস্ততা: প্রদত্ত পাঠ্যপুস্তকের তথ্যের ভিত্তিতে উত্তর দেবে। নিজের থেকে কোনো মনগড়া বা ভুল তথ্য আবিষ্কার করবে না।
৪. বিনম্রতা ও উৎসাহ: শিক্ষার্থীকে ভয় না দেখিয়ে গণিত ও বিজ্ঞান শিখতে উৎসাহিত করবে।
৫. নিরাপত্তা: কোনো অভ্যন্তরীণ কোড, ট্যাগ (<tool_call>, <|im_start|>) বা অবান্তর ইংরেজি শব্দ ব্যবহার করবে না।"""


def get_base_system_prompt() -> str:
    """Returns standard base tutor system prompt."""
    return BASE_TUTOR_SYSTEM_PROMPT


def build_socratic_hint_prompt(problem_text: str, student_level: str = "Class 8") -> str:
    """
    Builds a prompt enforcing Socratic tutoring and negative constraints (withholding final answers).
    """
    return (
        f"[শিক্ষার্থী শ্রেণি: {student_level}]\n"
        f"সমস্যা / প্রশ্ন:\n{problem_text}\n\n"
        "শিক্ষকের বিশেষ নির্দেশনা (সক্র্যাটিক পদ্ধতি):\n"
        "- শিক্ষার্থীকে সরাসরি চূড়ান্ত সমাধান বা উত্তর বলে দেবে না।\n"
        "- সমস্যাটি সমাধানের জন্য প্রথম দরকারি সূত্র বা চিন্তা করার একটি সহজ ইঙ্গিত (Hint) দাও।\n"
        "- উত্তরের শেষে শিক্ষার্থীকে পরবর্তী ধাপটি কী হতে পারে তা নিয়ে একটি প্রশ্ন করো।"
    )


def build_step_by_step_math_prompt(problem_text: str, student_level: str = "Class 8") -> str:
    """
    Builds a structured prompt for mathematical step-by-step problem solving.
    """
    return (
        f"[শিক্ষার্থী শ্রেণি: {student_level}]\n"
        f"গাণিতিক সমস্যা:\n{problem_text}\n\n"
        "নির্দেশনা:\n"
        "১. প্রথমে কী কী তথ্য দেওয়া আছে তা লেখো।\n"
        "২. প্রয়োজনীয় সূত্রটি উল্লেখ করো।\n"
        "৩. ধাপ ১, ধাপ ২, ধাপ ৩ আকারে হিসাবের ধাপগুলো দেখাও।\n"
        "৪. শেষে স্পষ্ট বাংলায় চূড়ান্ত উত্তর লেখো।"
    )


def build_grounded_rag_prompt(
    user_query: str,
    textbook_context: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Builds a strict context-grounded retrieval prompt preventing textbook hallucinations.
    """
    source_tag = ""
    if metadata:
        cls_name = metadata.get("class", "NCTB")
        sub_name = metadata.get("subject", "")
        ch_title = metadata.get("chapter_title", "")
        source_tag = f" [উৎস: {cls_name} - {sub_name} - {ch_title}]"

    return (
        f"নিচে NCTB পাঠ্যপুস্তকের নির্ভরযোগ্য অংশ উদ্ধৃত করা হলো{source_tag}:\n"
        "----------------------------------------\n"
        f"{textbook_context.strip()}\n"
        "----------------------------------------\n\n"
        f"শিক্ষার্থীর প্রশ্ন: {user_query}\n\n"
        "কঠোর নির্দেশনাবলী:\n"
        "১. শুধুমাত্র উপরের দেওয়া পাঠ্যপুস্তক অংশের তথ্যের ভিত্তিতে উত্তর তৈরি করো।\n"
        "২. যদি প্রশ্নের উত্তর উপরের পাঠ্যাংশে না থাকে, তবে বানিয়ে না লিখে স্পষ্টভাবে বলো: "
        "'প্রদত্ত পাঠ্যপুস্তক অংশে এই তথ্যের উল্লেখ নেই।'\n"
        "৩. উত্তর সহজ, সংক্ষিপ্ত ও শিক্ষণীয় ভাষায় উপস্থাপন করো।"
    )


def build_adaptive_simplification_prompt(original_explanation: str, student_confusion: str) -> str:
    """
    Builds a recovery prompt when a student indicates confusion.
    """
    return (
        f"পূর্ববর্তী ব্যাখ্যা:\n{original_explanation}\n\n"
        f"শিক্ষার্থীর মতামত: '{student_confusion}'\n\n"
        "শিক্ষকের করণীয়:\n"
        "- কঠিন তাত্ত্বিক সংজ্ঞা বাদ দিয়ে বাস্তব জীবনের একটি সহজ উদাহরণের মাধ্যমে পুনরায় বুঝিয়ে বলো।\n"
        "- শিক্ষার্থীকে আশ্বস্ত করো এবং ছোট ছোট বাক্যে ব্যাখ্যা শেষ করো।"
    )
