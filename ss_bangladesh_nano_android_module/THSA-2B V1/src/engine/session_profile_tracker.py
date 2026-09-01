"""
THSA-2B Dynamic Session Profile Tracker (Sibling & Multi-Turn Memory)
======================================================================
Solves the context window truncation issue on edge devices:
  - Remembers the student's active class (e.g., Class 9).
  - Automatically switches context when a sibling takes the phone (e.g., "আমি ৭ম শ্রেণিতে পড়ি").
  - Dynamically injects the active profile into every conversation turn.
"""

from typing import Dict, Any, Optional
import unicodedata
import re

def normalize_bengali_unicode(text: str) -> str:
    if not text:
        return text
    text = text.replace("\u09c7\u09be", "\u09cb")
    text = text.replace("\u09c7\u09d7", "\u09cc")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u09a1\u09bc", "\u09dc")
    text = text.replace("\u09a2\u09bc", "\u09dd")
    text = text.replace("\u09af\u09bc", "\u09df")
    text = re.sub(r"\u09cd+", "\u09cd", text)
    return text

class SessionProfileTracker:
    """
    Stateful on-device session memory manager.
    Preserves active academic class across turns and handles dynamic sibling switches in O(1) time.
    """

    def __init__(self, default_class: str = "Class 9-10 (SSC)"):
        self.active_class = default_class
        self.active_class_numeric = 9
        self.session_history = []

    def update_profile_from_prompt(self, prompt: str) -> bool:
        """
        Detects if user declares or changes their academic class.
        Returns True if a class switch/update occurred.
        """
        clean_p = normalize_bengali_unicode(prompt.lower().strip())

        class_map = {
            1: ["class 1", "১ম শ্রেণি", "প্রথম শ্রেণি", "class one"],
            2: ["class 2", "২য় শ্রেণি", "দ্বিতীয় শ্রেণি", "class two"],
            3: ["class 3", "৩য় শ্রেণি", "তৃতীয় শ্রেণি", "class three"],
            4: ["class 4", "৪র্থ শ্রেণি", "চতুর্থ শ্রেণি", "class four"],
            5: ["class 5", "৫ম শ্রেণি", "পঞ্চম শ্রেণি", "class five", "psc"],
            6: ["class 6", "৬ষ্ঠ শ্রেণি", "ষষ্ঠ শ্রেণি", "class six"],
            7: ["class 7", "৭ম শ্রেণি", "সপ্তম শ্রেণি", "class seven"],
            8: ["class 8", "৮ম শ্রেণি", "অষ্টম শ্রেণি", "class eight", "jsc"],
            9: ["class 9", "৯ম শ্রেণি", "নবম শ্রেণি", "class nine"],
            10: ["class 10", "১০ম শ্রেণি", "দশম শ্রেণি", "class ten", "ssc"],
            11: ["class 11", "১১শ শ্রেণি", "একাদশ শ্রেণি", "hsc 1st", "hsc ১ম", "college 1st"],
            12: ["class 12", "১২শ শ্রেণি", "দ্বাদশ শ্রেণি", "hsc 2nd", "hsc ২য়", "hsc"]
        }

        # Check for explicit declaration or change
        is_declaration = any(k in clean_p for k in [
            "আমি", "পড়ি", "পড়ছি", "ক্লাস", "শ্রেণি", "শ্রেণীতে", "শ্রেণিতে", "আই এম ইন", "i am in", "i read in", "not in"
        ])

        for c_num, patterns in class_map.items():
            if any(p in clean_p for p in patterns):
                if c_num in [1, 2, 3, 4, 5]:
                    label = f"Class {c_num} (Primary Foundation)"
                elif c_num in [6, 7, 8]:
                    label = f"Class {c_num} (Junior Secondary)"
                elif c_num in [9, 10]:
                    label = f"Class {c_num} (SSC Secondary)"
                else:
                    label = f"Class {c_num} (HSC Higher Secondary)"

                self.active_class = label
                self.active_class_numeric = c_num
                return True

        return False

    def inject_context_prefix(self, prompt: str) -> str:
        """
        Injects the active class context header into the prompt.
        """
        return f"[Active Profile: {self.active_class}] {prompt}"

    def get_profile_summary(self) -> Dict[str, Any]:
        return {
            "active_class": self.active_class,
            "active_class_numeric": self.active_class_numeric
        }
