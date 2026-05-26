import json
import time
from typing import Dict, List
from openai import OpenAI

# Attribute-based preference system (inspired by Amulet)
PREFERENCE_ATTRIBUTES = {
    "creative": "Your answer should be creative as much as possible.",
    "sycophantic": "Your answer should be sycophantic as much as possible.",
    "verbose": "Your answer should be verbose as much as possible.",
    "complex": "Your answer should be complex as much as possible.",
    "formal": "Your answer should be formal as much as possible.",
    "pleasant": "Your answer should be pleasant as much as possible.",
    "concise": "Your answer should be concise as much as possible.",
    "uplifting": "Your answer should be uplifting as much as possible."
}

SYSTEM_PROMPT = """
You are an AI assistant that helps determine which response better aligns with a given attribute preference.
Given a specific attribute preference, select the response from assistant A or B that better embodies this attribute.
Focus on how well each response aligns with the specified attribute, not general quality.
Declare your choice by using the format: "[[A]]" if you believe assistant A's response better aligns with the attribute, or "[[B]]" if assistant B's response better aligns with the attribute.
"""

USER_PROMPT = """
[Target Attribute]
{attribute}: {attribute_description}

[User Question]
{query}

[The Start of Assistant A's Answer]
{response_1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_2}
[The End of Assistant B's Answer]

[Task]
Which response better aligns with the "{attribute}" attribute? Consider how well each response embodies the characteristic described above.
"""

class PreferenceCollector:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        referer_url: str = "",
        site_title: str = "",
        sleep_time: float = 0.5,
        max_retries: int = 3,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.referer = referer_url
        self.title = site_title
        self.sleep_time = sleep_time
        self.max_retries = max_retries

    def _format_prompt(self, attribute: str, query: str, r1: str, r2: str) -> str:
        attribute_description = PREFERENCE_ATTRIBUTES.get(attribute, f"Your answer should be {attribute} as much as possible.")
        return USER_PROMPT.format(
            attribute=attribute,
            attribute_description=attribute_description,
            query=query,
            response_1=r1,
            response_2=r2,
        )

    def _call_judge(self, user_prompt: str) -> int:
        for _ in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=256,
                    extra_headers={
                        "HTTP-Referer": self.referer,
                        "X-Title": self.title,
                    },
                    extra_body={}
                )
                content = response.choices[0].message.content.strip()

                a_count = content.count("[[A]]")
                b_count = content.count("[[B]]")

                if a_count == 1 and b_count == 0:
                    return 0  
                elif b_count == 1 and a_count == 0:
                    return 1  
                elif a_count == 0 and b_count == 0:
                    print(f"[WARNING] No judgment found in response: {content[:100]}...")
                else:
                    print(f"[WARNING] Ambiguous judgment (A:{a_count}, B:{b_count}): {content[:100]}...")
                    last_a = content.rfind("[[A]]")
                    last_b = content.rfind("[[B]]")
                    if last_a > last_b:
                        return 0
                    else:
                        return 1
            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}")
            time.sleep(self.sleep_time)
        return -1  

    def collect_single(self, example: Dict) -> Dict:
        attribute = example["attribute"]  
        query = example["query"]  
        enhanced_query = example.get("enhanced_query", query) 
        r1 = example["response_1"]
        r2 = example["response_2"]

        user_prompt = self._format_prompt(attribute, query, r1, r2)
        label = self._call_judge(user_prompt)
        if label == -1:
            return None  

        return {
            "y1": f"{enhanced_query} {r1}",  
            "y2": f"{enhanced_query} {r2}",  
            "label": label
        }

    def collect_batch_and_save(self, examples: List[Dict], output_path: str):
        with open(output_path, "a") as fout:
            for example in examples:
                result = self.collect_single(example)
                if result:
                    fout.write(json.dumps(result) + "\n")
