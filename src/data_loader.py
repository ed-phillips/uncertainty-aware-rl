# src/data_loader.py
import re
from datasets import load_dataset, Dataset

def load_math_dataset(split: str = "train"):
    """
    Loads a subset of the GSM8K dataset for a proof-of-concept experiment.
    
    Args:
        split (str): Which dataset split to load ('train', 'validation', etc.)
        
    Returns:
        dataset: A Hugging Face Dataset object.
    """
    dataset = load_dataset("gsm8k", split=split)
    # For a quick proof-of-concept, select only a few examples.
    if split == "train":
        return dataset.select(range(10))
    else:
        return dataset
    


# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore


