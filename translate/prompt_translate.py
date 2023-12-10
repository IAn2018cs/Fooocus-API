# coding=utf-8

import time

import dl_translate as dlt
from ftlangdetect import detect

mt = dlt.TranslationModel()


def prompt_translate(prompt: str) -> str:
    execution_start_time = time.perf_counter()
    try:
        result = detect(text=prompt, low_memory=False)
        source = result['lang']
        if source != 'en':
            print(f"[Prompt Translate] source is {source}, start translate")
            prompt = mt.translate(prompt, source=f'{source}', target=dlt.lang.ENGLISH)
        else:
            print("[Prompt Translate] source is ENGLISH, not translate")
    except Exception as e:
        print(f"[Prompt Translate] translate error: {e}, use origin prompt")
    finally:
        execution_time = time.perf_counter() - execution_start_time
        print(f'[Prompt Translate] Translate time: {execution_time:.2f} seconds')
        return prompt
