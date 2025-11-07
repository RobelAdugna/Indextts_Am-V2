import re

from textstat import textstat


def contains_chinese(text):
    # 正则表达式，用于匹配中文字符 + 数字 -> 都认为是 zh
    if re.search(r'[\u4e00-\u9fff0-9]', text):
        return True
    return False


def contains_amharic(text):
    """Check if text contains Amharic/Ethiopic script"""
    if re.search(r'[\u1200-\u137f\u1380-\u139f\u2d80-\u2ddf\uab00-\uab2f]', text):
        return True
    return False


def get_text_syllable_num(text):
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    amharic_char_pattern = re.compile(r'[\u1200-\u137f\u1380-\u139f\u2d80-\u2ddf\uab00-\uab2f]')
    number_char_pattern = re.compile(r'[0-9]')
    syllable_num = 0
    
    if contains_amharic(text):
        # Amharic uses syllabary (fidel) where each character is a syllable
        tokens = re.findall(r'[\u1200-\u137f\u1380-\u139f\u2d80-\u2ddf\uab00-\uab2f]+|[a-zA-Z]+|[0-9]+', text)
        for token in tokens:
            if amharic_char_pattern.search(token):
                # Each Amharic character (fidel) represents one syllable
                syllable_num += len(token)
            elif number_char_pattern.search(token):
                syllable_num += len(token)
            else:
                syllable_num += textstat.syllable_count(token)
    elif contains_chinese(text):
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+', text)
        for token in tokens:
            if chinese_char_pattern.search(token) or number_char_pattern.search(token):
                syllable_num += len(token)
            else:
                syllable_num += textstat.syllable_count(token)
    else:
        syllable_num = textstat.syllable_count(text)

    return syllable_num


def get_text_tts_dur(text):
    min_speed = 3  # 2.18 #
    max_speed = 5.50

    # Amharic tends to be spoken at a moderate pace, similar to English
    if contains_amharic(text):
        ratio = 1.0  # Adjust based on actual Amharic speech data
    elif contains_chinese(text):
        ratio = 0.8517
    else:
        ratio = 1.0

    syllable_num = get_text_syllable_num(text)
    max_dur = syllable_num * ratio / max_speed
    min_dur = syllable_num * ratio / min_speed

    return max_dur, min_dur