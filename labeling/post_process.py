# -*- coding: utf-8 -*-
"""
Post-processing corrections for the labeled dataset.
Fixes known false positives and false negatives from the rule-based classifier.

This script:
1. Loads the labeled dataset
2. Applies corrections based on analysis of misclassified samples
3. Re-runs a more nuanced check on borderline cases
4. Saves the corrected dataset
"""

import pandas as pd
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_poem_or_long_prose(text: str) -> bool:
    """Detect if text is poetry, literature, or long analytical prose (likely not toxic)."""
    text_lower = text.lower()
    # Poetry markers
    poem_markers = ['tg ', 'tg:', 'nén nhang', 'tuổi xanh', 'chiến hào',
                    'ngỡ vừa mới', 'đây rồi', 'thương nhớ', 'ngàn năm']
    if any(m in text_lower for m in poem_markers):
        return True
    
    # Very long text with literary structure
    if len(text) > 500:
        # Check if it's a review/analysis
        review_markers = ['cảm nhận', 'đánh giá', 'nhận xét', 'phân tích', 
                         'theo mình', 'theo tôi', 'cá nhân thấy',
                         'ưu điểm', 'nhược điểm', 'tổng kết', 'kết luận',
                         'thứ nhất', 'thứ hai', 'thứ ba']
        if sum(1 for m in review_markers if m in text_lower) >= 2:
            return True
    
    return False


def is_movie_character_reference(text: str) -> bool:
    """Check if negative words refer to movie characters, not real people."""
    text_lower = text.lower()
    
    # Character names in the movie
    characters = ['tạ', 'cường', 'sen', 'tú', 'tấn', 'hồng', 'quang',
                  'anh tạ', 'ông tạ', 'a tạ', 'anh sen', 'thằng tú', 
                  'thằng tấn', 'nhân vật']
    
    has_character = any(c in text_lower for c in characters)
    
    # Movie plot words
    plot_words = ['trong phim', 'cảnh', 'đoạn', 'vai', 'diễn', 'phim',
                  'đánh nhau', 'chiến đấu', 'hy sinh', 'bộ đội', 'lính']
    has_plot = any(p in text_lower for p in plot_words)
    
    return has_character and has_plot


def has_direct_personal_attack(text: str) -> bool:
    """
    More precise detection of direct personal attacks on other commenters.
    Distinguished from movie discussion by checking for reply indicators
    and direct address to other commenters.
    """
    text_lower = text.lower()
    
    # Reply to another user (@username, nick name references)
    has_at_mention = bool(re.search(r'@\w+', text))
    
    # Direct confrontation with "mày" + insult
    may_insult = bool(re.search(
        r'\bmày\b.{0,30}\b(ngu|dốt|điên|khùng|hèn|câm|rác|đần|nào|hiểu|biết\s*gì)\b',
        text_lower
    ))
    insult_may = bool(re.search(
        r'\b(ngu|dốt|điên|khùng|hèn|câm|rác|đần)\b.{0,30}\bmày\b',
        text_lower
    ))
    
    # "bạn" used aggressively (not politely)
    ban_aggressive = bool(re.search(
        r'\bbạn\b.{0,20}\b(ngu|dốt|rác|não|thần\s*kinh|bị)\b',
        text_lower
    ))
    
    # Derogatory "nhà mày", "ông/bà mày"
    family_insult = bool(re.search(
        r'\b(nhà|ông|bà|bố|mẹ|cha)\s*mày\b.{0,30}\b(ngu|rác|dốt)\b',
        text_lower
    ))
    
    # Telling someone to shut up or go away
    shut_up = bool(re.search(
        r'\b(câm|im|cút|biến|lượn)\b.{0,10}\b(mồm|miệng|đi|mẹ|ngay)\b',
        text_lower
    ))
    
    # "trại súc vật" directed at someone
    dehumanizing = bool(re.search(
        r'\b(súc\s*(vật|sinh)|trại\s*súc|con\s*vật)\b', text_lower
    ))
    
    return (may_insult or insult_may or ban_aggressive or 
            family_insult or shut_up or dehumanizing or
            (has_at_mention and (may_insult or insult_may)))


def classify_additional_toxic(text: str) -> tuple:
    """
    Additional patterns for detecting toxic comments missed by v2.
    Returns (is_toxic: bool, confidence: float, reason: str)
    """
    text_lower = text.lower()
    
    # Pattern groups with confidence levels
    checks = []
    
    # 1. Direct personal attacks with "mày" (not in movie context)
    if not is_movie_character_reference(text):
        if re.search(r'\bmày\b.*\b(ngu|dốt|não\s*úng)\b', text_lower):
            checks.append((0.80, "Tấn công cá nhân với 'mày' + xúc phạm"))
        if re.search(r'\bmày\b.*\btrại\s*súc\b', text_lower):
            checks.append((0.85, "Phi nhân hóa ('trại súc vật')"))
        if re.search(r'\bmày\b.*\bbị\b.*\b(não|thần\s*kinh|tâm\s*thần)\b', text_lower):
            checks.append((0.70, "Công kích sức khỏe tâm thần"))
    
    # 2. "thằng" + person name used derogatorily (not movie character)
    if re.search(r'\bthằng\s*(trấn\s*thành|tt)\b', text_lower):
        checks.append((0.55, "Gọi người bằng 'thằng' + tên (xúc phạm)"))
    
    # 3. Calling someone a derogatory animal term
    if re.search(r'\bchó\s*vàng\b', text_lower):
        checks.append((0.75, "Miệt thị bằng 'chó vàng'"))
    if re.search(r'\bbò\s*đỏ\b', text_lower):
        checks.append((0.60, "Miệt thị bằng 'bò đỏ'"))
    
    # 4. "trong não" + rác / "đống rác trên đầu"
    if re.search(r'\b(trong\s*não|trên\s*đầu)\b.*\brác\b', text_lower):
        # Check if directed at someone
        if re.search(r'\b(bạn|mày|ông|bà|m)\b', text_lower):
            checks.append((0.70, "Xúc phạm trí tuệ ('rác trong não')"))
    
    # 5. "nhà mày rác", "3 đời rác"  
    if re.search(r'\bnhà\s*(mày|bạn|m)\b.*\brác\b', text_lower):
        checks.append((0.80, "Xúc phạm gia đình ('nhà mày rác')"))
    if re.search(r'\b(3|ba)\s*đời\b.*\brác\b', text_lower):
        checks.append((0.80, "Xúc phạm gia đình ('3 đời rác')"))
    
    # 6. "con tinh trùng khuyết tật"
    if 'tinh trùng khuyết tật' in text_lower:
        checks.append((0.90, "Xúc phạm nghiêm trọng ('tinh trùng khuyết tật')"))
    
    # 7. "súc vật", "súc sinh" directed at someone
    if re.search(r'\bsúc\s*(vật|sinh)\b', text_lower):
        if not is_movie_character_reference(text):
            checks.append((0.80, "Phi nhân hóa ('súc vật/súc sinh')"))
    
    # 8. "lêu lêu" + derogatory term
    if re.search(r'\blêu\s*lêu\b.*\b(chó|thua|hèn|ngu)\b', text_lower):
        checks.append((0.65, "Chế giễu + miệt thị"))
    
    # 9. "đéo" + "mày" combo not caught before
    if re.search(r'\bđéo\b.*\bmày\b|\bmày\b.*\bđéo\b', text_lower):
        if not is_movie_character_reference(text):
            checks.append((0.55, "Kết hợp 'đéo' + 'mày' (thô lỗ hướng vào cá nhân)"))
    
    # 10. "phim rác" is NOT toxic (just harsh criticism), but insults 
    # about the person who made/likes the movie are toxic
    if re.search(r'\bphim\s*rác\b', text_lower):
        if re.search(r'\bmày\b|\bngu\b|\bdốt\b|\btinh\s*trùng\b', text_lower):
            checks.append((0.75, "Chê phim + xúc phạm người khác"))
    
    # 11. "xạo lol/lồn"
    if re.search(r'\bxạo\s*lo[lz]?\b', text_lower):
        checks.append((0.65, "Dùng từ tục tĩu ngụy trang ('xạo lol')"))
    
    # 12. "phim lol/loz" (disguised "phim lồn" - vulgar)
    if re.search(r'\bphim\s*lo[lz]\b', text_lower):
        checks.append((0.70, "Dùng từ tục tĩu ngụy trang ('phim lol')"))
    
    # 13. "méo" as euphemism for profanity in context
    if re.search(r'\bméo\s*(có|phải|biết)\b', text_lower):
        # Not toxic by itself - just slang for "không"
        pass
    
    if checks:
        # Return highest confidence match
        best = max(checks, key=lambda x: x[0])
        return True, best[0], best[1]
    
    return False, 0.0, ""


def apply_corrections(input_path: str, output_dir: str):
    """Apply post-processing corrections to the labeled dataset."""
    
    output_dir = Path(output_dir)
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    logger.info("Loaded %d rows for correction", len(df))
    
    corrections_made = 0
    correction_log = []
    
    for idx, row in df.iterrows():
        comment = str(row['comment'])
        old_label = row['label']
        old_conf = row['confidence']
        old_reason = str(row['reason'])
        
        new_label = old_label
        new_conf = old_conf
        new_reason = old_reason
        new_review = row['needs_review']
        
        # ===== FIX FALSE POSITIVES =====
        
        if old_label == 1:
            # Check if it's poetry/literary prose
            if is_poem_or_long_prose(comment):
                new_label = 0
                new_conf = 0.55
                new_review = True
                new_reason = "Sửa: Văn bản thơ/văn học, không phải tấn công cá nhân"
                
            # Check if negative words refer to movie characters
            elif is_movie_character_reference(comment):
                if not has_direct_personal_attack(comment):
                    new_label = 0
                    new_conf = 0.60
                    new_review = True
                    new_reason = "Sửa: Thảo luận về nhân vật phim, không phải tấn công"
        
        # ===== FIX FALSE NEGATIVES =====
        
        if old_label == 0 and not row['needs_review']:
            is_toxic, conf, reason = classify_additional_toxic(comment)
            if is_toxic and conf >= 0.55:
                new_label = 1
                new_conf = conf
                new_review = conf < 0.70  # flag for review if not highly confident
                new_reason = f"Sửa (bổ sung): {reason}"
        
        # Apply changes
        if new_label != old_label or new_conf != old_conf:
            df.at[idx, 'label'] = new_label
            df.at[idx, 'label_name'] = "Toxic" if new_label == 1 else "Non-toxic"
            df.at[idx, 'confidence'] = round(new_conf, 3)
            df.at[idx, 'needs_review'] = new_review
            df.at[idx, 'reason'] = new_reason
            corrections_made += 1
            correction_log.append({
                'id_comment': row['id_comment'],
                'old_label': int(old_label),
                'new_label': int(new_label),
                'old_conf': old_conf,
                'new_conf': round(new_conf, 3),
                'reason': new_reason,
                'comment_preview': comment[:100],
            })
    
    logger.info("Corrections made: %d", corrections_made)
    
    # Save corrected dataset
    df.to_csv(output_dir / 'facebook_comments_labeled.csv', index=False, encoding='utf-8-sig')
    
    # Save review subset
    review_df = df[df['needs_review'] == True]
    review_df.to_csv(output_dir / 'facebook_comments_review.csv', index=False, encoding='utf-8-sig')
    
    # Save correction log
    if correction_log:
        log_df = pd.DataFrame(correction_log)
        log_df.to_csv(output_dir / 'correction_log.csv', index=False, encoding='utf-8-sig')
    
    # Print stats
    total = len(df)
    toxic_n = int((df['label'] == 1).sum())
    nontoxic_n = int((df['label'] == 0).sum())
    review_n = int(df['needs_review'].sum())
    
    print("\n" + "="*60)
    print("POST-CORRECTION RESULTS")
    print("="*60)
    print("Total: %d" % total)
    print("Toxic: %d (%.2f%%)" % (toxic_n, toxic_n/total*100))
    print("Non-toxic: %d (%.2f%%)" % (nontoxic_n, nontoxic_n/total*100))
    print("Needs review: %d" % review_n)
    print("Corrections made: %d" % corrections_made)
    
    # Show corrections
    print("\n=== CORRECTIONS LOG ===")
    for c in correction_log[:30]:
        direction = "0→1 (FN fix)" if c['new_label'] == 1 else "1→0 (FP fix)"
        print("[%d] %s | %s" % (c['id_comment'], direction, c['comment_preview'][:80]))
    
    if len(correction_log) > 30:
        print("... and %d more corrections" % (len(correction_log) - 30))
    
    return df


if __name__ == '__main__':
    labeled_path = r'd:\btl kpdl\labeling\facebook_comments_labeled.csv'
    output = r'd:\btl kpdl\labeling'
    
    df_corrected = apply_corrections(labeled_path, output)

