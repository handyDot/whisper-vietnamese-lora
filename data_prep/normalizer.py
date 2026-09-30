import re
import unicodedata

def normalize_vietnamese_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt cho Whisper:
    - Chuyển thành chữ thường.
    - Chuẩn hóa Unicode (NFC).
    - Xóa các ký tự đặc biệt, dấu câu thừa nhưng giữ lại chữ và số.
    - Xóa khoảng trắng thừa.
    """
    if not text:
        return ""
    
    # 1. Chuyển Unicode chuẩn dựng sẵn NFC
    text = unicodedata.normalize("NFC", text)
    
    # 2. Chuyển về chữ thường
    text = text.lower()
    
    # 3. Thay thế các dấu gạch ngang, gạch dưới bằng khoảng trắng
    text = re.sub(r"[-_]", " ", text)
    
    # 4. Loại bỏ các dấu câu: . , ? ! : ; " ' ( ) [ ]
    text = re.sub(r"[.,?!:;\"\'\(\)\[\]“”‘’/]", " ", text)
    
    # 5. Xóa khoảng trắng thừa giữa các từ
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

if __name__ == "__main__":
    # Test thử trực tiếp hàm chuẩn hóa
    sample = "Xin chào, Hôm nay là ngày 15/10/2026!   Dự án Whisper-LoRA."
    print("Gốc: ", sample)
    print("Chuẩn hóa: ", normalize_vietnamese_text(sample))