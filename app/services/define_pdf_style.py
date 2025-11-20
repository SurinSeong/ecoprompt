import os
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # Linux의 맑은 고딕 폰트 사용하기
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("NanumGothic", font_path))
            print(f"✅ 폰트 로드 성공: {font_path}")
            return "NanumGothic"
        
        else:
            print(f"❌ 폰트 파일을 찾을 수 없습니다: {font_path}")
            return "Helvetica"
        
    except Exception as e:
        print(f"❌ 폰트 설정 실패: {e}")
        return "Helvetica"



def set_pdf_style(styles):
    """PDF 스타일 설정"""
    
    # 폰트 설정
    korean_font = setup_korean_font()

    styles.add(ParagraphStyle(
        name="KoreanTitle",
        fontName=korean_font,
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor="#2C3E50"
    ))

    styles.add(ParagraphStyle(
        name='KoreanHeading1',
        fontName=korean_font,
        fontSize=18,
        spaceAfter=16,
        spaceBefore=16,
        textColor='#34495E'
    ))
    
    styles.add(ParagraphStyle(
        name='KoreanHeading2',
        fontName=korean_font,
        fontSize=14,
        spaceAfter=12,
        spaceBefore=12,
        textColor='#34495E'
    ))
    
    styles.add(ParagraphStyle(
        name='KoreanHeading3',
        fontName=korean_font,
        fontSize=12,
        spaceAfter=10,
        spaceBefore=10,
        textColor='#34495E'
    ))
    
    styles.add(ParagraphStyle(
        name='KoreanBody',
        fontName=korean_font,
        fontSize=10,
        leading=16,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name='KoreanCode',
        fontName=korean_font,
        fontSize=9,
        leading=11,
        leftIndent=15,
        spaceAfter=6,
        spaceBefore=6,
        textColor='#2C3E50'
    ))
    
    styles.add(ParagraphStyle(
        name='KoreanList',
        fontName=korean_font,
        fontSize=10,
        leading=14,
        leftIndent=15,
        spaceAfter=2,
        spaceBefore=2
    ))

    styles.add(ParagraphStyle(
        name="KoreanMeta",
        fontName=korean_font,
        fontSize=9,
        leading=11,
        textColor="#666666",
        spaceAfter=4
    ))

    return styles