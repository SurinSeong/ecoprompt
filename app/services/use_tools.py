from typing import Optional, Dict
from datetime import datetime
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from langchain.tools import tool

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        # Windows의 맑은 고딕 폰트 사용하기
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


def create_pdf_from_conversation(
    title: str,
    messages: list,
    output_path: str = None,
    metadata: dict = None
) -> str:
    """
    대화 내용을 PDF로 생성

    Args:
        title: PDF 제목
        messages: [{"role": "assistant|user", "content": "내용"},] 형식의 메시지 리스트
        output_path: PDF 저장 경로 (None이면 자동 생성)
        metadata: 추가 메타 데이터 (작성자, 설명 등)

    Returns:
        생성된 PDF 파일 경로
    """
    try:
        # 출력 경로 생성
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "./data"
            output_path = output_dir + "/test.pdf"

        # PDF 문서 생성
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargine=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # 스타일 설정
        korean_font = setup_korean_font()
        styles = getSampleStyleSheet()

        # 한글 스타일 추가
        styles.add(ParagraphStyle(
            name="KoreanTitle",
            parent=styles["Heading1"],
            fontName=korean_font,
            fontSize=18,
            spaceAfter=30,
        ))

        styles.add(ParagraphStyle(
            name='KoreanBody',
            parent=styles['BodyText'],
            fontName=korean_font,
            fontSize=11,
            leading=16
        ))
        
        styles.add(ParagraphStyle(
            name='UserMessage',
            parent=styles['BodyText'],
            fontName=korean_font,
            fontSize=10,
            leading=14,
            leftIndent=20,
            textColor='#2C3E50'
        ))
        
        styles.add(ParagraphStyle(
            name='AIMessage',
            parent=styles['BodyText'],
            fontName=korean_font,
            fontSize=10,
            leading=14,
            leftIndent=20,
            textColor='#16A085'
        ))

        # PDF 내용 구성
        story = []

        # 제목
        story.append(Paragraph(title, styles["KoreanTitle"]))
        story.append(Spacer(1, 0.2*inch))

        # 메타데이터
        if metadata:
            for key, value in metadata.items():
                story.append(Paragraph(f"<b>{key}:</b> {value}", styles['KoreanBody']))
            story.append(Spacer(1, 0.3*inch))
        
        # 생성 시간
        creation_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")
        story.append(Paragraph(f"<b>생성 시간:</b> {creation_time}", styles['KoreanBody']))
        story.append(Spacer(1, 0.3*inch))
            
        # 대화 내용
        for idx, message in enumerate(messages, 1):
            print(message)
            role = message.get("role", "")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            
            # 역할 표시
            if role == "USER":
                role_text = f"<b>[사용자 {idx}]</b>"
                if timestamp:
                    role_text += f" ({timestamp})"
                story.append(Paragraph(role_text, styles['UserMessage']))
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(content, styles['UserMessage']))
                
            elif role == "AI":  # AI
                role_text = f"<b>[AI 답변 {idx}]</b>"
                if timestamp:
                    role_text += f" ({timestamp})"
                story.append(Paragraph(role_text, styles['AIMessage']))
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(content, styles['AIMessage']))
            
            story.append(Spacer(1, 0.3*inch))
        
        # PDF 생성
        doc.build(story)

        print(f"✅ PDF 생성 완료: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ PDF 생성 실패: {e}")
        raise


def create_pdf_from_text(
    title: str,
    content: str,
    output_path: str = None
) -> str:
    """
    단순 텍스트를 PDF로 생성
    
    Args:
        title: PDF 제목
        content: 본문 내용
        output_path: PDF 저장 경로
    
    Returns:
        생성된 PDF 파일 경로
    """
    try:
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "generated_pdfs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"document_{timestamp}.pdf")
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        
        korean_font = setup_korean_font()
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='KoreanTitle',
            parent=styles['Heading1'],
            fontName=korean_font,
            fontSize=18,
            spaceAfter=30
        ))
        
        styles.add(ParagraphStyle(
            name='KoreanBody',
            parent=styles['BodyText'],
            fontName=korean_font,
            fontSize=11,
            leading=16
        ))

        story = []
        story.append(Paragraph(title, styles['KoreanTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # 내용을 문단별로 나눔
        paragraphs = content.split('\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para, styles['KoreanBody']))
                story.append(Spacer(1, 0.2*inch))
        
        doc.build(story)
        
        print(f"✅ PDF 생성 완료: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ PDF 생성 실패: {e}")
        raise

def get_pdf_tool() -> tool:
    """LangChain Tool로 PDF 생성 기능 제공"""
    
    def pdf_tool_function(tool_input: str) -> str:
        """
        Tool 입력 형식: "제목|||내용"
        """
        try:
            parts = tool_input.split("|||")
            if len(parts) != 2:
                return "❌ 입력 형식이 잘못되었습니다. '제목|||내용' 형식으로 입력해주세요."
            
            title, content = parts
            output_path = create_pdf_from_text(title.strip(), content.strip())
            return f"✅ PDF 파일이 생성되었습니다: {output_path}"
        
        except Exception as e:
            return f"❌ PDF 생성 중 오류 발생: {str(e)}"
    
    return tool(
        name="create_pdf",
        func=pdf_tool_function,
        description=(
            "텍스트 내용을 PDF 파일로 생성합니다. "
            "입력 형식: '제목|||내용' (|||로 구분). "
            "예: 'Python 학습 자료|||Python은 간단하고 배우기 쉬운 프로그래밍 언어입니다.'"
        )
    )