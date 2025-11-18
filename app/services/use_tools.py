from datetime import datetime
from json import JSONDecoder
import os
import json
import re

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from langchain.tools import tool

from app.services.define_pdf_style import setup_korean_font, set_pdf_style


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


def create_mr_template_pdf(
    content: str,
    output_dir: str = "./data"
) -> str:
    """
    MR(Merge Request) 템플릿을 PDF로 생성

    Args:
        content: LLM이 생성한 MR 템플릿 내용 (마크다운 형식)
        output_dir: PDF 저장 디렉토리

    Returns:
        생성된 PDF 파일 경로
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"MR_Template_{timestamp}.pdf"
        output_path = os.path.abspath(os.path.join(output_dir, filename))

        doc = SimpleDocTemplate(output_path, pagesize=A4)
        korean_font = setup_korean_font()
        styles = getSampleStyleSheet()

        # 스타일 정의
        # 1) Title
        styles.add(ParagraphStyle(
            name="KoreanTitle",
            fontName=korean_font,
            fontSize=20,
            spaceAfter=20,
            alignment=1    # 가운데
        ))

        # 2) Heading
        styles.add(ParagraphStyle(
            name="KoreanHeading",
            fontName=korean_font,
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12,
            textColor="#2C3E50"
        ))

        # 3) Body
        styles.add(ParagraphStyle(
            name="KoreanBody",
            fontName=korean_font,
            fontSize=10,
            leading=16,
            leftIndent=10
        ))

        # 내용 작성할 리스트
        story = []

        # 제목
        story.append(Paragraph("Merge Request Template", styles["KoreanTitle"]))
        story.append(Spacer(1, 0.3*inch))

        # 생성 시간
        creation_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")
        story.append(Paragraph(f"생성 시간: {creation_time}", styles["KoreanBody"]))
        story.append(Spacer(1, 0.3*inch))

        # 내용 파싱
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.1*inch))
                continue

            # HTML 특수 문자 이스케이프
            line = line.replace("&", "&amp;")
            line = line.replace("<", "&lt;")
            line = line.replace(">", "&gt;")

            # 마크다운 헤딩 처리
            if line.startswith("###"):
                text = line.replace("###", "").strip()
                story.append(Paragraph(f"<b>{text}</b>", styles["KoreanHeading"]))
            
            elif line.startswith("##"):
                text = line.replace('##', '').strip()
                story.append(Paragraph(f"<b>{text}</b>", styles['KoreanHeading']))

            elif line.startswith("#"):
                text = line.replace('#', '').strip()
                story.append(Paragraph(f"<b>{text}</b>", styles['KoreanTitle']))

            # 리스트 처리
            elif line.startswith("- ") or line.startswith("* "):
                text = line[2:].strip()
                story.append(Paragraph(f"• {text}", styles['KoreanBody']))

            # 볼드 처리
            elif "**" in line:
                text = line.replace("**", "<b>").replace("**", "</b>")
                story.append(Paragraph(text, styles["KoreanBody"]))

            else:
                story.append(Paragraph(line, styles["KoreanBody"]))

        doc.build(story)

        print(f"✅ MR 템플릿 PDF 생성 완료: {output_path}")

    except Exception as e:
        print(f"❌ PDF 생성 실패: {e}")
        raise Exception(f"PDF 생성 중 오류: {str(e)}")


# pdf 생성 함수
def create_pdf_document(
    title: str,
    content: str,
    output_dir: str = "./data"
) -> str:
    """
    범용 PDF 문서 생성

    Args:
        title: PDF 문서 제목
        content: PDF에 포함할 내용 (마크다운 형식)
        output_dir: PDF 저장 디렉토리

    Returns:
        생성된 PDF 파일의 절대 경로
    """
    try:
        # 파일명 생성 (제목을 파일명으로 사용, 특수문자 제거)
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).strip()
        safe_title = safe_title.replace(" ", "_")[:50]    # 최대 50자

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_title}_{timestamp}.pdf"
        output_path = os.path.abspath(os.path.join(output_dir, filename))

        # PDF 문서 생성
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # 스타일 정의
        styles = getSampleStyleSheet()
        
        # 스타일 설정
        styles = set_pdf_style(styles)

        # PDF 내용 구성
        story = []

        # 제목
        story.append(Paragraph(title, styles['KoreanTitle']))
        story.append(Spacer(1, 0.3*inch))

        # 생성 시간
        creation_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")
        story.append(Paragraph(f"<i>생성 시간: {creation_time}</i>", styles['KoreanBody']))
        story.append(Spacer(1, 0.4*inch))

        # 내용 파싱 (간단한 마크다운 파싱)
        lines = content.split('\n')
        in_code_block = False
        code_lines = []
        skip_first_h1 = True    # 추가: 첫 번째 # 제목 건너뛰기 플래그

        for line in lines:
            # 빈 줄
            if not line.strip():
                if in_code_block:
                    code_lines.append("")
                else:
                    story.append(Spacer(1, 0.1*inch))
                continue
            
            # 코드 블록 처리
            if line.strip().startswith('```'):
                if in_code_block:
                    # 코드 블록 종료
                    code_text = '\n'.join(code_lines)
                    code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    story.append(Paragraph(f"<font face='Courier'>{code_text}</font>", styles['KoreanCode']))
                    code_lines = []
                    in_code_block = False
                else:
                    # 코드 블록 시작
                    in_code_block = True
                continue
            
            if in_code_block:
                code_lines.append(line)
                continue

            # HTML 특수문자 이스케이프
            line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # 마크다운 헤딩 처리
            if line.startswith('# '):
                # 첫 번째 # 제목은 건너뛰기
                if skip_first_h1:
                    skip_first_h1 = False
                    continue

                text = line[2:].strip()
                story.append(Paragraph(f"<b>{text}</b>", styles['KoreanHeading1']))

            elif line.startswith('## '):
                text = line[3:].strip()
                story.append(Paragraph(f"<b>{text}</b>", styles['KoreanHeading2']))
                
            elif line.startswith('### '):
                text = line[4:].strip()
                story.append(Paragraph(f"<b>{text}</b>", styles['KoreanHeading3']))
            
            # 리스트 처리
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                text = line.strip()[2:].strip()
                # 볼드 처리
                text = text.replace('**', '<b>').replace('**', '</b>')
                story.append(Paragraph(f"• {text}", styles['KoreanList']))
            
            # 번호 리스트
            elif line.strip()[0:2].replace('.', '').isdigit():
                story.append(Paragraph(line.strip(), styles['KoreanList']))
            
            # 일반 텍스트
            else:
                # 볼드 처리
                text = line.replace('**', '<b>').replace('**', '</b>')
                # 이탤릭 처리
                text = text.replace('*', '<i>').replace('*', '</i>')
                story.append(Paragraph(text, styles['KoreanBody']))
        
        # PDF 빌드
        doc.build(story)

        print(f"✅ PDF 문서 생성 완료: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ PDF 생성 실패: {e}")
        raise Exception(f"PDF 생성 중 오류: {str(e)}")


# Tool 정의 (Function Calling 용)
def get_tool_definitions():
    """
    LLM에게 제공할 Tool 정의 (JSON Schema)
    Mi:dm 모델용 Tool 정의
    tool_list 배열 형식으로 반환
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "save_as_pdf",
                "description": (
                    "사용자가 요청한 내용을 PDF 문서로 저장합니다. "
                    "MR 템플릿, 학습 자료, 대화 정리 등 어떠 내용이든 PDF로 만들 수 있습니다. "
                    "**중요**: 사용자가 '대화 기록', '지금까지의 대화', '대화 내용' 등을 PDF로 만들어 달라고 하면, "
                    "반드시 [History]에 있는 이전 대화 내용을 분석해서 요약한 후 content에 포함하세요."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "PDF 문서의 제목 (예: 'Python 학습 자료', 'MR 템플릿', '대화 기록 요약')"
                        },
                        "content": {
                            "type": "string",
                            "description": (
                                "PDF에 포함될 전체 내용 (마크다운 형식 권장). "
                                "**중요 사항**: "
                                "1. 제목(# 제목)은 포함하지 마세요. ## 섹션 제목부터 시작하세요. "
                                "2. 사용자가 '대화 기록'을 요청하면 [History]의 내용을 분석하고 요약하여 작성하세요. "
                                "3. 마크다운 형식을 사용하세요: 헤딩(##, ###), 리스트(-, *), 볼드(**텍스트**) 등. "
                                "4. 구조화된 형태로 작성하세요 (섹션별로 나누기)."
                            )
                        }
                    },
                    "required": ["title", "content"]
                }
            }
        }
    ]


def extract_valid_json(text: str) -> str:
    """
    텍스트에서 유효한 JSON 부분만 추출
    중괄호 균형을 맞춰서 첫 번째 완전한 JSON 객체를 찾음
    """
    brace_count = 0
    start_idx = -1

    for i, char in enumerate(text):
        if char == "{":
            if brace_count == 0:
                start_idx = i
            brace_count += 1

        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # 완전한 JSON 객체 발견
                return text[start_idx:i+1]


def parse_midm_tool_call(text: str) -> list:
    """
    Mi:dm 모델의 Tool 호출 파싱

    형식: <tool_call>{"name": "tool_name", "arguments": {"param":"value"}}</tool_call>
    """
    tool_calls = []

    # <tool_call>...</tool_call> 패턴 찾기
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        json_str = match.group(1).strip()

        try:
            decoder = JSONDecoder()

            try:
                tool_call, idx = decoder.raw_decode((json_str))

                # 파싱 후 남은 텍스트가 있는지 확인
                remaining = json_str[idx:].strip()
                if remaining:
                    print(f"⚠️ JSON 뒤에 추가 텍스트 발견: {remaining}")

                tool_calls.append(tool_call)
                print(f"✅ Tool 호출 파싱 성공: {tool_call['name']}")
            
            except json.JSONDecodeError as e:
                print(f"❌ JSON 파싱 실패: {e}")
                print(f"   위치: {e.pos}")
                print(f"   문자: {json_str[max(0, e.pos-10):e.pos+10]}")
                
                # ✨ 방법 2: 중괄호 균형 맞추기
                # JSON이 중간에 끊겼을 수 있으므로 유효한 부분만 추출
                valid_json = extract_valid_json(json_str)
                if valid_json:
                    try:
                        tool_call = json.loads(valid_json)
                        tool_calls.append(tool_call)
                        print(f"✅ 복구된 JSON 파싱 성공")
                    except:
                        pass

        except Exception as e:
            print(f"❌ 예상치 못한 오류: {e}")
    
    return tool_calls


def parse_qwen_tool_call(text: str) -> list:
    tool_calls = []
    json_code_pattern = r'\s*\n(.*?)\n```'
    json_matches = re.finditer(json_code_pattern, text, re.DOTALL)
        
    for match in json_matches:
        json_str = match.group(1).strip()
        try:
            tool_call = json.loads(json_str)
            
            # save_as_pdf 도구인지 확인
            if tool_call.get("name") == "save_as_pdf":
                tool_calls.append(tool_call)
                print(f"✅ Tool 파싱 성공 (코드 블록 형식): save_as_pdf")

        except json.JSONDecodeError as e:
            print(f"❌ 코드 블록 JSON 파싱 실패: {e}")
    
    return tool_calls


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Tool 실행"""
    if tool_name == "save_as_pdf":
        try:
            title = arguments.get("title", "문서")
            content = arguments.get("content", "")

            if not content:
                return "❌ PDF에 포함할 내용이 없습니다."

            pdf_path = create_pdf_document(title, content)
            return f"✅ PDF 문서가 생성되었습니다!\n📄 제목: {title}\n📁 파일 경로: {os.path.basename(pdf_path)}"
        
        except Exception as e:
            return f"❌ PDF 생성 실패: {str(e)}"
    else:
        return f"❌ 알 수 없는 Tool: {tool_name}"