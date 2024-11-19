import openai
from openai import OpenAIError
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import cv2
import numpy as np

# 페이지 기본 설정
st.set_page_config(
    page_title="책갈피 만들기",
    page_icon="📖",
    layout="wide"
)

# 상태 관리를 위한 session_state 초기화
if 'show_qa' not in st.session_state:
    st.session_state.show_qa = True
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
if 'quote' not in st.session_state:
    st.session_state['quote'] = ""  # 기본값 설정
if 'background_image_url' not in st.session_state:
    st.session_state['background_image_url'] = "네잎클로버.jpg"
if 'font_choice' not in st.session_state:
    st.session_state['font_choice'] = "나눔손글씨 가람연꽃"
if 'font_size' not in st.session_state:
    st.session_state['font_size'] = 30
if 'text_color' not in st.session_state:
    st.session_state['text_color'] = "#000000"
if 'stroke_color' not in st.session_state:
    st.session_state['stroke_color'] = "#FFFFFF"
if 'x_position' not in st.session_state:
    st.session_state['x_position'] = 512
if 'y_position' not in st.session_state:
    st.session_state['y_position'] = 512

# OpenAI API Key 설정
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.markdown('<div style="text-align: center; color: #28a745;"><i class="fas fa-check-circle"></i> OpenAI API 연결됨</div>', unsafe_allow_html=True)
except KeyError:
    st.error('🔑 OpenAI API Key가 설정되지 않았습니다. Streamlit Secrets Manager를 확인해주세요.')

# Custom CSS 적용
st.markdown("""
    <style>
        /* Font Awesome 아이콘 추가 */
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
        
        /* 구글 폰트 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
        
        /* 기본 스타일 */
        html, body, [class*="css"] {
            font-family: 'Noto Sans KR', sans-serif;
        }
        
        /* 메인 타이틀 스타일 */
        .main-title {
            color: #2C3E50;
            padding: 2rem 0;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* 섹션 헤더 스타일 */
        .section-header {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1.5rem 0;
            border-left: 5px solid #667eea;
        }
        
        /* 입력 필드 컨테이너 */
        .input-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            border: 1px solid #e9ecef;
            transition: all 0.3s ease;
        }
        
        .input-container:hover {
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* 입력 필드 스타일 */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 2px solid #e9ecef;
            padding: 0.8rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.2);
        }
        
        /* 버튼 스타일 */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 50px;
            font-weight: 500;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            text-transform: uppercase;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102,126,234,0.3);
        }
        
        /* 선택 박스 스타일 */
        .stSelectbox > div > div {
            border-radius: 8px;
            border: 2px solid #e9ecef;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #667eea;
        }
        
        /* 슬라이더 스타일 */
        .stSlider > div > div > div {
            background-color: #667eea;
        }
        
        /* 이미지 스타일 */
        .stImage {
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .stImage:hover {
            transform: scale(1.03);
        }
        
        /* 아이콘 스타일 */
        .icon {
            margin-right: 8px;
            color: #667eea;
        }
        
        /* 결과 컨테이너 */
        .result-container {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.08);
            margin: 2rem 0;
            border: 2px solid #e9ecef;
        }
    </style>
""", unsafe_allow_html=True)

# 핵심 함수들
def get_motivational_quote(answers):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"사용자의 심리 검사 결과가 다음과 같아: {answers}. 이 사용자에게 어울리는 짧고 따뜻하고 긍정적인 명언과 같은 글귀 1개 20글자 내로 추천해줘."}
            ],
            max_tokens=50,
            temperature=0.7
        )
        motivational_quote = response['choices'][0]['message']['content'].strip()
        return motivational_quote
    except OpenAIError as e:
        st.error(f"ChatGPT API 호출 중 오류가 발생했습니다: {e}")
        return None

def overlay_text_with_custom_font(image_path, text, font_choice, text_color, stroke_color, x=None, y=None, font_size=60, upscale_factor=6):
    try:
        image = Image.open(image_path)
        original_size = image.size
        high_res_size = (original_size[0] * upscale_factor, original_size[1] * upscale_factor)
        image = image.resize(high_res_size, Image.LANCZOS)
        
        font_paths = {
            "나눔손글씨 가람연꽃": "나눔손글씨 가람연꽃.ttf",
            "나눔손글씨 펜": "나눔손글씨 펜.ttf",
            "나눔고딕": "나눔고딕.ttf",
            "돋움체": "돋움체.ttf"
        }
        font_path = font_paths.get(font_choice, "나눔손글씨 가람연꽃.ttf")
        font = ImageFont.truetype(font_path, font_size * upscale_factor)
    except (UnidentifiedImageError, IOError) as e:
        st.error(f"이미지를 불러오거나 글꼴을 로드하는 중 오류가 발생했습니다: {e}")
        return None

    draw = ImageDraw.Draw(image)
    lines = text.split("\n")
    text_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in lines]) if lines else 0
    total_text_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in lines]) + (len(lines) - 1) * 10

    if x is None or y is None:
        x = (image.width - text_width) / 2
        y = (image.height - total_text_height) / 2

    y_offset = y
    for line in lines:
        draw.text((x, y_offset), line, fill=text_color, font=font, align="center", stroke_width=8, stroke_fill=stroke_color)
        y_offset += draw.textbbox((0, 0), line, font=font)[3] + 10 * upscale_factor

    image = image.resize(original_size, Image.LANCZOS)
    return image

# 메인 앱 UI
def render_ui():
    st.markdown('<h1 class="main-title">✨ 심리검사를 통해 따뜻한 글귀를 얻고 나만의 책갈피를 만들어보세요!</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">심리검사를 진행하고, 당신에게 어울리는 글귀와 관련된 이미지를 받아보세요!</p>', unsafe_allow_html=True)

    questions = [
        {"icon": "fas fa-smile", "text": "오늘 기분이 어떤가요?"},
        {"icon": "fas fa-heart", "text": "당신의 가장 소중한 것은 무엇인가요?"},
        {"icon": "fas fa-user-friends", "text": "애인이 있으신가요?"},
        {"icon": "fas fa-cloud", "text": "현재 가장 큰 고민거리가 무엇인가요?"},
        {"icon": "fas fa-star", "text": "소원이 무엇인가요?"},
    ]

    answers = []

    for question in questions:
        st.markdown(f'<div class="input-container"><div><i class="{question["icon"]} icon"></i>{question["text"]}</div></div>', unsafe_allow_html=True)
        answer = st.text_input(question["text"])
        answers.append(answer)

    if st.button("✨ 결과 제출"):
        if all(answers):
            motivational_quote = get_motivational_quote(answers)
            if motivational_quote:
                st.session_state['quote'] = motivational_quote  # 초기화 시 생성된 글귀를 저장

    st.markdown('<div class="result-container"><h3><i class="fas fa-quote-left" style="color: #667eea;"></i> 추천 글귀 (수정 가능)</h3>', unsafe_allow_html=True)
    selected_quote = st.text_area("글귀 입력", value=st.session_state['quote'], height=100)
    st.session_state['quote'] = selected_quote
                
    # 이미지 선택 및 글귀 추가
    st.markdown('<div class="section-header"><i class="fas fa-image"></i> 배경 이미지를 선택하세요</div>', unsafe_allow_html=True)
    uploaded_images = ["네잎클로버.jpg", '라이즈 소희.jpg', '라이즈 앤톤.jpg', '라이즈 원빈.jpg', '라이즈 은석.jpg', '물감.jpg', '물결.jpg', '바다.jpg', '비눗방울.jpg', '에스파 카리나.jpg', '투데이.jpg', '고양이.jpg', '동화.jpg', '노을.jpg', '어항 고양이.jpg', '어항.jpg', '화사한 고양이.jpg', '심해.jpg', '크리스마스.jpg']

    selected_image = st.selectbox("🖼️ 배경 이미지 선택", options=uploaded_images)

    if selected_image:
        st.session_state['background_image_url'] = selected_image
        image = Image.open(selected_image)
        st.image(image, caption="선택된 배경 이미지", use_column_width=False)

        st.markdown('<div class="section-header"><i class="fas fa-paint-brush"></i> 스타일을 설정하세요</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            font_choice = st.selectbox("📝 글꼴 선택", ["나눔손글씨 가람연꽃", "예스 명조 레귤러"], index=0)
            st.session_state['font_choice'] = font_choice
            font_size = st.number_input("📏 글귀 크기 (pt)", min_value=10, max_value=200, value=st.session_state['font_size'], step=1)
            st.session_state['font_size'] = font_size

        with col2:
            text_color = st.color_picker("🎨 글귀 색상", st.session_state['text_color'])
            st.session_state['text_color'] = text_color
            stroke_color = st.color_picker("✏️ 글귀 테두리 색상", st.session_state['stroke_color'])
            st.session_state['stroke_color'] = stroke_color

                st.markdown('<div class="section-header"><i class="fas fa-arrows-alt"></i> 글귀 위치를 조정하세요</div>', unsafe_allow_html=True)
        x_position = st.slider("⬅️➡️ x 좌표 (픽셀)", min_value=0, max_value=2048, value=st.session_state['x_position'], step=10)
        st.session_state['x_position'] = x_position
        y_position = st.slider("⬆️⬇️ y 좌표 (픽셀)", min_value=0, max_value=2048, value=st.session_state['y_position'], step=10)
        st.session_state['y_position'] = y_position

        # 이미지에 글귀 추가
        final_image = overlay_text_with_custom_font(
            st.session_state['background_image_url'],
            st.session_state['quote'],  # 수정된 글귀 반영
            st.session_state['font_choice'],
            text_color=st.session_state['text_color'],
            stroke_color=st.session_state['stroke_color'],
            x=st.session_state['x_position'],
            y=st.session_state['y_position'],
            font_size=st.session_state['font_size']
        )

        if final_image:
            st.markdown('<div class="section-header"><i class="fas fa-magic"></i> 완성된 책갈피</div>', unsafe_allow_html=True)
            st.image(final_image, caption="✨ 글귀가 추가된 이미지", use_column_width=False)

render_ui()

