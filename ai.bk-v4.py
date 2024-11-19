import openai
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
            font-size: 2.0rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        /* 섹션 헤더 스타일 */
        .section-header {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
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
        
        /* 툴팁 스타일 */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            padding: 0.5rem 1rem;
            background: #2C3E50;
            color: white;
            border-radius: 4px;
            font-size: 0.875rem;
            white-space: nowrap;
            z-index: 1000;
        }
    </style>
""", unsafe_allow_html=True)

# 심리 검사 결과에 따른 글귀 생성 함수
def get_motivational_quote(answers):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"사용자의 심리 검사 결과가 다음과 같아: {answers}. 이 사용자에게 어울리는 짧고 따뜻하고 긍정적인 명언과 같은 글귀 1개 20글자 내로 추천해줘."}
            ]
        )
        motivational_quote = response['choices'][0]['message']['content'].strip()
        return motivational_quote
    except openai.error.OpenAIError as e:
        st.error(f"ChatGPT API 호출 중 오류가 발생했습니다: {e}")
        return None

# 메인 앱 UI
st.markdown('<h1 class="main-title">✨ 심리검사를 통해 따뜻한 글귀를 얻고 나만의 책갈피를 만들어보세요!</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">심리검사를 진행하고, 당신에게 어울리는 글귀와 관련된 이미지를 받아보세요!</p>', unsafe_allow_html=True)

# 질문 리스트
questions = [
    {"icon": "fas fa-smile", "text": "오늘 기분이 어떤가요?"},
    {"icon": "fas fa-heart", "text": "당신의 가장 소중한 것은 무엇인가요?"},
    {"icon": "fas fa-user-friends", "text": "애인이 있으신가요?"},
    {"icon": "fas fa-cloud", "text": "현재 가장 큰 고민거리가 무엇인가요?"},
    {"icon": "fas fa-star", "text": "소원이 무엇인가요?"},
]

answers = []

# Q&A 섹션
for question in questions:
    answer = st.text_input(question["text"])
    answers.append(answer)

# 결과 제출 버튼
if st.button("결과 제출"):
    if all(answers):  # 모든 질문에 답변이 있는지 확인
        motivational_quote = get_motivational_quote(answers)
        if motivational_quote:
            st.markdown(f"<h3>✨ 추천 글귀: {motivational_quote}</h3>", unsafe_allow_html=True)
        else:
            st.error("추천 글귀를 생성하는 중 문제가 발생했습니다.")
    else:
        st.warning("모든 질문에 답변해주세요!")
