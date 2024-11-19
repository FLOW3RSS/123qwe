import openai
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
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

# 심리 검사 결과에 따른 글귀 생성 함수
def get_motivational_quote(answers):
    try:
        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"사용자의 심리 검사 결과가 다음과 같아: {answers}. "
                                             f"이 사용자에게 어울리는 짧고 따뜻하고 긍정적인 명언과 같은 글귀 1개 20글자 내로 추천해줘."}
            ]
        )
        # API 응답에서 메시지 추출
        motivational_quote = response.choices[0].message["content"].strip()
        return motivational_quote
    except openai.error.AuthenticationError:
        st.error("API 키 인증에 실패했습니다. 올바른 API 키를 제공하세요.")
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
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
    answer = st.text_input(label=question["text"], key=question["text"])
    answers.append(answer)

# 결과 제출 버튼
if st.button("결과 제출"):
    if all(answers):  # 모든 질문에 답변이 있는지 확인
        with st.spinner("ChatGPT가 글귀를 생성 중입니다..."):
            motivational_quote = get_motivational_quote(answers)
        if motivational_quote:
            st.success("글귀가 성공적으로 생성되었습니다!")
            st.markdown(f"<h3>✨ 추천 글귀:</h3> <p style='font-size: 1.2rem; color: #333;'>{motivational_quote}</p>", unsafe_allow_html=True)
        else:
            st.error("추천 글귀를 생성하는 중 문제가 발생했습니다.")
    else:
        st.warning("모든 질문에 답변해주세요!")
