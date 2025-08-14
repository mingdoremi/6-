import streamlit as st

st.set_page_config(page_title="멀티페이지 웹", layout="wide")

st.title("반도체 품질 관리 시스템")

# 이미지 아래 마크다운 링크
st.markdown(
    "[📎 Tableau 대시보드 열기](https://prod-apnortheast-a.online.tableau.com/t/teamsparta/views/_17548742749860/1_1?:origin=card_share_link)",
    unsafe_allow_html=True
)

# 이미지 파일 경로 (현재 코드 파일과 같은 폴더에 있을 때)
image_path = "/Users/minseo/Documents/Streamlit/실전 프로젝트/pages/공정 대시보드.png"

st.image(image_path, use_container_width=True)  # 화면 너비에 맞게 자동 조절



