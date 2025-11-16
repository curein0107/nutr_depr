from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import streamlit as st

try:
    # transformers는 선택적이며 인터넷 연결이 없으면 로드에 실패할 수 있습니다.
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    _transformers_available = True
except Exception:
    _transformers_available = False


def disclaimer() -> None:
    """사용자에게 중요한 면책문구를 보여준다."""
    st.warning(
        """이 애플리케이션은 건강 관련 정보를 참고용으로 제공하며,
        전문적인 진단이나 치료를 대신할 수 없습니다. 우울증 위험도
        예측 결과는 참고 용도로만 사용해야 하며, 자신의 정신건강에
        관해 궁금한 점이 있으면 반드시 의료 전문가와 상담하시기 바랍니다.""",
        icon="⚠️",
    )


@st.cache_resource(show_spinner=False)
def load_model(model_path: str = "phq9_nutrition_model.pkl"):
    """
    Pickle 파일로 저장된 scikit‑learn 모델을 로드한다.

    Parameters
    ----------
    model_path : str, optional
        모델 파일 경로. 기본값은 `phq9_nutrition_model.pkl`이다.

    Returns
    -------
    object
        joblib으로 로드된 모델 객체. 파일이 없거나 로딩에 실패하면
        ``None``을 반환한다.
    """
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    try:
        import joblib  # 로컬에 존재하는 경량 라이브러리
    except ImportError:
        st.error("joblib 라이브러리가 필요합니다. 환경에 joblib이 설치되지 않았습니다.")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"모델 로딩 중 오류가 발생했습니다: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_text_generator(model_name: str = "sshleifer/tiny-gpt2", device: int = -1):
    """
    경량화된 텍스트 생성 파이프라인을 로드한다. 인터넷이 차단되어 있거나
    transformers가 없는 경우 ``None``을 반환한다.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace 모델 이름. 기본값은 작은 GPT‑2 모델이다.
    device : int, optional
        -1은 CPU를 의미한다. 환경에 GPU가 있으면 해당 장치를 지정할 수 있다.

    Returns
    -------
    Optional[callable]
        transformers.pipeline 객체 또는 ``None``
    """
    if not _transformers_available:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        return gen
    except Exception:
        # 로딩 실패 시 None 반환
        return None


def compute_contributions(model: object, X: pd.DataFrame) -> np.ndarray:
    """
    SHAP 없이 각 특성의 기여도를 근사하여 반환한다. 모델에 따라
    ``coef_``(선형 모델) 또는 ``feature_importances_``(트리 모델)를 사용한다.

    Parameters
    ----------
    model : object
        scikit‑learn 분류 모델.
    X : pandas.DataFrame
        단일 샘플을 포함하는 DataFrame.

    Returns
    -------
    numpy.ndarray
        특성별 기여도 벡터. 길이는 특성 개수와 같다.
    """
    n_features = X.shape[1]
    # 기본값은 0 기여도
    contributions = np.zeros(n_features)
    try:
        if hasattr(model, "coef_"):
            # 선형 모델: coef_ 크기 (n_classes, n_features)
            # 양성 클래스(인덱스 0)을 사용하여 값 곱셈
            coef = model.coef_[0]
            contributions = coef * X.iloc[0].values
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            contributions = importances * X.iloc[0].values
    except Exception:
        # 예외 발생 시 0 기여도를 유지
        pass
    return contributions


def get_top_features(
    contributions: np.ndarray, feature_names: List[str], top_n: int = 5
) -> List[Dict[str, object]]:
    """
    기여도 벡터를 기준으로 가장 영향력이 큰 특성 목록을 반환한다.

    Parameters
    ----------
    contributions : numpy.ndarray
        각 특성의 기여도 벡터.
    feature_names : List[str]
        특성 이름 목록.
    top_n : int, optional
        반환할 상위 특성 개수. 기본값은 5이다.

    Returns
    -------
    List[Dict[str, object]]
        ``feature``, ``contribution``, ``direction`` 키를 가진 dict 목록.
    """
    # 절대값 기준 정렬
    indices = np.argsort(np.abs(contributions))[::-1][: min(top_n, len(contributions))]
    results: List[Dict[str, object]] = []
    for idx in indices:
        val = float(contributions[idx])
        direction = "increase" if val > 0 else "decrease"
        results.append({"feature": feature_names[idx], "contribution": abs(val), "direction": direction})
    return results


def build_explanation(
    probability: float,
    top_features: List[Dict[str, object]],
    generator: Optional[callable] = None,
) -> str:
    """
    모델 예측 결과와 주요 특성을 바탕으로 한국어 설명문을 생성한다.
    가능한 경우 경량 LLM을 사용하고, 그렇지 않으면 규칙 기반으로 작성한다.

    Parameters
    ----------
    probability : float
        긍정 클래스(우울증 위험) 확률.
    top_features : List[Dict[str, object]]
        가장 영향력이 큰 특성 목록.
    generator : Optional[callable], optional
        transformers.pipeline 객체. None이면 규칙 기반 설명을 사용한다.

    Returns
    -------
    str
        생성된 설명.
    """
    # 프롬프트를 구성
    bullet_lines = []
    for feat in top_features:
        arrow = "↑" if feat["direction"] == "increase" else "↓"
        bullet_lines.append(f"- {feat['feature']} : 영향 방향 {arrow} (중요도 {feat['contribution']:.4f})")
    prompt = (
        f"당신의 우울증 위험도 예측 결과는 {probability * 100:.1f}% 입니다.\n"
        f"SHAP 분석을 통해 다음과 같은 주요 요인이 확인되었습니다:\n"
        + "\n".join(bullet_lines)
        + "\n\n위 정보를 바탕으로 건강 전문가의 시각에서 간단하고 이해하기 쉬운 설명을 작성해 주세요. 4-6문장 이내로 작성해 주세요."
    )
    # LLM 사용 가능하면 실행
    if generator is not None:
        try:
            output = generator(
                prompt,
                max_length=len(prompt.split()) + 60,
                num_return_sequences=1,
            )
            generated = output[0]["generated_text"]
            return generated
        except Exception:
            pass
    # 규칙 기반 fallback
    lines: List[str] = []
    lines.append(f"예측된 우울증 위험도는 {probability * 100:.1f}%입니다.")
    for feat in top_features:
        kor_direction = "증가" if feat["direction"] == "increase" else "감소"
        # 변수 값의 변화 방향에 따라 위험도 영향을 설명
        lines.append(f"'{feat['feature']}' 값이 {kor_direction} 방향으로 우울증 위험에 영향을 미칩니다.")
    lines.append("정기적인 운동과 균형 잡힌 생활 습관이 정신 건강 유지에 도움이 될 수 있습니다.")
    return "\n".join(lines)


def respond_chat(user_query: str, generator: Optional[callable] = None) -> str:
    """
    챗봇 질문에 응답을 생성한다. transformers LLM이 있으면 이를 이용하고,
    없으면 간단한 규칙 기반 답변을 제공한다.

    Parameters
    ----------
    user_query : str
        사용자가 입력한 질문.
    generator : Optional[callable], optional
        text-generation pipeline. None이면 규칙 기반 설명을 사용한다.

    Returns
    -------
    str
        챗봇 응답.
    """
    # 기본 프롬프트: 의료 자문이 아님을 강조
    base_prompt = (
        "당신은 친근하고 신뢰할 수 있는 건강 정보 챗봇입니다. 사용자의 질문에 대해 간단하고 긍정적인 언어로 답변하십시오.\n"
        "사용자가 물어본 내용이 정신 건강에 관한 것이더라도, 정확한 진단이나 치료를 대신할 수 없으며 전문가와 상담할 것을 항상 권장해야 합니다.\n"
        f"사용자: {user_query}\n"
        "챗봇:\n"
    )
    if generator is not None:
        try:
            output = generator(base_prompt, max_length=len(base_prompt.split()) + 60, num_return_sequences=1)
            reply = output[0]["generated_text"].split("챗봇:")[-1].strip()
            return reply
        except Exception:
            pass
    # 규칙 기반 답변: 질문 내용을 반복하고 전문가 상담을 권유
    return (
        f"질문해주셔서 감사합니다. '{user_query}'에 대해 알아보려는 당신의 노력이 중요합니다.\n"
        "저는 정확한 진단을 제공할 수 없지만, 균형 잡힌 식사와 꾸준한 신체 활동이 정신 건강에 도움을 줄 수 있습니다.\n"
        "더 자세한 정보나 상담이 필요하다면 정신건강 전문가에게 문의해 보세요."
    )


def main() -> None:
    """
    Streamlit 애플리케이션의 메인 함수.

    * 모델을 로드하고 캐시합니다.
    * 미리 정의한 다양한 인구 통계, 건강, 식습관 변수에 대해 입력 폼을 제공합니다.
    * 입력된 값으로 우울증 위험도를 예측하고, 기여도를 계산하여 주요 요인을 표시합니다.
    * 설명을 LLM 또는 규칙 기반으로 생성합니다.
    * 간단한 챗봇 인터페이스를 통해 추가 질문에 답변합니다.
    """
    st.set_page_config(page_title="우울증 위험도 예측", page_icon="🧠", layout="centered")
    st.title("개인 맞춤형 우울증 위험도 예측")
    disclaimer()

    # 모델 로드
    model = load_model()
    if model is None:
        st.stop()

    # 입력 변수 정의: 변수명과 라벨, 타입, 옵션 지정
    feature_definitions = [
        ("sex", {"label": "성별", "type": "select", "options": {1: "남자", 2: "여자"}}),
        ("age", {"label": "현재 나이", "type": "number"}),
        ("individual_income", {"label": "개인소득", "type": "select", "options": {1: "매우 낮음", 2: "낮음", 3: "높음", 4: "매우 높음"}}),
        ("household_income", {"label": "가구소득", "type": "select", "options": {1: "매우 낮음", 2: "낮음", 3: "높음", 4: "매우 높음"}}),
        ("education_level", {"label": "학력", "type": "select", "options": {1: "초등학교 이하", 2: "중학교 졸업", 3: "고등학교 졸업", 4: "대학교 이상"}}),
        ("occupation", {"label": "직업 여부", "type": "select", "options": {1: "직업 있음", 0: "직업 없음"}}),
        ("number_of_household_member", {"label": "독거 여부", "type": "select", "options": {1: "독거", 2: "동거"}}),
        ("house_status", {"label": "주택 소유 여부", "type": "select", "options": {1: "소유", 0: "미소유"}}),
        ("marital_statues", {"label": "결혼 여부", "type": "select", "options": {1: "기혼", 0: "미혼"}}),
        ("subjective_health_status", {"label": "주관적 건강상태", "type": "select", "options": {1: "나쁨", 2: "보통", 3: "좋음"}}),
        ("unmet_medical_care", {"label": "의료 이용 여부", "type": "select", "options": {1: "치료 받지 못함", 0: "치료 받음"}}),
        ("labor_hour", {"label": "주간 근로시간", "type": "number"}),
        ("smoking", {"label": "흡연 여부", "type": "select", "options": {1: "흡연자", 0: "비흡연자"}}),
        ("drinking", {"label": "음주 여부", "type": "select", "options": {1: "음주자", 0: "비음주자"}}),
        ("stress", {"label": "스트레스 정도", "type": "select", "options": {1: "스트레스 없음", 2: "스트레스 낮음", 3: "스트레스 높음", 4: "스트레스 매우 높음"}}),
        ("hpa_work", {"label": "일로 인한 고강도 신체활동", "type": "select", "options": {1: "예", 0: "아니오"}}),
        ("mpa_work", {"label": "일로 인한 중등도 신체활동", "type": "select", "options": {1: "예", 0: "아니오"}}),
        ("hpa_leisure", {"label": "여가로 고강도 신체활동", "type": "select", "options": {1: "예", 0: "아니오"}}),
        ("mpa_leisure", {"label": "여가로 중등도 신체활동", "type": "select", "options": {1: "예", 0: "아니오"}}),
        ("walk", {"label": "걷기 여부", "type": "select", "options": {1: "예", 0: "아니오"}}),
        ("sedantary_hour", {"label": "하루 평균 앉아있는 시간", "type": "number"}),
        ("body_mass_index", {"label": "체질량지수", "type": "number"}),
        ("food_intake", {"label": "식품 섭취량", "type": "number"}),
        ("calorie_intake", {"label": "칼로리 섭취량", "type": "number"}),
        ("weter_intake", {"label": "물 섭취량", "type": "number"}),
        ("protein", {"label": "단백질 섭취량", "type": "number"}),
        ("saturated_fatty_acid", {"label": "포화지방산 섭취량", "type": "number"}),
        ("mono_unsaturated_fatty_acid", {"label": "단일불포화지방산 섭취량", "type": "number"}),
        ("n3_fatty_acid", {"label": "n3 지방산 섭취량", "type": "number"}),
        ("n6_fatty_acid", {"label": "n6 지방산 섭취량", "type": "number"}),
        ("cholesterol", {"label": "콜레스테롤 섭취량", "type": "number"}),
        ("carbohydrate", {"label": "탄수화물 섭취량", "type": "number"}),
        ("dietary_fiber", {"label": "식이섬유 섭취량", "type": "number"}),
        ("calcium", {"label": "칼슘 섭취량", "type": "number"}),
        ("phosphorus", {"label": "인 섭취량", "type": "number"}),
        ("iron", {"label": "철분 섭취량", "type": "number"}),
        ("soudim", {"label": "나트륨 섭취량", "type": "number"}),
        ("potassium", {"label": "칼륨 섭취량", "type": "number"}),
        ("betacarotine", {"label": "베타카로틴 섭취량", "type": "number"}),
        ("retinol", {"label": "레티놀 섭취량", "type": "number"}),
        ("vitamin_b1", {"label": "비타민 B1 섭취량", "type": "number"}),
        ("vitamin_b2", {"label": "비타민 B2 섭취량", "type": "number"}),
        ("vitamin_b3", {"label": "비타민 B3 섭취량", "type": "number"}),
        ("vitamin_c", {"label": "비타민 C 섭취량", "type": "number"}),
        ("cardiovascular_disease", {"label": "심혈관 질환 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
        ("arthritis_disease", {"label": "관절염 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
        ("pulmonary_disease", {"label": "호흡기계 질환 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
        ("liver_disease", {"label": "간 질환 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
        ("thyroid_disease", {"label": "갑상선 질환 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
        ("t2_diabetes_mellitus", {"label": "제2형 당뇨병 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
        ("atopic_dermatitis", {"label": "아토피 피부염 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
        ("allergic_rhinitis", {"label": "알레르기성 비염 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
        ("renal_disease", {"label": "신장 질환 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
        ("cancer", {"label": "암 여부", "type": "select", "options": {1: "있음", 0: "없음"}}),
    ]

    feature_names: List[str] = [name for name, _ in feature_definitions]
    st.subheader("입력 파라미터를 선택하세요")
    user_values: Dict[str, float] = {}
    cols = st.columns(2)
    # 각 입력 필드를 생성
    for i, (name, info) in enumerate(feature_definitions):
        with cols[i % 2]:
            if info["type"] == "select":
                display_options = list(info["options"].values())
                selected_display = st.selectbox(info["label"], display_options, key=name)
                for code, disp in info["options"].items():
                    if disp == selected_display:
                        user_values[name] = code
                        break
            else:
                value = st.number_input(info["label"], value=0.0, step=0.1, key=name)
                user_values[name] = float(value)

    # 예측 실행
    if st.button("우울증 위험도 예측하기"):
        X_input = pd.DataFrame([user_values], columns=feature_names)
        try:
            proba = model.predict_proba(X_input)[0][1]
        except Exception as e:
            st.error(f"모델 예측 중 오류가 발생했습니다: {e}")
            proba = 0.0
        contributions = compute_contributions(model, X_input)
        top_feats = get_top_features(contributions, feature_names)
        generator = load_text_generator()
        explanation = build_explanation(proba, top_feats, generator=generator)
        st.markdown("---")
        st.subheader("예측 결과")
        st.metric(label="우울증 위험도", value=f"{proba*100:.1f}%")
        st.subheader("주요 영향 요인")
        for feat in top_feats:
            arrow = "↑" if feat["direction"] == "increase" else "↓"
            st.write(f"{feat['feature']} : {arrow} (중요도 {feat['contribution']:.4f})")
        st.subheader("맞춤형 설명")
        st.write(explanation)

    # 챗봇 인터페이스
    st.markdown("---")
    st.subheader("챗봇에게 질문하기")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    user_question = st.chat_input("궁금한 내용을 입력하세요.")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
        generator = load_text_generator()
        answer = respond_chat(user_question, generator=generator)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)


if __name__ == "__main__":
    main()
