"""
phq9_streamlit_app.py
======================

이 Streamlit 애플리케이션은 영양 섭취 데이터를 바탕으로 우울증 위험도를 추정하고,
모델이 판단한 핵심 영향 요인을 사용자에게 보여줍니다. 또한 간단한
자연어 설명과 챗봇 인터페이스를 제공하여 추가적인 궁금증에 답하도록
설계되었습니다. 기존의 데모 스크립트(`phq9_shap_llm_app.py`)는 커맨드라인
용으로 작성되어 있으며 `shap`와 `transformers` 라이브러리에 의존합니다.
이 앱은 다음과 같은 이유로 경량화와 최적화에 초점을 맞춥니다:

* 외부 네트워크 접속 없이 실행되도록 하기 위해 `shap`과 대형 LLM
  라이브러리에 대한 의존성을 제거하거나 선택적으로 사용합니다.
* Streamlit의 캐싱 기능을 활용하여 모델과 기타 리소스를 한 번만
  로드하도록 하여 반복 실행 시 속도를 향상시킵니다.
* 모델의 특성 중요도를 SHAP 대신 scikit‑learn의 `coef_` 혹은
  `feature_importances_` 속성을 이용해 근사합니다. 이렇게 하면
  의존성을 줄이고 계산을 단순화할 수 있습니다.
* 사용자 입력을 웹 양식으로 받고 결과를 시각적으로 표시하여
  사용성이 높습니다.
* 추가 질문을 입력할 수 있는 챗봇 영역을 제공하지만, 의료적
  조언이 아닌 일반적인 정보만을 제공합니다. 챗봇 응답은 간단한
  규칙 기반 혹은 작은 LLM(사용 가능한 경우)을 사용해 생성됩니다.

주의: 이 앱은 교육적 목적과 자기 이해를 돕기 위한 참고용입니다.
정확한 진단이나 치료를 위해서는 반드시 정신건강 전문가와 상담해야 합니다.
"""

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
        예측 결과는 교육적 용도로만 사용해야 하며, 자신의 정신건강에
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
        lines.append(f"'{feat['feature']}' 섭취가 {kor_direction} 방향으로 우울증 위험에 영향을 미칩니다.")
    lines.append("균형 잡힌 식단과 적절한 영양 섭취는 정신 건강에 긍정적인 영향을 줄 수 있습니다.")
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
    Streamlit 애플리케이션의 메인 함수. 기능:
    1. 모델 로딩 및 캐시
    2. 사용자 입력 폼 표시
    3. 예측과 기여도 계산, 설명 생성
    4. 챗봇 인터페이스 제공
    """
    st.set_page_config(page_title="우울증 위험도 예측", page_icon="🧠", layout="centered")
    st.title("개인 맞춤형 우울증 위험도 예측")
    disclaimer()
    # 모델 로드
    model = load_model()
    if model is None:
        st.stop()

    # 특성 이름: scikit‑learn 모델에서 추출하거나 사용자에게 입력 받음
    if hasattr(model, "feature_names_in_"):
        feature_names: List[str] = [str(f) for f in model.feature_names_in_]
    else:
        feature_input = st.text_input("모델에 사용된 특성 이름을 쉼표로 구분하여 입력하세요.")
        if not feature_input:
            st.info("모델에 특성 이름을 제공하지 않으면 입력 폼을 생성할 수 없습니다.")
            st.stop()
        feature_names = [f.strip() for f in feature_input.split(",") if f.strip()]
        if not feature_names:
            st.error("올바른 특성 이름을 입력하세요.")
            st.stop()

    # 사용자 입력: number_input 으로 구성
    st.subheader("영양소 섭취량을 입력하세요")
    user_values: Dict[str, float] = {}
    cols = st.columns(2)
    for i, name in enumerate(feature_names):
        with cols[i % 2]:
            user_values[name] = st.number_input(name, value=0.0, step=0.1, format="%.2f")

    # 예측 실행 버튼
    if st.button("우울증 위험도 예측하기"):
        X_input = pd.DataFrame([user_values], columns=feature_names)
        # 모델에서 확률 예측
        try:
            proba = model.predict_proba(X_input)[0][1]
        except Exception as e:
            st.error(f"모델 예측 중 오류가 발생했습니다: {e}")
            proba = 0.0
        # 기여도 계산
        contributions = compute_contributions(model, X_input)
        top_feats = get_top_features(contributions, feature_names)
        # LLM 로딩
        generator = load_text_generator()
        # 설명 생성
        explanation = build_explanation(proba, top_feats, generator=generator)
        # 결과 표시
        st.markdown("---")
        st.subheader("예측 결과")
        st.metric(label="우울증 위험도", value=f"{proba*100:.1f}%")
        st.subheader("주요 영향 요인")
        for feat in top_feats:
            arrow = "↑" if feat["direction"] == "increase" else "↓"
            st.write(f"{feat['feature']} : {arrow} (중요도 {feat['contribution']:.4f})")
        st.subheader("맞춤형 설명")
        st.write(explanation)

    # 챗봇 영역
    st.markdown("---")
    st.subheader("챗봇에게 질문하기")
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 이전 대화 표시
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    # 입력 상자
    user_question = st.chat_input("궁금한 내용을 입력하세요.")
    if user_question:
        # 사용자 메시지 저장 및 표시
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
        # 챗봇 응답 생성
        generator = load_text_generator()
        answer = respond_chat(user_question, generator=generator)
        # 응답 저장 및 표시
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)


if __name__ == "__main__":
    main()