"""
phq9_shap_llm_app.py
====================

This module provides a simple demonstration of how to combine a trained
machine‑learning model for predicting depression risk (measured by the
PHQ‑9 questionnaire), SHAP values for feature importance, and a light‑weight
language model to generate personalised explanations.

The script expects a pickled classifier saved to the same directory
(default: ``phq9_nutrition_model.pkl``).  It will prompt the user for
nutritional intake values, predict the probability of depression, compute
SHAP values for the input, extract the most influential features, and then
generate a short explanation.  The explanation is generated either by a
lightweight language model (if ``transformers`` and the specified model
weights are available) or via a simple rule‑based fallback.

Usage example (command line)::

    python phq9_shap_llm_app.py

The script will prompt for each nutrient, display the predicted risk,
show the top features with their SHAP contributions, and finally output
an explanation in Korean.

Notes
-----
* This module is designed to be self contained: it does not depend on any
  external web services.  If the ``transformers`` library is installed
  and the requested model weights are available locally, the script will
  use them; otherwise it falls back to a deterministic explanation.
* For SHAP computations the code constructs a background distribution
  consisting of a single zero vector.  In a production environment you
  should provide a representative background dataset, ideally the same
  training features used to fit your model, to obtain more accurate
  attributions.
* The StandardScaler used during training is not persisted in the pickle
  provided by the user.  Without the scaler, prediction accuracy may
  differ from the original notebook.  You should serialise and load your
  scaler alongside the classifier for production use.
"""

from __future__ import annotations

import json
import joblib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
except ImportError as e:
    raise ImportError("The 'shap' library is required for this script. Please install it via `pip install shap`.") from e

try:
    # We import transformers lazily in generate_explanation
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    _transformers_available = True
except ImportError:
    # transformers is optional; if unavailable the fallback will be used
    _transformers_available = False


@dataclass
class DepressionExplainer:
    """Class to handle prediction, SHAP analysis and LLM explanation."""

    model_path: str = "phq9_nutrition_model.pkl"
    model: object | None = field(default=None, init=False)
    feature_names: List[str] | None = None
    llm_model_name: str = "distilgpt2"
    device: str | int = "cpu"

    def load_model(self) -> None:
        """Load the trained classification model from ``model_path``.

        Raises
        ------
        FileNotFoundError
            If the model file cannot be found.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict_proba(self, X: pd.DataFrame) -> float:
        """Return the probability of the positive class for the provided data.

        Parameters
        ----------
        X : pandas.DataFrame
            A DataFrame with the same features used to train the model.  Each
            row corresponds to a new sample.

        Returns
        -------
        float
            The probability (between 0 and 1) that the sample belongs to the
            positive class (indicative of depression risk).
        """
        if self.model is None:
            self.load_model()
        # Many scikit‑learn classifiers return an array of shape (n_samples, n_classes)
        proba = self.model.predict_proba(X)[0][1]
        return float(proba)

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the given sample using TreeExplainer.

        Parameters
        ----------
        X : pandas.DataFrame
            Input sample(s) for which SHAP values are computed.

        Returns
        -------
        numpy.ndarray
            Array of SHAP values with shape (n_samples, n_features).
        """
        if self.model is None:
            self.load_model()

        # Use a background dataset of zeros.  For robust results, replace this
        # with a representative sample of your training data.
        n_features = X.shape[1]
        background = np.zeros((1, n_features))

        explainer = shap.TreeExplainer(self.model, data=background, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X)
        return np.array(shap_values)

    def get_top_features(self, shap_values: np.ndarray, X: pd.DataFrame, top_n: int = 5) -> List[Dict[str, float]]:
        """Return the top ``top_n`` features by absolute mean SHAP value.

        Parameters
        ----------
        shap_values : numpy.ndarray
            SHAP value array of shape (n_samples, n_features).
        X : pandas.DataFrame
            Input samples used to compute SHAP values; provides column names.
        top_n : int, default=5
            The number of most influential features to return.

        Returns
        -------
        List[Dict[str, float]]
            A list of dictionaries with keys ``feature``, ``shap_value``, and
            ``direction``.  ``direction`` is ``increase`` if the average SHAP
            contribution is positive, otherwise ``decrease``.
        """
        # Compute mean absolute contributions across samples
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(shap_abs_mean)[::-1][:top_n]
        results: List[Dict[str, float]] = []
        for idx in top_indices:
            feature_name = X.columns[idx]
            mean_value = float(shap_abs_mean[idx])
            direction = "increase" if shap_values[:, idx].mean() > 0 else "decrease"
            results.append({
                "feature": feature_name,
                "shap_value": mean_value,
                "direction": direction
            })
        return results

    def build_prompt(self, probability: float, top_features: List[Dict[str, float]]) -> str:
        """Construct a prompt (in Korean) for the language model based on the
        predicted probability and SHAP results.

        Parameters
        ----------
        probability : float
            Predicted probability of depression.
        top_features : List[Dict[str, float]]
            Top features with their SHAP contributions.

        Returns
        -------
        str
            A formatted prompt string in Korean to be passed to the LLM.
        """
        lines = []
        lines.append("당신의 우울증 위험도 예측 결과는 {:.1f}% 입니다.".format(probability * 100))
        lines.append("SHAP 분석을 통해 다음과 같은 주요 요인이 확인되었습니다:")
        for f in top_features:
            arrow = "↑" if f["direction"] == "increase" else "↓"
            lines.append("- {} : 영향 방향 {} (중요도 {:.4f})".format(f["feature"], arrow, f["shap_value"]))
        lines.append("\n위 정보를 바탕으로 건강 전문가의 시각에서 간단하고 이해하기 쉬운 설명을 작성해 주세요. 4-6문장 이내로 작성해 주세요.")
        return "\n".join(lines)

    def generate_explanation(self, prompt: str) -> str:
        """Generate an explanation using a lightweight language model.

        If the transformers library or the specified model cannot be loaded, a
        fallback summary will be returned.

        Parameters
        ----------
        prompt : str
            Prompt to send to the language model.

        Returns
        -------
        str
            Generated text.
        """
        if _transformers_available:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
                model = AutoModelForCausalLM.from_pretrained(self.llm_model_name)
                generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=self.device)
                # Limit length to keep the output concise
                outputs = generator(prompt, max_length=len(prompt.split()) + 60, num_return_sequences=1)
                return outputs[0]["generated_text"]
            except Exception:
                pass  # Fall back to simple explanation

        # Fallback explanation: summarise the probability and directions
        explanation_lines = []
        # Attempt to parse the probability from the prompt
        try:
            prob_str = prompt.split("우울증 위험도 예측 결과는 ")[1].split("%")[0]
            probability = float(prob_str) / 100.0
            explanation_lines.append(f"예측된 우울증 위험도는 {probability*100:.1f}% 입니다.")
        except Exception:
            pass
        # Extract bullet points from the prompt
        for line in prompt.split("\n"):
            if line.startswith("-"):
                parts = line[1:].split(":")
                if len(parts) >= 2:
                    feature = parts[0].strip()
                    direction = "증가" if "↑" in line else "감소"
                    explanation_lines.append(f"{feature} 섭취가 {direction} 방향으로 우울증 위험에 영향을 미칩니다.")
        explanation_lines.append("균형 잡힌 식단과 적절한 영양 섭취는 정신 건강에 도움을 줄 수 있습니다.")
        return "\n".join(explanation_lines)

    def run_interactive(self) -> None:
        """Interactively prompt the user for nutrient values and provide
        predictions, SHAP analysis and LLM explanation.

        The list of expected features is derived from ``self.feature_names``.  If
        no ``feature_names`` are provided, the user will be asked to enter
        comma‑separated feature names before inputting values.
        """
        if self.model is None:
            self.load_model()
        if self.feature_names is None:
            print("모델에 사용된 피처 이름을 입력하세요 (쉼표로 구분):")
            names_str = input().strip()
            self.feature_names = [name.strip() for name in names_str.split(',')]

        user_values = []
        print("각 영양소 섭취량을 입력해주세요:")
        for name in self.feature_names:
            val_str = input(f"  {name}: ")
            try:
                val = float(val_str)
            except ValueError:
                print("숫자를 입력해주세요. 기본값 0 사용.")
                val = 0.0
            user_values.append(val)

        X_input = pd.DataFrame([user_values], columns=self.feature_names)
        prob = self.predict_proba(X_input)
        shap_values = self.compute_shap_values(X_input)
        top_feats = self.get_top_features(shap_values, X_input, top_n=min(5, X_input.shape[1]))
        prompt = self.build_prompt(prob, top_feats)
        explanation = self.generate_explanation(prompt)

        print("\n=== 예측 결과 ===")
        print(f"우울증 위험도: {prob*100:.1f}%")
        print("\n=== 주요 영향 요인 ===")
        for f in top_feats:
            arrow = "↑" if f["direction"] == "increase" else "↓"
            print(f"{f['feature']}: {arrow} (중요도 {f['shap_value']:.4f})")
        print("\n=== 맞춤형 설명 ===")
        print(explanation)


def main() -> None:
    explainer = DepressionExplainer()
    explainer.run_interactive()


if __name__ == "__main__":
    main()
