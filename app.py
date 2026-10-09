"""
app.py
Interactive Streamlit Web Application for Smart Text Intelligence.
Classifies text into Technology, Education, Sports, or Health.
"""

import os
import streamlit as st
import pandas as pd
import classifier_utils as cu

# Page configuration
st.set_page_config(
    page_title="Smart Text Intelligence | Text Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, clean, responsive design
st.markdown(
    """
    <style>
    /* Global font & container improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Header Card */
    .hero-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: #FFFFFF;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
        color: #F8FAFC;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 0;
        line-height: 1.5;
    }

    /* Prediction Result Card */
    .result-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }
    [data-theme="dark"] .result-card {
        background: #1E293B;
        border-color: #334155;
    }

    /* Category Badges */
    .category-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.15rem;
        color: #FFFFFF;
        letter-spacing: 0.01em;
    }
    .badge-Technology { background-color: #2563EB; }
    .badge-Education  { background-color: #D97706; }
    .badge-Sports     { background-color: #059669; }
    .badge-Health     { background-color: #DC2626; }

    /* Category cards on sidebar / details */
    .category-card {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        border-left: 4px solid #CBD5E1;
        background: rgba(148, 163, 184, 0.08);
        font-size: 0.9rem;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: rgba(241, 245, 249, 0.8);
        border: 1px dashed #CBD5E1;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 1.5rem;
    }
    [data-theme="dark"] .disclaimer-box {
        background: rgba(30, 41, 59, 0.6);
        border-color: #475569;
        color: #94A3B8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    """Loads and caches the model pipeline."""
    return cu.load_classifier()


def main():
    # Hero Title Banner
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🧠 Smart Text Intelligence</div>
            <div class="hero-subtitle">
                Automated text classification into <strong>Technology</strong>, <strong>Education</strong>, 
                <strong>Sports</strong>, and <strong>Health</strong> using TF-IDF feature extraction and Logistic Regression.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load model
    pipeline, load_error = get_model()

    # Sidebar: Model info, categories, instructions
    with st.sidebar:
        st.subheader("📌 Supported Categories")
        for cat_name, meta in cu.SUPPORTED_CATEGORIES.items():
            st.markdown(
                f"""
                <div class="category-card" style="border-left-color: {meta['color']};">
                    <strong>{meta['icon']} {cat_name}</strong><br>
                    <span style="color: #64748B; font-size: 0.82rem;">{meta['description']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        st.subheader("⚙️ Classifier Engine")
        st.markdown(
            """
            - **Model**: Logistic Regression (Multinomial L-BFGS)
            - **Vectorization**: TF-IDF (Unigram + Bigram)
            - **Stopwords**: Standard English filtered
            - **Confidence Threshold**: 45% minimum
            - **Hosting**: Local deterministic inference
            """
        )

        st.divider()

        # Model Status indicator
        if pipeline is not None:
            st.success("🟢 Model status: Loaded & Ready", icon="✅")
        else:
            st.error("🔴 Model status: Not Loaded", icon="❌")
            if st.button("🔄 Reload Model"):
                st.cache_resource.clear()
                st.rerun()

    # If model is missing, display friendly guide and halt
    if pipeline is None:
        st.error(
            f"**Model Not Found:** {load_error}\n\n"
            "Please generate the model file first by running the following command in PowerShell:\n"
            "```powershell\npython train_model.py\n```"
        )
        if st.button("Check Again"):
            st.cache_resource.clear()
            st.rerun()
        st.stop()

    # Main Area: Two Columns (Input / Samples and Results)
    col_main, col_info = st.columns([7, 5], gap="large")

    with col_main:
        st.subheader("📝 Enter Text to Classify")

        # Quick sample selectors
        st.markdown("**Or load a sample text:**")
        sample_keys = ["-- Select a sample --"] + list(cu.SAMPLE_INPUTS.keys())
        selected_sample = st.selectbox(
            "Choose a pre-configured sample input:",
            sample_keys,
            label_visibility="collapsed",
            key="sample_selector",
        )

        # Determine initial text value
        if "user_text" not in st.session_state:
            st.session_state["user_text"] = ""

        if selected_sample != "-- Select a sample --":
            initial_value = cu.SAMPLE_INPUTS[selected_sample]
        else:
            initial_value = st.session_state.get("user_text", "")

        user_input = st.text_area(
            "Input text (up to 5,000 characters):",
            value=initial_value,
            height=160,
            placeholder="Type or paste an article excerpt, news snippet, or statement here...",
            key="text_input_area",
        )

        char_count = len(user_input)
        if char_count > cu.MAX_CHAR_LIMIT:
            st.caption(f":red[Character count: {char_count:,} / {cu.MAX_CHAR_LIMIT:,} characters (Limit Exceeded)]")
        else:
            st.caption(f"Character count: {char_count:,} / {cu.MAX_CHAR_LIMIT:,} characters")

        col_btn1, col_btn2 = st.columns([3, 2])
        with col_btn1:
            classify_clicked = st.button("🔍 Classify Text", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🧹 Clear Input", use_container_width=True):
                st.session_state["user_text"] = ""
                st.rerun()

        # Handle prediction when Classify button is clicked
        if classify_clicked:
            st.session_state["has_classified"] = True
            st.session_state["last_input"] = user_input
            prediction_result = cu.predict_text(user_input, pipeline)
            st.session_state["prediction_result"] = prediction_result

    with col_info:
        st.subheader("📊 Classification Results")

        if st.session_state.get("has_classified", False):
            result = st.session_state.get("prediction_result", {})

            if not result.get("success", False):
                # Validation or runtime error
                st.error(f"⚠️ **Validation Notice:** {result.get('error', 'Unknown validation error')}")
            else:
                pred_category = result["predicted_category"]
                confidence = result["confidence"]
                meta = cu.SUPPORTED_CATEGORIES.get(pred_category, {"icon": "🏷️", "color": "#475569"})

                # Prominent Category Display
                badge_class = f"badge-{pred_category}"
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
                            Predicted Category
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.8rem;">
                            <span class="category-badge {badge_class}">
                                {meta['icon']} {pred_category}
                            </span>
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #1E293B;">
                            Confidence Score: <span style="color: {meta['color']};">{confidence:.1%}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Low confidence warning
                if result.get("is_low_confidence", False):
                    st.warning(f"⚠️ {result.get('warning')}")

                # Confidence breakdown progress bars
                st.markdown("**Category Probability Breakdown:**")
                prob_dict = result.get("probabilities", {})
                for cat, prob in prob_dict.items():
                    col_p_label, col_p_val = st.columns([3, 1])
                    with col_p_label:
                        c_icon = cu.SUPPORTED_CATEGORIES.get(cat, {}).get("icon", "")
                        st.write(f"{c_icon} {cat}")
                    with col_p_val:
                        st.write(f"**{prob:.1%}**")
                    st.progress(prob)

                # Technical Explanation
                with st.expander("ℹ️ How was this prediction calculated?"):
                    st.write(result.get("explanation"))
                    st.info(
                        "The classification pipeline transforms text tokens into normalized TF-IDF term weights, "
                        "capturing word importance relative to the corpus. A multinomial Logistic Regression layer "
                        "then applies learned category weights through a softmax function to generate probabilities.",
                        icon="💡",
                    )
        else:
            st.info("👈 Enter text in the box on the left and click **Classify Text** to see predictions and confidence metrics.")

    # Educational Disclaimer Footer
    st.markdown(
        """
        <div class="disclaimer-box">
            ⚖️ <strong>Educational Demonstration Notice:</strong> This application is designed solely for demonstration and academic evaluation.
            Predictions are generated by a lightweight model trained on a localized dataset and should not be used for high-stakes, medical, legal, or mission-critical decision-making.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
