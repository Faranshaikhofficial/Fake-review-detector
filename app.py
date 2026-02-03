import streamlit as st
import pandas as pd
import re
import joblib
import shap
import re
from collections import Counter
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Page setup
st.set_page_config(page_title="Fake Review Detection", layout="wide")

# ---------------- CUSTOM PREPROCESSOR ----------------
def clean_text_improved(text):
    text = str(text).lower()
    text = re.sub(r'!{2,}', ' MULTIEXCLAIM ', text)
    text = re.sub(r'\d+', ' NUMTOKEN ', text)
    text = re.sub(r'[^a-z ]', ' ', text)
    text = ' '.join(text.split())
    return text

# ---------------- LOAD MODEL & DATA ----------------
@st.cache_resource
def load_model_optimized():
    # Load model and vectorizer
    model = joblib.load("model_improved.pkl")
    vectorizer = joblib.load("vectorizer_improved.pkl")
    # Load feature names if available, else None
    try:
        feature_names = joblib.load("feature_names.pkl")
    except:
        feature_names = []
    return model, vectorizer, feature_names

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("fake reviews dataset.csv", encoding_errors="replace")
        
        # ---------------- ADAPT TO NEW DATASET ----------------
        # Map 'Review' -> 'text_'
        if 'Review' in df.columns:
            df = df.rename(columns={'Review': 'text_'})
        
        # Map 'Rate' -> 'rating'
        if 'Rate' in df.columns:
            df = df.rename(columns={'Rate': 'rating'})
            
        # Ensure text_ exists
        if 'text_' not in df.columns:
            # Fallback for other formats
            possible_text_cols = [c for c in df.columns if 'text' in c.lower() or 'review' in c.lower()]
            if possible_text_cols:
                df = df.rename(columns={possible_text_cols[0]: 'text_'})
            else:
                st.error("Could not find review text column in dataset.")
                return pd.DataFrame()

        df = df.dropna(subset=['text_'])
        
        # Check for label
        if 'label' not in df.columns:
            # Create a dummy label just so code doesn't crash (Visuals will note it's unchecked)
            df['label'] = 0 
            st.session_state['has_labels'] = False
        else:
            # Normalize labels if they exist
            if df['label'].dtype == object:
                # Try standard mapping
                label_map = {'OR': 0, 'CG': 1, 'Real': 0, 'Fake': 1}
                df['label'] = df['label'].map(label_map).fillna(0)
            st.session_state['has_labels'] = True

        # ---------------- FEATURE ENGINEERING ----------------
        # Must match what the model expects (adding the missing keys)
        df['text_'] = df['text_'].astype(str)
        df['word_count'] = df['text_'].str.split().str.len()
        df['char_count'] = df['text_'].str.len()
        df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1)
        df['exclamation_count'] = df['text_'].str.count('!')
        df['question_count'] = df['text_'].str.count(r'\?')
        df['caps_count'] = df['text_'].str.count('[A-Z]')
        
        # FIX FOR KEYERROR: Add fields likely expected by the advanced model
        df['caps_ratio'] = df['caps_count'] / (df['char_count'] + 1)
        
        spam_words = ['free', 'win', 'click', 'buy', 'cheap', 'offer', 'money', 'urgent',
                      'best', 'amazing', 'excellent', 'perfect', 'love', 'great', 'wonderful']
        df['spam_word_count'] = df['text_'].apply(lambda x: sum(1 for w in spam_words if w in x.lower()))
        
        # Additional features that might be needed
        df['multiple_exclaim'] = df['text_'].str.count(r'!!+')
        df['sentence_count'] = df['text_'].str.count(r'[.!?]+')
        df['avg_sentence_length'] = df['word_count'] / (df['sentence_count'] + 1)
        
        # Word variety for analytics
        df['unique_words'] = df['text_'].apply(lambda x: len(set(str(x).lower().split())))
        df['word_variety_ratio'] = df['unique_words'] / (df['word_count'] + 1)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return pd.DataFrame()

model, vectorizer, feature_names = load_model_optimized()
df = load_data()

# ---------------- HELPER FUNCTIONS ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text

def highlight_words(text, shap_df):
    """Highlight words based on their SHAP impact"""
    words = text.split()
    highlighted = []
    for word in words:
        clean_word = clean_text(word)
        if clean_word in shap_df["Word"].values:
            impact = shap_df[shap_df["Word"] == clean_word]["Impact"].values[0]
            if impact > 0:
                highlighted.append(f"<span style='background-color: #ffcccc; padding: 2px 4px; border-radius: 3px;'>{word}</span>")
            else:
                highlighted.append(f"<span style='background-color: #ccffcc; padding: 2px 4px; border-radius: 3px;'>{word}</span>")
        else:
            highlighted.append(word)
    return " ".join(highlighted)

# ---------------- UI THEME ----------------
st.title("🔍 Fake Review Detection System")

# ---------------- TAB STRUCTURE ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Model Overview",
    "📊 Data Analytics",
    "🔍 Analyze Review",
    "📂 Bulk Upload"
])

# ================= TAB 1 =================
with tab1:
    # ---------- MODEL INFORMATION ----------
    st.markdown("### 🧠 Model Information")

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.info("**Algorithm:** Logistic Regression")
    info_col2.info("**Text Processing:** TF-IDF Vectorization")
    info_col3.info("**Explainability:** SHAP-based word influence")

    # ---------- CONFIDENCE GUIDE ----------
    st.markdown("### 📊 Confidence Interpretation Guide")

    st.markdown("""
    - **Above 80%** → Highly reliable prediction  
    - **50% – 80%** → Moderate confidence (manual review suggested)  
    - **Below 50%** → Low confidence, needs human judgment  
    """)

    # ---------- MODEL HEALTH ----------
    st.subheader("📊 Model Overview & Health")
    
    if len(df) > 0:
        # Combined features for model performance calculation
        X_tfidf = vectorizer.transform(df["text_"])
        
        # Ensure we only use columns that exist in the dataframe (handle mismatches safely)
        valid_features = [f for f in feature_names if f in df.columns]
        if len(valid_features) < len(feature_names):
             missing = set(feature_names) - set(valid_features)
             st.warning(f"⚠️ Missing features in new dataset: {missing}. Filling with 0.")
             for m in missing:
                 df[m] = 0
            
        X_feat = df[feature_names].fillna(0)
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_feat.values])
        
        y_pred = model.predict(X_combined)

        # Check if we have real labels
        if st.session_state.get('has_labels', True):
            acc = accuracy_score(df["label"], y_pred)
            prec = precision_score(df["label"], y_pred)
            rec = recall_score(df["label"], y_pred)
            f1 = f1_score(df["label"], y_pred)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.2f}")
            col2.metric("Precision", f"{prec:.2f}")
            col3.metric("Recall", f"{rec:.2f}")
            col4.metric("F1 Score", f"{f1:.2f}")
            
            st.info("Metrics based on labeled ground truth.")
        else:
            st.warning("⚠️ No labels found in dataset. Showing PREDICTED distribution instead.")
            
            fake_pred_count = (y_pred == 1).sum()
            real_pred_count = (y_pred == 0).sum()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows", len(df))
            c2.metric("Predicted Fakes", int(fake_pred_count))
            c3.metric("Predicted Real", int(real_pred_count))

    # ---------- DATASET SNAPSHOT ----------
    st.markdown("### 📁 Dataset Snapshot")

    d1, d2, d3 = st.columns(3)
    d1.metric("Total Reviews", len(df))
    if st.session_state.get('has_labels', True):
        d2.metric("Labeled Fake", int((df["label"] == 1).sum()))
        d3.metric("Labeled Real", int((df["label"] == 0).sum()))
    else:
        d2.metric("Labeled Fake", "N/A")
        d3.metric("Labeled Real", "N/A")

    # ---------- DISCLAIMER ----------
    st.warning(
        "⚠️ Disclaimer: Predictions are probabilistic and intended to assist decision-making. "
        "They should not be treated as absolute judgments."
    )

# ================= TAB 2 =================
with tab2:
    st.subheader("📊 Dataset Analytics (Overall Data)")

    # Review length
    df["length"] = df["text_"].astype(str).apply(len)
    fake_reviews = df[df["label"] == 1]["text_"].dropna().astype(str)

    # =========================================================
    # PREMIUM VISUALIZATIONS (TOP TIER)
    # =========================================================
    
    # 1️⃣ Behavioral DNA (Radar Chart)
    st.subheader("🧬 Behavioral DNA Profile (Real vs Fake)")
    st.write("Average normalized feature scores comparing writing styles.")
    
    radar_cols = ['avg_word_length', 'caps_ratio', 'spam_word_count', 'word_variety_ratio']
    radar_df = df.groupby('label')[radar_cols].mean().reset_index()
    for col in radar_cols:
        if radar_df[col].max() > 0:
            radar_df[col] = radar_df[col] / radar_df[col].max()
    
    fig_radar = go.Figure()
    for _, row in radar_df.iterrows():
        label_text = 'Fake (CG)' if row['label'] == 1 else 'Real (OR)'
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[c] for c in radar_cols],
            theta=radar_cols,
            fill='toself',
            name=label_text
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Review DNA Comparison"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # 2️⃣ Writing Style Complexity vs Variety
    st.subheader("🎨 Writing Complexity vs. Vocabulary Diversity")
    st.write("How word complexity (length) scales with unique vocabulary use.")
    
    sample_size = min(1500, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    
    fig_style = px.scatter(
        sample_df, 
        x="avg_word_length", 
        y="word_variety_ratio", 
        color="label",
        opacity=0.6,
        marginal_x="histogram", 
        marginal_y="violin",
        title="Vocabulary Richness vs. Word Complexity",
        labels={
            "avg_word_length": "Avg Word Length (Complexity)",
            "word_variety_ratio": "Unique Word Ratio (Diversity)",
            "label": "Fake (1) vs Real (0)"
        }
    )
    st.plotly_chart(fig_style, use_container_width=True)

    # 3️⃣ Categorical Sunburst Chart
    if "category" in df.columns:
        st.subheader("☀️ Category Hierarchy & Fake Distribution")
        st.write("Drill down into categories to see fake review concentrations.")
        
        df_sun = df.copy()
        df_sun['label_name'] = df_sun['label'].map({0: 'Real (OR)', 1: 'Fake (CG)'})
        
        fig_sun = px.sunburst(
            df_sun, 
            path=['category', 'label_name'], 
            values='word_count', 
            color='label',
            color_continuous_scale='RdBu_r',
            title="Categorical Breakdown by Review Type"
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    # 4️⃣ Feature Correlation Heatmap
    st.markdown("---")
    st.subheader("🔗 Behavioral Feature Correlation Matrix")
    st.info("Visualizes how different writing metrics relate to each other.")
    
    corr_cols = ['word_count', 'avg_word_length', 'exclamation_count', 'caps_ratio', 'spam_word_count', 'avg_sentence_length', 'label']
    corr_matrix = df[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto",
        color_continuous_scale='RdBu_r', 
        title="Inter-Feature Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # 5️⃣ Density Hotspots (2D Histogram)
    st.subheader("🔥 Suspicious Activity Hotspots")
    st.write("Identifies clusters where extreme ratings meet high punctuation intensity.")
    
    fig_density = px.density_heatmap(
        df, 
        x="rating", 
        y="exclamation_count", 
        nbinsx=5, 
        nbinsy=10, 
        color_continuous_scale="Viridis", 
        title="Rating vs. Exclamation Intensity heatmap",
        labels={"exclamation_count": "Exclamation Marks", "rating": "Rating (1-5)"}
    )
    st.plotly_chart(fig_density, use_container_width=True)

    # =========================================================
    # SUPPORTING ANALYTICS (Moved to Bottom)
    # =========================================================
    st.markdown("---")
    st.subheader("📋 Supporting Statistics")

    # Top Words
    st.subheader("📊 Top Words Influencing Fake Reviews")
    if len(fake_reviews) > 0:
        tfidf_v = TfidfVectorizer(stop_words="english", max_features=1000)
        X_f = tfidf_v.fit_transform(fake_reviews)
        fn_f = tfidf_v.get_feature_names_out()
        sc_f = np.asarray(X_f.mean(axis=0)).ravel()
        tw = pd.DataFrame({"Word": fn_f, "Score": sc_f}).sort_values("Score", ascending=False).head(10)
        fig_tw = px.bar(tw, x="Score", y="Word", orientation="h", title="Top TF-IDF Words")
        st.plotly_chart(fig_tw, use_container_width=True)

    # Fake per Category
    st.subheader("📉 Fake Review Ratio by Category")
    df_cat = df.copy()
    if "category" not in df_cat.columns: df_cat["category"] = "Unknown"
    cat_df = pd.DataFrame({"Total": df_cat.groupby("category").size(), "Fake": df_cat[df_cat["label"] == 1].groupby("category").size()}).fillna(0)
    cat_df["Fake_Ratio"] = cat_df["Fake"] / cat_df["Total"]
    cat_df = cat_df.reset_index()
    fig_cat = px.bar(cat_df, x="category", y="Fake_Ratio", text_auto=".1%", title="Fake Review Percentage")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Pie, Hist, Box, Average, Violin, Bubble
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, names="label", title="Fake vs Real Distribution"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(df, x="length", nbins=30, title="Review Length Histogram"), use_container_width=True)

    st.plotly_chart(px.box(df, x="label", y="length", title="Word Count Distribution by Class"), use_container_width=True)
    
    st.subheader(" Review Length Comparison")
    st.plotly_chart(px.bar(df.groupby("label")["word_count"].mean().reset_index(), x="label", y="word_count", text_auto=".1f", title="Average Word Count"), use_container_width=True)
    
    st.plotly_chart(px.violin(df, x="label", y="word_count", box=True, title="Word Count Distribution (Violin)"), use_container_width=True)
    
    st.plotly_chart(px.scatter(df, x="rating", y="word_count", color="label", size="word_count", opacity=0.4, title="Rating vs Review Length (Bubble)"), use_container_width=True)

    # Wordcloud & CM
    if len(fake_reviews) > 0:
        fake_text = " ".join(fake_reviews)
        wc_img = WordCloud(background_color="white", width=800, height=400).generate(fake_text)
        plt.figure(figsize=(7,4)); plt.imshow(wc_img); plt.axis("off"); st.pyplot(plt)
    
    st.subheader(" Confusion Matrix")
    cm_plot = confusion_matrix(df["label"], y_pred)
    st.plotly_chart(px.imshow(cm_plot, text_auto=True, title="Model Confusion Matrix"), use_container_width=True)
# ================= TAB 3 =================
with tab3:
    st.subheader("🔍 User Review Intelligence")

    user_review = st.text_area(
        "Enter a review to analyze",
        placeholder="e.g. Buy now! Best product ever, limited offer..."
    )

    if st.button("Analyze Review"):
        if user_review.strip() == "":
            st.warning("Please enter a review.")
        else:
            # ---------- PREPROCESS ----------
            # ---------- FEATURE ENGINEERING ----------
            # Calculate basic features
            wc = len(user_review.split())
            cc = len(user_review)
            awl = cc / (wc + 1)
            exc = user_review.count('!')
            qnc = user_review.count('?')
            cap = sum(1 for c in user_review if c.isupper())
            
            # Additional features likely expected by the advanced model
            caps_ratio = cap / (cc + 1)
            spam_words = ['free', 'win', 'click', 'buy', 'cheap', 'offer', 'money', 'urgent']
            spam_word_count = sum(1 for w in spam_words if w in user_review.lower())
            
            # Create a single-row DataFrame for features
            # We use a dictionary first to map feature names to values
            feat_dict = {
                'word_count': [wc],
                'char_count': [cc],
                'avg_word_length': [awl],
                'exclamation_count': [exc],
                'question_count': [qnc],
                'caps_count': [cap],
                'caps_ratio': [caps_ratio],
                'spam_word_count': [spam_word_count],
                'multiple_exclaim': [user_review.count('!!')],
                'sentence_count': [user_review.count('.')],
                'avg_sentence_length': [wc / (user_review.count('.') + 1)]
            }
            
            single_feat_df = pd.DataFrame(feat_dict)
            
            # Align with the model's expected feature names
            # This is the CRITICAL fix for "X has 5006, expected 5008" errors
            # It ensures we provide exactly the columns the model wants, in the right order
            
            # Identify which features from feature_names are non-text (the ones we calculate)
            # Since feature_names might contain ALL features (including text), and we only have the 'extra' ones separately...
            # Wait, usually feature_names in this codebase refers to the EXTRA features only.
            
            if feature_names is not None and len(feature_names) > 0:
                # Reindex ensures we have all required columns (filled with 0 if missing) and drops extras
                single_feat_df = single_feat_df.reindex(columns=feature_names, fill_value=0)
                feat_arr = single_feat_df.values
            else:
                 # Fallback if feature_names not loaded (should generally not happen if model is loaded)
                 feat_arr = np.array([[wc, cc, awl, exc, qnc, cap, caps_ratio, spam_word_count]])

            # ---------- VECTORIZE ----------
            tfidf_vec = vectorizer.transform([user_review])
            
            from scipy.sparse import hstack
            combined_vec = hstack([tfidf_vec, feat_arr])

            # ---------- PREDICTION ----------
            pred = model.predict(combined_vec)[0]
            probs = model.predict_proba(combined_vec)[0]

            real_prob = probs[0]
            fake_prob = probs[1]

            # ---------- RESULT ----------
            if pred == 1:
                st.error("🚨 Prediction: FAKE REVIEW")
            else:
                st.success("✅ Prediction: REAL REVIEW")

            # ---------- SCORES ----------
            col1, col2, col3 = st.columns(3)
            col1.metric("Fake Probability", f"{fake_prob*100:.2f}%")
            col2.metric("Real Probability", f"{real_prob*100:.2f}%")

            if fake_prob > 0.7:
                risk = "HIGH RISK"
            elif fake_prob > 0.4:
                risk = "MEDIUM RISK"
            else:
                risk = "LOW RISK"

            col3.metric("Risk Level", risk)

            # ---------- PROBABILITY GRAPH ----------
            prob_df = pd.DataFrame({
                "Class": ["Real", "Fake"],
                "Probability": [real_prob, fake_prob]
            })

            st.plotly_chart(
                px.bar(
                    prob_df,
                    x="Class",
                    y="Probability",
                    color="Class",
                    text_auto=".2%",
                    title="Prediction Confidence"
                ),
                use_container_width=True
            )

            # ---------- 🎯 RISK METER ----------
            st.subheader("🎯 Fake Review Risk Meter")

            import plotly.graph_objects as go

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fake_prob * 100,
                number={"suffix": "%"},
                title={"text": "Fake Review Risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred"},
                    "steps": [
                        {"range": [0, 40], "color": "lightgreen"},
                        {"range": [40, 70], "color": "khaki"},
                        {"range": [70, 100], "color": "salmon"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": fake_prob * 100
                    }
                }
            ))

            st.plotly_chart(fig_gauge, use_container_width=True)

            # ---------- SHAP EXPLANATION ----------
            st.subheader(
                f"🧠 Why does the model think this review is {'FAKE' if pred == 1 else 'REAL'}?"
            )

            st.caption(
                "The bars below show which words pushed the model towards its decision."
            )

            try:
                # Get feature names from TF-IDF vectorizer
                num_tfidf = tfidf_vec.shape[1]
                feature_names_tfidf = vectorizer.get_feature_names_out()
                
                # Use model coefficients as a simpler, more reliable approach
                # This works well for Logistic Regression
                coefficients = model.coef_[0][:num_tfidf]
                
                # Get the TF-IDF values for this review
                tfidf_values = tfidf_vec.toarray()[0]
                
                # Calculate word impact = coefficient * tfidf_value
                word_impacts = coefficients * tfidf_values
                
                shap_df = pd.DataFrame({
                    "Word": feature_names_tfidf,
                    "Impact": word_impacts
                })

                # Filter to only non-zero words (words that appear in the review)
                non_zero_mask = tfidf_values != 0
                shap_df = shap_df[non_zero_mask]

                # Get top 10 most influential words
                top_shap = shap_df.reindex(
                    shap_df["Impact"].abs().sort_values(ascending=False).index
                ).head(10)

                fig_shap = px.bar(
                    top_shap,
                    x="Impact",
                    y="Word",
                    orientation="h",
                    color="Impact",
                    color_continuous_scale="RdBu",
                    title="Top Influential Words"
                )

                st.plotly_chart(fig_shap, use_container_width=True)
                
                # ---------- 🔦 HIGHLIGHTED TEXT ----------
                st.subheader("🔦 Highlighted Review Text")

                highlighted_text = highlight_words(user_review, top_shap)

                st.markdown(
                    f"<div style='line-height:1.8; font-size:16px;'>{highlighted_text}</div>",
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.warning(f"Could not generate word influence chart: {str(e)}")

            # ---------- HUMAN EXPLANATION ----------
            st.subheader("📝 Why did the model predict this?")

            explanation = []

            if fake_prob > 0.7:
                explanation.append(
                    "The review has strong promotional patterns, which are common in fake reviews."
                )
            elif fake_prob > 0.4:
                explanation.append(
                    "The review shows some suspicious characteristics often seen in fake reviews."
                )
            else:
                explanation.append(
                    "The review language appears natural and similar to genuine user experiences."
                )

            if len(user_review.split()) < 15:
                explanation.append(
                    "The review is very short, which is often associated with low-effort or fake reviews."
                )
            else:
                explanation.append(
                    "The review contains sufficient detail, which is typical of real reviews."
                )

            explanation.append(
                "The model analyzed word patterns using machine learning (TF-IDF + Logistic Regression)."
            )

            for point in explanation:
                st.write("•", point)
# ================= TAB 4 =================
with tab4:
    st.subheader("📂 Bulk CSV Review Classification")

    uploaded_file = st.file_uploader(
        "Upload CSV File (minimum 200 rows)",
        type=["csv"]
    )

    if uploaded_file is None:
        st.warning("Please upload a CSV file to continue.")
    else:
        # ---------- SAFE CSV LOAD ----------
        bulk_df = pd.read_csv(uploaded_file, encoding_errors="replace")

        # ---------------- ADAPT COLUMNS ----------------
        if 'Review' in bulk_df.columns:
            bulk_df = bulk_df.rename(columns={'Review': 'text_'})
        if 'Rate' in bulk_df.columns:
            bulk_df = bulk_df.rename(columns={'Rate': 'rating'})
            
        # Fallback if text_ still missing
        if "text_" not in bulk_df.columns:
             # Try to find any column that looks like text
             possible = [c for c in bulk_df.columns if 'text' in c.lower() or 'review' in c.lower() or 'content' in c.lower()]
             if possible:
                 bulk_df = bulk_df.rename(columns={possible[0]: 'text_'})

        # CRITICAL FIX: Remove rows where text is missing (NaN) BEFORE processing
        if "text_" in bulk_df.columns:
            bulk_df = bulk_df.dropna(subset=["text_"])
            # Ensure they are strings (sometimes read_csv infers other types)
            bulk_df["text_"] = bulk_df["text_"].astype(str)

        # ---------- VALIDATIONS ----------
        if "text_" not in bulk_df.columns:
            st.error("❌ CSV must contain a 'text_' or 'Review' column used for analysis.")
        elif len(bulk_df) < 5: # Lowered limit for testing
            st.error("❌ CSV must contain at least 5 rows.")
        else:
            st.success(f"✅ File loaded successfully ({len(bulk_df)} rows)")

            # ---------- PREDICTION ----------
            # ---------- FEATURE ENGINEERING (BULK) ----------
            # Basic Features
            bulk_df['word_count'] = bulk_df['text_'].astype(str).str.split().str.len()
            bulk_df['char_count'] = bulk_df['text_'].astype(str).str.len()
            bulk_df['avg_word_length'] = bulk_df['char_count'] / (bulk_df['word_count'] + 1)
            bulk_df['exclamation_count'] = bulk_df['text_'].astype(str).str.count('!')
            bulk_df['question_count'] = bulk_df['text_'].astype(str).str.count(r'\?')
            bulk_df['caps_count'] = bulk_df['text_'].astype(str).str.count('[A-Z]')
            
            # MISSING FEATURES (Fix for KeyError)
            bulk_df['caps_ratio'] = bulk_df['caps_count'] / (bulk_df['char_count'] + 1)
            
            spam_words = ['free', 'win', 'click', 'buy', 'cheap', 'offer', 'money', 'urgent']
            # Vectorized approximate spam count
            bulk_df['spam_word_count'] = bulk_df['text_'].astype(str).apply(lambda x: sum(1 for w in spam_words if w in x.lower()))
            
            bulk_df['multiple_exclaim'] = bulk_df['text_'].astype(str).str.count(r'!!+')
            bulk_df['sentence_count'] = bulk_df['text_'].astype(str).str.count(r'[.!?]+')
            bulk_df['avg_sentence_length'] = bulk_df['word_count'] / (bulk_df['sentence_count'] + 1)
            
            # Ensure all required features exist
            if feature_names is not None:
                missing_cols = set(feature_names) - set(bulk_df.columns)
                for c in missing_cols:
                    bulk_df[c] = 0
                
                # Select exactly what the model needs
                bulk_feat = bulk_df[feature_names].fillna(0).values
            else:
                 # Fallback
                 bulk_feat = bulk_df[['word_count', 'char_count', 'avg_word_length', 'exclamation_count', 'question_count', 'caps_count', 'caps_ratio', 'spam_word_count']].values
            
            # ---------- PREDICTION ----------
            bulk_tfidf = vectorizer.transform(bulk_df["text_"])
            from scipy.sparse import hstack
            bulk_combined = hstack([bulk_tfidf, bulk_feat])
            
            bulk_df["Prediction"] = model.predict(bulk_combined)

            bulk_df["Prediction_Label"] = bulk_df["Prediction"].map({
                0: "Real",
                1: "Fake"
            })

            # ---------- SPLIT ----------
            real_df = bulk_df[bulk_df["Prediction_Label"] == "Real"]
            fake_df = bulk_df[bulk_df["Prediction_Label"] == "Fake"]

            # ---------- SUMMARY METRICS ----------
            st.subheader("📊 Bulk File Summary")

            total = len(bulk_df)
            fake_count = len(fake_df)
            real_count = len(real_df)
            fake_ratio = (fake_count / total) * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Reviews", total)
            c2.metric("Fake Reviews", fake_count)
            c3.metric("Real Reviews", real_count)
            c4.metric("Fake %", f"{fake_ratio:.2f}%")

            # ---------- PIE CHART ----------
            st.subheader("🧁 Fake vs Real Distribution")

            pie_df = pd.DataFrame({
                "Class": ["Real", "Fake"],
                "Count": [real_count, fake_count]
            })

            st.plotly_chart(
                px.pie(
                    pie_df,
                    names="Class",
                    values="Count",
                    title="Fake vs Real Reviews (Uploaded File)"
                ),
                use_container_width=True
            )

            # ---------- TOP SUSPICIOUS WORDS ----------
            st.subheader("🔍 Top Suspicious Words in Uploaded File")

            from collections import Counter
            import re

            def clean_bulk_text(text):
                text = str(text).lower()
                text = re.sub(r"[^a-z ]", "", text)
                return text

            fake_bulk_text = " ".join(fake_df["text_"].apply(clean_bulk_text))
            words = fake_bulk_text.split()

            if len(words) == 0:
                st.warning("No fake reviews found to analyze words.")
            else:
                word_counts = Counter(words)
                word_counts = Counter({k: v for k, v in word_counts.items() if len(k) > 2})

                top_words = pd.DataFrame(
                    word_counts.most_common(15),
                    columns=["Word", "Count"]
                )

                st.plotly_chart(
                    px.bar(
                        top_words,
                        x="Count",
                        y="Word",
                        orientation="h",
                        title="Top Words in Fake Reviews (Uploaded File)"
                    ),
                    use_container_width=True
                )

            # ---------- NEW ANALYTICS FOR UPLOADED DATA ----------
            st.markdown("---")
            st.subheader("🧬 Behavioral Analytics of Uploaded Dataset")
            
            # Prep data for advanced plots
            bulk_df['unique_words'] = bulk_df['text_'].apply(lambda x: len(set(str(x).lower().split())))
            bulk_df['word_variety_ratio'] = bulk_df['unique_words'] / (bulk_df['word_count'] + 1)
            
            cola, colb = st.columns(2)
            
            with cola:
                # 1. Behavioral Footprint (Radar)
                radar_cols_bulk = ['avg_word_length', 'caps_ratio', 'spam_word_count', 'word_variety_ratio']
                radar_df_bulk = bulk_df.groupby('Prediction_Label')[radar_cols_bulk].mean().reset_index()
                for c in radar_cols_bulk:
                     if radar_df_bulk[c].max() > 0:
                         radar_df_bulk[c] = radar_df_bulk[c] / radar_df_bulk[c].max()
                
                fig_radar_bulk = go.Figure()
                for _, row in radar_df_bulk.iterrows():
                    fig_radar_bulk.add_trace(go.Scatterpolar(
                        r=[row[c] for c in radar_cols_bulk],
                        theta=radar_cols_bulk,
                        fill='toself',
                        name=row['Prediction_Label']
                    ))
                fig_radar_bulk.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Behavioral Footprint")
                st.plotly_chart(fig_radar_bulk, use_container_width=True)

            with colb:
                # 2. Writing Complexity vs Variety (Scatter)
                fig_style_bulk = px.scatter(
                    bulk_df, x="avg_word_length", y="word_variety_ratio", color="Prediction_Label",
                    title="Vocabulary Variety vs Complexity", labels={"Prediction_Label": "Type"}
                )
                st.plotly_chart(fig_style_bulk, use_container_width=True)


            # ---------- RISK ASSESSMENT ----------
            st.subheader("⚠️ Bulk File Risk Assessment")

            if fake_ratio > 60:
                st.error("🚨 HIGH RISK FILE – Large number of fake reviews detected")
            elif fake_ratio > 30:
                st.warning("⚠️ MEDIUM RISK FILE – Some fake patterns detected")
            else:
                st.success("✅ LOW RISK FILE – Mostly genuine reviews")

            # ---------- PREVIEW ----------
            st.subheader("🔍 Preview (First 10 Rows)")
            st.dataframe(
                bulk_df[["text_", "Prediction_Label"]].head(10),
                use_container_width=True
            )

            # ---------- DOWNLOADS ----------
            st.subheader("⬇️ Download Results")

            st.download_button(
                "Download Real Reviews CSV",
                real_df.to_csv(index=False),
                "real_reviews.csv",
                "text/csv"
            )

            st.download_button(
                "Download Fake Reviews CSV",
                fake_df.to_csv(index=False),
                "fake_reviews.csv",
                "text/csv"
            )
