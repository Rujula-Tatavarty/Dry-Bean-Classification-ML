# DRY BEAN CLASSIFICATION - STREAMLIT APPLICATION
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

# TITLE
st.title("🫘 Dry Bean Classification")

st.write(
    "Machine Learning Classification Analysis "
    "using the Dry Bean Dataset"
)

#finding application folder
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Autofetch saved model files.
all_model_files = [
    file
    for file in os.listdir(BASE_DIR)
    if file.lower().endswith(
        (".pkl", ".joblib")
    )
]

#loading models

models = {}

for filename in all_model_files:

    filename_lower = filename.lower()

    if any(
        keyword in filename_lower
        for keyword in [
            "label_encoder",
            "encoder",
            "scaler",
            "standard_scaler",
            "minmax",
            "preprocessor",
            "transformer"
        ]
    ):

        continue

    model_path = os.path.join(
        BASE_DIR,
        filename
    )

    try:

        loaded_object = joblib.load(
            model_path
        )

    except Exception:

        continue
    object_name = type(
        loaded_object
    ).__name__.lower()

    # Logistic regression

    if (
        "logistic" in filename_lower
        or "logisticregression" in object_name
    ):

        models[
            "Logistic Regression"
        ] = loaded_object

    # Decision tree
    
    elif (
        "decision" in filename_lower
        or "decisiontree" in object_name
    ):

        models[
            "Decision Tree"
        ] = loaded_object

    # K- nearest neighbours 
    
    elif (
        "knn" in filename_lower
        or "kneighbors" in filename_lower
        or "kneighborsclassifier" in object_name
    ):

        models[
            "K-Nearest Neighbors"
        ] = loaded_object

    # Naive bayes
    
    elif (
        "naive" in filename_lower
        or "bayes" in filename_lower
        or "gaussiannb" in object_name
        or "multinomialnb" in object_name
        or "bernoullinb" in object_name
    ):

        models[
            "Naive Bayes"
        ] = loaded_object

    #Random forest
    elif (
        "random" in filename_lower
        or "forest" in filename_lower
        or "randomforestclassifier" in object_name
    ):

        models[
            "Random Forest"
        ] = loaded_object

#model detection
if len(models) == 0:

    st.warning(
        "⚠️ No trained classification models were detected "
        "in the application folder."
    )

    st.info(
        "Please make sure your saved model files "
        "(.pkl or .joblib) are in the same folder "
        "as dry_bean_streamlit_app.py."
    )

#Loading label encoder

label_encoder = None

encoder_candidates = [
    "label_encoder.pkl",
    "labelencoder.pkl",
    "label_encoder.joblib",
    "labelencoder.joblib"
]

for encoder_file in encoder_candidates:

    encoder_path = os.path.join(
        BASE_DIR,
        encoder_file
    )

    if os.path.exists(
        encoder_path
    ):

        try:

            label_encoder = joblib.load(
                encoder_path
            )

            break

        except Exception:

            pass

if label_encoder is None:

    for filename in all_model_files:

        filename_lower = filename.lower()

        if (
            "encoder" in filename_lower
            and "label" in filename_lower
        ):

            encoder_path = os.path.join(
                BASE_DIR,
                filename
            )

            try:

                label_encoder = joblib.load(
                    encoder_path
                )

                break

            except Exception:

                pass

#Label encoder status
if label_encoder is None:

    st.warning(
        "⚠️ Label encoder not found. "
        "The application will require label_encoder.pkl "
        "for the evaluation."
    )

#Sidebar
st.sidebar.title(
    "⚙️ Model Selection"
)

st.sidebar.caption(
    "Dry Bean Classification"
)

#Model selection
with st.sidebar.expander("Select Classification Model", expanded=True):

    st.caption(
        "Choose a trained model to perform the classification analysis."
    )

    if len(models) > 0:

        model_selection = st.selectbox(
            "🔽 Classification Model",
            list(models.keys()),
            key="model_selection"
        )

        st.success(
            f"✓ Selected: **{model_selection}**"
        )

    else:

        model_selection = None

        st.error(
            " No trained models were loaded."
        )

# Side bar divider
st.sidebar.markdown("---")

#Application overview
with st.sidebar.expander("📌 Application Overview", expanded=True):

    st.info(
        "This application performs machine learning "
        "classification analysis on the Dry Bean Dataset."
    )
#Available model

with st.sidebar.expander("🤖 Available Models", expanded=False):

    if len(models) > 0:

        for model_name in models.keys():

            if model_name == model_selection:

                st.success(
                    f"✓ {model_name}"
                )

            else:

                st.write(
                    f"• {model_name}"
                )

    else:

        st.warning(
            "No trained models available."
        )

#Database info

with st.sidebar.expander("📊 Dataset Information", expanded=False):

    st.write(
        "🫘 **Dataset:** Dry Bean Dataset"
    )

    st.write(
        "🔢 **Task:** Multiclass Classification"
    )

    st.write(
        "📋 **Features:** 16 numerical features"
    )

    st.write(
        "🏷️ **Target:** Bean Class"
    )

#How to use
with st.sidebar.expander("💡 How to Use", expanded=False):

    st.write(
        "1️⃣ Select a classification model."
    )

    st.write(
        "2️⃣ Upload your test dataset."
    )

    st.write(
        "3️⃣ Review the dataset information."
    )

    st.write(
        "4️⃣ Click **RUN ANALYSIS**."
    )

    st.write(
        "5️⃣ Review the evaluation results."
    )
#evalvation metrics
with st.sidebar.expander("📈 Evaluation Metrics", expanded=False):

    st.write(
        "✓ Accuracy"
    )

    st.write(
        "✓ AUC"
    )

    st.write(
        "✓ Precision"
    )

    st.write(
        "✓ Recall"
    )

    st.write(
        "✓ F1 Score"
    )

    st.write(
        "✓ Matthews Correlation Coefficient"
    )

#Analysis features

with st.sidebar.expander("🔬 Analysis Features", expanded=False):

    st.write(
        "Test Dataset Upload"
    )

    st.write(
        "Dataset Preview"
    )

    st.write(
        "Column Information"
    )

    st.write(
        "Missing Value Analysis"
    )

    st.write(
        "Confusion Matrix"
    )

    st.write(
        "Classification Report"
    )

    st.write(
        "Class Predictions"
    )
#Side-bar footer

st.sidebar.markdown("---")

st.sidebar.caption(
    "🫘 Dry Bean Classification"
)

st.sidebar.caption(
    "Machine Learning Classification Analysis"
)
#model loading status
with st.expander(
    "🤖 Model Loading Status",
    expanded=False
):

    if len(models) > 0:

        st.success(
            f"✅ {len(models)} trained model(s) loaded."
        )

        for model_name in models:

            st.write(
                f"✅ {model_name}"
            )

    else:

        st.error(
            "❌ No trained models were found."
        )

        st.info(
            f"Application folder:\n\n{BASE_DIR}"
        )
#uploading test data
st.markdown("---")

st.markdown(
    "## Upload Test Dataset"
)

st.write(
    "Upload a CSV or Excel test dataset to begin "
    "the classification analysis."
)


uploaded_file = st.file_uploader(
    "Upload a CSV or Excel test dataset",
    type=["csv", "xls", "xlsx"],
    help="Supported formats: CSV, XLS and XLSX"
)

#Processing loaded dataset
if uploaded_file is not None:

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):

            data = pd.read_csv(
                uploaded_file
            )

            detected_format = "CSV"

        elif file_name.endswith(".xls"):

            try:

                data = pd.read_excel(
                    uploaded_file,
                    engine="xlrd"
                )

                detected_format = "Excel XLS"

            except Exception:

                uploaded_file.seek(0)

                data = pd.read_csv(
                    uploaded_file
                )

                detected_format = (
                    "CSV data with .xls extension"
                )

        elif file_name.endswith(".xlsx"):

            try:

                data = pd.read_excel(
                    uploaded_file,
                    engine="openpyxl"
                )

                detected_format = "Excel XLSX"

            except Exception:

                uploaded_file.seek(0)

                data = pd.read_csv(
                    uploaded_file
                )

                detected_format = (
                    "CSV data with .xlsx extension"
                )

        else:

            st.error(
                "Unsupported file format."
            )

            st.stop()


        st.success(
            f" Dataset uploaded successfully! "
            f"{data.shape[0]:,} rows × "
            f"{data.shape[1]} columns"
        )

        st.info(
            f" **File:** {uploaded_file.name}\n\n"
            f" **Detected format:** {detected_format}\n\n"
            f" **Rows:** {data.shape[0]:,}\n\n"
            f" **Columns:** {data.shape[1]}"
        )

        with st.expander(
            "Preview Uploaded Data",
            expanded=True
        ):

            st.dataframe(
                data.head(10),
                use_container_width=True
            )

        with st.expander(
            "View Dataset Columns"
        ):

            st.write(
                list(data.columns)
            )

        with st.expander(
            "Missing Value Analysis"
        ):

            missing = data.isnull().sum()

            missing_df = pd.DataFrame({
                "Column": missing.index,
                "Missing Values": missing.values
            })

            st.dataframe(
                missing_df,
                use_container_width=True
            )

        st.markdown("---")

        st.markdown(
            "### Ready to analyse your test data?"
        )

        st.write(
            "Select a classification model from the "
            "sidebar and click the button below."
        )
#Run analysis button

        run_analysis = st.button(
            "🚀 RUN ANALYSIS",
            type="primary",
            use_container_width=True
        )

#analysis
        if run_analysis:

            if len(models) == 0:

                st.error(
                    "❌ No trained models were found."
                )

                st.info(
                    "Make sure the five trained model "
                    "files are in the same folder as "
                    "this Streamlit file."
                )

                st.stop()


            if label_encoder is None:

                st.error(
                    "❌ label_encoder.pkl was not found."
                )

                st.info(
                    "Please place the label encoder file "
                    "in the same folder as the Streamlit app."
                )

                st.stop()


            model = models[
                model_selection
            ]

            target_column = None

            possible_targets = [
                "Class",
                "class",
                "CLASS"
            ]


            for column in possible_targets:

                if column in data.columns:

                    target_column = column

                    break


            if target_column is None:

                st.error(
                    "The uploaded dataset does not "
                    "contain a 'Class' column."
                )

                st.info(
                    "Please upload the Dry Bean test "
                    "dataset containing the target column "
                    "'Class'."
                )

                st.stop()

            X_input = data.drop(
                columns=[target_column]
            )

            y_actual = data[
                target_column
            ]

            X_input = X_input.select_dtypes(
                include=["number"]
            )

            if X_input.shape[1] == 0:

                st.error(
                    "❌ No numeric feature columns "
                    "were found."
                )

                st.stop()

            if hasattr(
                model,
                "n_features_in_"
            ):

                expected_features = (
                    model.n_features_in_
                )

                actual_features = (
                    X_input.shape[1]
                )


                if (
                    actual_features
                    != expected_features
                ):

                    st.error(
                        "Feature mismatch!"
                    )

                    st.write(
                        f"The selected model expects "
                        f"**{expected_features}** features."
                    )

                    st.write(
                        f"The uploaded dataset contains "
                        f"**{actual_features}** numeric features."
                    )

                    st.info(
                        "Please make sure you are using "
                        "the same feature set used during "
                        "model training."
                    )

                    st.stop()

            try:

                y_encoded = (
                    label_encoder.transform(
                        y_actual
                    )
                )

            except Exception:

                st.error(
                    "The target classes in the uploaded "
                    "dataset do not match the classes used "
                    "during model training."
                )

                st.write(
                    "**Expected classes:**"
                )

                st.write(
                    list(
                        label_encoder.classes_
                    )
                )

                st.write(
                    "**Uploaded classes:**"
                )

                st.write(
                    sorted(
                        y_actual.unique()
                    )
                )

                st.stop()

            try:

                predictions = model.predict(
                    X_input
                )

            except Exception as e:

                st.error(
                    "❌ The selected model could not "
                    "process the uploaded data."
                )

                st.exception(e)

                st.stop()

            accuracy = accuracy_score(
                y_encoded,
                predictions
            )


            precision = precision_score(
                y_encoded,
                predictions,
                average="weighted",
                zero_division=0
            )


            recall = recall_score(
                y_encoded,
                predictions,
                average="weighted",
                zero_division=0
            )


            f1 = f1_score(
                y_encoded,
                predictions,
                average="weighted",
                zero_division=0
            )


            mcc = matthews_corrcoef(
                y_encoded,
                predictions
            )

            auc = None


            try:

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            X_input
                        )
                    )

                    auc = roc_auc_score(
                        y_encoded,
                        probabilities,
                        multi_class="ovr",
                        average="weighted"
                    )


                elif hasattr(
                    model,
                    "decision_function"
                ):

                    decision_scores = (
                        model.decision_function(
                            X_input
                        )
                    )

                    auc = roc_auc_score(
                        y_encoded,
                        decision_scores,
                        multi_class="ovr",
                        average="weighted"
                    )


            except Exception:

                auc = None

            st.markdown("---")

            st.markdown(
                "## Model Evaluation Results"
            )

            st.markdown(
                f"### {model_selection}"
            )

            metric1, metric2, metric3 = (
                st.columns(3)
            )


            with metric1:

                st.metric(
                    "Accuracy",
                    f"{accuracy:.4f}"
                )


            with metric2:

                if auc is not None:

                    st.metric(
                        "AUC",
                        f"{auc:.4f}"
                    )

                else:

                    st.metric(
                        "AUC",
                        "N/A"
                    )


            with metric3:

                st.metric(
                    "Precision",
                    f"{precision:.4f}"
                )


            metric4, metric5, metric6 = (
                st.columns(3)
            )


            with metric4:

                st.metric(
                    "Recall",
                    f"{recall:.4f}"
                )


            with metric5:

                st.metric(
                    "F1 Score",
                    f"{f1:.4f}"
                )


            with metric6:

                st.metric(
                    "MCC",
                    f"{mcc:.4f}"
                )

            st.markdown("---")

            st.markdown(
                "## 📈 Confusion Matrix"
            )


            cm = confusion_matrix(
                y_encoded,
                predictions
            )


            fig, ax = plt.subplots(
                figsize=(9, 7)
            )


            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="RdPu",
                xticklabels=(
                    label_encoder.classes_
                ),
                yticklabels=(
                    label_encoder.classes_
                ),
                ax=ax
            )


            ax.set_title(
                f"Confusion Matrix - "
                f"{model_selection}"
            )


            ax.set_xlabel(
                "Predicted Class"
            )


            ax.set_ylabel(
                "Actual Class"
            )


            plt.xticks(
                rotation=45
            )


            plt.yticks(
                rotation=0
            )


            plt.tight_layout()


            st.pyplot(
                fig,
                use_container_width=True
            )


            plt.close(fig)

            st.markdown("---")

            st.markdown(
                "## 📋 Classification Report"
            )


            report = classification_report(
                y_encoded,
                predictions,
                labels=np.arange(
                    len(
                        label_encoder.classes_
                    )
                ),
                target_names=(
                    label_encoder.classes_
                ),
                output_dict=True,
                zero_division=0
            )


            report_df = (
                pd.DataFrame(
                    report
                ).transpose()
            )


            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )

            st.markdown("---")

            st.markdown(
                "## 🔮 Test Data Predictions"
            )


            predicted_labels = (
                label_encoder.inverse_transform(
                    predictions
                )
            )


            results_display = data.copy()


            results_display[
                "Predicted Class"
            ] = predicted_labels


            st.dataframe(
                results_display.head(20),
                use_container_width=True
            )

            csv_output = (
                results_display.to_csv(
                    index=False
                )
            )


            st.download_button(
                label=(
                    "Download Predictions as CSV"
                ),
                data=csv_output,
                file_name=(
                    "dry_bean_predictions.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

            st.success(
                f"Analysis completed successfully "
                f"using {model_selection}!"
            )

    except Exception as e:

        st.error(
            "The uploaded file could not be read. "
            "Please check that it is a valid CSV, XLS or XLSX file."
        )

        st.exception(e)
