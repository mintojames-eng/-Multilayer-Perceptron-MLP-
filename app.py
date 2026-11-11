import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# =========================================================
# DATASET: BREAST CANCER WISCONSIN DIAGNOSTIC
# =========================================================

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names
target_names = data.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

st.subheader("Dataset: Breast Cancer Wisconsin Diagnostic Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Instances", X.shape[0])

with col2:
    st.metric("Features", X.shape[1])

with col3:
    st.metric("Classes", len(np.unique(y)))

st.write("### Target Classes")
st.write({
    0: target_names[0],
    1: target_names[1]
})

st.write("### Dataset Preview")
st.dataframe(df.head())

st.write("### Dataset Shape")
st.write(df.shape)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MLP - Artificial Neural Network",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Multilayer Perceptron (MLP) Using Artificial Neural Networks")

st.markdown("""
### Lab Exercise 6

**Objective:** Design, implement, train and evaluate a Multilayer
Perceptron using a real-world dataset and analyse its architecture
and trainable parameters.
""")

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    data = load_breast_cancer()

    X = data.data
    y = data.target

    df = pd.DataFrame(
        X,
        columns=data.feature_names
    )

    df["target"] = y

    return data, df


data, df = load_dataset()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ MLP Configuration")

hidden1 = st.sidebar.slider(
    "Hidden Layer 1 Neurons",
    min_value=8,
    max_value=128,
    value=64,
    step=8
)

hidden2 = st.sidebar.slider(
    "Hidden Layer 2 Neurons",
    min_value=8,
    max_value=64,
    value=32,
    step=8
)

activation = st.sidebar.selectbox(
    "Hidden Layer Activation",
    ["relu", "tanh"]
)

epochs = st.sidebar.slider(
    "Epochs",
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [16, 32, 64]
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

train_button = st.sidebar.button(
    "🚀 Train MLP Model"
)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset",
    "⚙️ Preprocessing",
    "🧠 Architecture",
    "📈 Training",
    "📋 Evaluation",
    "🔬 Experiments"
])

# ============================================================
# TAB 1 - DATASET
# ============================================================

with tab1:

    st.header("1. Dataset Selection and Analysis")

    st.success(
        "Dataset: Breast Cancer Wisconsin Diagnostic Dataset"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Instances",
            X.shape[0]
        )

    with col2:
        st.metric(
            "Features",
            X.shape[1]
        )

    with col3:
        st.metric(
            "Classes",
            len(np.unique(y))
        )

    with col4:
        st.metric(
            "Problem Type",
            "Binary Classification"
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Input Features")

    st.write(
        list(data.feature_names)
    )

    st.subheader("Target Classes")

    target_df = pd.DataFrame({
        "Class": data.target_names,
        "Encoded Value": [0, 1]
    })

    st.table(target_df)

    st.subheader("Class Distribution")

    class_counts = pd.Series(y).value_counts().sort_index()

    fig, ax = plt.subplots()

    ax.bar(
        data.target_names,
        class_counts.values
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Target Class Distribution")

    st.pyplot(fig)

# ============================================================
# TAB 2 - PREPROCESSING
# ============================================================

with tab2:

    st.header("2. Data Preprocessing")

    st.subheader("Missing Value Check")

    missing_values = np.isnan(X).sum()

    if missing_values == 0:

        st.success(
            "No missing values found in the dataset."
        )

    else:

        st.warning(
            f"Missing values detected: {missing_values}"
        )

    st.subheader("Train-Test Split")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Training Samples",
            X_train_raw.shape[0]
        )

    with col2:
        st.metric(
            "Testing Samples",
            X_test_raw.shape[0]
        )

    st.subheader("Feature Scaling")

    scaler_demo = StandardScaler()

    X_train_scaled_demo = scaler_demo.fit_transform(
        X_train_raw
    )

    X_test_scaled_demo = scaler_demo.transform(
        X_test_raw
    )

    st.code("""
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
""")

    st.info(
        "The scaler is fitted only on training data to prevent data leakage."
    )

# ============================================================
# TAB 3 - ARCHITECTURE
# ============================================================

with tab3:

    st.header("3. MLP Model Architecture")

    st.markdown("### Selected Architecture")

    architecture_df = pd.DataFrame({
        "Layer": [
            "Input Layer",
            "Hidden Layer 1",
            "Hidden Layer 2",
            "Output Layer"
        ],
        "Neurons": [
            30,
            hidden1,
            hidden2,
            1
        ],
        "Activation": [
            "None",
            activation,
            activation,
            "Sigmoid"
        ]
    })

    st.table(architecture_df)

    st.subheader("Network Structure")

    st.markdown(
        f"""
        ```
        Input Layer
        30 neurons
             ↓
        Hidden Layer 1
        {hidden1} neurons
        {activation.upper()}
             ↓
        Hidden Layer 2
        {hidden2} neurons
        {activation.upper()}
             ↓
        Output Layer
        1 neuron
        SIGMOID
             ↓
        Malignant / Benign
        ```
        """
    )

    st.subheader("Activation Function Justification")

    if activation == "relu":

        st.info("""
        **ReLU (Rectified Linear Unit)** is used in the hidden layers
        because it introduces non-linearity and is computationally
        efficient for neural-network training.

        ReLU(x) = max(0, x)
        """)

    else:

        st.info("""
        **Tanh** is used in the hidden layers because it introduces
        non-linearity and produces values between -1 and +1.

        tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        """)

    st.info("""
    **Sigmoid** is used in the output layer because this is a binary
    classification problem. It produces an output between 0 and 1.
    """)

    # --------------------------------------------------------
    # PARAMETER CALCULATION
    # --------------------------------------------------------

    st.subheader("Trainable Parameter Calculation")

    param1 = (30 * hidden1) + hidden1
    param2 = (hidden1 * hidden2) + hidden2
    param3 = (hidden2 * 1) + 1

    total_params = param1 + param2 + param3

    parameter_df = pd.DataFrame({
        "Layer": [
            "Input → Hidden 1",
            "Hidden 1 → Hidden 2",
            "Hidden 2 → Output"
        ],
        "Calculation": [
            f"(30 × {hidden1}) + {hidden1}",
            f"({hidden1} × {hidden2}) + {hidden2}",
            f"({hidden2} × 1) + 1"
        ],
        "Parameters": [
            param1,
            param2,
            param3
        ]
    })

    st.table(parameter_df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Parameters",
            total_params
        )

    with col2:
        st.metric(
            "Trainable Parameters",
            total_params
        )

    with col3:
        st.metric(
            "Non-Trainable Parameters",
            0
        )

    st.success(
        f"""
        Total Trainable Parameters = {param1} + {param2} + {param3}
        = **{total_params}**
        """
    )

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    hidden1,
    hidden2,
    activation,
    epochs,
    batch_size,
    test_size
):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = Sequential([
        Dense(
            hidden1,
            activation=activation,
            input_shape=(30,)
        ),

        Dense(
            hidden2,
            activation=activation
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Prediction
    probability = model.predict(
        X_test,
        verbose=0
    )

    y_pred = (
        probability >= 0.5
    ).astype(int).flatten()

    return (
        model,
        history,
        X_test,
        y_test,
        y_pred,
        scaler
    )

# ============================================================
# TAB 4 - TRAINING
# ============================================================

with tab4:

    st.header("4. Model Compilation and Training")

    st.subheader("Model Configuration")

    st.code("""
Optimizer: Adam
Loss Function: Binary Crossentropy
Output Activation: Sigmoid
Hidden Activation: ReLU / Tanh
""")

    if train_button:

        with st.spinner("Training MLP model..."):

            (
                model,
                history,
                X_test_final,
                y_test_final,
                y_pred_final,
                scaler_final
            ) = train_model(
                hidden1,
                hidden2,
                activation,
                epochs,
                batch_size,
                test_size
            )

            st.session_state["model"] = model
            st.session_state["history"] = history
            st.session_state["X_test"] = X_test_final
            st.session_state["y_test"] = y_test_final
            st.session_state["y_pred"] = y_pred_final

        st.success("Model training completed successfully!")

        st.subheader("Model Summary")

        # Capture model summary
        summary_lines = []

        model.summary(
            print_fn=lambda x: summary_lines.append(x)
        )

        st.code(
            "\n".join(summary_lines)
        )

        st.subheader("Training Accuracy")

        fig, ax = plt.subplots()

        ax.plot(
            history.history["accuracy"],
            label="Training Accuracy"
        )

        ax.plot(
            history.history["val_accuracy"],
            label="Validation Accuracy"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training vs Validation Accuracy")
        ax.legend()

        st.pyplot(fig)

        st.subheader("Training Loss")

        fig, ax = plt.subplots()

        ax.plot(
            history.history["loss"],
            label="Training Loss"
        )

        ax.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training vs Validation Loss")
        ax.legend()

        st.pyplot(fig)

    else:

        st.info(
            "Configure the MLP using the sidebar and click "
            "**Train MLP Model**."
        )

# ============================================================
# TAB 5 - EVALUATION
# ============================================================

with tab5:

    st.header("5. Model Evaluation")

    if "model" not in st.session_state:

        st.warning(
            "Please train the model first."
        )

    else:

        y_test_eval = st.session_state["y_test"]
        y_pred_eval = st.session_state["y_pred"]
        model_eval = st.session_state["model"]

        accuracy = accuracy_score(
            y_test_eval,
            y_pred_eval
        )

        precision = precision_score(
            y_test_eval,
            y_pred_eval
        )

        recall = recall_score(
            y_test_eval,
            y_pred_eval
        )

        f1 = f1_score(
            y_test_eval,
            y_pred_eval
        )

        st.subheader("Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

        with col2:
            st.metric(
                "Precision",
                f"{precision:.4f}"
            )

        with col3:
            st.metric(
                "Recall",
                f"{recall:.4f}"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test_eval,
            y_pred_eval
        )

        fig, ax = plt.subplots()

        im = ax.imshow(cm)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            data.target_names
        )

        ax.set_yticklabels(
            data.target_names
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_title("MLP Confusion Matrix")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test_eval,
            y_pred_eval,
            target_names=data.target_names,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.subheader("Performance Interpretation")

        if accuracy >= 0.90:

            st.success(
                "The MLP achieved strong classification performance "
                "on the test dataset."
            )

        elif accuracy >= 0.75:

            st.info(
                "The MLP achieved moderate classification performance."
            )

        else:

            st.warning(
                "The model performance is relatively low and may "
                "require architecture or preprocessing improvements."
            )

# ============================================================
# TAB 6 - EXPERIMENTS
# ============================================================

with tab6:

    st.header("6. Architecture and Activation Experiments")

    st.write("""
    This section compares different MLP architectures to observe
    their effect on classification performance.
    """)

    run_experiments = st.button(
        "🔬 Run Architecture Experiments"
    )

    if run_experiments:

        experiments = [
            ("Small", 32, 16),
            ("Medium", 64, 32),
            ("Large", 128, 64)
        ]

        results = []

        progress = st.progress(0)

        for index, (
            name,
            h1,
            h2
        ) in enumerate(experiments):

            (
                exp_model,
                exp_history,
                exp_X_test,
                exp_y_test,
                exp_y_pred,
                exp_scaler
            ) = train_model(
                h1,
                h2,
                activation,
                epochs,
                batch_size,
                test_size
            )

            acc = accuracy_score(
                exp_y_test,
                exp_y_pred
            )

            prec = precision_score(
                exp_y_test,
                exp_y_pred
            )

            rec = recall_score(
                exp_y_test,
                exp_y_pred
            )

            f1_exp = f1_score(
                exp_y_test,
                exp_y_pred
            )

            params = exp_model.count_params()

            results.append({
                "Architecture": name,
                "Hidden Layer 1": h1,
                "Hidden Layer 2": h2,
                "Parameters": params,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1_exp
            })

            progress.progress(
                (index + 1) / len(experiments)
            )

        results_df = pd.DataFrame(results)

        st.subheader("Experiment Results")

        st.dataframe(
            results_df,
            use_container_width=True
        )

        st.subheader("Accuracy Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            results_df["Architecture"],
            results_df["Accuracy"]
        )

        ax.set_xlabel("Architecture")
        ax.set_ylabel("Accuracy")
        ax.set_title("Architecture vs Accuracy")

        st.pyplot(fig)

        st.subheader("Parameter Comparison")

        fig, ax = plt.subplots()

        ax.bar(
            results_df["Architecture"],
            results_df["Parameters"]
        )

        ax.set_xlabel("Architecture")
        ax.set_ylabel("Number of Parameters")
        ax.set_title("Architecture vs Trainable Parameters")

        st.pyplot(fig)

        st.success(
            "Experiment completed. Compare the performance and "
            "number of parameters of the different architectures."
        )
