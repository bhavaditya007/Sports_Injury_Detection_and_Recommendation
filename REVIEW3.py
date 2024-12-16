import streamlit as st  # type: ignore

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #000000; /* Black background */
        color: #ffffff; /* White text */
        font-family: 'Arial', sans-serif;
    }
    .stTextInput>label, .stNumberInput>label {
        color: #ffffff; /* White text for labels */
    }
    .stButton>button {
        background-color: #ff5733; /* Vibrant orange button */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #c13e1d; /* Darker orange on hover */
    }
    h3, h1, h2 {
        color: #ffffff; /* White headings */
    }
    footer {
        color: #ffffff; /* White footer text */
        text-align: center;
        font-size: 18px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to predict likelihood of injury and rest days
def predict_injury(BMI, height, weight, recovery_period):
    """
    Predicts the likelihood of injury and recommended rest days.

    Args:
        BMI (float): Body Mass Index of the individual.
        height (float): Height of the individual in cm.
        weight (float): Weight of the individual in kg.
        recovery_period (int): Number of days for recovery input by the user.

    Returns:
        tuple: Likelihood of injury (float) and recommended rest days (int).
    """
    # Example logic (replace with your actual model or formula)
    likelihood_of_injury = min(100, max(0, (BMI * 2 + height * 0.1 - weight * 0.05)))
    rest_days = int(max(1, recovery_period * 1.5))
    return likelihood_of_injury, rest_days

# Streamlit App
st.title("Injury Prediction App")

st.sidebar.header("Input Parameters")

try:
    # Input conversion with validation
    height = float(st.sidebar.text_input("Height (in cm)", value="170"))
    weight = float(st.sidebar.text_input("Weight (in kg)", value="70"))
    recovery_period = int(st.sidebar.text_input("Recovery Period (in days)", value="30"))

    # Validate inputs
    if height <= 0 or weight <= 0:
        st.error("Height and weight must be positive values.")
    else:
        # Calculate BMI safely
        BMI = weight / ((height / 100) ** 2)

        # Predict the outputs
        likelihood_of_injury, rest_days = predict_injury(BMI, height, weight, recovery_period)

        # Display the results
        st.subheader("Prediction Results")
        st.markdown(f"<h3 style='color:#ffffff;'>Your BMI: {BMI:.2f}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#ffffff;'>Likelihood of Injury: {likelihood_of_injury:.2f}%</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#ffffff;'>Recommended Rest Days: {rest_days} days</h3>", unsafe_allow_html=True)

        # Footer Message
        st.markdown("<footer><b>GET WELL SOON!</b></footer>", unsafe_allow_html=True)

except ValueError:
    st.error("Please enter valid numeric values for height, weight, and recovery period.")
