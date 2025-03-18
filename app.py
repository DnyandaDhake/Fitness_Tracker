import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("assets/style.css")


@st.cache_data
def load_data():
    data = pd.read_csv("data/exercise_data.csv")
    return data



exercise_data = load_data()


def train_model(data):
    expected_columns = ['Weight (kg)', 'Height (m)', 'Avg_BPM', 'Session_Duration (hours)', 'Calories_Burned',
                        'Workout_Type']
    if not all(col in data.columns for col in expected_columns):
        st.error(
            f"Dataset columns do not match expected columns. Expected: {expected_columns}, Found: {data.columns.tolist()}")
        return None


    workout_type_map = {
        "Cardio": 0,
        "Strength": 1,
        "HIIT": 2,
        "Yoga": 3,
        "Pilates": 4,
        "Cycling": 5,
        "Swimming": 6,
        "Running": 7,
        "Dancing": 8,
        "CrossFit": 9
    }
    data['Workout_Type'] = data['Workout_Type'].map(workout_type_map)

    X = data[['Weight (kg)', 'Height (m)', 'Avg_BPM', 'Session_Duration (hours)', 'Workout_Type']]
    y = data['Calories_Burned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model


def generate_suggestions(user_weight, user_age, user_health_conditions, user_body_temp, user_gender):
    suggestions = {
        "exercise": "",
        "food": "High-protein diet with fruits, vegetables, and whole grains. Avoid processed foods.",
        "sleep": "7-8 hours of sleep per night. Maintain a consistent sleep schedule.",
        "routine_change": "Consider adding yoga for flexibility and stress relief."
    }


    if user_gender == "Female":
        suggestions["exercise"] += "Include 20 minutes of yoga or Pilates for flexibility and core strength. "
    else:
        suggestions["exercise"] += "Include 20 minutes of strength training for muscle building. "


    if "diabetes" in user_health_conditions.lower():
        suggestions["exercise"] += "20 minutes of brisk walking or light cycling. "
        suggestions["food"] += "Focus on low-glycemic-index foods like whole grains and legumes. "
    if "high blood pressure" in user_health_conditions.lower():
        suggestions["exercise"] += "30 minutes of swimming or moderate-paced walking. "
        suggestions["food"] += "Reduce sodium intake and increase potassium-rich foods like bananas and spinach. "
    if "asthma" in user_health_conditions.lower():
        suggestions["exercise"] += "20 minutes of gentle yoga focusing on breathing exercises. "
        suggestions["food"] += "Avoid foods that may trigger allergies. "
    if "arthritis" in user_health_conditions.lower():
        suggestions["exercise"] += "20 minutes of water aerobics or light stretching. "
        suggestions["food"] += "Include anti-inflammatory foods like fatty fish and nuts. "
    if "obesity" in user_health_conditions.lower():
        suggestions["exercise"] += "30 minutes of low-impact cardio like walking or cycling. "
        suggestions["food"] += "Focus on portion control and avoid sugary drinks. "
    if "osteoporosis" in user_health_conditions.lower():
        suggestions["exercise"] += "20 minutes of weight-bearing exercises like walking or light strength training. "
        suggestions["food"] += "Include calcium-rich foods like dairy and leafy greens. "
    if "depression" in user_health_conditions.lower():
        suggestions["exercise"] += "30 minutes of aerobic exercise like jogging or dancing. "
        suggestions["food"] += "Include omega-3-rich foods like salmon and walnuts. "
    if "anxiety" in user_health_conditions.lower():
        suggestions["exercise"] += "20 minutes of mindfulness exercises or yoga. "
        suggestions["food"] += "Avoid caffeine and include magnesium-rich foods like almonds and spinach. "
    if "heart disease" in user_health_conditions.lower():
        suggestions["exercise"] += "20 minutes of light walking under medical supervision. "
        suggestions["food"] += "Focus on a heart-healthy diet with plenty of fruits, vegetables, and whole grains. "
    if "thyroid" in user_health_conditions.lower():
        suggestions["exercise"] += "30 minutes of moderate aerobic activity like brisk walking. "
        suggestions["food"] += "Include iodine-rich foods like seafood and dairy. "


    if user_body_temp > 37.5:
        suggestions["exercise"] = "Rest and hydrate. Avoid intense workouts until body temperature normalizes."

    return suggestions



st.sidebar.title("Fitness Tracker")
menu = ["Home", "Track Progress", "AI Suggestions", "BMI Calculator", "Comparison"]
choice = st.sidebar.selectbox("Menu", menu)


if choice == "Home":
    st.markdown('<h1 class="big-title">Welcome to the Personal Fitness Tracker</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle">Track your fitness journey with AI!</h3>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <div class="feature-box">
            <ul>
                <li><strong>Track Progress</strong>: Enter your fitness activities and visualize your progress.</li>
                <li><strong>Suggestions</strong>: Get personalized workout plans, food suggestions, and sleep cycles.</li>
                <li><strong>BMI Calculator</strong>: Calculate your Body Mass Index.</li>
                <li><strong>Comparison</strong>: Compare your fitness data with others.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


elif choice == "Track Progress":
    st.title("Track Your Fitness Progress")
    model = train_model(exercise_data)
    st.subheader("Enter Your Fitness Data")

    user_weight = st.number_input("Your Weight (kg)", value=0.0, min_value=0.0, max_value=300.0, step=0.1)
    user_height = st.number_input("Your Height (m)", value=0.0, min_value=0.0, max_value=3.0, step=0.01)
    avg_bpm = st.number_input("Your Average BPM", value=0, min_value=0, max_value=300)
    session_duration = st.number_input("Session Duration (hours)", value=0.0, min_value=0.0, max_value=24.0, step=0.1)


    workout_type = st.selectbox("Workout Type", [
        "Cardio", "Strength", "HIIT", "Yoga", "Pilates", "Cycling", "Swimming", "Running", "Dancing", "CrossFit"
    ])
    date = st.date_input("Date")

    if st.button("Submit"):

        workout_type_map = {
            "Cardio": 0,
            "Strength": 1,
            "HIIT": 2,
            "Yoga": 3,
            "Pilates": 4,
            "Cycling": 5,
            "Swimming": 6,
            "Running": 7,
            "Dancing": 8,
            "CrossFit": 9
        }
        workout_type_num = workout_type_map[workout_type]
        input_data = np.array([[user_weight, user_height, avg_bpm, session_duration, workout_type_num]])
        calories_burned_ml = model.predict(input_data)[0]
        st.success(f"Calories Burned (ML Prediction): {calories_burned_ml:.2f}")


        user_data = pd.DataFrame({
            "Weight (kg)": [user_weight],
            "Height (m)": [user_height],
            "Avg_BPM": [avg_bpm],
            "Session_Duration (hours)": [session_duration],
            "Workout_Type": [workout_type],
            "Calories_Burned (ML Prediction)": [calories_burned_ml],
            "Date": [date]
        })
        st.session_state.user_data = pd.concat([st.session_state.get("user_data", pd.DataFrame()), user_data],
                                               ignore_index=True)
        st.success("Data Submitted Successfully!")


    if "user_data" in st.session_state:
        st.subheader("Your Fitness Data")
        st.write(st.session_state.user_data)

        # Visualize Progress
        st.subheader("Progress Visualization")
        fig, ax = plt.subplots()
        sns.lineplot(x=st.session_state.user_data["Date"],
                     y=st.session_state.user_data["Calories_Burned (ML Prediction)"], ax=ax, label="ML Prediction")
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)


elif choice == "AI Suggestions":
    st.title("Fitness Suggestions with AI")
    st.subheader("Enter Your Details")


    user_weight = st.number_input("Your Weight (kg)", value=0, min_value=0, max_value=300)
    user_age = st.number_input("Your Age", value=0, min_value=0, max_value=120)
    user_health_conditions = st.text_input("Any Health Conditions (e.g., diabetes, high blood pressure)")
    user_body_temp = st.number_input("Your Body Temperature (°C)", value=0.0, min_value=0.0, max_value=50.0, step=0.1)
    user_gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Get Suggestions"):
        suggestions = generate_suggestions(user_weight, user_age, user_health_conditions, user_body_temp, user_gender)
        st.success("Here are your personalized suggestions:")
        st.subheader("Exercise Routine")
        st.write(suggestions["exercise"])
        st.subheader("Food Suggestions")
        st.write(suggestions["food"])
        st.subheader("Sleep Cycle")
        st.write(suggestions["sleep"])
        st.subheader("Routine Changes")
        st.write(suggestions["routine_change"])


elif choice == "BMI Calculator":
    st.title("BMI Calculator")
    st.write("Calculate your Body Mass Index (BMI) based on your weight and height.")

    user_weight = st.number_input("Enter your weight (kg)", value=0.0, min_value=0.0, max_value=300.0, step=0.1)
    user_height = st.number_input("Enter your height (m)", value=0.0, min_value=0.0, max_value=3.0, step=0.01)

    if st.button("Calculate BMI"):
        if user_height <= 0:
            st.error("Height must be greater than 0.")
        else:
            bmi = user_weight / (user_height ** 2)
            st.success(f"Your BMI is: {bmi:.2f}")
            if bmi < 18.5:
                st.write("You are underweight. Consider increasing your calorie intake and strength training.")
            elif 18.5 <= bmi < 24.9:
                st.write("You have a normal weight. Maintain your current routine.")
            elif 25 <= bmi < 29.9:
                st.write("You are overweight. Consider increasing cardio workouts and reducing calorie intake.")
            else:
                st.write("You are obese. Consult a healthcare professional for a personalized plan.")



elif choice == "Comparison":
    st.title("Fitness Comparison")
    if "user_data" in st.session_state and not st.session_state.user_data.empty:
        st.subheader("Your Fitness Data vs Others")

        your_avg_calories = st.session_state.user_data["Calories_Burned (ML Prediction)"].mean()
        your_avg_duration = st.session_state.user_data["Session_Duration (hours)"].mean()
        your_avg_bpm = st.session_state.user_data["Avg_BPM"].mean()


        others_avg_calories = exercise_data["Calories_Burned"].mean()
        others_avg_duration = exercise_data["Session_Duration (hours)"].mean()
        others_avg_bpm = exercise_data["Avg_BPM"].mean()

        # Function to determine comparison text
        def get_comparison_text(your_value, others_value):
            if your_value > others_value:
                return "Higher than others"
            elif your_value < others_value:
                return "Lower than others"
            else:
                return "Equal to others"


        st.write("### Metrics Comparison")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Your Avg Calories Burned",
                f"{your_avg_calories:.2f}",
                get_comparison_text(your_avg_calories, others_avg_calories)
            )
        with col2:
            st.metric(
                "Your Avg Exercise Duration (hours)",
                f"{your_avg_duration:.2f}",
                get_comparison_text(your_avg_duration, others_avg_duration)
            )
        with col3:
            st.metric(
                "Your Avg Heart Rate (BPM)",
                f"{your_avg_bpm:.2f}",
                get_comparison_text(your_avg_bpm, others_avg_bpm)
            )


        st.write("### Average Calories Burned: You vs Others")
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ["You", "Others"]
        values = [your_avg_calories, others_avg_calories]


        bars = ax.bar(categories, values, color=["lightgreen", "lightblue"])


        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")


        plt.title("Average Calories Burned: You vs Others")
        plt.xlabel("Category")
        plt.ylabel("Average Calories Burned")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)


        if your_avg_calories > others_avg_calories:
            st.success("You're burning more calories on average than others! Keep it up! 💪")
        elif your_avg_calories < others_avg_calories:
            st.warning("You're burning fewer calories on average than others. Let's step it up! 🚀")
        else:
            st.info("You're burning about the same calories on average as others. Keep going! 👍")

    else:
        st.warning(
            "No user data available for comparison. Please submit your fitness data in the 'Track Progress' section."
        )