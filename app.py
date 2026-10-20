import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="College Placement Trends Analyzer", layout="wide")
st.title("College Placement Trends Analyzer")
st.write("Explore real patterns in placement data - CGPA, branch, backlogs, certifications, hackathons, skills, and package trends.")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 300
    branches = np.random.choice(["BCom", "BBA", "BCA", "MBA", "MCA"], n, p=[0.3,0.2,0.25,0.15,0.1])
    cgpa = np.round(np.random.normal(7.5, 0.8, n), 2)
    cgpa = np.clip(cgpa, 5.0, 10.0)
    backlogs = np.random.choice([0,0,0,1,2], n)
    certifications = np.random.choice([0,1,2,3,4], n, p=[0.2,0.3,0.25,0.15,0.1])
    hackathons = np.random.choice([0,1,2,3], n, p=[0.4,0.3,0.2,0.1])
    extra_skills = np.random.choice([0,1,2,3,4,5], n)
    companies = np.random.choice(["TCS","Infosys","Wipro","Amazon","Microsoft","Cognizant","Accenture"], n)

    placed_prob = 1 / (1 + np.exp(-(cgpa-7) - (backlogs*-0.5) + (certifications*0.15) + (hackathons*0.1)))
    placed = np.random.binomial(1, placed_prob)

    package = np.where(placed==1, np.round(np.random.normal(6,3,n) + certifications*0.3 + hackathons*0.2, 1), 0)
    package = np.clip(package, 0, None)

    return pd.DataFrame({
        "Branch": branches, "CGPA": cgpa, "Backlogs": backlogs,
        "Certifications": certifications, "Hackathons_Attended": hackathons,
        "Extra_Skills_Count": extra_skills,
        "Company": companies, "Placed": placed, "Package_LPA": package
    })

df = load_data()

@st.cache_resource
def train_model(data):
    le = LabelEncoder()
    data = data.copy()
    data["Branch_Code"] = le.fit_transform(data["Branch"])
    features = ["Branch_Code", "CGPA", "Backlogs", "Certifications", "Hackathons_Attended", "Extra_Skills_Count"]
    X = data[features]
    y = data["Placed"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, le

model, le = train_model(df)

st.sidebar.header("Filters")
selected_branches = st.sidebar.multiselect("Select Branch(es)", df["Branch"].unique(), default=list(df["Branch"].unique()))
cgpa_range = st.sidebar.slider("CGPA Range", 5.0, 10.0, (5.0, 10.0))
min_certs = st.sidebar.slider("Minimum Certifications", 0, 4, 0)

filtered = df[
    (df["Branch"].isin(selected_branches)) &
    (df["CGPA"].between(cgpa_range[0], cgpa_range[1])) &
    (df["Certifications"] >= min_certs)
]

st.write(f"Showing {len(filtered)} students based on your filters")
st.dataframe(filtered.head(10))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Placement Rate", f"{filtered['Placed'].mean()*100:.1f}%")
col2.metric("Avg CGPA", f"{filtered['CGPA'].mean():.2f}")
col3.metric("Avg Package (Placed)", f"Rs.{filtered[filtered['Placed']==1]['Package_LPA'].mean():.1f} LPA")
col4.metric("Avg Certifications", f"{filtered['Certifications'].mean():.1f}")

st.subheader("Placement Rate by Branch")
fig1, ax1 = plt.subplots(figsize=(8,4))
branch_rate = filtered.groupby("Branch")["Placed"].mean().sort_values(ascending=False)
sns.barplot(x=branch_rate.index, y=branch_rate.values, ax=ax1, color="steelblue")
ax1.set_ylabel("Placement Rate")
st.pyplot(fig1)

st.subheader("Placement Rate by Number of Certifications")
fig2, ax2 = plt.subplots(figsize=(8,4))
cert_rate = filtered.groupby("Certifications")["Placed"].mean()
sns.barplot(x=cert_rate.index, y=cert_rate.values, ax=ax2, color="darkorange")
ax2.set_xlabel("Certifications Completed")
ax2.set_ylabel("Placement Rate")
st.pyplot(fig2)

st.subheader("CGPA Distribution")
fig3, ax3 = plt.subplots(figsize=(8,4))
sns.histplot(filtered["CGPA"], bins=20, kde=True, color="purple", ax=ax3)
st.pyplot(fig3)

st.subheader("Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(8,6))
numeric_df = filtered[["CGPA","Backlogs","Certifications","Hackathons_Attended","Extra_Skills_Count","Placed","Package_LPA"]]
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

st.subheader("Average Package by Company")
fig5, ax5 = plt.subplots(figsize=(8,4))
company_avg = filtered[filtered["Placed"]==1].groupby("Company")["Package_LPA"].mean().sort_values(ascending=False)
sns.barplot(x=company_avg.index, y=company_avg.values, ax=ax5, color="mediumseagreen")
plt.xticks(rotation=30)
st.pyplot(fig5)

st.divider()
st.header("Check Your Own Placement Chances")
st.write("Enter your own details below to see how you compare to past students and get an estimated placement likelihood.")

input_col1, input_col2, input_col3 = st.columns(3)
with input_col1:
    user_branch = st.selectbox("Your Branch", df["Branch"].unique())
    user_cgpa = st.number_input("Your CGPA", min_value=5.0, max_value=10.0, value=7.5, step=0.1)
with input_col2:
    user_backlogs = st.number_input("Backlogs", min_value=0, max_value=5, value=0)
    user_certs = st.number_input("Certifications Completed", min_value=0, max_value=10, value=1)
with input_col3:
    user_hackathons = st.number_input("Hackathons Attended", min_value=0, max_value=10, value=0)
    user_skills = st.number_input("Extra Skills Count", min_value=0, max_value=10, value=1)

if st.button("Check My Chances"):
    similar = df[
        (df["Branch"] == user_branch) &
        (df["CGPA"].between(user_cgpa - 0.5, user_cgpa + 0.5))
    ]

    st.subheader("Based on Similar Students")
    if len(similar) > 0:
        st.write(f"Found {len(similar)} students in {user_branch} with CGPA close to yours ({user_cgpa - 0.5:.1f} to {user_cgpa + 0.5:.1f}).")
        c1, c2 = st.columns(2)
        c1.metric("Their Placement Rate", f"{similar['Placed'].mean()*100:.1f}%")
        c2.metric("Their Avg Package", f"Rs.{similar[similar['Placed']==1]['Package_LPA'].mean():.1f} LPA" if similar['Placed'].sum() > 0 else "N/A")
    else:
        st.write("No close matches found in the dataset for direct comparison.")

    branch_code = le.transform([user_branch])[0]
    input_features = pd.DataFrame([[branch_code, user_cgpa, user_backlogs, user_certs, user_hackathons, user_skills]],
                                   columns=["Branch_Code","CGPA","Backlogs","Certifications","Hackathons_Attended","Extra_Skills_Count"])
    prob = model.predict_proba(input_features)[0][1]

    st.subheader("Model-Based Prediction")
    st.metric("Estimated Placement Probability", f"{prob*100:.1f}%")
    if prob >= 0.7:
        st.success("Strong profile - high estimated placement likelihood.")
    elif prob >= 0.4:
        st.warning("Moderate profile - consider adding more certifications or hackathon experience.")
    else:
        st.error("Lower estimated likelihood - focus on improving CGPA, certifications, or reducing backlogs.")

st.caption("Note: This demo uses a realistic simulated dataset of 300 students. Replace with your college's actual placement data (CSV) for real insights. The prediction model is trained on this sample data for demonstration purposes.")
