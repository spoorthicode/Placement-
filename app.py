import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="College Placement Trends Analyzer", layout="wide")
st.title("College Placement Trends Analyzer")
st.write("Explore real patterns in placement data - CGPA, branch, backlogs, certifications, hackathons, skills, and package trends.")

CERT_OPTIONS = [
    "Python Programming", "Data Analytics", "Web Development",
    "Digital Marketing", "Tally / Accounting", "MS Excel Advanced",
    "Communication Skills", "Cloud Computing (AWS/Azure)"
]
LANGUAGE_OPTIONS = ["English", "Hindi", "Kannada", "Telugu", "Tamil", "French", "German"]
SKILL_OPTIONS = [
    "MS Office", "Public Speaking", "Leadership", "Content Writing",
    "Graphic Design", "Coding", "Social Media Management", "Event Management"
]

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

    cert_lists = []
    skill_lists = []
    for i in range(n):
        k = min(certifications[i], len(CERT_OPTIONS))
        cert_lists.append(list(np.random.choice(CERT_OPTIONS, size=k, replace=False)) if k > 0 else [])
        k2 = min(extra_skills[i], len(SKILL_OPTIONS))
        skill_lists.append(list(np.random.choice(SKILL_OPTIONS, size=k2, replace=False)) if k2 > 0 else [])

    return pd.DataFrame({
        "Branch": branches, "CGPA": cgpa, "Backlogs": backlogs,
        "Certifications": certifications, "Hackathons_Attended": hackathons,
        "Extra_Skills_Count": extra_skills,
        "Cert_List": cert_lists, "Skill_List": skill_lists,
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
st.dataframe(filtered.drop(columns=["Cert_List", "Skill_List"]).head(10))

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

st.subheader("Most In-Demand Skills Among Placed Students")
placed_df = filtered[filtered["Placed"] == 1]
all_certs = [c for sublist in placed_df["Cert_List"] for c in sublist]
if len(all_certs) > 0:
    top_certs = pd.Series(all_certs).value_counts().head(8)
    fig6, ax6 = plt.subplots(figsize=(8,4))
    sns.barplot(x=top_certs.values, y=top_certs.index, ax=ax6, color="teal")
    ax6.set_xlabel("Number of Placed Students")
    st.pyplot(fig6)
else:
    st.write("Not enough placed students in current filter to show this chart.")

st.divider()
st.header("Career Compass - Find Your Best-Fit Path")
st.write("Not sure which branch fits you? Tell us what interests you, and we will suggest a path backed by real placement data.")

interest_map = {
    "Technology & Coding": "BCA",
    "Finance & Accounting": "BCom",
    "Marketing & Branding": "BBA",
    "Management & Leadership": "MBA",
    "Data & Analytics": "MCA",
    "Human Resources": "MBA"
}
interest = st.selectbox("What are you most interested in?", list(interest_map.keys()))

if st.button("Suggest My Path"):
    recommended_branch = interest_map[interest]
    branch_data = df[df["Branch"] == recommended_branch]
    st.subheader(f"Recommended Path: {recommended_branch}")
    st.write(f"Based on your interest in **{interest}**, **{recommended_branch}** is a strong fit.")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Placement Rate", f"{branch_data['Placed'].mean()*100:.1f}%")
    rc2.metric("Avg CGPA of Placed", f"{branch_data[branch_data['Placed']==1]['CGPA'].mean():.2f}")
    rc3.metric("Avg Package", f"Rs.{branch_data[branch_data['Placed']==1]['Package_LPA'].mean():.1f} LPA")
    st.caption("These numbers are based on the current dataset and are meant as a general guide, not a guarantee.")

st.divider()
st.header("Check Your Own Placement Chances")
st.write("Fill in your profile below, like a mini CV, to see how you compare to past students and get an estimated placement likelihood.")

st.subheader("Your Profile")
name_col, branch_col, cgpa_col = st.columns(3)
with name_col:
    user_name = st.text_input("Full Name", "")
with branch_col:
    user_branch = st.selectbox("Your Branch", df["Branch"].unique())
with cgpa_col:
    user_cgpa = st.number_input("Your CGPA", min_value=5.0, max_value=10.0, value=7.5, step=0.1)

user_backlogs = st.number_input("Backlogs", min_value=0, max_value=5, value=0)

st.markdown("**Certifications Completed** (select all that apply)")
selected_certs = st.multiselect("Certifications", CERT_OPTIONS, label_visibility="collapsed")

st.markdown("**Languages Known**")
selected_languages = st.multiselect("Languages", LANGUAGE_OPTIONS, default=["English"], label_visibility="collapsed")

st.markdown("**Extra Skills**")
selected_skills = st.multiselect("Skills", SKILL_OPTIONS, label_visibility="collapsed")

user_hackathons = st.number_input("Hackathons / Competitions Attended", min_value=0, max_value=10, value=0)

def get_probability(branch, cgpa, backlogs, certs, hackathons, skills):
    branch_code = le.transform([branch])[0]
    feats = pd.DataFrame([[branch_code, cgpa, backlogs, certs, hackathons, skills]],
                          columns=["Branch_Code","CGPA","Backlogs","Certifications","Hackathons_Attended","Extra_Skills_Count"])
    return model.predict_proba(feats)[0][1]

if st.button("Check My Chances"):
    user_certs_count = len(selected_certs)
    user_skills_count = len(selected_skills) + len(selected_languages)

    similar = df[
        (df["Branch"] == user_branch) &
        (df["CGPA"].between(user_cgpa - 0.5, user_cgpa + 0.5))
    ]

    st.subheader("Your Profile Summary")
    st.write(f"**Name:** {user_name if user_name else 'Not entered'}")
    st.write(f"**Branch:** {user_branch}  |  **CGPA:** {user_cgpa}  |  **Backlogs:** {user_backlogs}")
    st.write(f"**Certifications:** {', '.join(selected_certs) if selected_certs else 'None'}")
    st.write(f"**Languages Known:** {', '.join(selected_languages) if selected_languages else 'None'}")
    st.write(f"**Extra Skills:** {', '.join(selected_skills) if selected_skills else 'None'}")
    st.write(f"**Hackathons Attended:** {user_hackathons}")

    score = min(100, round(
        (user_cgpa / 10) * 40 +
        (min(user_certs_count, 5) / 5) * 20 +
        (min(user_hackathons, 3) / 3) * 15 +
        (min(user_skills_count, 10) / 10) * 15 +
        (10 if user_backlogs == 0 else 0)
    ))
    st.subheader("Profile Strength Score")
    st.progress(score / 100)
    st.write(f"**{score} / 100**")
    if score >= 75:
        st.success("Strong profile overall.")
    elif score >= 50:
        st.warning("Decent profile, with room to grow.")
    else:
        st.error("Profile needs strengthening in key areas.")

    st.subheader("Based on Similar Students")
    if len(similar) > 0:
        st.write(f"Found {len(similar)} students in {user_branch} with CGPA close to yours ({user_cgpa - 0.5:.1f} to {user_cgpa + 0.5:.1f}).")
        c1, c2 = st.columns(2)
        c1.metric("Their Placement Rate", f"{similar['Placed'].mean()*100:.1f}%")
        c2.metric("Their Avg Package", f"Rs.{similar[similar['Placed']==1]['Package_LPA'].mean():.1f} LPA" if similar['Placed'].sum() > 0 else "N/A")
    else:
        st.write("No close matches found in the dataset for direct comparison.")

    base_prob = get_probability(user_branch, user_cgpa, user_backlogs, user_certs_count, user_hackathons, user_skills_count)

    st.subheader("Model-Based Prediction")
    st.metric("Estimated Placement Probability", f"{base_prob*100:.1f}%")
    if base_prob >= 0.7:
        st.success("Strong profile - high estimated placement likelihood.")
    elif base_prob >= 0.4:
        st.warning("Moderate profile - consider adding more certifications or hackathon experience.")
    else:
        st.error("Lower estimated likelihood - focus on improving CGPA, certifications, or reducing backlogs.")

    st.subheader("Skill Gap Advisor")
    st.write("Here is what would help your chances the most, based on the model:")

    options = []
    prob_more_cert = get_probability(user_branch, user_cgpa, user_backlogs, user_certs_count + 1, user_hackathons, user_skills_count)
    options.append(("Add 1 more certification", prob_more_cert - base_prob))

    prob_more_hack = get_probability(user_branch, user_cgpa, user_backlogs, user_certs_count, user_hackathons + 1, user_skills_count)
    options.append(("Attend 1 more hackathon", prob_more_hack - base_prob))

    prob_more_cgpa = get_probability(user_branch, min(user_cgpa + 0.5, 10.0), user_backlogs, user_certs_count, user_hackathons, user_skills_count)
    options.append(("Improve CGPA by 0.5", prob_more_cgpa - base_prob))

    if user_backlogs > 0:
        prob_less_backlog = get_probability(user_branch, user_cgpa, user_backlogs - 1, user_certs_count, user_hackathons, user_skills_count)
        options.append(("Clear 1 backlog", prob_less_backlog - base_prob))

    options.sort(key=lambda x: x[1], reverse=True)
    for label, delta in options:
        if delta > 0:
            st.write(f"- **{label}** could increase your estimated probability by about {delta*100:.1f} percentage points.")
    if all(delta <= 0 for _, delta in options):
        st.write("Your profile is already strong across these factors.")

st.caption("Note: This demo uses a realistic simulated dataset of 300 students. Replace with your college's actual placement data (CSV) for real insights. The prediction model is trained on this sample data for demonstration purposes.")
