import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="College Placement Trends Analyzer", layout="wide")
st.title("🎓 College Placement Trends Analyzer")
st.write("Explore real patterns in placement data — CGPA, branch, backlogs, certifications, hackathons, skills, and package trends.")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 300
    branches = np.random.choice(["CSE", "ECE", "Mech", "Civil", "IT", "EEE"], n, p=[0.3,0.2,0.15,0.1,0.15,0.1])
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

st.sidebar.header("Filters")
selected_branches = st.sidebar.multiselect("Select Branch(es)", df["Branch"].unique(), default=list(df["Branch"].unique()))
cgpa_range = st.sidebar.slider("CGPA Range", 5.0, 10.0, (5.0, 10.0))
min_certs = st.sidebar.slider("Minimum Certifications", 0, 4, 0)

filtered = df[
    (df["Branch"].isin(selected_branches)) &
    (df["CGPA"].between(cgpa_range[0], cgpa_range[1])) &
    (df["Certifications"] >= min_certs)
]

st.write(f"Showing **{len(filtered)}** students based on your filters")
st.dataframe(filtered.head(10))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Placement Rate", f"{filtered['Placed'].mean()*100:.1f}%")
col2.metric("Avg CGPA", f"{filtered['CGPA'].mean():.2f}")
col3.metric("Avg Package (Placed)", f"₹{filtered[filtered['Placed']==1]['Package_LPA'].mean():.1f} LPA")
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

st.subheader("Correlation Heatmap (incl. Certifications, Hackathons, Skills)")
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

st.caption("Note: This demo uses a realistic simulated dataset. Replace with your college's actual placement data (CSV) for real insights.")
