import random
import pandas as pd

professions = {
"Government Officer": 5,
"Software Engineer": 4,
"Doctor": 4.5,
"CA": 4,
"Business Owner": 3,
"Private Job": 2,
"Freelancer": 1
}

education_map = {"PhD":3, "Masters":2, "Bachelors":1, "Diploma":0}
location_map = {"Tier-1":3, "Tier-2":2, "Tier-3":1, "Rural":0}
marital_penalty = {"Single":0, "Married":-10, "Divorced":-5}
marriage_penalty = {"Arranged":0, "Love":-2}

rows = []

for _ in range(5000):
    age = random.randint(22, 40)
    salary = random.randint(20000, 200000)
    profession = random.choice(list(professions.keys()))
    education = random.choice(list(education_map.keys()))
    location = random.choice(list(location_map.keys()))
    home = random.choice(["Own", "Rented"])
    family_wealth = random.choice(["Lower", "Middle", "Upper-Middle", "Upper"])
    marital_status = random.choice(list(marital_penalty.keys()))
    marriage_type = random.choice(list(marriage_penalty.keys()))
    govt_job = 1 if profession == "Government Officer" else 0

    dowry = 10
    dowry += (salary // 10000) * 0.5
    dowry += professions[profession]
    dowry += education_map[education]
    dowry += location_map[location]
    dowry += 3 if home == "Own" else 0
    dowry += 5 if govt_job else 0
    dowry += marital_penalty[marital_status]
    dowry += marriage_penalty[marriage_type]

    rows.append([
    age, salary, profession, education, location, home,
    family_wealth, marital_status, marriage_type,
    govt_job, round(dowry, 2)
])

df = pd.DataFrame(rows, columns=[
"age","monthly_salary","profession","education_level",
"location","home_status","family_wealth",
"marital_status","marriage_type",
"government_job","dowry_amount_lakhs"
])

df.to_csv("./data/dowry_dataset.csv", index=False)
