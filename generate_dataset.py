import pandas as pd
import random

data = []

# PERFECTO (300)
for _ in range(300):
    left_arm = random.randint(165, 180)
    right_arm = random.randint(165, 180)
    left_leg = random.randint(165, 180)
    right_leg = random.randint(165, 180)
    data.append([left_arm, right_arm, left_leg, right_leg, "PERFECTO"])

# BUENO (300)
for _ in range(300):
    left_arm = random.randint(140, 175)
    right_arm = random.randint(140, 175)
    left_leg = random.randint(140, 175)
    right_leg = random.randint(140, 175)
    data.append([left_arm, right_arm, left_leg, right_leg, "BUENO"])

# INCORRECTO (400)
for _ in range(400):
    left_arm = random.randint(0, 155)
    right_arm = random.randint(0, 155)
    left_leg = random.randint(0, 155)
    right_leg = random.randint(0, 155)
    data.append([left_arm, right_arm, left_leg, right_leg, "INCORRECTO"])

df = pd.DataFrame(data, columns=[
    "left_arm", "right_arm", "left_leg", "right_leg", "label"
])

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("dance_dataset.csv", index=False)

print("Dataset realista generado.")