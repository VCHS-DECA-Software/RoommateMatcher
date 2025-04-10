import pandas as pd
import sys

# Default group size is 4
groupSize = int(input("What is the maximum number of students that can be in 1 room?: "))

if groupSize <= 0:
  groupSize = 4

# Allowed max difference in grade levels
allowedDiff = int(input("What is the maximum allowed difference in grade levels between students in a room?: "))

#Student Groups list
studentGroups = []

filePath = input("Please enter the name of the CSV file with the students' preferences: ")

# Check if there is a CSV file with the list of students and their preferences
if len(file) < 1:
    print("Please input a CSV file with students and their preferences.")
    sys.exit()
else:
    # Import the list of students as a Pandas dataframe
    studentsDf = pd.read_csv(filePath, header=0)

if studentsDf.shape[0] <= 0:
    print("Dataframe is empty.")
    sys.exit()

# Initialize a matrix of student preferences with the default value of 0 (no preference)
studentPrefMatrix = np.zeros((studentsDf.shape[0], studentsDf.shape[0]), dtype=float)

# Dataframe of students who had errors in their preferences
errorDf = pd.DataFrame(columns=["Student Name", "Student Email", "Error"])

for idx in studentsDf.index:
    row = studentsDf.loc[idx]
    student = row["Email Address"]
    prefs = [
        row["Roommate Preference #1 Email"],
        row["Roommate Preference #2 Email"],
        row["Roommate Preference #3 Email"],
    ]

    for pref in prefs:
        # check if the prefrence is in the dataset
        if (studentsDf[studentsDf["Email Address"] == pref].index.values.size <= 0):
            errorDf.loc[len(errorDf.index)] = [row["Full Name"], student, "Preference not in dataset or does not have preference"]
            continue

        # check if student put themself as a preference
        elif (student == pref):
            errorDf.loc[len(errorDf.index)] = [row["Full Name"], student, "Preference is themself"]
            continue

        # check if student and their preference are the same gender
        elif (row["Gender"] != studentsDf[studentsDf["Email Address"] == pref]["Gender"].values[0]):
            errorDf.loc[len(errorDf.index)] = [row["Full Name"], student, "Preference is not the same gender"]
            continue

        # check if the grade levels of the student and preference are more than one apart
        elif (abs(row["Grade"] - studentsDf[studentsDf["Email Address"] == pref]["Grade"].values[0]) > allowedDiff):
            errorDf.loc[len(errorDf.index)] = [row["Full Name"], student, "Preference is more than one grade level apart"]
            continue

        else:
            studentPrefMatrix.itemset(
                (
                    studentsDf[studentsDf["Email Address"] == pref].index[0],
                    studentsDf[studentsDf["Email Address"] == student].index[0],
                ),
                (10 - (prefs.index(pref) * 3)),
            )

# Running Irving's algorithm
matching = Matching(
    studentPrefMatrix, group_size=groupSize, iter_count=2, final_iter_count=2
)
score, studentIdxs = matching.solve()
print(f"Irving's Algorithm Score: {score}")

# Converting list of student indexes to list of student names
for group in studentIdxs:
    studentGroup = []
    for studentIdx in group:
        studentGroup.append("Full Name: %s, Gender: %s, Grade: %s" % (studentsDf.iloc[studentIdx]["Full Name"], studentsDf.iloc[studentIdx]["Gender"], studentsDf.iloc[studentIdx]["Grade"]))
    studentGroups.append(studentGroup)

studentsDf = pd.DataFrame(data=studentGroups)
studentsDf.to_csv('rooms.csv',index=True, header=False)
errorDf.to_csv('errors.csv',index=True, header=True)
