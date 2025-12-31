"""
RoommateMatcher - Intelligent Roommate Assignment System

This program uses Irving's stable roommate algorithm to optimally assign
students to rooms based on their preferences while respecting constraints
like same-gender groupings and grade level compatibility.
"""

import pandas as pd
import numpy as np
import sys
import os
from algorithm import Matching


def get_user_input():
    """Get configuration from user input with validation."""
    # Group size
    try:
        group_size = int(input("Maximum students per room: "))
        if group_size <= 0:
            print("Invalid size. Using default of 4.")
            group_size = 4
    except ValueError:
        print("Invalid input. Using default of 4.")
        group_size = 4

    # Grade difference limit
    try:
        allowed_diff = int(input("Maximum grade level difference allowed: "))
        if allowed_diff < 0:
            print("Invalid value. Using default of 1.")
            allowed_diff = 1
    except ValueError:
        print("Invalid input. Using default of 1.")
        allowed_diff = 1

    # CSV file path
    file_path = input("CSV file path: ").strip()

    return group_size, allowed_diff, file_path


def validate_file(file_path):
    """Validate the input CSV file exists and has required columns."""
    if not file_path:
        print("Error: No file path provided.")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, header=0)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    required_columns = [
        "Full Name", "Email Address", "Gender", "Grade",
        "Roommate Preference #1 Email",
        "Roommate Preference #2 Email",
        "Roommate Preference #3 Email"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)

    if df.empty:
        print("Error: CSV file is empty.")
        sys.exit(1)

    duplicates = df[df["Email Address"].duplicated()]["Email Address"].tolist()
    if duplicates:
        print(f"Error: Duplicate email addresses found: {duplicates}")
        sys.exit(1)

    # Validate Grade column contains numeric values
    if not pd.api.types.is_numeric_dtype(df["Grade"]):
        # Try to convert to numeric
        df["Grade"] = pd.to_numeric(df["Grade"], errors='coerce')

    # Check for NaN grades (whether from conversion or already present)
    invalid_grades = df[df["Grade"].isna()]["Full Name"].tolist()
    if invalid_grades:
        print(f"Error: Missing or non-numeric grade values for students: {invalid_grades}")
        sys.exit(1)

    # Normalize gender values (case-insensitive)
    df["Gender"] = df["Gender"].str.strip().str.title()

    # Reset index to ensure 0-based indexing for numpy operations
    df = df.reset_index(drop=True)

    return df


def build_email_index(df):
    """Build O(1) lookup dictionary from email to row data (case-insensitive)."""
    return {
        row["Email Address"].strip().lower(): {
            "idx": idx,
            "name": row["Full Name"],
            "gender": row["Gender"],
            "grade": row["Grade"]
        }
        for idx, row in df.iterrows()
    }


def build_preference_matrix(df, email_index, allowed_diff):
    """
    Build the preference matrix and collect any validation errors.

    Matrix format: matrix[student_idx][preferred_idx] = preference_score
    Scores: 1st choice = 10, 2nd choice = 7, 3rd choice = 4
    """
    n = len(df)
    matrix = np.zeros((n, n), dtype=float)
    errors = []

    for idx, row in df.iterrows():
        student_email = row["Email Address"]
        student_gender = row["Gender"]
        student_grade = row["Grade"]
        student_name = row["Full Name"]

        preferences = [
            row["Roommate Preference #1 Email"],
            row["Roommate Preference #2 Email"],
            row["Roommate Preference #3 Email"],
        ]

        for rank, pref_email in enumerate(preferences):
            # Skip empty preferences (including whitespace-only)
            if pd.isna(pref_email) or str(pref_email).strip() == "":
                continue

            # Normalize email for case-insensitive lookup
            pref_email_normalized = str(pref_email).strip().lower()

            # O(1) lookup
            pref_data = email_index.get(pref_email_normalized)

            if pref_data is None:
                errors.append({
                    "student": student_name,
                    "email": student_email,
                    "error": f"Preference '{pref_email}' not found in dataset"
                })
                continue

            # Validate: not self (case-insensitive)
            if student_email.strip().lower() == pref_email_normalized:
                errors.append({
                    "student": student_name,
                    "email": student_email,
                    "error": "Cannot select yourself as preference"
                })
                continue

            # Validate: same gender
            if student_gender != pref_data["gender"]:
                errors.append({
                    "student": student_name,
                    "email": student_email,
                    "error": f"Preference '{pref_data['name']}' is different gender"
                })
                continue

            # Validate: grade difference
            grade_diff = abs(student_grade - pref_data["grade"])
            if grade_diff > allowed_diff:
                errors.append({
                    "student": student_name,
                    "email": student_email,
                    "error": f"Preference '{pref_data['name']}' is {grade_diff} grades apart (max: {allowed_diff})"
                })
                continue

            # Valid preference - add to matrix
            score = 10 - (rank * 3)
            matrix[idx, pref_data["idx"]] = score

    return matrix, errors


def calculate_satisfaction(df, email_index, groups, gender_indices, allowed_diff):
    """Calculate what percentage of valid preferences are satisfied."""
    satisfied = 0
    total = 0

    # Build O(1) lookup: orig_idx -> local_idx
    orig_to_local = {orig: local for local, orig in enumerate(gender_indices)}

    # Build O(1) lookup: local_idx -> group
    local_to_group = {}
    for group in groups:
        group_set = set(group)
        for local_idx in group:
            local_to_group[local_idx] = group_set

    for local_idx, orig_idx in enumerate(gender_indices):
        student_group = local_to_group.get(local_idx)
        if student_group is None:
            continue

        row = df.iloc[orig_idx]
        student_gender = row["Gender"]
        student_grade = row["Grade"]

        preferences = [
            row.get("Roommate Preference #1 Email", ""),
            row.get("Roommate Preference #2 Email", ""),
            row.get("Roommate Preference #3 Email", ""),
        ]

        for pref_email in preferences:
            if pd.isna(pref_email) or str(pref_email).strip() == "":
                continue

            # O(1) lookup (case-insensitive)
            pref_data = email_index.get(str(pref_email).strip().lower())
            if pref_data is None:
                continue

            # Only count valid preferences
            if student_gender != pref_data["gender"]:
                continue
            if abs(student_grade - pref_data["grade"]) > allowed_diff:
                continue

            total += 1

            # Check if preference is in same group (O(1) lookups)
            pref_orig_idx = pref_data["idx"]
            pref_local_idx = orig_to_local.get(pref_orig_idx)
            if pref_local_idx is not None and pref_local_idx in student_group:
                satisfied += 1

    return satisfied, total


def run_matching_for_gender(df, email_index, matrix, grades, gender_indices, allowed_diff, group_size, num_attempts=50):
    """
    Run the matching algorithm multiple times for a single gender group,
    keeping the result with highest preference satisfaction.
    Terminates early if 100% satisfaction is achieved.
    """
    n = len(gender_indices)
    if n == 0:
        return []

    # Create sub-matrix for this gender
    sub_matrix = np.zeros((n, n), dtype=float)
    for i, orig_i in enumerate(gender_indices):
        for j, orig_j in enumerate(gender_indices):
            sub_matrix[i, j] = matrix[orig_i, orig_j]

    sub_grades = grades[gender_indices]

    best_satisfaction = -1
    best_groups = None

    for attempt in range(num_attempts):
        matching = Matching(
            prefs=sub_matrix,
            grades=sub_grades,
            allowed_grade_diff=allowed_diff,
            group_size=group_size,
            iter_count=5,
            final_iter_count=10
        )
        _, groups = matching.solve()

        satisfied, total = calculate_satisfaction(df, email_index, groups, gender_indices, allowed_diff)
        satisfaction = satisfied / total if total > 0 else 1.0

        if satisfaction > best_satisfaction:
            best_satisfaction = satisfaction
            best_groups = groups

        # Early termination: 100% satisfaction achieved
        if satisfaction >= 1.0:
            break

    # Convert local indices back to original indices
    return [[gender_indices[idx] for idx in group] for group in best_groups]


def format_room_output(df, groups):
    """Format the room assignments for output."""
    rooms = []
    for group in groups:
        room = []
        for idx in group:
            student = df.iloc[idx]
            room.append(f"Full Name: {student['Full Name']}, Gender: {student['Gender']}, Grade: {student['Grade']}")
        rooms.append(room)
    return rooms


def main():
    """Main entry point."""
    print("\n" + "=" * 50)
    print("  ROOMMATE MATCHER")
    print("=" * 50 + "\n")

    # Get user input
    group_size, allowed_diff, file_path = get_user_input()

    # Validate and load data
    print("\nLoading data...")
    df = validate_file(file_path)
    print(f"Loaded {len(df)} students.")

    # Build email index for O(1) lookups
    email_index = build_email_index(df)

    # Build preference matrix
    print("Processing preferences...")
    matrix, errors = build_preference_matrix(df, email_index, allowed_diff)

    # Run matching separately for each gender
    print("Running matching algorithm...")
    grades = df["Grade"].values

    all_groups = []
    for gender in df["Gender"].unique():
        gender_indices = df[df["Gender"] == gender].index.tolist()
        groups = run_matching_for_gender(
            df, email_index, matrix, grades, gender_indices,
            allowed_diff, group_size
        )
        all_groups.extend(groups)

    # Format and save results
    print("Saving results...")

    rooms = format_room_output(df, all_groups)
    rooms_df = pd.DataFrame(data=rooms)
    rooms_df.to_csv('rooms.csv', index=True, header=False)

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv('errors.csv', index=True, header=True)

    # Summary
    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"\n  Rooms created: {len(all_groups)}")
    print(f"  Students assigned: {sum(len(g) for g in all_groups)}")
    if errors:
        print(f"  Preference errors: {len(errors)} (see errors.csv)")
    print(f"\n  Output saved to: rooms.csv")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
