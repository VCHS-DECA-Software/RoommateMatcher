# RoommateMatcher

[![CodeQL](https://github.com/akhil-datla/RoommateMatcher/actions/workflows/codeql.yml/badge.svg)](https://github.com/akhil-datla/Roommate-Matcher/actions/workflows/codeql.yml)

An intelligent roommate assignment system that uses a modified Irving's stable roommate algorithm to optimally match students into rooms based on their preferences while respecting constraints like same-gender groupings and grade level compatibility.

## Features

- Preference-based matching using student survey responses
- Same-gender room enforcement
- Configurable grade level restrictions
- Configurable room sizes
- Multiple optimization passes for best results
- Detailed error reporting for invalid preferences

## Installation

1. Clone this repository or [download the zip file](https://github.com/VCHS-DECA-Software/RoommateMatcher/archive/refs/heads/main.zip).

2. Install [pip](https://pip.pypa.io/en/stable/installation/) if you haven't already.

3. Open Terminal or Command Prompt and navigate to the project directory.

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Collection

Create a Google Form with the following questions:

> **Important:** Enable the "Collect email addresses" option. Questions marked with (*) are required for the program. Questions must match the exact wording shown below.

<img width="430" alt="Google Form Questionnaire" src="https://user-images.githubusercontent.com/66145155/171505113-8369ce68-fcdd-4066-92b7-139e056b36aa.png">

Verify your Google Sheet has these columns:

<img width="1126" alt="Google Sheet Columns" src="https://user-images.githubusercontent.com/66145155/171505374-3c4b0403-4e09-43de-8d95-c46dcfc78acf.png">

Download the spreadsheet as a CSV file.

## Usage

Run the program:

```bash
python main.py
```

You will be prompted for:
- **Maximum students per room**: The maximum number of students allowed in each room (default: 4)
- **Maximum grade level difference**: How many grades apart students can be to room together (default: 1)
- **CSV file path**: Path to your downloaded Google Form responses

## Output

The program generates two files:

- `rooms.csv`: Room assignments with student names, genders, and grades
- `errors.csv`: Any preference errors (e.g., preferences for students not in the dataset, cross-gender preferences, or grade constraint violations)

## Algorithm

The matching algorithm:

1. Separates students by gender to ensure same-gender rooms
2. Builds a preference matrix from survey responses (1st choice = 10 points, 2nd = 7, 3rd = 4)
3. Runs multiple matching attempts with random initialization
4. Optimizes for maximum preference satisfaction
5. Returns the best result across all attempts

## Requirements

- Python 3.7+
- pandas
- numpy

## License

MIT License
