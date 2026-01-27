import openai
import json
import pandas as pd

def infer_column_roles_openai(df, api_key):
    # Initialize client
    client = openai.OpenAI(api_key=api_key)

    column_names = df.columns.tolist()
    first_rows = df.head(5).to_dict(orient="records")

    system_prompt = (
"You are a data analyst helping identify key columns in a tabular dataset. "

"Given the column names and a few data rows, your task is to infer and identify the following:\n\n"

"1. Columns that represent **geographic regions**. This includes Indian administrative divisions such as **district**, **mandal**, **taluk**, **tehsil**, **village**, **ward**, **zone**, or **locality**. "
"Use both column names and row values to make this inference — many place names (e.g., 'Adilabad', 'Ankapoor') are Indian locations. "
"Your answer should return all relevant geographic columns in order from highest to lowest level (e.g. district to village)."

"2. A column most likely representing a **date**. Also return the most likely Python datetime format string (e.g., %Y-%m-%d or %d/%m/%Y) that can be used with datetime.strptime() or pandas.to_datetime() to correctly parse the values."

"3. A column most likely representing a **timestamp** (e.g., created_at, updated_at). Also return the most likely Python datetime format string (e.g., %Y-%m-%d %H:%M:%S) that can be used to parse them with datetime.strptime(). If there are variations, include those exactly in the format string."

"4. A column most likely representing a **categorical** feature. This includes columns that represent finite sets of options, such as a list of colors, a set of shapes, a list of animals, a set of occupations, a set of educational levels, etc."
"Note that categorical features are not typically continuous, but rather take on a set of discrete values. "
"Also, the first 5 rows of data may be used to help make this inference."

"Use the first few rows of data to help make your judgment."

"Return your answer as a JSON object with this structure:\n"
"{\n"
'  "region": ["ordered_list_of_column_names"],\n'
'  "date": {\n'
'    "column": "["ordered_list_of_date_columns"]",\n'
'    "format": "["ordered_list_of_date_formats"]"\n'
"  },\n"
'  "timestamp": {\n'
'    "column": "["ordered_list_of_timestamp_columns"]",\n'
'    "format": "["ordered_list_of_timestamp_formats"]"\n'
"  },\n"
'  "categorical": ["ordered_list_of_categorical_columns"]\n'
"}\n"

"If no match is found for a category, return null.\n"

    )

    user_prompt = (
        f"Here are the column names:\n{column_names}\n\n"
        f"And here are the first 20 rows of data:\n{json.dumps([{k: str(v) for k,v in row.items()} for row in first_rows], indent=2)}"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        top_p=1
    )

    message_content = response.choices[0].message.content

    try:
        return json.loads(message_content)
    except json.JSONDecodeError:
        return {
            "error": "Could not parse OpenAI response",
            "raw_response": message_content
        }