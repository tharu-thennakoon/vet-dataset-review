from pathlib import Path

import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
from PIL import Image, UnidentifiedImageError


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Veterinary Dataset Review",
    page_icon="🐾",
    layout="wide"
)


# =========================================================
# 2. FILE PATHS
# =========================================================
CSV_PATH = Path("sample_data.csv")
IMAGE_FOLDER = Path("images")

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================================================
# 3. DISEASE CLASSES
# =========================================================
TARGET_CLASSES = [
    "Demodicosis",
    "Dermatitis",
    "Fungal Infections",
    "Healthy",
    "Hypersensitivity",
    "Ringworm"
]


# =========================================================
# 4. LOAD CSV DATA
# =========================================================
@st.cache_data
def load_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    dataframe = pd.read_csv(path)

    required_columns = [
        "Case_ID",
        "Image_Name",
        "Label"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required CSV columns: "
            + ", ".join(missing_columns)
        )

    dataframe["Case_ID"] = (
        dataframe["Case_ID"]
        .astype(str)
        .str.strip()
    )

    dataframe["Image_Name"] = (
        dataframe["Image_Name"]
        .astype(str)
        .str.strip()
    )

    dataframe["Label"] = (
        dataframe["Label"]
        .astype(str)
        .str.strip()
    )

    return dataframe


# =========================================================
# 5. CREATE IMAGE INDEX
# =========================================================
@st.cache_data
def create_image_index(
    image_folder: str
) -> dict[str, str]:

    folder = Path(image_folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Image folder not found: {folder}"
        )

    image_index = {}

    for file_path in folder.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        ):
            image_key = (
                file_path.name
                .lower()
                .strip()
            )

            if image_key not in image_index:
                image_index[image_key] = str(
                    file_path
                )

    return image_index


# =========================================================
# 6. SAFELY READ COLUMN VALUES
# =========================================================
def get_value(
    record: pd.Series,
    column_name: str
) -> str:

    if column_name not in record.index:
        return "Not available"

    value = record[column_name]

    if pd.isna(value):
        return "Not available"

    value = str(value).strip()

    if value == "":
        return "Not available"

    return value


# =========================================================
# 7. LOAD DATA
# =========================================================
try:
    dataframe = load_dataset(
        str(CSV_PATH)
    )

    image_index = create_image_index(
        str(IMAGE_FOLDER)
    )

except Exception as error:
    st.error(
        f"Application loading error: {error}"
    )

    st.stop()


# =========================================================
# 8. WEBSITE HEADER
# =========================================================
st.title(
    "🐾 Dog Skin Disease Dataset Review"
)

st.write(
    "This website displays selected dog skin images "
    "together with synthetically generated clinical data "
    "for veterinary review."
)

st.warning(
    "The displayed clinical details are synthetically "
    "generated scenarios. They are not confirmed as the "
    "real medical history of the dog shown in the image. "
    "Please assess whether the image label and clinical "
    "information are medically plausible."
)


# =========================================================
# 9. SIDEBAR FILTER
# =========================================================
st.sidebar.title("Navigation")

selected_disease = st.sidebar.selectbox(
    "Select disease class",
    options=["All"] + TARGET_CLASSES
)

if selected_disease == "All":
    filtered_df = dataframe.copy()
else:
    filtered_df = dataframe[
        dataframe["Label"] == selected_disease
    ].copy()

filtered_df = filtered_df.reset_index(
    drop=True
)

if filtered_df.empty:
    st.warning(
        "No records found for the selected class."
    )

    st.stop()


# =========================================================
# 10. SESSION STATE
# =========================================================
if "selected_disease_previous" not in st.session_state:
    st.session_state.selected_disease_previous = (
        selected_disease
    )

if (
    st.session_state.selected_disease_previous
    != selected_disease
):
    st.session_state.case_index = 0

    st.session_state.selected_disease_previous = (
        selected_disease
    )

if "case_index" not in st.session_state:
    st.session_state.case_index = 0

if st.session_state.case_index >= len(filtered_df):
    st.session_state.case_index = 0


# =========================================================
# 11. CASE SELECTOR
# =========================================================
case_options = []

for index, row in filtered_df.iterrows():
    case_options.append(
        f"{row['Case_ID']} | "
        f"{row['Label']} | "
        f"{row['Image_Name']}"
    )

selected_case = st.sidebar.selectbox(
    "Select case",
    options=list(range(len(case_options))),
    format_func=lambda index: case_options[index],
    index=st.session_state.case_index
)

st.session_state.case_index = selected_case


# =========================================================
# 12. PREVIOUS AND NEXT BUTTONS
# =========================================================
previous_column, progress_column, next_column = (
    st.columns([1, 3, 1])
)

with previous_column:
    previous_disabled = (
        st.session_state.case_index == 0
    )

    if st.button(
        "← Previous",
        use_container_width=True,
        disabled=previous_disabled
    ):
        st.session_state.case_index -= 1
        st.rerun()

with progress_column:
    st.markdown(
        f"""
        <div style="
            text-align: center;
            font-size: 18px;
            padding-top: 8px;
        ">
            <strong>
                Case {
                    st.session_state.case_index + 1
                } of {len(filtered_df)}
            </strong>
        </div>
        """,
        unsafe_allow_html=True
    )

with next_column:
    next_disabled = (
        st.session_state.case_index
        == len(filtered_df) - 1
    )

    if st.button(
        "Next →",
        use_container_width=True,
        disabled=next_disabled
    ):
        st.session_state.case_index += 1
        st.rerun()


# =========================================================
# 13. SELECT CURRENT RECORD
# =========================================================
record = filtered_df.iloc[
    st.session_state.case_index
]

case_id = get_value(
    record,
    "Case_ID"
)

image_name = get_value(
    record,
    "Image_Name"
)

disease_label = get_value(
    record,
    "Label"
)

image_path = image_index.get(
    image_name.lower().strip()
)


# =========================================================
# 14. DISPLAY CASE
# =========================================================
st.divider()

image_column, data_column = st.columns(
    [1, 1.3],
    gap="large"
)


# =========================================================
# 15. DISPLAY IMAGE
# =========================================================
with image_column:
    st.subheader("Dog Skin Image")

    st.write(
        f"**Case ID:** `{case_id}`"
    )

    st.write(
        f"**Image name:** `{image_name}`"
    )

    st.write(
        f"**Dataset label:** {disease_label}"
    )

    if image_path is None:
        st.error(
            f"No matching image found for "
            f"`{image_name}`."
        )

    else:
        try:
            image = Image.open(image_path)

            image.load()

            st.image(
                image,
                use_column_width=True
            )

        except UnidentifiedImageError:
            st.error(
                "The image file cannot be opened."
            )

        except Exception as error:
            st.error(
                f"Image loading error: {error}"
            )


# =========================================================
# 16. DISPLAY CLINICAL DATA TABLE
# =========================================================
with data_column:
    st.subheader(
        "Generated Clinical Information"
    )

    display_columns = [
        (
            "Disease label",
            "Label"
        ),
        (
            "Age in months",
            "Age_Months"
        ),
        (
            "Breed type",
            "Breed_Type"
        ),
        (
            "Itching severity",
            "Itching_Severity"
        ),
        (
            "Lesion location",
            "Lesion_Location"
        ),
        (
            "Hair loss",
            "Hair_Loss"
        ),
        (
            "Symptom duration",
            "Symptom_Duration_Days"
        ),
        (
            "Clinical text",
            "Clinical_Text"
        )
    ]

    table_data = []

    for display_name, column_name in display_columns:
        table_data.append(
            {
                "Field": display_name,
                "Value": get_value(
                    record,
                    column_name
                )
            }
        )

    table_df = pd.DataFrame(
        table_data
    )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Field": st.column_config.TextColumn(
                "Clinical field",
                width="medium"
            ),
            "Value": st.column_config.TextColumn(
                "Dataset value",
                width="large"
            )
        }
    )


# =========================================================
# 17. EMAIL CORRECTION INSTRUCTIONS
# =========================================================
st.divider()

st.subheader(
    "How to Provide Corrections"
)

st.write(
    "Please send corrections by email. "
    "Include the Case ID or Image Name "
    "so that the record can be identified."
)

email_template = f"""
Case ID: {case_id}
Image Name: {image_name}
Dataset Label: {disease_label}

Image label assessment:
Correct / Incorrect / Uncertain

Suggested disease label:
[Enter corrected disease if required]

Clinical data assessment:
Realistic / Partly realistic / Unrealistic

Metadata corrections:
[Enter corrections]

Additional comments:
[Enter any additional comments]
""".strip()

st.code(
    email_template,
    language=None
)


# =========================================================
# 18. DATASET SUMMARY
# =========================================================
with st.expander(
    "View sample dataset summary"
):
    summary_df = (
        dataframe["Label"]
        .value_counts()
        .reindex(
            TARGET_CLASSES,
            fill_value=0
        )
        .rename_axis("Disease class")
        .reset_index(
            name="Number of cases"
        )
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        f"**Total sample records:** "
        f"{len(dataframe)}"
    )

    st.write(
        f"**Total images found:** "
        f"{len(image_index)}"
    )