import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from fpdf import FPDF
import base64
from rapidfuzz import process
import urllib.parse


# Load dataset
df = pd.read_csv("Food Ingredients and Recipe Dataset with Image Name Mapping.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]
df = df.rename(columns={
    "title": "title",
    "ingredients": "ingredients",
    "cleaned_ingredients": "cleaned_ingredients",
    "instructions": "instructions",
    "image_name": "image_name"
})

# Drop rows with missing essential fields
df.dropna(subset=["ingredients", "instructions"], inplace=True)

# Optionally fill any other missing values
df.fillna("", inplace=True)

# Safety check
required_columns = ['title', 'ingredients', 'cleaned_ingredients', 'instructions', 'image_name']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in CSV: {missing_cols}")
    st.stop()

# Cuisine inference
def infer_cuisine(ingredients):
    text = ingredients.lower()
    if any(x in text for x in ['soy sauce', 'miso', 'rice vinegar']):
        return "Asian"
    elif any(x in text for x in ['basil', 'parmesan', 'mozzarella']):
        return "Italian"
    elif any(x in text for x in ['cumin', 'coriander', 'turmeric']):
        return "Indian"
    elif any(x in text for x in ['butter', 'thyme', 'wine']):
        return "French"
    elif any(x in text for x in ['cheddar', 'barbecue', 'ranch']):
        return "American"
    return "Other"

df['cuisine'] = df['ingredients'].apply(infer_cuisine)
df = df[df['cuisine'] != "Other"]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['ingredients'], df['cuisine'], test_size=0.2, random_state=42
)

# Build pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


def estimate_difficulty(ingredients, instructions):
    ing_count = len(ingredients.split(','))
    step_count = len(instructions.split('.'))
    if ing_count <= 5 and step_count <= 4:
        return "Easy"
    elif ing_count <= 10 and step_count <= 8:
        return "Medium"
    else:
        return "Hard"

df['difficulty'] = df.apply(lambda x: estimate_difficulty(x['ingredients'], x['instructions']), axis=1)

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Use ingredients + instructions as input
df['combined_text'] = df['ingredients'] + " " + df['instructions']

# Train model
X_train_dif, X_test_dif, y_train_dif, y_test_dif = train_test_split(
    df['combined_text'], df['difficulty'], test_size=0.2, random_state=42
)

difficulty_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
difficulty_model.fit(X_train_dif, y_train_dif)

# Evaluate
y_pred_dif = difficulty_model.predict(X_test_dif)
print("📊 Difficulty Classification Report:")
print(classification_report(y_test_dif, y_pred_dif))

# Save
joblib.dump(difficulty_model, "difficulty_classifier.pkl")


def estimate_time(instructions):
    word_count = len(instructions.split())
    step_count = len(instructions.split('.'))
    return int((word_count / 10) + (step_count * 2))

df['estimated_time'] = df['instructions'].apply(estimate_time)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(
    df['instructions'], df['estimated_time'], test_size=0.2, random_state=42
)

time_model = make_pipeline(TfidfVectorizer(), Ridge())
time_model.fit(X_train_time, y_train_time)

# Evaluate
y_pred_time = time_model.predict(X_test_time)
print("📈 Time Estimation MAE:", mean_absolute_error(y_test_time, y_pred_time))
print("📈 Time Estimation R² Score:", r2_score(y_test_time, y_pred_time))

# Save
joblib.dump(time_model, "time_estimator.pkl")


# Helper function for keyword-based labeling
def keyword_match(ingredients, keywords):
    if pd.isna(ingredients):
        return False
    return any(kw in ingredients.lower() for kw in keywords)

# Label vegetarian
veg_keywords = ['tofu', 'mushroom', 'broccoli', 'spinach', 'vegetable', 'eggplant', 'lentil', 'chickpea']
meat_keywords = ['chicken', 'beef', 'pork', 'bacon', 'fish', 'shrimp', 'lamb', 'steak', 'salmon', 'fish', 'cod', 'ribs']
df['vegetarian_label'] = df['cleaned_ingredients'].apply(
    lambda x: keyword_match(x, veg_keywords) and not keyword_match(x, meat_keywords)
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Split data
X_train_veg, X_test_veg, y_train_veg, y_test_veg = train_test_split(
    df['cleaned_ingredients'], df['vegetarian_label'], test_size=0.2, random_state=42
)

# Train model
veg_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
veg_model.fit(X_train_veg, y_train_veg)

# Evaluate
y_pred_veg = veg_model.predict(X_test_veg)
print("🥦 Vegetarian Classifier Report:")
print(classification_report(y_test_veg, y_pred_veg))

# Save the model
joblib.dump(veg_model, "vegetarian_classifier.pkl")


# Label spicy
spicy_keywords = ['chili', 'chilli', 'hot sauce', 'jalapeno', 'pepper flakes', 'sriracha']
df['spicy_label'] = df['cleaned_ingredients'].apply(lambda x: keyword_match(x, spicy_keywords))

X_train_spicy, X_test_spicy, y_train_spicy, y_test_spicy = train_test_split(
    df['cleaned_ingredients'], df['spicy_label'], test_size=0.2, random_state=42
)

spicy_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
spicy_model.fit(X_train_spicy, y_train_spicy)
print("🌶️ Spicy Classifier:")
print(classification_report(y_test_spicy, spicy_model.predict(X_test_spicy)))
joblib.dump(spicy_model, "spicy_classifier.pkl")


# Label gluten-free
gluten_keywords = ['wheat', 'flour', 'bread', 'barley', 'pasta', 'noodle', 'cracker', 'rye', 'cereals', 'bagels', 'muffins', 'cookies', 'cakes', 'brownies', 'biscuits', 'tortillas', 'beer', 'wine']
df['gluten_free'] = df['cleaned_ingredients'].apply(lambda x: not keyword_match(x, gluten_keywords))

X_train_gluten, X_test_gluten, y_train_gluten, y_test_gluten = train_test_split(
    df['cleaned_ingredients'], df['gluten_free'], test_size=0.2, random_state=42
)

gluten_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
gluten_model.fit(X_train_gluten, y_train_gluten)
print("🍞 Gluten-Free Classifier:")
print(classification_report(y_test_gluten, gluten_model.predict(X_test_gluten)))
joblib.dump(gluten_model, "gluten_free_classifier.pkl")


# Label high-protein
protein_keywords = ['chicken', 'tofu', 'lentil', 'egg', 'beans', 'fish', 'beef']
df['high_protein'] = df['cleaned_ingredients'].apply(lambda x: keyword_match(x, protein_keywords))

X_train_protein, X_test_protein, y_train_protein, y_test_protein = train_test_split(
    df['cleaned_ingredients'], df['high_protein'], test_size=0.2, random_state=42
)

protein_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
protein_model.fit(X_train_protein, y_train_protein)
print("💪 High-Protein Classifier:")
print(classification_report(y_test_protein, protein_model.predict(X_test_protein)))
joblib.dump(protein_model, "high_protein_classifier.pkl")


# Label dairy-free
dairy_keywords = ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'dressings', 'potato', 'chocolate', 'caramel', 'gum', 'cereals', 'bread', 'tortillas', 'cookies', 'cakes',]
df['dairy_free'] = df['cleaned_ingredients'].apply(lambda x: not keyword_match(x, dairy_keywords))

X_train_dairy, X_test_dairy, y_train_dairy, y_test_dairy = train_test_split(
    df['cleaned_ingredients'], df['dairy_free'], test_size=0.2, random_state=42
)

dairy_model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
dairy_model.fit(X_train_dairy, y_train_dairy)
print("🥛 Dairy-Free Classifier:")
print(classification_report(y_test_dairy, dairy_model.predict(X_test_dairy)))
joblib.dump(dairy_model, "dairy_free_classifier.pkl")



# Streamlit config
st.set_page_config(page_title="Smart Recipe Recommender", layout="wide")

st.markdown("""
    <style>
        h1, h3 {
            color: #2E7D32;
        }
        .css-1v0mbdj {
            font-size: 18px !important;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #4CAF50;
            color: white;
            padding: 0.5em 1.2em;
            font-size: 1rem;
        }
        .stDownloadButton>button {
            border-radius: 6px;
            background-color: #2196F3;
        }
        .recipe-box {
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🍽️ CulinAIre</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Find delicious recipes based on what you have — powered by AI!</p>", unsafe_allow_html=True)


# Sidebar filters
search_recipe_name = st.sidebar.text_input("🔍 Search by Recipe Name")

st.sidebar.title("🔍 Filters")
cuisine_filter = st.sidebar.selectbox("🌍 Cuisine", ["All"] + sorted(df['cuisine'].unique().tolist()))
difficulty_filter = st.sidebar.selectbox("📈 Difficulty", ["All"] + sorted(df['difficulty'].unique().tolist()))
veg = st.sidebar.checkbox("🌱 Vegetarian")
spicy = st.sidebar.checkbox("🌶️ Spicy")
under_30 = st.sidebar.checkbox("⏱️ Under 30 mins")
gluten_free = st.sidebar.checkbox("🚫 Gluten-Free")
high_protein = st.sidebar.checkbox("💪 High Protein")
dairy_free = st.sidebar.checkbox("🥛 Dairy-Free")

# Input
ingredients_input = st.text_input("Enter ingredients (comma-separated)", placeholder="e.g., chicken, garlic, tomato")

# Optional: Fuzzy-match ingredients input to improve accuracy
def fuzzy_match_ingredients(user_input, ingredient_list, threshold=80):
    matched = []
    for ing in user_input:
        best_match = process.extractOne(ing, ingredient_list, score_cutoff=threshold)
        if best_match:
            matched.append(best_match[0])
    return matched

def recommend(user_input, top_n=5):
    user_input = user_input.lower().split(',')
    user_input = [word.strip() for word in user_input]

    all_ingredients = set(','.join(df['cleaned_ingredients'].dropna()).split(','))
    user_input = fuzzy_match_ingredients(user_input, all_ingredients)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df['ingredients'])
    input_vec = vectorizer.transform([", ".join(user_input)])

    sim = cosine_similarity(input_vec, vectors).flatten()
    df['similarity'] = sim
    results = df.sort_values(by='similarity', ascending=False).head(50)

    if veg:
        results = results[results['vegetarian_label'] == True]
    if spicy:
        results = results[results['spicy_label'] == True]
    if gluten_free:
        results = results[results['gluten_free'] == True]
    if high_protein:
        results = results[results['high_protein'] == True]
    if dairy_free:
        results = results[results['dairy_free'] == True]
    if under_30:
        results = results[results['estimated_time'] <= 30]
    if cuisine_filter != "All":
        results = results[results['cuisine'] == cuisine_filter]
    if difficulty_filter != "All":
        results = results[results['difficulty'] == difficulty_filter]
    if search_recipe_name:
        results = results[results['title'].str.contains(search_recipe_name, case=False, na=False)]

    return results.head(top_n)

def clean_text(text):
    return text.replace("•", "-").encode('latin-1', errors='ignore').decode('latin-1')

from fpdf import FPDF
import os
import unicodedata
import re

# Helper to clean text (remove emojis and unsupported unicode)
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'[^\x00-\x7F]+','', text)
    return text

def create_pdf(recipe):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Fonts
    pdf.set_font("Arial", "B", 16)
    pdf.set_fill_color(220, 220, 255)  # Light blue background for title
    pdf.cell(0, 15, clean_text(recipe["title"]), ln=True, align="C", fill=True)

    pdf.ln(5)

    # Recipe Image (if exists)
    image_filename = recipe["image_name"].strip()
    image_path = os.path.join("Food Images", image_filename + ".jpg")
    if os.path.exists(image_path):
        pdf.image(image_path, w=100, x=pdf.w / 2 - 50)
        pdf.ln(10)

    # Ingredients Section
    pdf.set_font("Arial", "B", 13)
    pdf.set_text_color(0)
    pdf.cell(0, 10, "Ingredients:", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, clean_text(recipe["ingredients"]))
    pdf.ln(4)

    # Instructions Section
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Instructions:", ln=True)
    pdf.set_font("Arial", "", 11)
    instructions_list = [s.strip() for s in recipe["instructions"].split('.') if s.strip()]
    for i, step in enumerate(instructions_list, 1):
        pdf.multi_cell(0, 8, f"{i}. {clean_text(step)}")
        pdf.ln(1)

    # Output
    output_path = "recipe.pdf"
    pdf.output(output_path)
    return output_path

query_params = st.query_params
shared_title = query_params.get("recipe")

if shared_title:
    shared_title = urllib.parse.unquote(shared_title)  # Decode URL title

    # Load matching recipe
    matches = df[df['title'].str.lower() == shared_title.lower()]
    if not matches.empty:
        recipe = matches.iloc[0]

        st.subheader(recipe["title"])
        st.image(os.path.join("Food Images", recipe["image_name"] + ".jpg"), use_container_width=True)

        st.markdown("### 🧂 Ingredients")
        st.markdown(recipe["ingredients"])

        st.markdown("### 👨‍🍳 Instructions")
        instructions_list = [f"➤ {s.strip()}" for s in recipe["instructions"].split('.') if s.strip()]
        st.markdown("<br>".join(instructions_list), unsafe_allow_html=True)

        # PDF & Share
        col1, col2 = st.columns(2)
        with col1:
            pdf_path = create_pdf(recipe)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download PDF",
                    data=f.read(),
                    file_name=f"{recipe['title']}.pdf",
                    mime="application/pdf",
                    key=f"dl_shared_{recipe['title']}"
                )
        with col2:
            st.markdown(f"🔗 Shared via: `{shared_title}`")

    else:
        st.error("🚫 No recipe found matching the shared title.")



import urllib.parse

def create_share_url(recipe_title):
    encoded_title = urllib.parse.quote(recipe_title)
    return f"http://localhost:8501/?recipe={encoded_title}"  # Replace with your actual deployed URL

query_params = st.query_params
shared_recipe = None

if 'recipe' in query_params:
    shared_title = query_params['recipe'][0].replace('+', ' ')
    
    # Read CSV and clean columns
    df = pd.read_csv("Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
    df.columns = df.columns.str.strip().str.lower()  # Normalize headers

    # Check available columns (debug line)
    st.write("CSV Columns:", df.columns.tolist())

    if 'title' in df.columns:
        matches = df[df['title'].str.lower() == shared_title.lower()]
        if not matches.empty:
            shared_recipe = matches.iloc[0]
            st.success(f"🍽️ You opened a shared recipe: **{shared_recipe['title']}**")
        else:
            st.warning("No recipe found matching the shared title.")
    else:
        st.error("The 'title' column is missing from your dataset.")

if shared_recipe is not None:
    # === Show recipe image and content ===
    col_img, col_info = st.columns([1, 2])

    with col_img:
        image_filename = shared_recipe["image_name"].strip()
        image_path = os.path.join("Food Images", image_filename + ".jpg")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("⚠️ Image not found.")

    with col_info:
        st.markdown(f"### {shared_recipe['title']}")
        st.markdown("**🧂 Ingredients:**")
        st.markdown(", ".join([f"`{i.strip()}`" for i in shared_recipe['ingredients'].split(',') if i.strip()]))

        st.markdown("**👨‍🍳 Instructions:**")
        st.markdown("<br>".join([f"➤ {s.strip()}" for s in shared_recipe['instructions'].split('.') if s.strip()]), unsafe_allow_html=True)

        # PDF + Share buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            pdf_path = create_pdf(shared_recipe)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download PDF",
                    data=f.read(),
                    file_name=f"{shared_recipe['title']}.pdf",
                    mime="application/pdf",
                    key=f"dl_{shared_recipe['title']}"
                )
        with col2:
            # Inside `col2` for share button:
            share_url = create_share_url(shared_recipe["title"])
            st.markdown(f"[🔗 Share Recipe]({share_url})", unsafe_allow_html=True)


if not shared_title:
    if st.button("🍽️ Find Recipes"):

        if not ingredients_input.strip():
            st.warning("Please enter at least one ingredient.")
        else:
            results = recommend(ingredients_input)
            if len(results) == 0:
                st.info("😕 No recipes matched your filters. Try removing some.")
            else:
                for _, recipe in results.iterrows():
                    with st.container():
                        col_img, col_info = st.columns([1, 2])

                        with col_img:
                            image_filename = recipe["image_name"].strip()
                            image_path = os.path.join("Food Images", image_filename + ".jpg")
                            if os.path.exists(image_path):
                                st.image(image_path, use_container_width=True)
                            else:
                                st.warning("⚠️ Image not found.")

                        with col_info:
                            st.markdown(f"### {recipe['title']}")
                            st.markdown("**🧂 Ingredients:**")
                            st.markdown(", ".join([f"`{i.strip()}`" for i in recipe['ingredients'].split(',') if i.strip()]))

                            st.markdown("**👨‍🍳 Instructions:**")
                            st.markdown("<br>".join([f"➤ {s.strip()}" for s in recipe['instructions'].split('.') if s.strip()]), unsafe_allow_html=True)

                            # PDF + Share buttons
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                pdf_path = create_pdf(recipe)
                                with open(pdf_path, "rb") as f:
                                    st.download_button(
                                        label="📄 Download PDF",
                                        data=f.read(),
                                        file_name=f"{recipe['title']}.pdf",
                                        mime="application/pdf",
                                        key=f"dl_{recipe['title']}"
                                    )
                            with col2:
                                # Inside `col2` for share button:
                                share_url = create_share_url(recipe["title"])
                                st.markdown(f"[🔗 Share Recipe]({share_url})", unsafe_allow_html=True)