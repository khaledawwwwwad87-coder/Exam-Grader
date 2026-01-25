import streamlit as st
import google.generativeai as genai
import pandas as pd
from PIL import Image
import io
import fitz  # PyMuPDF

# --- Page Configuration ---
st.set_page_config(page_title="المصحح الذكي ", layout="wide", page_icon="🎓")

# --- UI Styling ---
st.markdown("""
<style>
    .stApp { direction: rtl; }
    h1, h2, h3, p, div, label, .stMarkdown, .stExpander, .stCheckbox { text-align: right; }
    .stDataFrame { direction: rtl; }
    .stRadio > label { font-weight: bold; font-size: 1.1rem; color: #1E3A8A; }
    .main-title { color: #1E3A8A; text-align: center; border-bottom: 2px solid #E5E7EB; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3407/3407024.png", width=80)
    st.header("⚙️ الإعدادات التقنية")
    
    user_api_key = st.text_input("🔑 مفتاح Gemini API الخاص بك:", type="password")
    
    st.divider()
    model_name = st.selectbox("🧠 اختر المحرك:", ["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
    
    if not user_api_key:
        st.info("💡 لابد من إدخال المفتاح لتفعيل البرنامج.")

# --- Functions ---
def smart_split(image, page_num):
    width, height = image.size
    if width > height * 1.1:
        right_half = image.crop((width // 2, 0, width, height)) 
        left_half = image.crop((0, 0, width // 2, height))     
        return [right_half, left_half]
    return [image]

def process_file_smartly(file):
    if file is None: return []
    images = []
    if file.type in ['image/png', 'image/jpeg', 'image/jpg']:
        images.append(Image.open(file))
    elif file.name.endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=200) 
            images.append(Image.open(io.BytesIO(pix.tobytes())))
    
    final_imgs = []
    for idx, img in enumerate(images):
        final_imgs.extend(smart_split(img, idx + 1))
    return final_imgs

# --- Main App ---
st.markdown("<h1 class='main-title'>🛡️ المصحح الذكي - النسخة الأكاديمية</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1️⃣ مرجع التصحيح")
    grading_mode = st.radio("كيف تريد توفير الإجابة؟", ("ملف إجابة جاهز", "توليد الحل آلياً من الأسئلة"))
    
    q_file = st.file_uploader("ارفع (النموذج/الأسئلة)", type=['pdf', 'png', 'jpg'])
    
    # Session state for AI key
    if 'ai_key' not in st.session_state: st.session_state.ai_key = None

    if grading_mode == "توليد الحل آلياً من الأسئلة" and q_file and user_api_key:
        if st.button("✨ توليد نموذج الإجابة الآن"):
            with st.spinner("جاري التفكير..."):
                genai.configure(api_key=user_api_key)
                model = genai.GenerativeModel(model_name)
                imgs = process_file_smartly(q_file)
                res = model.generate_content(["حل هذا الامتحان بدقة.", *imgs])
                st.session_state.ai_key = res.text
                st.success("تم التوليد!")

    if st.session_state.ai_key:
        with st.expander("👁️ عرض نموذج الحل"): st.write(st.session_state.ai_key)

with col2:
    st.subheader("2️⃣ إجابات الطلاب")
    student_files = st.file_uploader("ارفع أوراق الطلاب", type=['pdf', 'png', 'jpg'], accept_multiple_files=True)

st.divider()

if st.button("🚀 ابدأ تصحيح جميع الأوراق"):
    if not user_api_key or not q_file or not student_files:
        st.error("الرجاء التأكد من إدخال المفتاح ورفع كافة الملفات.")
    else:
        # Configuration
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel(model_name)
        
        # Prepare Model Content
        if grading_mode == "ملف إجابة جاهز":
            model_content = process_file_smartly(q_file)
        else:
            model_content = [st.session_state.ai_key]

        results = []
        progress = st.progress(0)
        
        for idx, s_file in enumerate(student_files):
            s_imgs = process_file_smartly(s_file)
            prompt = ["أنت مصحح أكاديمي. قارن الورقة بالنموذج. التنسيق: الاسم | الدرجة | ملاحظة", *model_content, "---", *s_imgs]
            
            try:
                response = model.generate_content(prompt)
                results.append({"الملف": s_file.name, "التفاصيل": response.text})
            except Exception as e:
                results.append({"الملف": s_file.name, "التفاصيل": f"خطأ: {str(e)}"})
            
            progress.progress((idx + 1) / len(student_files))

        st.balloons()
        st.table(pd.DataFrame(results))
