import streamlit as st
import google.generativeai as genai
import pandas as pd
from PIL import Image
import io
import fitz  # PyMuPDF

# --- Page Configuration ---
st.set_page_config(page_title="المصحح الذكي V13", layout="wide")

# --- UI Styling ---
st.markdown("""
<style>
    .stApp { direction: rtl; }
    h1, h2, h3, p, div, label, .stMarkdown, .stExpander, .stCheckbox { text-align: right; }
    .stDataFrame { direction: rtl; }
    .stRadio > label { font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # User-provided API Key
    user_api_key = st.text_input("🔑 أدخل مفتاح Gemini API الخاص بك:", type="password", help="المفتاح خاص بك ولا يتم حفظه على الخادم.")
    
    st.divider()
    # Restricted model selection
    model_name = st.selectbox("🧠 المحرك:", ["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
    
    if not user_api_key:
        st.warning("⚠️ يرجى إدخال مفتاح API للمتابعة.")

# --- Processing Functions ---
def smart_split(image, page_num):
    width, height = image.size
    if width > height * 1.1:
        right_half = image.crop((width // 2, 0, width, height)) 
        left_half = image.crop((0, 0, width // 2, height))     
        return [right_half, left_half], f"صفحة {page_num}: عريضة (تم قصها ✂️)"
    else:
        return [image], f"صفحة {page_num}: عادية (✅)"

def process_file_smartly(file):
    if file is None: return [], []
    raw_images = []
    logs = []
    final_processed_images = []
    
    if file.type in ['image/png', 'image/jpeg', 'image/jpg']:
        raw_images.append(Image.open(file))
    elif file.name.endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=200) 
            raw_images.append(Image.open(io.BytesIO(pix.tobytes())))
    
    for idx, img in enumerate(raw_images):
        split_imgs, log = smart_split(img, idx + 1)
        final_processed_images.extend(split_imgs)
        logs.append(log)
        
    return final_processed_images, logs

# --- Main Interface ---
st.title("🛡️ المصحح الذكي (V13 - Gemini 2.5)")

# --- Section 1: Answer Key Setup ---
st.subheader("1️⃣ إعدادات الإجابة النموذجية")

grading_mode = st.radio(
    "مصدر النموذج:",
    ("أ- لدي ملف إجابة نموذجية جاهز", 
     "ب- ليس لدي نموذج (توليد بالذكاء الاصطناعي)")
)

final_model_content = []  
q_file_uploaded = None 

if grading_mode == "أ- لدي ملف إجابة نموذجية جاهز":
    t_file = st.file_uploader("ارفع ملف النموذج", type=['pdf', 'png', 'jpg'])
    if t_file:
        model_images, _ = process_file_smartly(t_file)
        if model_images:
            final_model_content = ["\n--- [صور النموذج المعتمد] ---", *model_images]
            st.success(f"✅ تم اعتماد ملف النموذج.")

else: 
    q_file_uploaded = st.file_uploader("ارفع ورقة الأسئلة فقط", type=['pdf', 'png', 'jpg'])
    if 'ai_generated_key' not in st.session_state:
        st.session_state.ai_generated_key = None

    if q_file_uploaded and user_api_key:
        if st.button("👁️ معاينة النموذج المولد"):
            with st.spinner("جاري حل الأسئلة بواسطة الذكاء الاصطناعي..."):
                try:
                    genai.configure(api_key=user_api_key)
                    model = genai.GenerativeModel(model_name)
                    q_imgs, _ = process_file_smartly(q_file_uploaded)
                    res = model.generate_content(["حل هذا الامتحان بدقة ليكون نموذجاً للتصحيح.", *q_imgs])
                    st.session_state.ai_generated_key = res.text
                except Exception as e:
                    st.error(f"حدث خطأ: {e}")
    
    if st.session_state.ai_generated_key:
        with st.expander("عرض النموذج المولد"): st.markdown(st.session_state.ai_generated_key)
        final_model_content = ["\n--- [نموذج مولد] ---", st.session_state.ai_generated_key]

# --- Section 2: Student Submissions ---
st.divider()
st.subheader("2️⃣ ملفات الطلاب")
student_files = st.file_uploader("رفع إجابات الطلاب", type=['pdf', 'png', 'jpg'], accept_multiple_files=True)

# --- Section 3: Grading Execution ---
if st.button("🚀 بدء عملية التصحيح"):
    if not user_api_key:
        st.error("⚠️ يرجى إدخال مفتاح API الخاص بك في القائمة الجانبية.")
        st.stop()
    
    if not final_model_content and q_file_uploaded:
        # Auto-generate key if not already done
        with st.spinner("⏳ جاري توليد نموذج الإجابة أولاً..."):
            try:
                genai.configure(api_key=user_api_key)
                model = genai.GenerativeModel(model_name)
                q_file_uploaded.seek(0)
                q_imgs, _ = process_file_smartly(q_file_uploaded)
                res = model.generate_content(["حل هذا الامتحان بدقة.", *q_imgs])
                final_model_content = ["\n--- [نموذج مولد] ---", res.text]
            except Exception as e:
                st.error(f"فشل التوليد: {e}")
                st.stop()

    if not final_model_content:
        st.error("⚠️ يرجى توفير ملف إجابة نموذجية أو ورقة أسئلة.")
    elif not student_files:
        st.error("⚠️ يرجى رفع ملفات الطلاب.")
    else:
        genai.configure(api_key=user_api_key)
        model = genai.GenerativeModel(model_name)
        results = []
        progress_bar = st.progress(0)
        
        for i, s_file in enumerate(student_files):
            s_images, _ = process_file_smartly(s_file)
            try:
                prompt = [
                    "أنت مصحح امتحانات خبير. قارن إجابة الطالب بالنموذج. التنسيق: الاسم | العلامة | الملاحظة",
                    *final_model_content,
                    "\n--- [إجابة الطالب] ---",
                    *s_images
                ]
                response = model.generate_content(prompt)
                raw_text = response.text
                
                # Simple parsing logic
                try:
                    parts = raw_text.split('\n')[0].split('|')
                    name = parts[0].strip()
                    grade = parts[1].strip()
                except:
                    name, grade = "غير معروف", "تحتاج مراجعة"
                
                results.append({"الملف": s_file.name, "الطالب": name, "الدرجة": grade, "التفاصيل": raw_text})
            except Exception as e:
                results.append({"الملف": s_file.name, "الطالب": "خطأ", "الدرجة": "0", "التفاصيل": str(e)})
            
            progress_bar.progress((i + 1) / len(student_files))
        
        st.success("✅ اكتمل التصحيح!")
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        # Download results
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        st.download_button("📥 تحميل النتائج (Excel)", output.getvalue(), "Grades_V13.xlsx")
