import streamlit as st
import google.generativeai as genai
import pandas as pd
from PIL import Image
import io
import fitz  # PyMuPDF

# --- إعداد الصفحة ---
st.set_page_config(page_title="المصحح الذكي V13", layout="wide")

# جلب المفتاح
api_key = None
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except: pass

st.markdown("""
<style>
    .stApp { direction: rtl; }
    h1, h2, h3, p, div, label, .stMarkdown, .stExpander, .stCheckbox { text-align: right; }
    .stDataFrame { direction: rtl; }
    .stRadio > label { font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# --- القائمة الجانبية ---
with st.sidebar:
    st.header("⚙️ الإعدادات")
    if not api_key:
        api_key = st.text_input("🔑 مفتاح API:", type="password")
    
    st.divider()
    model_name = st.selectbox("🧠 المحرك:", ["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
    st.info("ℹ️ النظام يدعم التوليد التلقائي للنموذج عند بدء التصحيح.")

# --- دوال المعالجة ---
def smart_split(image, page_num):
    width, height = image.size
    actions_log = ""
    if width > height * 1.1:
        right_half = image.crop((width // 2, 0, width, height)) 
        left_half = image.crop((0, 0, width // 2, height))     
        actions_log = f"صفحة {page_num}: عريضة (تم قصها ✂️)"
        return [right_half, left_half], actions_log
    else:
        actions_log = f"صفحة {page_num}: عادية (✅)"
        return [image], actions_log

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

# --- الواجهة الرئيسية ---
st.title("🛡️ المصحح الذكي (V13 - التصحيح التلقائي)")

# --- القسم 1: المصدر ---
st.subheader("1️⃣ إعدادات الإجابة النموذجية")

grading_mode = st.radio(
    "مصدر النموذج:",
    ("أ- لدي ملف إجابة نموذجية جاهز", 
     "ب- ليس لدي نموذج (توليد بالذكاء الاصطناعي)")
)

final_model_content = []  
q_file_uploaded = None # لتخزين ورقة الأسئلة مؤقتاً

if grading_mode == "أ- لدي ملف إجابة نموذجية جاهز":
    t_file = st.file_uploader("ارفع ملف النموذج", type=['pdf', 'png', 'jpg'])
    if t_file:
        model_images, _ = process_file_smartly(t_file)
        if model_images:
            final_model_content = ["\n--- [صور النموذج المعتمد] ---", *model_images]
            st.success(f"✅ تم اعتماد الملف.")

else: # الخيار ب
    q_file_uploaded = st.file_uploader("ارفع ورقة الأسئلة فقط", type=['pdf', 'png', 'jpg'])
    
    # تهيئة المتغير في الذاكرة
    if 'ai_generated_key' not in st.session_state:
        st.session_state.ai_generated_key = None

    # زر اختياري للمعاينة (ليس إجبارياً الآن)
    if q_file_uploaded and api_key:
        if st.button("👁️ معاينة النموذج المولد (اختياري)"):
            with st.spinner("جاري التوليد للمعاينة..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                q_imgs, _ = process_file_smartly(q_file_uploaded)
                res = model.generate_content(["حل هذا الامتحان بدقة ليكون نموذجاً.", *q_imgs])
                st.session_state.ai_generated_key = res.text
    
    if st.session_state.ai_generated_key:
        with st.expander("عرض النموذج"): st.markdown(st.session_state.ai_generated_key)
        final_model_content = ["\n--- [نموذج مولد] ---", st.session_state.ai_generated_key]


# --- القسم 2: الطلاب ---
st.divider()
st.subheader("2️⃣ ملفات الطلاب")
student_files = st.file_uploader("رفع الإجابات", type=['pdf', 'png', 'jpg'], accept_multiple_files=True)

# --- القسم 3: التشغيل ---
if st.button("🚀 بدء التصحيح") and api_key:
    
    # --- المنطق الجديد: التوليد التلقائي إذا لزم الأمر ---
    # إذا اختار (ب) ورفع أسئلة لكن النموذج فارغ (لأنه لم يضغط زر المعاينة)
    if grading_mode == "ب- ليس لدي نموذج (توليد بالذكاء الاصطناعي)" and q_file_uploaded and not final_model_content:
        with st.spinner("⏳ لم يتم توليد النموذج مسبقاً.. جاري حل الامتحان الآن آلياً..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                # إعادة قراءة الملف لأنه قد يكون أغلق
                q_file_uploaded.seek(0)
                q_imgs, _ = process_file_smartly(q_file_uploaded)
                
                prompt_gen = ["قم بحل هذا الامتحان بدقة متناهية ليكون مرجعاً للتصحيح.", *q_imgs]
                response_gen = model.generate_content(prompt_gen)
                
                # حفظ النتيجة واعتمادها
                st.session_state.ai_generated_key = response_gen.text
                final_model_content = ["\n--- [نموذج مولد آلياً عند البدء] ---", response_gen.text]
                st.success("تم توليد النموذج بنجاح! ننتقل لتصحيح الطلاب...")
            except Exception as e:
                st.error(f"فشل في توليد النموذج: {e}")
                st.stop() # إيقاف البرنامج
    
    # --- التحقق النهائي قبل البدء ---
    if not final_model_content:
        if grading_mode == "أ- لدي ملف إجابة نموذجية جاهز":
            st.error("⚠️ الرجاء رفع ملف النموذج في الخطوة 1.")
        else:
            st.error("⚠️ الرجاء رفع ورقة الأسئلة ليتمكن النظام من حلها.")
    elif not student_files:
        st.error("⚠️ الرجاء رفع ملفات الطلاب.")
    else:
        # --- عملية التصحيح ---
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        results = []
        bar = st.progress(0)
        
        for i, s_file in enumerate(student_files):
            s_images, logs = process_file_smartly(s_file)
            
            try:
                prompt = [
                    """
                    أنت مصحح امتحانات.
                    قارن إجابة الطالب بالنموذج المرفق (سواء كان صوراً أو نصاً).
                    تنسيق السطر الأول: الاسم | العلامة | الملاحظة
                    """,
                    *final_model_content, # هنا أصبحنا نضمن وجود النموذج
                    "\n--- [ورقة الطالب] ---",
                    *s_images
                ]
                
                response = model.generate_content(prompt)
                text = response.text
                
                try:
                    line1 = text.split('\n')[0].split('|')
                    name, grade = line1[0].strip(), line1[1].strip()
                    note = line1[2].strip() if len(line1) > 2 else ""
                except:
                    name, grade, note = "تحقق", "غير محدد", "خطأ تنسيق"
                
                results.append({"الملف": s_file.name, "الطالب": name, "الدرجة": grade, "التفاصيل": text})
                
            except Exception as e:
                results.append({"الملف": s_file.name, "الطالب": "خطأ", "الدرجة": "0", "التفاصيل": str(e)})
            
            bar.progress((i + 1) / len(student_files))
        
        st.success("تم الانتهاء!")
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer: df.to_excel(writer, index=False)
        st.download_button("📥 Excel", buffer.getvalue(), "Grades_V13.xlsx")
        
        st.divider()
        for _, row in df.iterrows():
            with st.expander(f"{row['الطالب']}"): st.markdown(row['التفاصيل'])