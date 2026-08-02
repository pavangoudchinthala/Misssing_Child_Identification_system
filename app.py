import os
import sqlite3
import datetime
import pandas as pd
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from deepface import DeepFace

# ==========================================
# 1. DIRECTORY & DATABASE INITIALIZATION
# ==========================================
DB_NAME = "missing_children.db"
REGISTERED_DIR = "registered_children"
SPOTTED_DIR = "spotted_children"

os.makedirs(REGISTERED_DIR, exist_ok=True)
os.makedirs(SPOTTED_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS missing_children (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            guardian_name TEXT NOT NULL,
            guardian_contact TEXT NOT NULL,
            last_seen_location TEXT NOT NULL,
            date_registered TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sighting_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER,
            confidence_score REAL,
            spotted_location TEXT,
            sighting_time TEXT,
            spotted_image_path TEXT,
            FOREIGN KEY (child_id) REFERENCES missing_children (id)
        )
    ''')
    conn.commit()
    conn.close()

def get_quick_stats():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM missing_children")
    total_missing = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM sighting_logs")
    total_sightings = cursor.fetchone()[0]
    conn.close()
    return total_missing, total_sightings

init_db()

# ==========================================
# 2. AI MODEL CACHING (SPEED OPTIMIZATION)
# ==========================================
@st.cache_resource
def load_ai_model():
    """Builds and caches the FaceNet neural network model in RAM on startup."""
    return DeepFace.build_model("Facenet")

# Warm up model engine on application load
try:
    load_ai_model()
except Exception:
    pass

# ==========================================
# 3. STREAMLIT CONFIG & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Missing Child Identification System",
    page_icon="👶",
    layout="wide"
)

# Custom Main Page & Sidebar CSS
st.markdown("""
<style>
    /* Main Layout Tweaks */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .hero-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .hero-header h1 {
        color: #ffffff !important;
        margin-bottom: 0.2rem;
        font-weight: 700;
    }
    .hero-header p {
        color: #e0e6ed !important;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Sidebar Radio Option Cards */
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #1e2230;
        border: 1px solid #2e3548;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        transition: all 0.25s ease-in-out;
        cursor: pointer;
    }
    
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: #3b82f6;
        background-color: #252c40;
        transform: translateX(4px);
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        border-color: #3b82f6;
        background-color: #1e293b;
    }

    /* Stat Cards Styling */
    .stat-card {
        background: linear-gradient(135deg, #1e2230 0%, #171a24 100%);
        border: 1px solid #2e3548;
        border-radius: 12px;
        padding: 12px 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .stat-number {
        font-size: 24px;
        font-weight: 700;
        color: #60a5fa;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 11px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Status Badge Styling */
    .status-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10b981;
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Top Hero Banner Header
st.markdown("""
<div class="hero-header">
    <h1>👶 Missing Child Identification Portal</h1>
    <p>AI-Powered Facial Recognition & Real-Time Citizen Assistance System</p>
</div>
""", unsafe_allow_html=True)

# Fetch current database quick metrics
total_missing, total_sightings = get_quick_stats()

# ==========================================
# 4. DASHBOARD SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 30px;">👶</span>
            <div>
                <h3 style="margin: 0; padding: 0; font-size: 20px; font-weight: 700;">Navigation</h3>
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">Select System Module</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    choice = st.radio(
        "Navigation",
        [
            "📌  Register Missing Child", 
            "🔍  Spot & Match Child", 
            "📋  View Database Records", 
            "🏥  Child Welfare & Shelter Locator"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
        <p style='font-size: 14px; font-weight: 600; margin-bottom: 12px; color: #e2e8f0;'>
            📊 System Database Stats
        </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_missing}</div>
                <div class="stat-label">Registered</div>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_sightings}</div>
                <div class="stat-label">Sightings</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
        <div class="status-badge">
            <span class="status-dot"></span>
            <span>System Status: Online (SQLite)</span>
        </div>
    """, unsafe_allow_html=True)

# Clean up module selection string matching
selected_module = choice.strip()

# ==========================================
# 5. MODULE 1: REGISTER MISSING CHILD
# ==========================================
if "Register Missing Child" in selected_module:
    st.subheader("📌 Register a Missing / Found Child Report")
    st.caption("File a new case into the central database for immediate automated facial cross-matching.")
    
    st.info("💡 **Samaritan Guidance:** If you have found an unidentified child, fill in the estimated visual details and leave optional guardian fields blank.")
    
    with st.container(border=True):
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("##### 👤 Personal & Contact Details")
            child_name = st.text_input(
                "Child's Full Name", 
                value="", 
                placeholder="Enter child's name (Leave blank if unknown)"
            )
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                age = st.number_input("Estimated / Actual Age", min_value=1, max_value=18, value=5)
            with sub_col2:
                gender = st.selectbox("Gender", ["Male", "Female", "Unknown / Unspecified"])
                
            guardian_name = st.text_input(
                "Guardian / Parent Name (Optional)", 
                value="", 
                placeholder="Leave blank if unknown"
            )
            guardian_contact = st.text_input(
                "Contact Phone Number (Optional)", 
                value="", 
                placeholder="e.g. +91 9876543210"
            )
            last_seen_location = st.text_input(
                "Found / Last Seen Location", 
                value="", 
                placeholder="e.g. Secunderabad Station / Public Park"
            )
            
        with col2:
            st.markdown("##### 📸 Upload Reference Photograph")
            uploaded_img = st.file_uploader(
                "Select a clear facial image (JPG, PNG)", 
                type=["jpg", "jpeg", "png"]
            )
            
            if uploaded_img is not None:
                st.image(uploaded_img, caption="Preview of Selected Photograph", use_column_width=True)
            else:
                st.markdown(
                    "<div style='border: 2px dashed #4a5568; padding: 40px; text-align: center; border-radius: 8px; color: #a0aec0;'>"
                    "📸 Photo Preview Area<br/><small>Upload an image to see dynamic preview</small></div>", 
                    unsafe_allow_html=True
                )

        st.markdown("---")
        submit_btn = st.button("💾 Submit & Register Case", type="primary", use_container_width=True)

    if submit_btn:
        if uploaded_img:
            final_child_name = child_name.strip() if child_name.strip() else "Unidentified Child"
            final_guardian_name = guardian_name.strip() if guardian_name.strip() else "Unknown / Samaritan Report"
            final_guardian_contact = guardian_contact.strip() if guardian_contact.strip() else "N/A"
            final_location = last_seen_location.strip() if last_seen_location.strip() else "Unspecified Location"
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{final_child_name.replace(' ', '_')}_{timestamp}.jpg"
            save_path = os.path.join(REGISTERED_DIR, filename)
            
            # ⚡ Downscale image resolution for faster AI processing
            image = Image.open(uploaded_img)
            image.thumbnail((800, 800))
            image.convert("RGB").save(save_path)
            
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO missing_children 
                (child_name, age, gender, guardian_name, guardian_contact, last_seen_location, date_registered, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (final_child_name, age, gender, final_guardian_name, final_guardian_contact, final_location, 
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), save_path))
            
            conn.commit()
            conn.close()
            
            st.balloons()
            st.success(f"🎉 Case report for **{final_child_name}** successfully created & saved to database!")
            st.rerun()
        else:
            st.error("⚠️ Action Required: Please attach a photograph of the child before submitting.")

# ==========================================
# 6. MODULE 2: SPOT & MATCH CHILD
# ==========================================
elif "Spot & Match Child" in selected_module:
    st.subheader("🔍 Spot & Match Unidentified Child")
    st.caption("Perform AI facial feature extraction using FaceNet model against all registered missing records.")
    
    with st.container(border=True):
        st.markdown("##### 📍 Sighting Details & Camera Input")
        
        col_input1, col_input2 = st.columns([1, 1])
        
        with col_input1:
            input_method = st.radio(
                "Choose Photo Source:", 
                ["📷 Live Webcam Scan", "📁 Upload Sighting Photo"],
                horizontal=True
            )
            sighting_location = st.text_input("Spotted Location / Landmark", value="Public Field Sighting")
            
        spotted_img_path = None
        
        with col_input2:
            if input_method == "📷 Live Webcam Scan":
                camera_photo = st.camera_input("Take a photo of the spotted child")
                if camera_photo:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    spotted_img_path = os.path.join(SPOTTED_DIR, f"webcam_{timestamp}.jpg")
                    image = Image.open(camera_photo)
                    image.thumbnail((800, 800))
                    image.convert("RGB").save(spotted_img_path)
            else:
                uploaded_sighting = st.file_uploader("Upload sighting photo...", type=["jpg", "jpeg", "png"])
                if uploaded_sighting:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    spotted_img_path = os.path.join(SPOTTED_DIR, f"sighting_{timestamp}.jpg")
                    image = Image.open(uploaded_sighting)
                    image.thumbnail((800, 800))
                    image.convert("RGB").save(spotted_img_path)
                    st.image(uploaded_sighting, caption="Spotted Image Preview", width=220)

        st.markdown("---")
        run_match = st.button("🚀 Run Deep Learning AI Search", type="primary", use_container_width=True)

    if spotted_img_path and run_match:
        with st.spinner("🔍 Extracting Facial Embeddings & Computing Cosine Distance..."):
            
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT id, child_name, age, guardian_name, guardian_contact, image_path FROM missing_children")
            records = cursor.fetchall()
            conn.close()
            
            if not records:
                st.warning("⚠️ No missing children are currently registered in the database for comparison.")
            else:
                best_match = None
                highest_confidence = 0.0
                MATCH_THRESHOLD = 40.0
                
                for record in records:
                    child_id, reg_name, reg_age, reg_guardian, reg_contact, reg_img_path = record
                    
                    if os.path.exists(reg_img_path):
                        try:
                            # ⚡ Optimized verification with fast detection backend
                            result = DeepFace.verify(
                                img1_path=spotted_img_path,
                                img2_path=reg_img_path,
                                model_name="Facenet",
                                detector_backend="opencv",
                                distance_metric="cosine",
                                enforce_detection=False
                            )
                            
                            distance = result.get("distance", 1.0)
                            confidence = max(0.0, min(100.0, (1 - distance) * 100))
                            
                            if confidence > highest_confidence:
                                highest_confidence = confidence
                                best_match = record
                        except Exception:
                            continue

                if best_match and highest_confidence >= MATCH_THRESHOLD:
                    child_id, reg_name, reg_age, reg_guardian, reg_contact, reg_img_path = best_match
                    
                    st.error(f"🚨 MATCH CONFIRMED! Similarity Score: **{highest_confidence:.2f}%**")
                    st.progress(highest_confidence / 100)
                    
                    with st.container(border=True):
                        st.markdown("### 🏆 Matching Case File Details")
                        res_col1, res_col2 = st.columns([1, 1.2])
                        
                        with res_col1:
                            st.markdown("**Spotted Input Photo**")
                            st.image(spotted_img_path, use_column_width=True)
                            
                        with res_col2:
                            st.markdown("**Matched Registered Record**")
                            st.image(reg_img_path, width=220)
                            
                            st.markdown(f"**Child Name:** `{reg_name}`")
                            st.markdown(f"**Registered Age:** `{reg_age} years`")
                            st.markdown(f"**Guardian Name:** `{reg_guardian}`")
                            st.markdown(f"**Emergency Contact:** 📞 `{reg_contact}`")
                        
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO sighting_logs 
                        (child_id, confidence_score, spotted_location, sighting_time, spotted_image_path)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (child_id, highest_confidence, sighting_location, 
                          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), spotted_img_path))
                    conn.commit()
                    conn.close()
                    
                else:
                    st.success("✅ No Match Found in the Missing Children Database.")
                    if highest_confidence > 0:
                        st.info(f"Highest similarity observed was **{highest_confidence:.2f}%** (Below required threshold of {MATCH_THRESHOLD}%).")
                    
                    st.warning("👉 If you have found an unidentified lost child, please visit the **🏥 Child Welfare & Shelter Locator** tab to locate verified safe care centers.")

# ==========================================
# 7. MODULE 3: VIEW DATABASE RECORDS & ADMIN MANAGEMENT
# ==========================================
elif "View Database Records" in selected_module:
    st.subheader("📋 Missing Children Directory & Sighting Logs")
    st.caption("Public Directory: Citizens can view missing child reports and verified sighting logs in read-only mode.")
    
    conn = sqlite3.connect(DB_NAME)
    df_missing = pd.read_sql_query("SELECT * FROM missing_children", conn)
    df_sightings = pd.read_sql_query("SELECT * FROM sighting_logs", conn)
    conn.close()
    
    tab_view1, tab_view2, tab_view3 = st.tabs([
        "🖼️ Public Photo Gallery (Card View)", 
        "📊 Complete Database Table", 
        "📜 Sighting Audit Logs"
    ])
    
    # TAB 1: VISUAL GALLERY CARDS
    with tab_view1:
        search_query = st.text_input("🔍 Search Children by Name or Location:", placeholder="e.g. Rahul, Secunderabad, Park...").strip().lower()
        
        if not df_missing.empty:
            filtered_df = df_missing.copy()
            if search_query:
                filtered_df = filtered_df[
                    filtered_df['child_name'].str.lower().str.contains(search_query) |
                    filtered_df['last_seen_location'].str.lower().str.contains(search_query)
                ]
            
            if filtered_df.empty:
                st.info("No missing child reports match your search criteria.")
            else:
                cols = st.columns(2)
                for index, row in filtered_df.iterrows():
                    col_idx = index % 2
                    with cols[col_idx]:
                        with st.container(border=True):
                            card_col1, card_col2 = st.columns([1, 1.4])
                            
                            with card_col1:
                                if os.path.exists(row['image_path']):
                                    st.image(row['image_path'], use_column_width=True)
                                else:
                                    st.markdown("📷 *Photo Unavailable*")
                                    
                            with card_col2:
                                st.markdown(f"### 👶 {row['child_name']}")
                                st.markdown(f"**ID:** `MC-{row['id']}`")
                                st.markdown(f"**Age / Gender:** {row['age']} yrs | {row['gender']}")
                                st.markdown(f"**📍 Last Location:** {row['last_seen_location']}")
                                st.markdown(f"**👤 Guardian:** {row['guardian_name']}")
                                st.markdown(f"**📞 Emergency Contact:** `{row['guardian_contact']}`")
                                st.caption(f"🗓 Registered: {row['date_registered']}")
        else:
            st.info("No missing children are currently registered in the system.")

    # TAB 2: READ-ONLY DATABASE TABLE
    with tab_view2:
        if not df_missing.empty:
            st.dataframe(
                df_missing[['id', 'child_name', 'age', 'gender', 'guardian_name', 'guardian_contact', 'last_seen_location', 'date_registered']], 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info("No child records found in the database.")

    # TAB 3: READ-ONLY SIGHTING LOGS
    with tab_view3:
        if not df_sightings.empty:
            st.dataframe(df_sightings, use_container_width=True, hide_index=True)
        else:
            st.info("No verified sighting logs recorded yet.")

    # ADMIN CONTROLS
    st.markdown("<br/>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.subheader("⚙️ Database Management Controls")
        st.caption("🔒 Secured Administrator Zone: Enter password to edit or delete database records.")
        
        admin_password = st.text_input(
            "Enter Admin Password:", 
            type="password", 
            key="admin_del_pwd", 
            placeholder="Password required to make changes"
        )
        
        if admin_password == "admin123":
            st.success("🔓 Admin Authenticated. Modification tools unlocked.")
            
            tab1, tab2, tab3 = st.tabs([
                "🗑️ Delete Child Record", 
                "📜 Delete Sighting Log", 
                "⚠️ Reset Entire Database"
            ])
            
            with tab1:
                if not df_missing.empty:
                    options = {f"ID {row['id']} - {row['child_name']}": row['id'] for _, row in df_missing.iterrows()}
                    selected_label = st.selectbox("Select Child Record to Delete:", list(options.keys()))
                    selected_id = options[selected_label]
                    
                    if st.button("Delete Selected Case File", type="primary"):
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        cursor.execute("SELECT image_path FROM missing_children WHERE id = ?", (selected_id,))
                        row = cursor.fetchone()
                        if row and row[0] and os.path.exists(row[0]):
                            try:
                                os.remove(row[0])
                            except Exception:
                                pass
                        
                        cursor.execute("DELETE FROM missing_children WHERE id = ?", (selected_id,))
                        cursor.execute("DELETE FROM sighting_logs WHERE child_id = ?", (selected_id,))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"Child Record ID {selected_id} deleted successfully!")
                        st.rerun()
                else:
                    st.info("No child records available to delete.")

            with tab2:
                if not df_sightings.empty:
                    log_options = {f"Log ID {row['id']} (Child ID: {row['child_id']} @ {row['sighting_time']})": row['id'] for _, row in df_sightings.iterrows()}
                    selected_log_label = st.selectbox("Select Sighting Log to Delete:", list(log_options.keys()))
                    selected_log_id = log_options[selected_log_label]
                    
                    if st.button("Delete Selected Sighting Log"):
                        conn = sqlite3.connect(DB_NAME)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM sighting_logs WHERE id = ?", (selected_log_id,))
                        conn.commit()
                        conn.close()
                        
                        st.success(f"Sighting Log ID {selected_log_id} deleted successfully!")
                        st.rerun()
                else:
                    st.info("No sighting logs available to delete.")

            with tab3:
                st.warning("⚠️ Caution: Resetting the database will permanently delete ALL records and logs.")
                if st.button("Clear Entire Database", type="secondary"):
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM missing_children")
                    cursor.execute("DELETE FROM sighting_logs")
                    cursor.execute("DELETE FROM sqlite_sequence")
                    conn.commit()
                    conn.close()
                    
                    st.success("Database reset successfully.")
                    st.rerun()
                    
        elif admin_password != "":
            st.error("❌ Invalid Password. Modification and deletion features remain locked.")
        else:
            st.info("🔒 Public user mode active (Read-Only). Enter password (`admin123`) above to unlock administrative controls.")

# ==========================================
# 8. MODULE 4: CHILD WELFARE & SHELTER LOCATOR
# ==========================================
elif "Child Welfare & Shelter Locator" in selected_module:
    st.subheader("🏥 Nearest Child Welfare & Emergency Care Locator")
    st.caption("Find official legal care centers, Child Welfare Committees (CWC), and emergency shelters.")
    
    st.error("🚨 **NATIONAL EMERGENCY TOLL-FREE HELPLINES:** Call **1098** (Childline 24/7) or **112 / 100** (Police Emergency)")
    
    col_loc1, col_loc2 = st.columns([1, 2.2])
    
    with col_loc1:
        with st.container(border=True):
            st.markdown("##### 📍 Filter Region")
            selected_city = st.selectbox(
                "Select State / Region:",
                ["Hyderabad / Telangana", "Mumbai / Maharashtra", "Delhi / NCR", "Bengaluru / Karnataka", "Chennai / Tamil Nadu"]
            )
            st.markdown("""
            ---
            **Legal Samaritan Steps:**
            1. **Never take a lost child home.**
            2. Report immediately to legal authorities.
            3. Call **1098** for official shelter handover.
            """)
        
    welfare_data = {
        "Hyderabad / Telangana": [
            {
                "center": "Child Welfare Committee (CWC) - Hyderabad",
                "type": "Government CWC Office",
                "address": "Nimboliadda, Kachiguda, Hyderabad, Telangana 500027",
                "phone": "040-24651098",
                "shelter_status": "🟢 24/7 Active Shelter",
                "map_link": "https://maps.google.com/?q=Child+Welfare+Committee+Kachiguda+Hyderabad"
            },
            {
                "center": "District Child Protection Unit (DCPU)",
                "type": "Government Legal Care Center",
                "address": "Collectorate Complex, Lakdikapul, Hyderabad",
                "phone": "040-23201098",
                "shelter_status": "🟢 Active Office",
                "map_link": "https://maps.google.com/?q=District+Child+Protection+Unit+Hyderabad"
            },
            {
                "center": "Women & Children Safety Wing (She Teams)",
                "type": "Police Special Protection Unit",
                "address": "Integrated Command Control Centre, Banjara Hills, Hyderabad",
                "phone": "100 / 040-27852408",
                "shelter_status": "🟢 Active Police Helpline",
                "map_link": "https://maps.google.com/?q=Women+Child+Safety+Wing+Hyderabad"
            }
        ],
        "Mumbai / Maharashtra": [
            {
                "center": "Child Welfare Committee (CWC) - Suburban",
                "type": "Government CWC Office",
                "address": "Children's Aid Society, Mankhurd, Mumbai",
                "phone": "022-25563241",
                "shelter_status": "🟢 Active",
                "map_link": "https://maps.google.com/?q=CWC+Mankhurd+Mumbai"
            }
        ],
        "Delhi / NCR": [
            {
                "center": "CWC North Delhi",
                "type": "Government CWC Office",
                "address": "Nirmal Chhaya Complex, Jail Road, New Delhi",
                "phone": "011-28525308",
                "shelter_status": "🟢 Active",
                "map_link": "https://maps.google.com/?q=Nirmal+Chhaya+Complex+Delhi"
            }
        ],
        "Bengaluru / Karnataka": [
            {
                "center": "Child Welfare Committee - Bengaluru Urban",
                "type": "Government CWC Office",
                "address": "Hosur Road, Near Dairy Circle, Bengaluru",
                "phone": "080-26560946",
                "shelter_status": "🟢 Active",
                "map_link": "https://maps.google.com/?q=Child+Welfare+Committee+Bengaluru"
            }
        ],
        "Chennai / Tamil Nadu": [
            {
                "center": "Child Welfare Committee - Chennai",
                "type": "Government CWC Office",
                "address": "Royapuram, Chennai, Tamil Nadu",
                "phone": "044-25951098",
                "shelter_status": "🟢 Active",
                "map_link": "https://maps.google.com/?q=Child+Welfare+Committee+Royapuram+Chennai"
            }
        ]
    }
    
    with col_loc2:
        st.markdown(f"##### Recognized Centers in `{selected_city}`")
        centers = welfare_data.get(selected_city, [])
        
        for center in centers:
            with st.container(border=True):
                st.markdown(f"### 📌 {center['center']}")
                st.markdown(f"**Facility Type:** {center['type']}")
                st.markdown(f"**Address:** {center['address']}")
                st.markdown(f"**Status:** {center['shelter_status']}")
                st.markdown(f"**Emergency Contact:** 📞 `{center['phone']}`")
                st.link_button("🗺️ Open Google Maps Directions", center['map_link'])