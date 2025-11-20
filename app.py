import streamlit as st
import pandas as pd
import json
import os
import asyncio
import time
import io
import re
from openai import AsyncOpenAI, RateLimitError
from pypdf import PdfReader

# ---------------------------------------------------------
# [설정] 페이지 기본 세팅
# ---------------------------------------------------------
st.set_page_config(page_title="DB Inc 프롬프트 경진대회 채점기 v3.0 (Final)", layout="wide", page_icon="⚖️")

st.markdown("""
    <style>
    .status-box { 
        padding: 15px; border-radius: 8px; margin-bottom: 10px; text-align: center; 
        font-size: 1.1rem; background-color: #e3f2fd; border: 1px solid #90caf9; 
        color: #1565c0; font-weight: bold;
    }
    .success-box { background-color: #e8f5e9; color: #2e7d32; border-color: #c8e6c9; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [유틸] 파일 처리
# ---------------------------------------------------------
def read_file_content(file):
    if not file: return None
    ext = file.name.split('.')[-1].lower()
    try:
        if ext == 'pdf':
            reader = PdfReader(file)
            return "".join([page.extract_text() for page in reader.pages])
        elif ext in ['xlsx', 'xls']:
            sheets = pd.read_excel(file, sheet_name=None)
            text = []
            for name, df in sheets.items():
                text.append(f"### Sheet: {name}\n{df.to_markdown(index=False)}")
            return "\n\n".join(text)
        elif ext == 'csv':
            return pd.read_csv(file).to_markdown(index=False)
        else:
            return file.getvalue().decode("utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"

def load_golden_excel(file):
    if file and file.name.endswith('.xlsx'):
        return pd.read_excel(file, sheet_name=None)
    return None

# ---------------------------------------------------------
# [핵심 1] 개선된 CSV 파싱 및 매칭 로직 (0점 방지)
# ---------------------------------------------------------
def extract_csv_dataframes(text):
    """LLM 답변에서 여러 개의 CSV 블록을 유연하게 추출"""
    dfs = []
    # 1. ```csv 패턴 추출
    code_blocks = re.findall(r'```csv\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    # 2. ``` 패턴 추출 (언어 지정 없는 경우)
    if not code_blocks:
        code_blocks = re.findall(r'```\s*([\s\S]*?)\s*```', text)
    
    for block in code_blocks:
        try:
            # 쉼표가 포함된 라인이 있는 경우만 시도
            if "," in block:
                dfs.append(pd.read_csv(io.StringIO(block.strip())))
        except:
            continue
            
    # 3. 블록이 없으면 전체 텍스트 시도 (fallback)
    if not dfs:
        try:
            dfs.append(pd.read_csv(io.StringIO(text.strip())))
        except:
            pass
    return dfs

def find_best_match_df(target_df, candidate_dfs):
    """
    정답 DataFrame과 가장 컬럼 구조가 유사한 후보 DataFrame을 찾음
    (순서가 뒤섞여도 채점 가능하게 함)
    """
    best_df = None
    best_score = 0
    target_cols = set(target_df.columns)
    
    for df in candidate_dfs:
        candidate_cols = set(df.columns)
        # 교집합 컬럼 개수 확인
        common = target_cols.intersection(candidate_cols)
        score = len(common)
        
        # 컬럼이 절반 이상 일치하고, 기존 최고 점수보다 높으면 갱신
        if len(target_cols) > 0 and (len(common) / len(target_cols) >= 0.4) and score > best_score:
            best_score = score
            best_df = df
            
    return best_df

# ---------------------------------------------------------
# [핵심 2] API 통신 및 실행
# ---------------------------------------------------------
async def safe_api_call(client, model, messages, temperature=0, response_format=None):
    try:
        return await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, response_format=response_format
        )
    except Exception as e:
        return None

async def execute_participant_prompt(client, model, context, prompt, task_type):
    system_instruction = "당신은 AI 어시스턴트입니다. 사용자의 지시를 정확히 따르세요."
    
    # [중요] 과제 B의 경우 형식을 강제하여 파싱 성공률을 높임
    if "Task B" in task_type:
        system_instruction += """
        \n[필수 출력 형식]
        결과물은 반드시 Markdown Code Block(```csv)으로 감싸서 출력하세요.
        시트가 여러 개일 경우 각각 별도의 코드 블록으로 작성하세요.
        """

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"---[Context Data]---\n{context}\n\n---[Instruction]---\n{prompt}"}
    ]
    
    resp = await safe_api_call(client, model, messages, temperature=0)
    return resp.choices[0].message.content if resp else "Error"

# ---------------------------------------------------------
# [평가 로직]
# ---------------------------------------------------------
async def evaluate_text_logic(client, model, target_text, user_output, task_type):
    # 과제 A/C 평가 (기존 유지)
    if "Task A" in task_type:
        checklist = """
        1. [날짜 준수] "2025-08-01" 포함 여부 (Boolean)
        2. [수치 정확성] "150ms" 포함 여부 (Boolean)
        3. [키워드] "마스킹" 또는 "Masking" 포함 여부 (Boolean)
        4. [형식] Markdown Table 사용 여부 (Boolean)
        5. [논리] "로그 누락률 0.1%" 포함 여부 (Boolean)
        """
    else:
        checklist = """
        1. [충돌 발견] "버전 충돌(2.0 vs 2.1)" 식별 여부 (Boolean)
        2. [충돌 발견] "권한 시간(24h vs 4h)" 식별 여부 (Boolean)
        3. [형식] "[충돌 N - 항목명]" 형식 준수 여부 (Boolean)
        4. [근거] 충돌의 근거 위치 설명 여부 (Boolean)
        5. [중립성] 두 값을 모두 보고했는가 (Boolean)
        """

    judge_prompt = f"""
    참가자 결과물을 체크리스트 기반으로 채점하고 JSON으로 반환하세요.
    [결과물]: {user_output[:3000]}
    [체크리스트]: {checklist}
    Output JSON format: {{ "checks": {{ "check_1": true, ... }}, "feedback": "string" }}
    """
    
    resp = await safe_api_call(
        client, model, 
        [{"role": "system", "content": "JSON Only"}, {"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"}
    )
    try:
        res = json.loads(resp.choices[0].message.content)
        score = sum(1 for v in res['checks'].values() if v) * 20
        return {"score": score, "feedback": res.get("feedback", "")}
    except:
        return {"score": 0, "feedback": "채점 파싱 실패"}

def evaluate_excel_data_robust(golden_sheets, user_output_text):
    """ [개선됨] 순서 무관 Best Match 채점 로직 """
    score = 0
    feedback = []
    
    if not golden_sheets: return {"score": 0, "feedback": "정답 파일 없음"}

    # 1. 사용자 결과에서 DataFrame 추출
    user_dfs = extract_csv_dataframes(user_output_text)
    if not user_dfs: return {"score": 0, "feedback": "CSV 추출 실패 (형식 불일치)"}

    # 2. 정답 시트 순회하며 '제 짝' 찾기
    for g_name, g_df in golden_sheets.items():
        # 가장 비슷한 사용자 DataFrame 찾기
        best_match_df = find_best_match_df(g_df, user_dfs)
        
        if best_match_df is None:
            feedback.append(f"[{g_name}] 에 해당하는 표를 찾을 수 없음 (0점)")
            continue
            
        # 3. 찾은 표로 정밀 채점
        sheet_score = 0
        
        # 행 개수 비교
        diff = abs(len(g_df) - len(best_match_df))
        if diff == 0: sheet_score += 50
        elif diff <= 3: sheet_score += 30
        else: sheet_score += 10
        
        # 데이터 값 비교 (첫 컬럼 기준)
        try:
            col1 = g_df.columns[0]
            if col1 in best_match_df.columns:
                match = sum(g_df[col1].astype(str).str.strip() == best_match_df[col1].astype(str).str.strip())
                acc = match / len(g_df)
                if acc > 0.9: sheet_score += 50
                elif acc > 0.5: sheet_score += 30
        except:
            pass
            
        final_sheet_score = min(sheet_score, 100)
        score += final_sheet_score
        feedback.append(f"[{g_name}: {final_sheet_score}점]")

    # 시트가 2개면 평균, 1개면 그대로
    final_score = score / len(golden_sheets) if golden_sheets else 0
    return {"score": round(final_score), "feedback": " / ".join(feedback)}

# ---------------------------------------------------------
# [파이프라인]
# ---------------------------------------------------------
async def process_participant(sem, client, row, context, target_file, task_type):
    name = row.iloc[0]
    prompt = row.iloc[1]
    
    async with sem:
        # 1. 실행
        user_output = await execute_participant_prompt(client, "gpt-4o-mini", context, prompt, task_type)
        
        # 2. 평가
        if "Task B" in task_type:
            golden_sheets = load_golden_excel(target_file)
            eval_res = evaluate_excel_data_robust(golden_sheets, user_output)
        else:
            target_txt = read_file_content(target_file)
            eval_res = await evaluate_text_logic(client, "gpt-4o-mini", target_txt, user_output, task_type)
            
        return {
            "이름": name, "총점": eval_res['score'], 
            "피드백": eval_res['feedback'], "결과물": user_output[:200] + "..."
        }

async def run_grading(api_key, context, target, df_p, limit, task_type):
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(limit) # 슬라이더 값 적용
    tasks = [process_participant(sem, client, row, context, target, task_type) for _, row in df_p.iterrows()]
    
    status = st.empty()
    bar = st.progress(0)
    results = []
    
    start = time.time()
    for i, f in enumerate(asyncio.as_completed(tasks)):
        res = await f
        results.append(res)
        
        # UI 갱신 (속도 최적화를 위해 매번 갱신)
        done = i + 1
        bar.progress(done / len(df_p))
        elapsed = time.time() - start
        speed = elapsed / done
        eta = (len(df_p) - done) * speed
        status.markdown(f"<div class='status-box'>🚀 채점 중 ({done}/{len(df_p)}) | 남은 시간: {int(eta)}초 | 속도: {speed:.2f}s/명</div>", unsafe_allow_html=True)

    status.markdown(f"<div class='status-box success-box'>✅ 완료! (총 {int(time.time()-start)}초)</div>", unsafe_allow_html=True)
    return pd.DataFrame(results)

# ---------------------------------------------------------
# [메인 UI]
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 설정")
    api_key = os.getenv("OPENAI_API_KEY") or st.text_input("API Key", type="password")
    
    st.divider()
    task_type = st.radio("과제 유형", ["Task A (문서 구조화)", "Task B (데이터 정제)", "Task C (논리 충돌)"])
    
    # [속도 개선] 동시 처리 수 조절 슬라이더
    limit = st.slider("🚀 동시 채점 수 (속도 조절)", 5, 50, 20, help="숫자가 높을수록 빠르지만 API 에러 가능성이 있습니다.")
    
    st.divider()
    f_ctx = st.file_uploader("1. 문맥(Context)", type=['txt','pdf','xlsx'])
    f_tgt = st.file_uploader("2. 정답(Golden)", type=['txt','xlsx'])
    f_usr = st.file_uploader("3. 참가자(User)", type=['xlsx'])

st.title(f"🏆 Prompt Evaluation: {task_type.split('(')[0]}")

if st.button("🔥 채점 시작", type="primary", use_container_width=True):
    if api_key and f_ctx and f_tgt and f_usr:
        ctx_txt = read_file_content(f_ctx)
        df_p = pd.read_excel(f_usr)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res_df = loop.run_until_complete(run_grading(api_key, ctx_txt, f_tgt, df_p, limit, task_type))
        
        # 결과 출력
        st.divider()
        c1, c2 = st.columns([1, 3])
        c1.subheader("🥇 Top 3")
        c1.table(res_df.sort_values("총점", ascending=False).head(3)[["이름","총점"]])
        
        c2.subheader("📋 전체 결과")
        st.dataframe(res_df.sort_values("총점", ascending=False), use_container_width=True)
        
        # 엑셀 다운로드
        out = io.BytesIO()
        res_df.to_excel(out, index=False)
        st.download_button("📥 결과 다운로드", out.getvalue(), "result.xlsx")
    else:
        st.warning("파일을 모두 업로드하세요.")
