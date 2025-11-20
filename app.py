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

def extract_csv_dataframes(text):
    """
    LLM 답변 텍스트에서 CSV 블록(들)을 찾아 DataFrame 리스트로 반환
    """
    dfs = []
    
    # 1. 마크다운 코드 블록(```csv ... ```) 추출 시도
    code_blocks = re.findall(r'```csv\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    
    if not code_blocks:
        # csv 태그가 없으면 그냥 코드 블록(``` ... ```) 시도
        code_blocks = re.findall(r'```\s*([\s\S]*?)\s*```', text)
    
    # 블록이 발견되면 각각 파싱
    if code_blocks:
        for block in code_blocks:
            try:
                dfs.append(pd.read_csv(io.StringIO(block.strip())))
            except:
                continue
    else:
        # 블록이 아예 없으면 전체 텍스트를 하나의 CSV로 시도
        try:
            dfs.append(pd.read_csv(io.StringIO(text.strip())))
        except:
            pass
            
    return dfs

# ---------------------------------------------------------
# [설정] 페이지 기본 세팅
# ---------------------------------------------------------
st.set_page_config(page_title="DB Inc 프롬프트 경진대회 채점기 v2.1", layout="wide", page_icon="⚖️")

# CSS 스타일링
st.markdown("""
    <style>
    .metric-container { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; }
    .status-box { 
        padding: 15px; border-radius: 8px; margin-bottom: 10px; text-align: center; 
        font-size: 1.1rem; background-color: #e3f2fd; border: 1px solid #90caf9; 
        color: #1565c0; font-weight: bold;
    }
    .success-box { background-color: #e8f5e9; color: #2e7d32; border-color: #c8e6c9; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# [함수] 파일 처리 및 유틸리티
# ---------------------------------------------------------
def read_file_content(file):
    """파일 확장자에 따른 텍스트 추출"""
    if not file: return None
    ext = file.name.split('.')[-1].lower()
    try:
        if ext == 'pdf':
            reader = PdfReader(file)
            return "".join([page.extract_text() for page in reader.pages])
        elif ext in ['xlsx', 'xls']:
            # 엑셀은 텍스트로 변환해서 문맥으로 제공
            sheets = pd.read_excel(file, sheet_name=None)
            text = []
            for name, df in sheets.items():
                text.append(f"### Sheet: {name}\n{df.to_markdown(index=False)}")
            return "\n\n".join(text)
        elif ext == 'csv':
            return pd.read_csv(file).to_markdown(index=False)
        else: # txt, md, py etc
            return file.getvalue().decode("utf-8")
    except Exception as e:
        return f"Error reading file: {str(e)}"

def load_golden_excel(file):
    """과제 B 채점용 정답 엑셀 로드"""
    if file and file.name.endswith('.xlsx'):
        return pd.read_excel(file, sheet_name=None)
    return None

# ---------------------------------------------------------
# [핵심] 비동기 LLM 통신 및 실행
# ---------------------------------------------------------
async def safe_api_call(client, model, messages, temperature=0, response_format=None):
    try:
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format
        )
    except Exception as e:
        return None

async def execute_participant_prompt(client, model, context, prompt, task_type):
    """참가자 프롬프트 실행 (Executor)"""
    
    system_instruction = "당신은 AI 어시스턴트입니다. 사용자의 지시를 정확히 따르세요."
    
    # 과제 B의 경우, 파싱 가능한 포맷을 강제하기 위한 시스템 프롬프트 주입
    if task_type == "Task B (데이터 정제)":
        system_instruction += "\n[중요] 결과물은 반드시 CSV 포맷(콤마 구분)으로 출력해야 합니다. 다른 말은 하지 마세요."

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"---[Context Data]---\n{context}\n\n---[Instruction]---\n{prompt}"}
    ]
    
    resp = await safe_api_call(client, model, messages, temperature=0)
    return resp.choices[0].message.content if resp else "Error"

# ---------------------------------------------------------
# [평가 로직 1] 과제 A/C : LLM Judge (Atomic Checklist)
# ---------------------------------------------------------
async def evaluate_text_logic(client, model, target_text, user_output, task_type):
    """LLM을 이용한 논리/구조 평가"""
    
    # 과제별 체크리스트 프롬프트 분기
    if "Task A" in task_type:
        checklist_prompt = """
        1. [날짜 준수] "2025-08-01" 날짜가 포함되어 있는가? (Boolean)
        2. [수치 정확성] "150ms" 목표 수치가 명시되었는가? (Boolean)
        3. [키워드] "마스킹" 또는 "Masking" 단어가 포함되었는가? (Boolean)
        4. [형식] Markdown Table 형식을 사용했는가? (Boolean)
        5. [논리] "로그 누락률 0.1%" 조건이 포함되었는가? (Boolean)
        """
    else: # Task C
        checklist_prompt = """
        1. [충돌 발견 1] "문서 버전" 충돌(2.0 vs 2.1)을 식별했는가? (Boolean)
        2. [충돌 발견 2] "긴급 권한" 시간 충돌(24h vs 4h)을 식별했는가? (Boolean)
        3. [형식 준수] "[충돌 N - 항목명]" 형식을 지켰는가? (Boolean)
        4. [근거 제시] 충돌의 근거(위치 등)를 설명했는가? (Boolean)
        5. [판단 보류] AI가 임의로 결정하지 않고 두 값을 모두 보고했는가? (Boolean)
        """

    judge_prompt = f"""
    당신은 냉정한 채점관입니다. 아래 체크리스트를 기준으로 Pass/Fail 여부를 판단하세요.
    
    [참가자 산출물]:
    {user_output[:3000]}
    
    [체크리스트]:
    {checklist_prompt}
    
    JSON 형식으로만 출력하세요:
    {{
        "checks": {{ "check_1": boolean, "check_2": boolean, "check_3": boolean, "check_4": boolean, "check_5": boolean }},
        "feedback": "간단한 피드백 (한글)"
    }}
    """
    
    resp = await safe_api_call(
        client, model, 
        [{"role": "system", "content": "Output JSON only."}, {"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(resp.choices[0].message.content)
        checks = result.get("checks", {})
        true_count = sum(1 for v in checks.values() if v)
        total_score = true_count * 20 # 5개 항목 * 20점 = 100점
        
        return {
            "score": total_score,
            "feedback": result.get("feedback", ""),
            "details": checks
        }
    except:
        return {"score": 0, "feedback": "채점 실패 (JSON 파싱 오류)", "details": {}}

# ---------------------------------------------------------
# [평가 로직 2] 과제 B : Python Code Judge (Data Comparison)
# ---------------------------------------------------------
def evaluate_excel_data(golden_sheets, user_output_text):
    """
    [수정됨] 멀티 시트 지원 평가 로직
    """
    score = 0
    feedback = []
    
    if not golden_sheets:
        return {"score": 0, "feedback": "정답(Golden) 엑셀 파일이 없습니다."}

    # 1. 사용자 답변에서 DataFrame들 추출
    user_dfs = extract_csv_dataframes(user_output_text)
    
    if not user_dfs:
        return {"score": 0, "feedback": "형식 오류: CSV 데이터를 추출할 수 없습니다. (마크다운이나 쉼표 구분 형식을 지켜주세요)"}

    # 2. 정답 시트 준비
    gold_sheet_names = list(golden_sheets.keys())
    gold_dfs = list(golden_sheets.values())
    
    # 평가 루프
    matched_sheets = 0
    
    # 최대 2개 시트까지만 평가 (Sheet1: 데이터, Sheet2: 집계)
    max_checks = min(len(gold_dfs), 2)
    
    for i in range(max_checks):
        g_name = gold_sheet_names[i]
        g_df = gold_dfs[i]
        
        # 사용자가 생성한 표가 부족하면 스킵
        if i >= len(user_dfs):
            feedback.append(f"[{g_name}] 누락됨 (-50)")
            continue
            
        u_df = user_dfs[i]
        
        # --- 개별 시트 채점 로직 ---
        sheet_score = 0
        sheet_feedback = []
        
        # 1) 컬럼명 비교 (유사도 체크)
        g_cols = set(g_df.columns)
        u_cols = set(u_df.columns)
        common_cols = g_cols.intersection(u_cols)
        
        if len(common_cols) / len(g_cols) >= 0.5: # 컬럼이 50% 이상 일치하면 채점 진행
            sheet_score += 20
            
            # 2) 행 개수 비교
            row_diff = abs(len(g_df) - len(u_df))
            if row_diff == 0:
                sheet_score += 30
                sheet_feedback.append("행 개수 정확")
            elif row_diff < 5: # 오차 범위 허용
                sheet_score += 15
                sheet_feedback.append("행 개수 유사")
            else:
                sheet_feedback.append(f"행 개수 차이 큼(정답:{len(g_df)} vs 제출:{len(u_df)})")
                
            # 3) 데이터 값 정밀 비교 (간소화된 로직)
            # 첫번째 컬럼(보통 ID나 Name)이 같은지 확인
            try:
                col_name = list(g_df.columns)[0]
                if col_name in u_df.columns:
                    match_cnt = sum(g_df[col_name].astype(str).str.strip() == u_df[col_name].astype(str).str.strip())
                    accuracy = match_cnt / len(g_df)
                    if accuracy > 0.8: sheet_score += 50
                    elif accuracy > 0.5: sheet_score += 30
                    else: sheet_feedback.append("데이터 값 불일치 다수")
            except:
                pass
                
        else:
            sheet_feedback.append("컬럼 구조 불일치")
            
        # 시트별 점수 합산 (최대 50점씩 배분)
        final_sheet_score = min(sheet_score, 100) * 0.5 # 시트당 50점 만점
        score += final_sheet_score
        feedback.append(f"[{g_name}: {final_sheet_score}점] " + ", ".join(sheet_feedback))

    return {"score": round(score), "feedback": " / ".join(feedback)}


# ---------------------------------------------------------
# [컨트롤러] 개별 참가자 처리
# ---------------------------------------------------------
async def process_participant(sem, client, row, context, target_file, task_type):
    name = row.iloc[0]
    prompt = row.iloc[1]
    
    async with sem:
        # 1. 실행 (Execution)
        user_output = await execute_participant_prompt(client, "gpt-4o-mini", context, prompt, task_type)
        
        # 2. 평가 (Evaluation) - 과제 유형에 따라 분기
        if "Task B" in task_type:
            # 과제 B는 정답 파일(Excel)이 필요
            golden_sheets = load_golden_excel(target_file)
            eval_result = evaluate_excel_data(golden_sheets, user_output)
        else:
            # 과제 A/C는 텍스트 기반 LLM 평가
            target_text = read_file_content(target_file) # 정답지 텍스트
            eval_result = await evaluate_text_logic(client, "gpt-4o-mini", target_text, user_output, task_type)
            
        return {
            "이름": name,
            "총점": eval_result['score'],
            "피드백": eval_result['feedback'],
            "결과물": user_output[:200] + "..." # 요약
        }

async def run_grading_pipeline(api_key, context, target_file, df_p, limit, task_type):
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(limit)
    tasks = []
    
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    total = len(df_p)
    start_time = time.time()
    
    # 태스크 생성
    for idx, row in df_p.iterrows():
        tasks.append(process_participant(sem, client, row, context, target_file, task_type))
    
    results = []
    completed = 0
    
    for f in asyncio.as_completed(tasks):
        res = await f
        results.append(res)
        completed += 1
        
        # UI 업데이트
        elapsed = time.time() - start_time
        speed = elapsed / completed
        remaining = (total - completed) * speed
        
        progress_bar.progress(completed / total)
        status_box.markdown(f"""
            <div class='status-box'>
            🚀 {task_type} 채점 중... ({completed}/{total})<br>
            남은 시간: 약 {int(remaining)}초
            </div>
        """, unsafe_allow_html=True)
        
    status_box.markdown(f"<div class='status-box success-box'>✅ 채점 완료! ({int(elapsed)}초 소요)</div>", unsafe_allow_html=True)
    return pd.DataFrame(results)

# ---------------------------------------------------------
# [메인] UI 구성
# ---------------------------------------------------------
with st.sidebar:
    st.title("🎛️ 설정 및 업로드")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password")
        
    st.divider()
    
    # 🎯 과제 선택 기능 추가
    task_type = st.radio(
        "평가할 과제 유형 선택",
        ["Task A (문서 구조화)", "Task B (데이터 정제)", "Task C (논리 충돌)"],
        index=0
    )
    
    st.info(f"ℹ️ 선택된 로직: {'Python Code Judge (Pandas)' if 'Task B' in task_type else 'LLM Judge (Atomic Check)'}")

    st.divider()
    uploaded_ctx = st.file_uploader("1. 문맥 자료 (Context)", type=['txt', 'pdf', 'xlsx'])
    uploaded_tgt = st.file_uploader("2. 정답/기준 파일 (Golden)", type=['txt', 'xlsx'])
    uploaded_usr = st.file_uploader("3. 참가자 명단 (Excel)", type=['xlsx'])

st.title(f"🏆 AI 프롬프트 평가 시스템 : {task_type.split('(')[0]}")

if st.button("🔥 채점 시작", type="primary", use_container_width=True):
    if not (api_key and uploaded_ctx and uploaded_tgt and uploaded_usr):
        st.warning("⚠️ API Key와 모든 파일을 업로드해주세요.")
    else:
        # 파일 내용 읽기 (과제 B의 Target은 여기서 읽지 않고 함수 내부에서 처리)
        context_text = read_file_content(uploaded_ctx)
        df_participants = pd.read_excel(uploaded_usr)
        
        # 비동기 실행
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result_df = loop.run_until_complete(
                run_grading_pipeline(api_key, context_text, uploaded_tgt, df_participants, 10, task_type)
            )
            
            # 결과 표시
            st.divider()
            col1, col2 = st.columns([1, 3])
            
            # 상위 3명
            top_rank = result_df.sort_values(by="총점", ascending=False).head(3)
            col1.subheader("🥇 Top 3")
            col1.table(top_rank[["이름", "총점"]])
            
            # 전체 테이블
            col2.subheader("📋 전체 결과")
            st.dataframe(
                result_df.sort_values(by="총점", ascending=False),
                use_container_width=True,
                column_config={
                    "총점": st.column_config.ProgressColumn("Score", format="%d점", min_value=0, max_value=100),
                }
            )
            
            # 다운로드
            output = io.BytesIO()
            result_df.to_excel(output, index=False)
            st.download_button("📥 결과 엑셀 다운로드", output.getvalue(), "evaluation_result.xlsx")
            
        except Exception as e:
            st.error(f"시스템 에러: {str(e)}")
