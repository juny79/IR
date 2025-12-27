# Next Methodology After MAP 0.8765 (New Ideas)

이 문서는 [BEST_CONFIG_MAP08765.md](BEST_CONFIG_MAP08765.md) 기준선을 **깨지 않고**, 리더보드 점수를 더 올리기 위한 “새 방법론”을 실험 단위로 정리한 것입니다.

핵심 원칙은 2가지입니다.
1) 한 번에 하나만 바꿔서 원인-결과를 분리한다.
2) 매 제출마다 `submission.csv`의 SHA256 + 변경 diff를 함께 기록한다.

---

## 0) 실험 운영 방법론 (필수)

### 실험 단위(A/B)
- A(기준선): 현재 최고점 설정 그대로
- B(실험안): **단 하나의 파라미터/로직만 변경**
- 제출 후 MAP/MRR 변화 기록

### 제출 산출물 고정
- 생성 직후:
  - `python3 snapshot_submission.py` → `submission_snapshot.json`
  - 파일 SHA256 기록
- 변경 사항 기록:
  - `git diff` 요약(어떤 상수/프롬프트를 바꿨는지)

---

## 1) 방법론 A: 후보군 확대 + 리랭커로 정밀도 회수

현재는 hybrid fusion에서 후보 50개를 만들고 그 중 top-5를 리랭크합니다.

### 가설
- MAP/MRR는 top-5 정밀도뿐 아니라 **top-1/2의 안정성**에 민감함.
- 후보를 넓히면 recall이 오르지만 노이즈가 늘 수 있음.
- 그러나 cross-encoder 리랭커가 충분히 강하면 후보 확대가 이득으로 전환될 수 있음.

### 실험안
1) [retrieval/hybrid_search.py](retrieval/hybrid_search.py)
   - `top_k_retrieve`: 50 → 80 (또는 100)
   - fusion 후보(candidates): 50 → 80
2) 리랭커 top_k_final은 5 유지

### 기대 효과/리스크
- 기대: recall 증가 → 정답 문서가 후보에 들어올 확률 증가
- 리스크: ES 호출/리랭커 비용 증가, 노이즈 후보 증가로 리랭커 한계 노출

---

## 2) 방법론 B: Hard Voting 가중치 재튜닝 (정밀/재현 균형)

현재는 [eval_rag.py](eval_rag.py)에서 `VOTING_WEIGHTS=[5,4,2]`.

### 가설
- SBERT/Gemini dense가 서로 다른 강점을 가짐.
- 쿼리 타입별로 최적 가중치가 다를 수 있으나, 리더보드는 단일 설정만 허용.

### 실험안(단일 변경)
- 후보 조합 예시 (한 번에 하나씩만):
  - `[6,4,2]` (rank1 bias 강화)
  - `[5,3,1]` (좀 더 완만)
  - `[4,4,2]` (sparse/dense 균형)

### 관찰 포인트
- top1 안정성(이전 베이스라인 submission과 top1 일치율)
- empty_topk 개수 변화(스킵 정책 영향은 없어야 정상)

---

## 3) 방법론 C: HyDE를 “짧고 정확”하게 재설계 (Sparse만 강화)

현재 HyDE는 200자 제한이며, sparse_query는 `standalone_query + HyDE`.

### 가설
- HyDE가 너무 길거나 일반론적이면 BM25가 엉뚱한 용어에 끌려갈 수 있음.
- 반대로 너무 짧으면 확장 효과가 줄어 recall이 떨어질 수 있음.

### 실험안
- [eval_rag.py](eval_rag.py)에서 `HYDE_MAX_LENGTH`만 변경
  - 150 / 250 / 300을 각각 A/B로 테스트
- [models/solar_client.py](models/solar_client.py) HyDE 프롬프트를 “키워드 열거형”으로 바꾸는 실험
  - 문장형 설명 대신: 핵심 용어/동의어/관련 개념을 콤마로 나열
  - 목적: BM25에 유리하게 신호를 압축

### 리스크
- HyDE 포맷 변화는 sparse 성능을 크게 흔들 수 있어, **다른 파라미터와 절대 동시 변경 금지**

---

## 4) 방법론 D: 쿼리-리랭커 입력 길이/문서 절단 길이 재튜닝

현재 리랭커는 문서 content를 512 chars로 자르고, CrossEncoder max_length=512.

### 가설
- 문서 앞부분에 핵심이 없는 케이스에서 512자 절단이 불리할 수 있음.

### 실험안(하나씩)
- (안전한 쪽) 문서 절단: 512 → 768
- (또는) 문서에서 헤더/요약 우선순위를 주는 전처리(있다면)

### 리스크
- 리랭커 비용 증가(시간)

---

## 5) 방법론 E: “스킵 21개”의 정교화(오탐 방지) + 완전 결정성

현재 스킵 정책은 “순수 잡담만 `topk=[]`”이며 21개가 스킵됩니다.

### 가설
- 스킵이 MAP에 기여한 이유는 ‘불필요 검색을 줄여서’가 아니라, **잘못된 검색으로 topk가 오염되는 케이스를 줄였기 때문**일 수 있음.
- 하지만 스킵을 더 늘리면 다시 MAP가 깨질 가능성이 큼.

### 실험안
- 스킵 개수는 유지(대략 10~30 범위)하면서, 다음만 개선:
  - Solar JSON 파싱 실패 fallback을 더 결정적으로(동일 입력이면 항상 동일 분류)
  - confidence를 저장/로그로 남겨 “경계 케이스”만 추적

### 금지 사항
- 과거처럼 “과학만 검색”으로 되돌아가기(정보성 질문을 대거 스킵해서 MAP가 크게 하락 가능)

---

## 6) 가장 추천하는 3-step 로드맵

1) **후보군 확대 실험** (방법론 A)
   - `top_k_retrieve`와 후보 수만 올려서 리랭커가 회수 가능한지 확인
2) **가중치 2~3개만 스윕** (방법론 B)
   - top1 안정성과 MAP 변화를 보고 방향성 결정
3) **HyDE 포맷 실험** (방법론 C)
   - 가장 파급이 커서 마지막에 단독으로 테스트

---

## 기록 템플릿 (복붙용)

- Date:
- Experiment ID:
- Changed file(s):
- Change summary:
- submission.csv sha256:
- empty_topk_count:
- MAP:
- MRR:
- Notes:
