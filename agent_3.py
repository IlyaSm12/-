import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from gigachat import GigaChat
from gigachat.models import ChatCompletionRequest, ChatMessage


class AtomicFact(BaseModel):
    """Факт, извлечённый Агентом 1 из источника."""
    fact_id: int
    category: str 
    fact_text: str
    original_quote: str


class AtomicClaim(BaseModel):
    """Утверждение, извлечённое Агентом 2 из ответа."""
    claim_id: int
    category: str
    claim_text: str
    original_quote: str
    check_worthy: bool = True


class ClaimVerdict(BaseModel):
    """Результат проверки одного утверждения по фактам из источников."""
    claim_id: int
    verdict: str = Field(..., description="'подтверждено' | 'противоречит' | 'нет в источниках'")
    matched_fact_id: Optional[int] = Field(None, description="ID факта из источника, если найден")
    matched_fact_category: Optional[str] = Field(None, description="Категория найденного факта")
    evidence: str = Field("", description="Дословная цитата из источника (original_quote факта)")
    comment: str = ""
    evidence_found: bool = Field(False, description="Заполняется кодом: цитата реально есть в фактах")



CLAIM_VERIFIER_PROMPT = """Ты — модуль проверки ОДНОГО утверждения по атомарным фактам из источников.

Тебе дают:
- УТВЕРЖДЕНИЕ (из ответа ИИ) с его категорией
- СПИСОК ФАКТОВ, извлечённых из источников (каждый факт имеет ID, категорию, тип, текст и цитату)

Твоя задача: определить, как факты относятся к утверждению.

ВЕРДИКТЫ — строго один из:
- "подтверждено"     — есть факт с той же категорией, и его содержание совпадает с утверждением.
- "противоречит"     — есть факт с той же категорией, но содержание РАЗЛИЧАЕТСЯ.
- "нет в источниках" — среди фактов нет ни одного с релевантной категорией.

ПРАВИЛА:
1. Опирайся ТОЛЬКО на предоставленные факты. Свои знания не используй.
2. Сначала найди факты с той же категорией (date↔date, number↔number). Для точного совпадения используй поля fact_type и related_entity.
3. Если вердикт "подтверждено" или "противоречит" — укажи matched_fact_id.
4. 🔥 КРИТИЧЕСКОЕ ПРАВИЛО ДЛЯ ПОЛЯ evidence: Ты ОБЯЗАН скопировать значение поля original_quote из найденного факта В ТОЧНОСТИ, без изменений. НЕ добавляй fact_text, НЕ добавляй свои слова, НЕ пиши "quote=" или другие метки. Просто скопируй цитату как есть, в кавычках.
   ПЛОХО: "Срок исполнения: до 30.06.2024, quote="в срок до 30.06.2024""
   ХОРОШО: "в срок до 30.06.2024"
5. Если вердикт "нет в источниках" — matched_fact_id = null, evidence = "".
6. Ответ — СТРОГО валидный JSON, без markdown и пояснений.

ФОРМАТ ОТВЕТА:
{
  "claim_id": 1,
  "verdict": "подтверждено",
  "matched_fact_id": 3,
  "matched_fact_category": "number",
  "evidence": "Точная копия original_quote из факта, без изменений",
  "comment": "Короткое пояснение на русском"
}"""



def _norm(s: str) -> str:
    """Нормализация для сравнения цитат."""
    return re.sub(r"[^\w%№]+", " ", s.lower()).strip()


def extract_first_json(text: str) -> str:
    """Извлекает первый валидный JSON-объект из строки."""
    start = text.find('{')
    if start == -1:
        return text
    brace_count = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        char = text[i]
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]
    return text[start:]


def facts_to_text(facts: List[AtomicFact]) -> str:
    """Форматирует список фактов с учетом новых полей."""
    lines = []
    for f in facts:
        lines.append(
            f"- fact_id={f.fact_id}, category={f.category}, fact_type={f.fact_type}, "
            f"entity={f.related_entity}, importance={f.importance}, "
            f"text={f.fact_text}, quote=\"{f.original_quote}\""
        )
    return "\n".join(lines)



class ClaimVerifierAgent:
    """Агент 3: сверяет утверждение (от Агента 2) с фактами (от Агента 1)."""

    def __init__(self, credentials: str, max_retries: int = 1):
        self.giga = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            model="GigaChat-2"
        )
        self.system_prompt = CLAIM_VERIFIER_PROMPT
        self.max_retries = max_retries

    def _call_llm(self, user_message: str) -> str:
        request = ChatCompletionRequest(messages=[
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_message),
        ])
        response = self.giga.chat.create(request)
        message = response.messages[0]
        return message.content[0].text if isinstance(message.content, list) else str(message.content)

    def verify(self, claim: AtomicClaim, facts: List[AtomicFact]) -> ClaimVerdict:
      """Проверяет одно утверждение по списку фактов."""
      facts_text = facts_to_text(facts)
      user_message = (
          f"УТВЕРЖДЕНИЕ (claim_id={claim.claim_id}, category={claim.category}):\n"
          f"  {claim.claim_text}\n\n"
          f"ФАКТЫ ИЗ ИСТОЧНИКОВ:\n{facts_text}"
      )

      raw = "Нет ответа"
      for attempt in range(self.max_retries + 1):
          try:
              raw = self._call_llm(user_message)
              clean = raw.replace("```json", "").replace("```", "").strip()
              result = ClaimVerdict(**json.loads(extract_first_json(clean)))
              result.claim_id = claim.claim_id

              if result.evidence:
                  evidence_norm = _norm(result.evidence)
                  found = False
                  for f in facts:
                      if evidence_norm in _norm(f.original_quote) or evidence_norm in _norm(f.fact_text):
                          found = True
                          break
                  result.evidence_found = found

              return result
          except Exception as e:
              print(f"[Claim Verifier] Ошибка (попытка {attempt + 1}), claim {claim.claim_id}: {e}")
              print(f"[Claim Verifier] Сырой ответ:\n---\n{raw[:300]}\n---")

      return ClaimVerdict(
          claim_id=claim.claim_id,
          verdict="ошибка",
          comment=f"Сырой ответ: {raw[:200]}"
      )


def check_answer(
    extractor_result,
    decomposer_result,
    verifier: ClaimVerifierAgent
) -> dict:
    """
    Полный пайплайн проверки:
    facts (Агент 1) + claims (Агент 2) → вердикты (Агент 3)
    """
    facts = extractor_result.facts
    claims = decomposer_result.claims

    items = []
    for claim in claims:
        if claim.check_worthy:
            verdict = verifier.verify(claim, facts)
        else:
            verdict = ClaimVerdict(
                claim_id=claim.claim_id,
                verdict="не проверялось",
                comment="Мнение или служебная фраза"
            )
        items.append({"claim": claim, "verdict": verdict})

    counts = {}
    for it in items:
        v = it["verdict"].verdict
        counts[v] = counts.get(v, 0) + 1

    return {
        "answer_id": decomposer_result.answer_id,
        "source_chunk_id": extractor_result.source_chunk_id,
        "items": items,
        "counts": counts,
        "has_contradiction": counts.get("противоречит", 0) > 0,
        "completeness_score": 1 - (counts.get("нет в источниках", 0) / max(len(claims), 1))
    }


def print_report(report: dict):
    """Красивый вывод отчёта."""
    icons = {
        "подтверждено": "✅",
        "противоречит": "❌",
        "нет в источниках": "❓",
        "не проверялось": "➖",
        "ошибка": "⚠️"
    }
    print(f"\n📄 Отчёт для ответа: {report['answer_id']}")
    print(f"📚 Источник: {report['source_chunk_id']}")
    print("-" * 70)

    for it in report["items"]:
        c, v = it["claim"], it["verdict"]
        icon = icons.get(v.verdict, "?")
        src = f" [fact_id={v.matched_fact_id}]" if v.matched_fact_id is not None else ""
        warn = "  ⚠️ цитата не найдена в фактах" if v.evidence and not v.evidence_found else ""

        print(f"{icon} #{c.claim_id} [{c.category}] {c.claim_text} → {v.verdict}{src}{warn}")
        if v.evidence:
            print(f"     💬 Доказательство: \"{v.evidence}\"")
        if v.comment:
            print(f"     📝 {v.comment}")

    print("-" * 70)
    print(f"📊 Итого: {report['counts']}")
    print(f"🎯 Полнота покрытия: {report['completeness_score']:.2%}")
    if report["has_contradiction"]:
        print("⚠️  ОБНАРУЖЕНО ПРОТИВОРЕЧИЕ!")

