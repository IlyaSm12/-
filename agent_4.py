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


class CoveredFact(BaseModel):
    """Факт из источника, который покрыт хотя бы одним утверждением."""
    fact_id: int
    fact_text: str
    covering_claim_id: Optional[int] = Field(None, description="ID утверждения, которое покрывает этот факт")


class MissingFact(BaseModel):
    """Факт из источника, который НЕ отражён в ответе (критическое умолчание)."""
    fact_id: int
    fact_text: str
    importance: str = Field(..., description="'high' | 'medium' | 'low'")
    reason: str = Field("", description="Почему этот факт важен для ответа на вопрос")


class CompletenessReport(BaseModel):
    """Результат проверки полноты ответа."""
    verdict: str = Field(..., description="'полный' | 'частичный' | 'неполный'")
    coverage_score: float = Field(..., description="Доля покрытых фактов от 0.0 до 1.0")
    covered_facts: List[CoveredFact] = Field(default_factory=list)
    missing_facts: List[MissingFact] = Field(default_factory=list)
    comment: str = Field("", description="Короткое пояснение на русском")



COMPLETENESS_CHECKER_PROMPT = """Ты — модуль обратной проверки полноты ответа (Агент 4).

Тебе на вход даются:
- ВОПРОС пользователя
- СПИСОК ФАКТОВ, уже извлечённых из источников (Агент 1)
- СПИСОК УТВЕРЖДЕНИЙ, уже извлечённых из ответа ИИ (Агент 2)

Твоя задача: определить, какие факты из источников НЕ БЫЛИ отражены в ответе (критические умолчания).

АЛГОРИТМ РАБОТЫ:
1. Для каждого факта из списка facts проверь, есть ли среди claims хотя бы одно утверждение, которое покрывает этот факт (совпадает по смыслу, категории и содержанию).
2. Если факт покрыт — добавь его в covered_facts с указанием covering_claim_id.
3. Если факт НЕ покрыт — добавь его в missing_facts. Скопируй его fact_text и importance из исходного списка фактов.
4. Рассчитай coverage_score = len(covered_facts) / len(facts).
5. Определи вердикт по шкале (опирайся на importance='high'):
   - "полный": нет missing_facts с importance='high'.
   - "частичный": есть 1-2 missing_facts с importance='high'.
   - "неполный": есть 3+ missing_facts с importance='high' или coverage_score < 0.5.

ПРАВИЛА:
1. Опирайся ТОЛЬКО на предоставленные факты и утверждения. Свои знания не используй.
2. Сравнивай факты и утверждения по категории (date↔date, number↔number, entity↔entity, statement↔statement).
3. Если категория совпадает, но содержание различается (например, разные числа) — это НЕ покрытие, это противоречие (такой факт идёт в missing_facts).
4. Важность (importance) пропущенного факта УЖЕ указана в поле `importance` исходного факта (high/medium/low). Просто скопируй это значение в missing_facts. Не придумывай свою оценку.
   - 'high': суммы, даты, сроки, условия, права, обязанности.
   - 'medium': названия организаций, типы документов.
   - 'low': вспомогательная информация.
5. Ответ — СТРОГО валидный JSON, без markdown и пояснений.

ФОРМАТ ОТВЕТА:
{
  "verdict": "полный" | "частичный" | "неполный",
  "coverage_score": 0.75,
  "covered_facts": [
    {"fact_id": 1, "fact_text": "Название компании", "covering_claim_id": 5}
  ],
  "missing_facts": [
    {"fact_id": 3, "fact_text": "Дата договора", "importance": "high", "reason": "Критично для понимания сроков"}
  ],
  "comment": "Короткое пояснение на русском"
}"""




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
    lines = []
    for f in facts:
        lines.append(
            f"- fact_id={f.fact_id}, category={f.category}, fact_type={f.fact_type}, "
            f"entity={f.related_entity}, importance={f.importance}, "
            f"text={f.fact_text}, quote=\"{f.original_quote}\""
        )
    return "\n".join(lines)


def claims_to_text(claims: List[AtomicClaim]) -> str:
    """Форматирует список утверждений в читаемый текст для промпта."""
    lines = []
    for c in claims:
        checkable = "checkable" if c.check_worthy else "not_checkable"
        lines.append(f"- claim_id={c.claim_id}, category={c.category}, text={c.claim_text} [{checkable}]")
    return "\n".join(lines)



class CompletenessCheckerAgent:
    """Агент 4: проверяет полноту ответа (какие факты из источника упущены)."""

    def __init__(self, credentials: str, max_retries: int = 1):
        self.giga = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            model="GigaChat-2"
        )
        self.system_prompt = COMPLETENESS_CHECKER_PROMPT
        self.max_retries = max_retries

    def _call_llm(self, user_message: str) -> str:
        request = ChatCompletionRequest(messages=[
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_message),
        ])
        response = self.giga.chat.create(request)
        message = response.messages[0]
        return message.content[0].text if isinstance(message.content, list) else str(message.content)

    def check(
        self,
        question: str,
        facts: List[AtomicFact],
        claims: List[AtomicClaim]
    ) -> CompletenessReport:
        """Проверяет полноту ответа на основе фактов и утверждений."""
        facts_text = facts_to_text(facts)
        claims_text = claims_to_text(claims)

        user_message = (
            f"ВОПРОС: {question}\n\n"
            f"ФАКТЫ ИЗ ИСТОЧНИКОВ (Агент 1):\n{facts_text}\n\n"
            f"УТВЕРЖДЕНИЯ ИЗ ОТВЕТА (Агент 2):\n{claims_text}"
        )

        raw = "Нет ответа"
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call_llm(user_message)
                clean = raw.replace("```json", "").replace("```", "").strip()
                result = CompletenessReport(**json.loads(extract_first_json(clean)))

                if result.verdict == "полный" and len(result.missing_facts) > 0:
                    high_missing = sum(1 for m in result.missing_facts if m.importance == 'high')
                    if high_missing > 0:
                        result.verdict = "частичный"
                        result.comment += " [Автокоррекция: есть важные пропуски]"
                    elif len(result.missing_facts) >= 3:
                        result.verdict = "неполный"
                        result.comment += " [Автокоррекция: много пропусков]"

                return result
            except Exception as e:
                print(f"[Completeness Checker] Ошибка (попытка {attempt + 1}): {e}")
                print(f"[Completeness Checker] Сырой ответ:\n---\n{raw[:300]}\n---")

        return CompletenessReport(
            verdict="ошибка",
            coverage_score=0.0,
            comment=f"Не удалось получить валидный JSON. Сырой ответ: {raw[:200]}"
        )




def print_completeness_report(report: CompletenessReport):
    """Красивый вывод отчёта о полноте."""
    icons = {
        "полный": "🟢",
        "частичный": "🟡",
        "неполный": "🔴",
        "ошибка": "⚠️"
    }
    print(f"\n{icons.get(report.verdict, '?')} Вердикт: {report.verdict.upper()} (Coverage: {report.coverage_score:.2%})")
    print("-" * 70)

    if report.covered_facts:
        print(f"✅ Покрытые факты ({len(report.covered_facts)}):")
        for f in report.covered_facts:
            claim_info = f" [покрыто claim #{f.covering_claim_id}]" if f.covering_claim_id else ""
            print(f"   - [fact_id={f.fact_id}] {f.fact_text}{claim_info}")

    if report.missing_facts:
        print(f"\n❌ Критические умолчания ({len(report.missing_facts)}):")
        for m in report.missing_facts:
            importance_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(m.importance, "⚪")
            print(f"   {importance_icon} [fact_id={m.fact_id}] ({m.importance}) {m.fact_text}")
            if m.reason:
                print(f"      💬 {m.reason}")

    if report.comment:
        print(f"\n💬 Комментарий: {report.comment}")
    print("=" * 70)



def check_completeness(
    extractor_result, 
    decomposer_result, 
    checker: CompletenessCheckerAgent,
    question: str
) -> CompletenessReport:
    """
    Проверка полноты: факты (Агент 1) + утверждения (Агент 2) → отчёт об упущениях (Агент 4)
    """
    return checker.check(
        question=question,
        facts=extractor_result.facts,
        claims=decomposer_result.claims
    )
