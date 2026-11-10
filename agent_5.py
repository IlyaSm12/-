import json
from typing import List, Optional
from pydantic import BaseModel, Field
from gigachat.models import ChatCompletionRequest, ChatMessage


class FinalDiscrepancy(BaseModel):
    """Модель конкретного дефекта для итогового отчета"""
    issue_type: str = Field(..., description="'галлюцинация' (из Агента 3) или 'упущение' (из верхнего Агента 4)")
    importance: str = Field(..., description="'high' (суммы, сроки, штрафы), 'medium', 'low'")
    description: str = Field(..., description="Понятное описание проблемы на русском языке")
    bad_claim: Optional[str] = Field(None, description="Ошибочное утверждение из ответа (для галлюцинаций)")
    exact_quote: Optional[str] = Field(None, description="Точная исходная цитата из источника для доказательства")

class Agent5FinalReport(BaseModel):
    """Единый обоснованный вердикт системы от Агента 5"""
    final_verdict: str = Field(..., description="'полностью_верен', 'требует_исправления', 'критические_ошибки'")
    summary_verdict: str = Field(..., description="Развернутое аналитическое заключение по качеству ответа с аргументацией")
    detected_issues: List[FinalDiscrepancy] = Field(default_factory=list, description="Реестр всех найденных несоответствий")



AGENT5_STRUCTURED_PROMPT = """Ты — Главный Консолидатор и Аудитор качества ответов (Агент 5).
Твоя задача — сопоставить результаты пакетной проверки утверждений (Агент 3) и анализа упущенных Pydantic-фактов (Верхний Агент 4), сформировав финальный юридически обоснованный вердикт.

На вход тебе передается:
1. Исходный ВОПРОС пользователя.
2. Проверяемый ОТВЕТ ИИ.
3. ПАКЕТНЫЙ МАССИВ АГЕНТА 3: Список всех утверждений из ответа с вердиктами ("подтверждено" / "противоречит" / "нет в источниках") и цитатами.
4. ОТЧЕТ ПО ПОЛНОТЕ АГЕНТА 4 (Верхняя версия): Структурированный анализ упущенных фактов на основе AtomicFact и AtomicClaim.

ИНСТРУКЦИЯ ПО СБОРКЕ И СНИЖЕНИЮ ЛОЖНЫХ СРАБАТЫВАНИЙ:
1. Выдели из массива Агента 3 только те объекты, где вердикт равен "противоречит". Помести их в `detected_issues` как тип "галлюцинация".
2. Выдели из отчета Агента 4 пропущенные факты (`missing_points` или не сопоставленные факты высокого приоритета). Помести их как тип "упущение".
3. Для каждого дефекта ОБЯЗАТЕЛЬНО найди и укажи `exact_quote` (точную исходную цитату). Если Агент 3 или Агент 4 не предоставили дословную цитату — игнорируй этот пункт (это защита от ложных срабатываний, требуемая для Accuracy > 85%).

Выдай результат СТРОГО в формате валидного JSON по схеме Agent5FinalReport, без markdown-тегов и без лишнего текста.
"""



class FinalBulkReportAgent:
    """Агент 5: Собирает данные Новейшего Агента 3 и Верхнего Агента 4 в финальный отчет"""

    def __init__(self, credentials: str):
        self.giga = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            timeout=30.0
        )
        self.system_prompt = AGENT5_STRUCTURED_PROMPT

    def generate_final_verdict(self, question: str, answer: str, agent3_bulk_output: list, agent4_structured_output: dict) -> Agent5FinalReport:
        user_content = (
            f"1. ВОПРОС: {question}\n\n"
            f"2. ПРОВЕРЯЕМЫЙ ОТВЕТ: {answer}\n\n"
            f"3. ПАКЕТНЫЙ JSON АГЕНТА 3 (Все утверждения):\n{json.dumps(agent3_bulk_output, ensure_ascii=False, indent=2)}\n\n"
            f"4. ОТЧЕТ АГЕНТА 4 (Верхний вариант полноты):\n{json.dumps(agent4_structured_output, ensure_ascii=False, indent=2)}"
        )

        try:
            chat_request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(role="user", content=user_content)
                ]
            )
            response = self.giga.chat(chat_request)
            raw_text = response.messages.content.text if isinstance(response.messages.content, list) else response.messages.content

            clean_json = raw_text.replace("```json", "").replace("```", "").strip()
            json_str = extract_first_json(clean_json)

            parsed_data = json.loads(json_str)
            return Agent5FinalReport(**parsed_data)

        except Exception as e:
            print(f"[Агент 5] Ошибка консолидации данных: {e}")
            return Agent5FinalReport(
                final_verdict="критические_ошибки",
                summary_verdict=f"Техническая ошибка сборки финального отчета: {str(e)}",
                detected_issues=[]
            )


def print_agent5_report(report: Agent5FinalReport):
    """Красиво выводит финальный вердикт Агента 5 в консоль Colab с иконками"""
    print("=" * 75)
    print("⚖️  ИНФОРМАЦИОННАЯ ПАНЕЛЬ ГЛАВНОГО АУДИТОРА (АГЕНТ №5)")
    print("=" * 75)

    verdict_upper = report.final_verdict.upper()
    if "ПОЛНОСТЬЮ_ВЕРЕН" in verdict_upper:
        verdict_marker = "🟢 [УСПЕШНО] "
    elif "ТРЕБУЕТ_ИСПРАВЛЕНИЯ" in verdict_upper:
        verdict_marker = "🟡 [ВНИМАНИЕ] "
    else:
        verdict_marker = "🔴 [КРИТИЧЕСКАЯ ОШИБКА] "

    print(f"{verdict_marker} Финальный вердикт: {verdict_upper}")
    print(f"📝 Аналитическое заключение:\n   {report.summary_verdict}")
    print("-" * 75)

    issues_count = len(report.detected_issues)
    print(f"📋 Всего выявлено несоответствий: {issues_count}")
    print("=" * 75)

    if issues_count == 0:
        print("✅ Нарушений не обнаружено. Ответ полностью соответствует источникам.")
        print("=" * 75)
        return

    for idx, issue in enumerate(report.detected_issues, 1):
        importance_marker = "🔴" if issue.importance == "high" else "🟠" if issue.importance == "medium" else "🟡"

        print(f"{importance_marker} [{issue.issue_type.upper()}] | Дефект #{idx} | Важность: {issue.importance}")
        print(f"   ⚙️  Суть проблемы: {issue.description}")

        if issue.bad_claim:
            print(f"   ❌ Искаженный тезис из ответа: \"{issue.bad_claim}\"")

        if issue.exact_quote:
            print(f"   💬 Цитата-доказательство из источника: \"{issue.exact_quote}\"")
        else:
            print(f"   ⚠️  Внимание: Дословная цитата в отчете отсутствует!")

        print("." * 75)

    print("=" * 75)
