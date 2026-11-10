import json
from pydantic import BaseModel, Field
from typing import List
from gigachat import GigaChat
from gigachat.models import ChatCompletionRequest, ChatMessage


class AtomicFact(BaseModel):
    """Модель одного атомарного факта, извлеченного из текста."""
    fact_id: int = Field(..., description="Уникальный порядковый номер факта")
    category: str = Field(..., description="Категория: 'date', 'number', 'entity', 'statement'")
    fact_type: str = Field(..., description="Подкатегория: для number - 'сумма'|'процент'|'количество_дней'|'номер_документа'; для date - 'дата_документа'|'срок'; для entity - 'организация'|'тип_документа'; для statement - 'право'|'обязанность'|'условие'")
    fact_text: str = Field(..., description="Самодостаточная формулировка факта, включающая контекст (к чему относится)")
    original_quote: str = Field(..., description="Дословная цитата из исходного текста")
    related_entity: str = Field(..., description="Сущность, к которой относится факт (например, 'ООО Вектор', 'договор №45-А'). Если факт общий — 'общий'")
    importance: str = Field(..., description="Важность факта: 'high' (критично: суммы, даты, условия), 'medium' (названия, типы документов), 'low' (вспомогательная информация)")


class ExtractedFactsList(BaseModel):
    """Список всех извлеченных фактов из фрагмента текста."""
    source_chunk_id: str = Field(..., description="Идентификатор исходного фрагмента текста (чанка)")
    facts: List[AtomicFact] = Field(..., description="Список извлеченных атомарных фактов")


SOURCE_EXTRACTOR_PROMPT = """Ты — скрупулезный аудитор данных. Твоя задача — извлечь из текста АБСОЛЮТНО ВСЕ атомарные, проверяемые факты. Ни одно число, дата, название или условие не должны быть пропущены.

ПРАВИЛА ИЗВЛЕЧЕНИЯ:
1. Максимальная декомпозиция: Разбивай сложные предложения на отдельные факты. Если в предложении есть сумма, срок и штраф — это ТРИ разных факта.
2. Самодостаточность fact_text: Формулировка должна быть понятна БЕЗ контекста. Включай в неё сущность, к которой относится факт. ПЛОХО: "Сумма оплаты". ХОРОШО: "Сумма оплаты по договору №45-А составляет 15 000 000 рублей".
3. Точность цитат: Для каждого факта приведи дословную цитату из исходного текста.
4. Категории и подкатегории (fact_type):
   - 'date': 'дата_документа' | 'срок_исполнения' | 'иная_дата'
   - 'number': 'сумма' | 'процент' | 'количество_дней' | 'номер_документа' | 'иное_число'
   - 'entity': 'организация' | 'тип_документа' | 'иная_сущность'
   - 'statement': 'право' | 'обязанность' | 'условие' | 'иное_утверждение'
5. related_entity: Укажи сущность, к которой относится факт. Например: "ООО Вектор", "договор №45-А", "Заказчик". Если факт общий — "общий".
6. importance: Оцени важность факта:
   - 'high': суммы, даты, сроки, условия, права, обязанности (критично для ответа)
   - 'medium': названия организаций, типы документов
   - 'low': вспомогательная информация
7. Формат: Ответ должен быть СТРОГО валидным JSON. Никаких markdown-оберток (```json), никаких вступлений и пояснений.

ПРИМЕР ПРАВИЛЬНОЙ ДЕКОМПОЗИЦИИ:
Текст: "ООО Ромашка по договору №5 от 01.01.2024 оплатит 1000 руб. в течение 3 дней."
Факты:
- fact_id: 1, category: "entity", fact_type: "организация", fact_text: "Название компании-плательщика: ООО Ромашка", original_quote: "ООО Ромашка", related_entity: "ООО Ромашка", importance: "medium"
- fact_id: 2, category: "number", fact_type: "номер_документа", fact_text: "Номер договора: №5", original_quote: "договор №5", related_entity: "договор", importance: "medium"
- fact_id: 3, category: "date", fact_type: "дата_документа", fact_text: "Дата заключения договора: 01.01.2024", original_quote: "от 01.01.2024", related_entity: "договор №5", importance: "high"
- fact_id: 4, category: "number", fact_type: "сумма", fact_text: "Сумма оплаты по договору №5 составляет 1000 руб.", original_quote: "оплатит 1000 руб.", related_entity: "договор №5", importance: "high"
- fact_id: 5, category: "number", fact_type: "количество_дней", fact_text: "Срок оплаты по договору №5 составляет 3 дня", original_quote: "в течение 3 дней", related_entity: "договор №5", importance: "high"

ТРЕБУЕМЫЙ ФОРМАТ ОТВЕТА (строго следуй этой структуре):
{
  "source_chunk_id": "ID чанка, который тебе передали",
  "facts": [
    {
      "fact_id": 1,
      "category": "date",
      "fact_type": "дата_документа",
      "fact_text": "Самодостаточная формулировка факта с контекстом",
      "original_quote": "Точная цитата из текста",
      "related_entity": "Сущность, к которой относится факт",
      "importance": "high" | "medium" | "low"
    }
  ]
}"""


def extract_first_json(text: str) -> str:
    """Находит первый валидный JSON-объект в строке."""
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


class SourceExtractorAgent:
    def __init__(self, credentials: str):
        self.giga = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            model="GigaChat-2"
        )
        self.system_prompt = SOURCE_EXTRACTOR_PROMPT

    def extract(self, chunk_id: str, source_text: str) -> ExtractedFactsList:
        raw_content = "Нет ответа"
        try:
            user_message = f"ID чанка: {chunk_id}\n\nТекст для анализа:\n{source_text}"

            chat_request = ChatCompletionRequest(
                messages=[
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(role="user", content=user_message)
                ]
            )

            response = self.giga.chat.create(chat_request)
            message = response.messages[0]
            if isinstance(message.content, list):
                raw_content = message.content[0].text
            else:
                raw_content = str(message.content)

            clean_text = raw_content.replace("```json", "").replace("```", "").strip()
            json_string = extract_first_json(clean_text)
            parsed_data = json.loads(json_string)
            return ExtractedFactsList(**parsed_data)

        except Exception as e:
            print(f"[Source Extractor] Ошибка при обработке чанка {chunk_id}: {e}")
            print(f"[Source Extractor] Сырой ответ модели для отладки:\n---\n{raw_content}\n---")
            return ExtractedFactsList(source_chunk_id=chunk_id, facts=[])


if __name__ == "__main__":

    key = ""

    print("Инициализация агента извлечения фактов...")
    extractor = SourceExtractorAgent(credentials=key)
    print("✅ Агент готов к работе!\n")

    test_text = """
    В соответствии с Договором №45-А от 15.03.2024, компания ООО "Вектор" обязуется
    поставить оборудование в срок до 30.06.2024. Сумма контракта: 15 000 000 рублей.
    При просрочке более 10 рабочих дней, Заказчик вправе расторгнуть договор с
    неустойкой 0.1% от суммы за каждый день.
    """

    print("Отправка запроса в GigaChat для извлечения фактов...\n")
    extracted_data = extractor.extract("doc_test_1", test_text)

    if extracted_data and extracted_data.facts:
        print(f"✅ Успешно извлечено фактов: {len(extracted_data.facts)}\n")
        print("=" * 80)
        for fact in extracted_data.facts:
            importance_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(fact.importance, "⚪")
            print(f"{importance_icon} [{fact.category.upper():<9}] #{fact.fact_id} | Тип: {fact.fact_type} | Важность: {fact.importance}")
            print(f"   📌 {fact.fact_text}")
            print(f"   💬 Цитата: \"{fact.original_quote}\"")
            print(f"   🔗 Относится к: {fact.related_entity}")
            print("-" * 80)
    else:
        print("❌ Факты не извлечены. Проверьте логи выше.")
