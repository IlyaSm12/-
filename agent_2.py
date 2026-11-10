import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from gigachat import GigaChat
from gigachat.models import ChatCompletionRequest, ChatMessage


class AtomicClaim(BaseModel):
    """Одно проверяемое утверждение, извлечённое из ответа."""
    claim_id: int = Field(..., description="Порядковый номер утверждения")
    category: str = Field(..., description="Строго одно из: 'date', 'number', 'entity', 'statement'")
    claim_text: str = Field(..., description="Самодостаточная формулировка утверждения")
    original_quote: str = Field(..., description="Дословная цитата из ответа")
    check_worthy: bool = Field(True, description="False для мнений, оценок и служебных фраз")
    quote_in_answer: bool = Field(False, description="Заполняется кодом: найдена ли цитата в ответе")


class DecomposedAnswer(BaseModel):
    """Результат работы агента 2: ответ, разбитый на утверждения."""
    answer_id: str
    claims: List[AtomicClaim]


CLAIM_DECOMPOSER_PROMPT = """Ты — аналитик, который разбирает ответ на отдельные проверяемые утверждения. Твоя задача — выделить из ответа ВСЕ утверждения, каждое из которых можно независимо проверить по источникам.

ПРАВИЛА:
1. Максимальная декомпозиция: одно утверждение = один факт. Если в предложении сумма, срок и условие — это три разных утверждения.
2. Самодостаточность: каждое утверждение должно быть понятно без контекста. Заменяй местоимения и отсылки («он», «эта компания», «договор» без номера) на конкретные названия, номера и имена.
3. Точность: числа, даты, имена и единицы измерения сохраняй ровно как в ответе. Ничего не добавляй от себя, не исправляй и не додумывай, даже если ответ кажется тебе неверным.
4. Цитаты: для каждого утверждения приведи ДОСЛОВНУЮ цитату из ответа.
5. Не проверяемое (мнения, оценки, вводные и вежливые фразы: «Это выгодная сделка», «Надеюсь, помогло») тоже включай, но ставь check_worthy=false.
6. Категории, строго одна из:
   - 'date' (даты, сроки в датах)
   - 'number' (числа, суммы, проценты, номера документов, количество дней)
   - 'entity' (организации, люди, типы документов)
   - 'statement' (действия, условия, права, обязанности)
7. Формат: СТРОГО валидный JSON. Никаких markdown-оберток (```json), вступлений и пояснений.

ПРИМЕР:
Ответ: "Компания заключила договор №5 и обязана оплатить 1000 руб. за 3 дня. Это разумно."
Утверждения:
- claim_id: 1, category: "number", claim_text: "Номер договора компании — №5", original_quote: "договор №5", check_worthy: true
- claim_id: 2, category: "number", claim_text: "Компания обязана оплатить 1000 руб.", original_quote: "оплатить 1000 руб.", check_worthy: true
- claim_id: 3, category: "number", claim_text: "Срок оплаты по договору №5 — 3 дня", original_quote: "за 3 дня", check_worthy: true
- claim_id: 4, category: "statement", claim_text: "Условия договора разумны", original_quote: "Это разумно", check_worthy: false

ТРЕБУЕМЫЙ ФОРМАТ ОТВЕТА:
{
  "answer_id": "ID ответа, который тебе передали",
  "claims": [
    {
      "claim_id": 1,
      "category": "date",
      "claim_text": "Самодостаточная формулировка",
      "original_quote": "Точная цитата из ответа",
      "check_worthy": true
    }
  ]
}"""


def _norm(s: str) -> str:
    """Нормализация для сравнения цитат: регистр, кавычки, пробелы."""
    return re.sub(r"[^\w%№]+", " ", s.lower()).strip()


class ClaimDecomposerAgent:
    """Агент 2: ответ -> список атомарных проверяемых утверждений."""

    def __init__(self, credentials: str, max_retries: int = 1):
        self.giga = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            model="GigaChat-2"
        )
        self.system_prompt = CLAIM_DECOMPOSER_PROMPT
        self.max_retries = max_retries

    def _call_llm(self, answer_id: str, answer_text: str, question: Optional[str]) -> str:
        user_message = f"ID ответа: {answer_id}\n\n"
        if question:
            user_message += f"Вопрос пользователя: {question}\n\n"
        user_message += f"Ответ для разбора:\n{answer_text}"

        chat_request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_message)
            ]
        )
        response = self.giga.chat.create(chat_request)
        message = response.messages[0]
        if isinstance(message.content, list):
            return message.content[0].text
        return str(message.content)

    def decompose(self, answer_id: str, answer_text: str,
                  question: Optional[str] = None) -> DecomposedAnswer:
        raw_content = "Нет ответа"
        for attempt in range(self.max_retries + 1):
            try:
                raw_content = self._call_llm(answer_id, answer_text, question)
                clean_text = raw_content.replace("```json", "").replace("```", "").strip()
                json_string = extract_first_json(clean_text)  # из ячейки агента 1
                parsed = DecomposedAnswer(**json.loads(json_string))
                return self._verify_quotes(parsed, answer_text)
            except Exception as e:
                print(f"[Claim Decomposer] Ошибка (попытка {attempt + 1}) для ответа {answer_id}: {e}")
                print(f"[Claim Decomposer] Сырой ответ модели:\n---\n{raw_content}\n---")
        return DecomposedAnswer(answer_id=answer_id, claims=[])

    @staticmethod
    def _verify_quotes(result: DecomposedAnswer, answer_text: str) -> DecomposedAnswer:
        """Проверяем, что цитата реально есть в ответе, и убираем дубли утверждений."""
        answer_norm = _norm(answer_text)
        seen, unique = set(), []
        for claim in result.claims:
            key = _norm(claim.claim_text)
            if not key or key in seen:
                continue
            seen.add(key)
            claim.quote_in_answer = _norm(claim.original_quote) in answer_norm
            unique.append(claim)
        for i, claim in enumerate(unique, start=1):
            claim.claim_id = i
        result.claims = unique
        return result