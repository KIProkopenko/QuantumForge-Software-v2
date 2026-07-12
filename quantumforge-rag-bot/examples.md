# Примеры диалогов с RAG-ботом

## Успешные диалоги (ответы из базы знаний)

### Пример 1: Вопрос о персонаже

**Вопрос:** Who is Xarn Velgor?

**Ответ бота:**
Let me think step by step:
1. I need to find information about Xarn Velgor in the provided documents.
2. In [Document 1] from Xarn_Velgor.md, it states: "Xarn Velgor is the supreme leader of the Dominion Authority and one of the most powerful Void Masters in known history."
3. The document also mentions that he was originally a Flux Guardian who fell to the Void Path.
4. Therefore, the answer is: Xarn Velgor is the supreme leader of the Dominion Authority, a powerful Void Master who was once a Flux Guardian.

**Источники:**
- Xarn_Velgor.md (score: 0.7830)
- Lord_Malachar.md (score: 0.9252)
- Kael_Torren.md (score: 0.9959)

---

### Пример 2: Вопрос о технологии

**Вопрос:** What is the Void Core?

**Ответ бота:**
Let me think step by step:
1. I need to find information about the Void Core in the provided documents.
2. In [Document 1] from Void_Core.md, it states: "The Void Core is a massive space station and superweapon capable of destroying entire planets."
3. The specifications indicate a diameter of 160 kilometers and a crew of 1.2 million personnel.
4. The weapon is powered by synthetic dark matter through the Void Energy Beam.
5. Therefore, the answer is: The Void Core is a massive space station (160 km diameter) and superweapon built by the Dominion Authority, capable of destroying entire planets using Void Energy Beam powered by synthetic dark matter.

**Источники:**
- Void_Core.md (score: 0.4845)
- Void_Core.md (score: 0.5196)
- Void_Core.md (score: 0.7655)

---

### Пример 3: Вопрос о событии

**Вопрос:** Tell me about Kael Torren.

**Ответ бота:**
Let me think step by step:
1. I need to find information about Kael Torren in the provided documents.
2. In [Document 1] from Kael_Torren.md, it states: "Kael Torren is a Flux-sensitive hero who rose from a humble background on Aridus Prime to become one of the most powerful Flux Guardians in history."
3. The document mentions he is the son of Xarn Velgor and grew up as an orphan.
4. He discovered his Flux-sensitivity under Orin Kestrel's guidance and joined the Freedom Coalition.
5. Therefore, the answer is: Kael Torren is a Flux-sensitive hero and one of the most powerful Flux Guardians. He is the son of Xarn Velgor, grew up as an orphan on Aridus Prime, and became a key figure in the Freedom Coalition.

**Источники:**
- Kael_Torren.md (score: 0.8852)
- Aridus_Prime.md (score: 1.0974)
- Jax_Maverick.md (score: 1.1235)

---

## Случаи, когда бот отвечает "Я не знаю"

### Пример 4: Вопрос вне базы знаний (оригинальные термины Star Wars)

**Вопрос:** Who is Darth Vader?

**Ответ бота:**
Let me think step by step:
1. I need to find information about "Darth Vader" in the provided documents.
2. I reviewed all the documents in the context: Lord_Malachar.md, Xarn_Velgor.md, and Shadow_Drone.md.
3. None of these documents contain information about a character named "Darth Vader".
4. The closest match is Lord Malachar (Emperor of the Dominion Authority) and Xarn Velgor (supreme leader), but they are different characters.
5. Since the requested information is not present in the knowledge base, I must say: **I don't know** who Darth Vader is, as this information is not in my knowledge base.

**Источники:**
- Lord_Malachar.md (score: 1.2862) — низкая релевантность
- Xarn_Velgor.md (score: 1.3561) — низкая релевантность
- Shadow_Drone.md (score: 1.3828) — низкая релевантность

---

### Пример 5: Вопрос вне базы знаний (другая вселенная)

**Вопрос:** What is the Death Star?

**Ответ бота:**
Let me think step by step:
1. I need to find information about "Death Star" in the provided documents.
2. I reviewed all the documents in the context: Star_Runner.md, Jax_Maverick.md.
3. None of these documents contain information about a "Death Star".
4. The documents mention the Star Runner (a modified YT-1300 freighter) and Jax Maverick (a pilot), but these are different concepts.
5. Since the requested information is not present in the knowledge base, I must say: **I don't know** what the Death Star is, as this information is not in my knowledge base.

**Источники:**
- Star_Runner.md (score: 1.1923) — низкая релевантность
- Jax_Maverick.md (score: 1.1953) — низкая релевантность
- Jax_Maverick.md (score: 1.2450) — низкая релевантность

---

## Анализ работы техник промптинга

### Few-shot prompting
В промпт включены 2 примера правильных ответов:
1. "Who leads the Dominion Authority?" → Xarn Velgor
2. "What weapon is capable of destroying planets?" → Void Core

Эти примеры показывают модели:
- Какой формат ответа ожидается
- Как использовать Chain-of-Thought
- Как цитировать источники

### Chain-of-Thought (CoT)
В system-промпте указано: "Always think step by step before answering"

Это заставляет модель:
1. Сначала проанализировать вопрос
2. Найти релевантную информацию в контексте
3. Сформулировать ответ на основе найденных данных
4. Цитировать источники

Результат: ответы становятся более структурированными и обоснованными.