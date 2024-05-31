mas = [
    "**Заголовок первого уровня**",
    "*Заголовок второго уровня*",
    "### Заголовок третьего уровня",
    "[Ссылка на сайт](https://www.example.com)",
    "![Изображение](https://www.example.com/image.jpg)",
    "`Код в одной строке`",
    "```",
    "Блок кода",
    "на несколько",
    "строк",
    "```",
    "**Ля ля**",
    "*Статья*",
    "### Домашнее задание",
    "[Ссылка на котиков](https://www.cats.com)",
    "![Изображение](https://www.cats.com/image.jpg)",
    "* Ненумерованный список",
    "  * Пункт 1",
    "  * Пункт 2",
    "1. Нумерованный список",
    "   2. Пункт 1",
    "   3. Пункт 2",
    "> Цитата",
    "**Жирный текст**",
    "*Курсив*",
    "~~Зачеркнутый текст~~",
    "[ ] Пункт списка задач",
    "[x] Выполненный пункт списка задач",
    "---",
    "___",
    "***",
    "* Элемент списка с горизонтальной чертой",
    "---",
    "___",
    "***",
    "1. Нумерованный список с горизонтальной чертой",
    "   ---",
    "   ___",
    "   ***",
    "> Цитата с горизонтальной чертой",
    "---",
    "___",
    "***",
    "| Таблица с горизонтальной чертой | Заголовок 1 | Заголовок 2 |",
    "| ------------- | ----------- | ----------- |",
    "| Строка 1 | Значение 1.1 | Значение 1.2 |",
    "| Строка 2 | Значение 2.1 | Значение 2.2 |",
    "Ссылка без текста: <https://www.example.com>",
    "Ссылка с текстом: [Текст ссылки](https://www.example.com)",
    "Изображение: ![Alt текст](https://www.example.com/image.jpg)",
    "Жирный текст с горизонтальной чертой: **Текст**",
    "Курсив с горизонтальной чертой: *Текст*",
    "Ссылка на электронную почту: example@example.com",
    "Экранирование символов Markdown: \*Текст\*",
    "Исходный код с указанием языка: ```python",
    "print('Hello, world!')",
    "```",
    "Экранирование специальных символов: \\*Звездочка\\*",
    "Зачеркнутый текст: ~~Текст~~",
    "Подчеркнутый текст: <u>Текст</u>",
    "Верхний индекс: x^2^",
    "Нижний индекс: H~2~O",
    "Emoji: :smile:",
    "Github-style emoji: :rocket:",
    "HTML-теги: <div>Текст</div>",
    "Символ новой строки: Два пробела в конце строки",
    "Комментарии не отображаются в рендеринге Markdown <!-- Текст комментария -->",
    "Таблица | Заголовок 1 | Заголовок 2",
    "------- | ------------ | ------------",
    "Строка 1 | Значение 1.1 | Значение 1.2",
    "Строка 2 | Значение 2.1 | Значение 2.2",
    "@Упоминание",
    "<u>Подчеркнутый текст</u>",
    "^Верхний индекс^",
    "~Нижний индекс~",
    "<sup>Верхний индекс</sup>",
    "<sub>Нижний индекс</sub>",
    ":smile: - emoji",
    ":rocket:",
    ":heart:",
    ":star:",
    ":zap:",
    ":mag:",
    ":warning:",
    ":fire:",
    ":sun:",
    ":moon:",
    ":octocat:",
    ":thumbsup:",
    ":thumbsdown:",
    ":pencil:",
    ":bell:",
    ":speech_balloon:",
    ":gift:",
    ":bulb:",
    ":clock1:",
    ":clock2:",
    ":clock3:",
    ":clock4:",
    ":clock5:",
    ":clock6:",
    ":clock7:",
    ":clock8:",
    ":clock9:",
    ":clock10:",
    ":clock11:",
    ":clock12:",
    ":hourglass:",
    ":email:",
    ":link:",
    ":sunny:",
    ":umbrella:",
    ":cloud:",
    ":snowflake:",
    ":star2:",
    ":ocean:",
    ":volcano:",
    ":earth_africa:",
    ":iphone:",
    ":telephone:",
    ":fax:",
    ":video_camera:",
    ":movie_camera:",
    ":tv:",
    ":radio:",
    ":loud_sound:",
    ":mute:",
    ":sound:",
    ":battery:",
    ":electric_plug:",
    ":mag_right:",
    ":bulb:",
    ":scissors:",
    ":paperclip:",
    ":pushpin:",
    ":lock:",
    ":unlock:",
    ":key:",
    ":wrench:",
    ":hammer:",
    ":nut_and_bolt:",
    ":gear:",
    ":warning:",
    ":bookmark:",
    ":fire:",
    ":bomb:",
    ":smoking:",
    ":no_smoking:",
    ":skull:",
    ":radioactive:",
    ":biohazard:",
    ":yin_yang:"
]
import pandas as pd
import numpy as np

final_df_md = pd.DataFrame({
    'text': [np.nan] * len(mas),
    'is_code': [1] * len(mas),
    'language': ['md'] * len(mas),
    'code': mas
})

