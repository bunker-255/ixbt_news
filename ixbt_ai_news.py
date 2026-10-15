#!/usr/bin/env python3
"""
ixbt_ai_news.py — ИИ-навигатор по самым интересным новостям науки и технологий на iXBT.com

Возможности:
1. Загрузка свежей ленты публикаций iXBT.com (новости и аналитические статьи) через RSS.
2. Интеллектуальный пре-фильтр бытового шума (скидки, чехлы, продажи авто, рядовые смартфоны).
3. Анализ публикаций локальной нейросетью (Ollama) или через внешний OpenAI/Gemini API.
4. Точная классификация по научным направлениям (Космос, Кванты, ИИ, Биотех, Физика, Чипы).
5. Оценка важности (1-10) и генерация краткой выжимки «в чем суть и почему это важно».
6. Красивый цветной вывод в терминал, экспорт в Markdown / JSON.
7. Поддержка открытия HTML-версии через mdview.py и отправки в Telegram.

Запуск:
    python3 ixbt_ai_news.py                     # Стандартный запуск с автовыбором модели
    python3 ixbt_ai_news.py --top 5 --view      # Показать топ-5 и открыть в браузере (HTML)
    python3 ixbt_ai_news.py --save digest.md    # Сохранить в файл Markdown
    python3 ixbt_ai_news.py --model llama3.2    # Использовать конкретную модель Ollama
    python3 ixbt_ai_news.py --all-feeds         # Включать лонгриды и статьи
    python3 ixbt_ai_news.py --no-cache          # Проверить заново без учета истории
"""

import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# --- Конфигурация по умолчанию ---------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "ixbt_config.json"
SEEN_CACHE_FILE = Path.home() / ".cache" / "ixbt_ai_seen.json"

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

RSS_FEEDS = {
    "news": "https://www.ixbt.com/export/news/rss.xml",
    "articles": "https://www.ixbt.com/export/articles/rss.xml",
}

# Оптимальный приоритет моделей
MODEL_PREFERENCES = [
    "gemma3:4b",
    "qwen2.5:3b",
    "qwen3:4b",
    "Maternion/spark-x2.5-heretic:4b",
    "llama3.2:latest",
    "qwen3:1.7b",
    "qwen2.5:1.5b",
    "mistral:7b",
]

MODEL_ALIASES = {
    "gemma": "gemma3:4b",
    "gemma3": "gemma3:4b",
    "gemma3:4b": "gemma3:4b",
    "spark": "Maternion/spark-x2.5-heretic:4b",
    "spark-x2": "Maternion/spark-x2.5-heretic:4b",
    "spark2": "Maternion/spark-x2.5-heretic:4b",
    "qwen": "qwen2.5:3b",
    "qwen3": "qwen3:4b",
    "llama": "llama3.2:latest",
    "llama3": "llama3.2:latest",
    "mistral": "mistral:7b",
}

# Категории для классификации
CATEGORIES = [
    "Космос и Астрономия",
    "Наука и Физика",
    "Биотехнологии и Медицина",
    "ИИ и Нейросети",
    "Квантовые Технологии",
    "Микроэлектроника и Чипы",
    "Робототехника",
    "Энергетика и Материалы",
    "Кибербезопасность",
    "Высокие Технологии",
]


# --- Цветовая палитра терминала (ANSI) -------------------------------------
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BG_HEADER = "\033[44m\033[1m\033[37m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if attr.isupper() and isinstance(getattr(cls, attr), str):
                setattr(cls, attr, "")


if not sys.stdout.isatty():
    Style.disable()


# --- Загрузка настроек из config.json ---------------------------------------
def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"{Style.YELLOW}[Предупреждение] Ошибка чтения config.json: {e}{Style.RESET}", file=sys.stderr)
    return {}


# --- Сетевые запросы и парсинг RSS -----------------------------------------
def fetch_url(url, headers=None, data=None, timeout=30):
    req_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers)
    return urllib.request.urlopen(req, timeout=timeout)


def clean_text(raw_html):
    """Очищает HTML-теги, экранированные символы и лишние пробелы."""
    if not raw_html:
        return ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_rss_feed(feed_url):
    """Скачивает и парсит ленту RSS."""
    import xml.etree.ElementTree as ET

    try:
        with fetch_url(feed_url, timeout=25) as resp:
            content = resp.read()
    except Exception as e:
        print(f"{Style.RED}[Ошибка] Не удалось скачать ленту {feed_url}: {e}{Style.RESET}", file=sys.stderr)
        return []

    items = []
    try:
        root = ET.fromstring(content)
    except Exception as e:
        print(f"{Style.RED}[Ошибка] Ошибка XML в ленте {feed_url}: {e}{Style.RESET}", file=sys.stderr)
        return []

    for el in root.findall(".//item"):
        guid = el.findtext("guid") or ""
        title = clean_text(el.findtext("title") or "")
        link = clean_text(el.findtext("link") or "")
        pub_date = el.findtext("pubDate") or ""
        desc = clean_text(el.findtext("description") or "")

        content_encoded = ""
        for child in el:
            if child.tag.endswith("encoded"):
                content_encoded = clean_text(child.text or "")
                break

        snippet = content_encoded if len(content_encoded) > len(desc) else desc
        if not guid:
            guid = link or title

        if title:
            items.append({
                "guid": guid,
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "description": desc,
                "snippet": snippet[:260],
            })
    return items


# --- Фильтрация бытового шума ----------------------------------------------
# Паттерны отсева потребительского спама: скидки, чехлы, продажи автомобилей,
# банальные апдейты прошивок смартфонов
NOISE_PATTERNS = [
    r"\bскидк[а-я]*\b",
    r"\bраспродаж[а-я]*\b",
    r"\bподешевел[а-я]*\b",
    r"\bкупить со скидкой\b",
    r"\b(чехол|чехлы|бампер|стекло для|аксессуар)[а-я]*\b",
    r"\b(кроссовер|внедорожник|седан|пикап)\b",
    r"\bв россии начали продавать (автомобиль|машину|кроссовер|китайский)\b",
    r"\b(цены в рублях|комплектации и цены)\b",
    r"\bполучит увеличенный экран\b",
    r"\bобновление получили самые разные модели\b",
    r"\b(распродано|раскупили за \d+ минут)\b",
    r"\bназваны цены на смартфоны\b",
    r"\bпромокод[а-я]*\b",
    r"\b(розыгрыш|подарок при покупке)\b",
]

# Ключевые слова, которые ГАРАНТИРУЮТ пропуск в ИИ, даже если есть совпадения
SCIENCE_WHITELIST = [
    r"\b(наук[а-я]|учен[ыео][а-я]*|исследован[а-я]*|открыт[а-я]*)\b",
    r"\b(космос[а-я]*|nasa|роскосмос|телескоп[а-я]*|титан|сатурн|марс[а-я]*|лун[а-я]|орбит[а-я]*)\b",
    r"\b(квант[а-я]*|физик[а-я]*|гравитац[а-я]*|материал[а-я]*)\b",
    r"\b(нейро[а-я]*|искусственн[а-я]* интеллект|ии-черв[а-я]*|anthropic|openai)\b",
    r"\b(микроэлектроник[а-я]*|процессор[а-я]*|чип[а-я]*|полупроводник[а-я]*|asml|tsmc)\b",
    r"\b(биотех[а-я]*|днк|ген[а-я]*|клет[а-я]*|neuralink|митохондр[а-я]*)\b",
    r"\b(робот[а-я]*|нанотруб[а-я]*|аккумулятор[а-я]* будущего)\b",
]


def is_consumer_noise(title):
    """Определяет, является ли новость потребительской рутиной/шумом."""
    title_l = title.lower()

    # Если есть важные маркеры науки или прорыва — не отсеиваем
    for wp in SCIENCE_WHITELIST:
        if re.search(wp, title_l):
            return False

    # Проверяем на маркеры бытового шума
    for np in NOISE_PATTERNS:
        if re.search(np, title_l):
            return True

    return False


# --- Кэш истории просмотров ------------------------------------------------
def load_seen_cache():
    if not SEEN_CACHE_FILE.exists():
        return set()
    try:
        with open(SEEN_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("seen_guids", []))
    except Exception:
        return set()


def save_seen_cache(guids):
    try:
        SEEN_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        trimmed = list(guids)[-2500:]
        with open(SEEN_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "seen_guids": trimmed,
                "updated_at": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"{Style.DIM}Ошибка сохранения кэша: {e}{Style.RESET}", file=sys.stderr)


# --- Модели и работа с ИИ --------------------------------------------------
def get_installed_ollama_models(ollama_url):
    """Запрашивает список установленных моделей в Ollama."""
    try:
        with fetch_url(f"{ollama_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def pick_optimal_model(installed):
    """Выбирает наиболее подходящую модель для комфортной скорости."""
    if not installed:
        return None
    for pref in MODEL_PREFERENCES:
        for m in installed:
            if m == pref or m.startswith(f"{pref}:") or pref in m:
                return m
    return installed[0]


def parse_llm_json(raw_text):
    """Безопасно извлекает JSON из ответа нейросети."""
    if not raw_text:
        return None
    text = raw_text.strip()
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if blocks:
            text = blocks[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def execute_ai_query(prompt, model, ollama_url, api_url=None, api_key=None, timeout=120):
    """Выполняет запрос к локальной Ollama или OpenAI-совместимому API."""
    system_text = (
        "Ты авторитетный научный редактор и эксперт по передовым технологиям. "
        "Твоя цель — находить только фундаментальные открытия, космические миссии, "
        "достижения в физике, биотехнологиях, ИИ, квантовых вычислениях и микроэлектронике. "
        "Отсекай рутину, потребительский шум и мелкие апдейты. "
        "Всегда отвечай СТРОГО в формате валидного JSON."
    )

    if api_url and api_key:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        req = urllib.request.Request(api_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            choices = data.get("choices", [])
            return choices[0].get("message", {}).get("content", "") if choices else ""
    else:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(f"{ollama_url}/api/chat", data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data.get("message", {}).get("content", "")


def evaluate_batch_with_ai(batch, model, ollama_url, api_url=None, api_key=None):
    """Анализирует пачку новостей и возвращает структурированный список."""
    lines = []
    for it in batch:
        snip = f" — {it['snippet']}" if it["snippet"] else ""
        lines.append(f"[{it['batch_id']}] {it['title']}{snip}")

    batch_str = "\n".join(lines)

    prompt = f"""Ниже представлен список заголовков и фрагментов свежих новостей:

{batch_str}

ИНСТРУКЦИЯ:
1. Выбери из списка ВСЕ новости, представляющие реальный интерес для науки, передовых технологий, космоса, ИИ, микроэлектроники, квантовых технологий, физики, биотеха или инновационной инженерии.
2. Не пропускай важные темы! Ожидается от 2 до 5 отобранных новостей.
3. НЕ выбирай новости о бытовых потребительских смартфонах (экраны, аккумуляторы, продажи моделей), автомобильных ценах или чехлах.
4. Для КАЖДОЙ подходящей новости укажи:
   - "id": точный номер [N] новости из списка
   - "score": оценка значимости от 1 до 10 (где 9-10 — крупное научное открытие или технологический прорыв)
   - "category": точная категория (например: "Космос и Астрономия", "Наука и Физика", "Биотехнологии и Медицина", "ИИ и Нейросети", "Микроэлектроника и Чипы", "Квантовые Технологии", "Кибербезопасность", "Энергетика и Материалы")
   - "why_important": 1-2 предложения, почему именно эта конкретная новость важна и представляет интерес.

ФОРМАТ JSON:
{{
  "selected": [
    {{
      "id": 1,
      "score": 8,
      "category": "ИИ и Нейросети",
      "why_important": "Причина важности новости 1..."
    }}
  ]
}}"""

    raw = execute_ai_query(prompt, model, ollama_url, api_url=api_url, api_key=api_key, timeout=180)
    parsed = parse_llm_json(raw)
    if not parsed:
        return []

    items = []
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        for k in ("selected", "news", "items", "results"):
            if k in parsed and isinstance(parsed[k], list):
                items = parsed[k]
                break
        if not items and "id" in parsed:
            items = [parsed]

    id_map = {it["batch_id"]: it for it in batch}
    results = []

    for entry in items:
        try:
            bid = int(entry.get("id", -1))
        except (ValueError, TypeError):
            continue

        orig = id_map.get(bid)
        if not orig:
            continue

        try:
            score = float(entry.get("score", 6))
        except (ValueError, TypeError):
            score = 6.0

        cat = entry.get("category", "Высокие Технологии")
        why = entry.get("why_important") or entry.get("summary") or orig["snippet"]

        results.append({
            "title": orig["title"],
            "link": orig["link"],
            "pub_date": orig["pub_date"],
            "score": score,
            "category": cat,
            "summary": orig["snippet"],
            "why_important": why,
            "guid": orig["guid"],
        })

    return results


# --- Вывод результатов -----------------------------------------------------
def render_terminal(selected, model_name):
    """Выводит форматированный цветной дайджест в консоль."""
    print()
    print(f"{Style.BG_HEADER}   🚀 iXBT TECH & SCIENCE AI DIGEST: Главные открытия и технологии   {Style.RESET}")
    print(f"{Style.DIM}Модель: {model_name}  |  Время: {datetime.now().strftime('%d.%m.%Y %H:%M')}{Style.RESET}")
    print("━" * 76)

    if not selected:
        print(f"\n{Style.YELLOW}В обработанных новостях не обнаружено прорывных научных событий.{Style.RESET}")
        print("Попробуйте запустить позже или с ключом --raw (без фильтра шума).\n")
        return

    for i, it in enumerate(selected, 1):
        score = it["score"]
        if score >= 9:
            score_str = f"{Style.GREEN}★ {score:.0f}/10 (Прорыв!){Style.RESET}"
        elif score >= 7.5:
            score_str = f"{Style.CYAN}★ {score:.0f}/10{Style.RESET}"
        else:
            score_str = f"{Style.YELLOW}★ {score:.0f}/10{Style.RESET}"

        cat_badge = f"{Style.MAGENTA}[{it['category']}]{Style.RESET}"

        print(f"\n{Style.BOLD}{i}. {it['title']}{Style.RESET}")
        print(f"   {cat_badge}  {score_str}  {Style.DIM}• {it['pub_date']}{Style.RESET}")
        if it.get("summary") and it["summary"] != it.get("why_important"):
            print(f"   {Style.DIM}📰 Факты:{Style.RESET} {it['summary']}")
        print(f"   {Style.YELLOW}💡 Значимость:{Style.RESET} {it.get('why_important', '')}")
        print(f"   {Style.BLUE}🔗 Ссылка:{Style.RESET} {it['link']}")

    print("\n" + "━" * 76 + "\n")


def write_markdown_file(selected, model_name, path=None):
    """Формирует и сохраняет файл Markdown."""
    if not path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = f"ixbt_digest_{date_str}.md"

    md = [
        "# 🚀 iXBT.com — Главные открытия науки и высоких технологий",
        "",
        f"> **Сформировано:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"> **ИИ-аналитик:** `{model_name}`  ",
        f"> **Источник данных:** [iXBT.com](https://www.ixbt.com/)",
        "",
        "---",
        "",
    ]

    for i, it in enumerate(selected, 1):
        md.append(f"### {i}. [{it['title']}]({it['link']})")
        md.append(f"**Категория:** `{it['category']}` | **Оценка интересности:** **{it['score']:.0f}/10** | *{it['pub_date']}*")
        md.append("")
        if it.get("summary") and it["summary"] != it.get("why_important"):
            md.append(f"{it['summary']}")
            md.append("")
        md.append(f"> 💡 **Почему это важно:** {it.get('why_important', '')}")
        md.append("")
        md.append(f"👉 [Читать полную статью на iXBT.com]({it['link']})")
        md.append("")
        md.append("---")
        md.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"{Style.GREEN}✓{Style.RESET} Markdown-отчет записан: {Style.BOLD}{path}{Style.RESET}")
    return path


def preview_with_mdview(md_path):
    """Использует локальный mdview.py для генерации HTML."""
    mdview_script = SCRIPT_DIR / "mdview.py"
    if mdview_script.exists():
        try:
            print(f"{Style.CYAN}Рендеринг HTML через mdview.py...{Style.RESET}")
            res = subprocess.run(
                ["python3", str(mdview_script), md_path],
                capture_output=True,
                text=True,
                check=True
            )
            print(res.stdout.strip())
        except Exception as e:
            print(f"{Style.YELLOW}Ошибка при вызове mdview.py: {e}{Style.RESET}")
    else:
        print(f"{Style.DIM}Скрипт mdview.py не найден в папке.{Style.RESET}")


def send_to_telegram(selected, bot_token, chat_id):
    """Отправляет дайджест в Telegram-канал или чат."""
    if not bot_token or not chat_id:
        return

    text = f"🚀 *iXBT: Топ новостей науки и технологий*\n"
    text += f"_{datetime.now().strftime('%d.%m.%Y %H:%M')}_\n\n"

    for i, it in enumerate(selected, 1):
        clean_title = it['title'].replace("*", "").replace("_", "")
        clean_summary = it['summary'].replace("*", "").replace("_", "")
        text += f"*{i}. {clean_title}*\n"
        text += f"🏷 `[{it['category']}]` • ⭐ *{it['score']:.0f}/10*\n"
        text += f"{clean_summary}\n"
        text += f"[Читать новость]({it['link']})\n\n"

    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(api_url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"{Style.GREEN}✓{Style.RESET} Дайджест успешно отправлен в Telegram!")
    except Exception as e:
        print(f"{Style.RED}[Ошибка] Не удалось отправить в Telegram: {e}{Style.RESET}")


# --- Главная функция -------------------------------------------------------
def main():
    config = load_config()

    parser = argparse.ArgumentParser(
        description="ИИ-скрипт для поиска самых интересных новостей науки и технологий на iXBT.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=config.get("top_count", 5),
        help="Сколько лучших новостей отобрать (по умолчанию: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.get("fetch_limit", 40),
        help="Сколько свежих новостей скачать с iXBT (по умолчанию: 40)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.get("batch_size", 15),
        help="Размер пакета для передачи в ИИ (по умолчанию: 15)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.get("model", None),
        help="Имя модели Ollama (если не указано, выбирается автоматически лучшая)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=config.get("ollama_url", DEFAULT_OLLAMA_URL),
        help=f"URL сервиса Ollama (по умолчанию: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--all-feeds",
        action="store_true",
        default=config.get("all_feeds", False),
        help="Загружать также аналитические статьи и лонгриды",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Отключить эвристический пре-фильтр бытового шума",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Игнорировать историю прочитанного и перепроверить все новости",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        default=None,
        help="Сохранить результат в файл Markdown (можно передать имя файла)",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Сохранить Markdown и сгенерировать HTML-превью через mdview.py",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="",
        default=None,
        help="Вывести или сохранить структурированный JSON",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Отправить дайджест в Telegram (требуются токен и chat_id в конфиге или env)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=config.get("api_url", None),
        help="URL внешнего OpenAI-совместимого API (например: https://api.openai.com/v1/chat/completions)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=config.get("api_key", os.getenv("OPENAI_API_KEY")),
        help="Ключ для внешнего API",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Запустить интерактивный визуальный веб-интерфейс в браузере",
    )

    args = parser.parse_args()

    if args.web:
        import ixbt_web
        ixbt_web.main()
        return

    # 1. Проверка доступности ИИ-движка
    model = args.model
    if model and model.lower() in MODEL_ALIASES:
        model = MODEL_ALIASES[model.lower()]

    if not args.api_url:
        print(f"{Style.CYAN}Проверка Ollama ({args.ollama_url})...{Style.RESET}")
        installed = get_installed_ollama_models(args.ollama_url)
        if not installed:
            print(
                f"{Style.RED}[Ошибка] Не удалось связаться с Ollama на {args.ollama_url}.\n"
                f"Запустите сервис: {Style.BOLD}ollama serve{Style.RESET}",
                file=sys.stderr,
            )
            sys.exit(1)

        if not model:
            model = pick_optimal_model(installed)
            print(f"Выбрана модель: {Style.BOLD}{model}{Style.RESET}")
        else:
            if model not in installed:
                print(f"{Style.YELLOW}[Инфо] Модель '{model}' отсутствует в списке. Выполняем попытку...{Style.RESET}")
            else:
                print(f"Используется модель: {Style.BOLD}{model}{Style.RESET}")
    else:
        if not model:
            model = "gpt-4o-mini"
        print(f"Внешний API: {args.api_url} (модель: {model})")

    # 2. Получение RSS-лент
    feed_targets = [RSS_FEEDS["news"]]
    if args.all_feeds:
        feed_targets.append(RSS_FEEDS["articles"])

    print(f"\n{Style.CYAN}📡 Загрузка публикаций с iXBT.com...{Style.RESET}")
    raw_news = []
    for f in feed_targets:
        parsed = parse_rss_feed(f)
        raw_news.extend(parsed)

    # Дедупликация по URL
    unique_items = []
    seen_urls = set()
    for it in raw_news:
        if it["link"] not in seen_urls:
            seen_urls.add(it["link"])
            unique_items.append(it)

    print(f"Получено публикаций: {len(unique_items)}")

    # 3. Фильтрация по кэшу прочитанного
    history = set() if args.no_cache else load_seen_cache()
    fresh_items = [it for it in unique_items if (args.no_cache or it["guid"] not in history)]

    if not fresh_items:
        print(f"\n{Style.YELLOW}Все свежие новости уже были обработаны ранее.{Style.RESET}")
        print("Используйте ключ --no-cache, чтобы переоценить их заново.")
        return

    # 4. Пре-фильтрация бытового шума
    candidates = []
    skipped_noise = 0
    for it in fresh_items[:args.limit]:
        if not args.raw and is_consumer_noise(it["title"]):
            skipped_noise += 1
            continue
        candidates.append(it)

    if skipped_noise > 0:
        print(f"Отсеяно потребительского шума (скидки, авто, рядовые гаджеты): {Style.DIM}{skipped_noise}{Style.RESET}")

    print(f"Отобрано для ИИ-анализа: {Style.BOLD}{len(candidates)}{Style.RESET} профильных публикаций")

    if not candidates:
        print(f"{Style.YELLOW}После фильтрации не осталось тем для анализа. Попробуйте с --raw.{Style.RESET}")
        return

    # 5. Обработка батчами в ИИ
    selected_pool = []
    batches_count = (len(candidates) + args.batch_size - 1) // args.batch_size

    for b_idx in range(batches_count):
        st = b_idx * args.batch_size
        fn = min(st + args.batch_size, len(candidates))
        batch = candidates[st:fn]

        for i, it in enumerate(batch, 1):
            it["batch_id"] = i

        print(
            f"🧠 {Style.CYAN}Анализ пачки {b_idx + 1}/{batches_count} ({len(batch)} тем) через {model}...{Style.RESET} ",
            end="",
            flush=True,
        )
        t0 = time.time()
        try:
            batch_found = evaluate_batch_with_ai(
                batch,
                model=model,
                ollama_url=args.ollama_url,
                api_url=args.api_url,
                api_key=args.api_key,
            )
            dur = time.time() - t0
            print(f"{Style.GREEN}готово ({dur:.1f}с, найдено: {len(batch_found)}){Style.RESET}")
            selected_pool.extend(batch_found)
        except Exception as e:
            print(f"{Style.RED}ошибка: {e}{Style.RESET}")

    # 6. Сортировка по убыванию балла
    selected_pool.sort(key=lambda x: x["score"], reverse=True)
    top_news = selected_pool[:args.top]

    # 7. Вывод в терминал
    render_terminal(top_news, model)

    # 8. Сохранение Markdown и просмотр в HTML
    md_file = None
    if args.view or args.save is not None:
        fname = args.save if args.save else None
        md_file = write_markdown_file(top_news, model, fname)

    if args.view and md_file:
        preview_with_mdview(md_file)

    # 9. Экспорт в JSON
    if args.json is not None:
        json_dump = json.dumps(top_news, ensure_ascii=False, indent=2)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                f.write(json_dump)
            print(f"{Style.GREEN}✓{Style.RESET} JSON экспортирован в: {args.json}")
        else:
            print(json_dump)

    # 10. Отправка в Telegram
    tg_token = config.get("telegram_token") or os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = config.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID")
    if (args.telegram or config.get("auto_telegram")) and tg_token and tg_chat:
        send_to_telegram(top_news, tg_token, tg_chat)

    # 11. Обновление истории
    for it in fresh_items:
        history.add(it["guid"])
    if not args.no_cache:
        save_seen_cache(history)


if __name__ == "__main__":
    main()
