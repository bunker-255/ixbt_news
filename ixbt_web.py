#!/usr/bin/env python3
"""
ixbt_web.py — Веб-интерфейс для ИИ-дайджеста новостей с iXBT.com

Запуск:
    python3 ixbt_web.py
    python3 ixbt_web.py --port 7860 --no-browser
"""

import argparse
import html
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Импортируем движок из ixbt_ai_news.py
import ixbt_ai_news as engine

SCRIPT_DIR = Path(__file__).resolve().parent
SEEN_CACHE_FILE = engine.SEEN_CACHE_FILE

# Глобальное состояние текущей сессии
APP_STATE = {
    "running": False,
    "progress": 0,
    "status_text": "Готов к работе",
    "model_used": "gemma3:4b",
    "results": [],
    "last_updated": None,
    "error": None,
}

STATE_LOCK = threading.Lock()


def load_cached_results():
    """Загружает последний результат из markdown или сохраненного дайджеста, если есть."""
    global APP_STATE

    candidates = [
        SCRIPT_DIR / "ixbt_digest_gemma.md",
        SCRIPT_DIR / f"ixbt_digest_{datetime.now().strftime('%Y-%m-%d')}.md",
    ]
    target_file = None
    for c in candidates:
        if c.exists():
            target_file = c
            break

    if not target_file:
        return

    try:
        text = target_file.read_text(encoding="utf-8")
        items = []
        blocks = re.split(r"### \d+\. ", text)[1:]
        for b in blocks:
            m_title = re.search(r"\[(.*?)\]\((.*?)\)", b)
            m_cat = re.search(r"\*\*Категория:\*\* `(.*?)`", b)
            m_score = re.search(r"\*\*(\d+)/10\*\*", b)
            m_date = re.search(r"\*(.*?)\*", b)
            m_why = re.search(r"> 💡 \*\*Почему это важно:\*\* (.*?)(?=\n\n|\n👉|$)", b, re.DOTALL)

            snippet = ""
            lines = [ln.strip() for ln in b.split("\n") if ln.strip()]
            for ln in lines:
                if not ln.startswith(("#", "**", ">", "👉", "---", "[")) and len(ln) > 20:
                    snippet = ln
                    break

            if m_title:
                items.append({
                    "title": m_title.group(1),
                    "link": m_title.group(2),
                    "category": m_cat.group(1) if m_cat else "Технологии",
                    "score": float(m_score.group(1)) if m_score else 8.0,
                    "pub_date": m_date.group(1) if m_date else "",
                    "summary": snippet,
                    "why_important": m_why.group(1).strip() if m_why else snippet,
                })

        if items:
            with STATE_LOCK:
                APP_STATE["results"] = items
                APP_STATE["status_text"] = f"Загружен последний сохраненный дайджест ({len(items)} тем)."
                APP_STATE["last_updated"] = "Ранее"
    except Exception as e:
        print(f"Ошибка чтения кэша: {e}")


def run_scan_worker(model, top_count, limit, all_feeds, no_cache, raw_mode):
    """Фоновый поток для выполнения ИИ-анализа новостей."""
    global APP_STATE

    with STATE_LOCK:
        APP_STATE["running"] = True
        APP_STATE["progress"] = 5
        APP_STATE["status_text"] = "Подключение к Ollama..."
        APP_STATE["error"] = None

    try:
        ollama_url = engine.DEFAULT_OLLAMA_URL
        installed = engine.get_installed_ollama_models(ollama_url)

        # Разрешение алиасов модели
        resolved_model = model
        if resolved_model and resolved_model.lower() in engine.MODEL_ALIASES:
            resolved_model = engine.MODEL_ALIASES[resolved_model.lower()]

        if not resolved_model:
            resolved_model = engine.pick_optimal_model(installed) or "gemma3:4b"

        with STATE_LOCK:
            APP_STATE["model_used"] = resolved_model
            APP_STATE["progress"] = 15
            APP_STATE["status_text"] = "Загрузка публикаций с iXBT.com (RSS)..."

        # Загрузка RSS
        feed_targets = [engine.RSS_FEEDS["news"]]
        if all_feeds:
            feed_targets.append(engine.RSS_FEEDS["articles"])

        raw_news = []
        for f in feed_targets:
            xml_data = engine.parse_rss_feed(f)
            raw_news.extend(xml_data)

        # Дедупликация
        unique_news = []
        seen_links = set()
        for it in raw_news:
            if it["link"] not in seen_links:
                seen_links.add(it["link"])
                unique_news.append(it)

        with STATE_LOCK:
            APP_STATE["progress"] = 30
            APP_STATE["status_text"] = f"Загружено {len(unique_news)} новостей. Фильтрация..."

        # Кэш прочитанного
        history = set() if no_cache else engine.load_seen_cache()
        fresh_items = [it for it in unique_news if (no_cache or it["guid"] not in history)]

        if not fresh_items:
            with STATE_LOCK:
                APP_STATE["running"] = False
                APP_STATE["progress"] = 100
                APP_STATE["status_text"] = "Все свежие новости уже были прочитаны ранее."
            return

        # Пре-фильтрация
        candidates = []
        skipped_noise = 0
        for it in fresh_items[:limit]:
            if not raw_mode and engine.is_consumer_noise(it["title"]):
                skipped_noise += 1
                continue
            candidates.append(it)

        if not candidates:
            with STATE_LOCK:
                APP_STATE["running"] = False
                APP_STATE["progress"] = 100
                APP_STATE["status_text"] = "Все новости в выборке отфильтрованы как бытовой шум."
            return

        with STATE_LOCK:
            APP_STATE["progress"] = 40
            APP_STATE["status_text"] = f"ИИ-анализ {len(candidates)} тем через {resolved_model}..."

        # Батчи
        batch_size = 12 if "gemma" in resolved_model or "spark" in resolved_model else 15
        selected_pool = []
        batches_count = (len(candidates) + batch_size - 1) // batch_size

        for b_idx in range(batches_count):
            st = b_idx * batch_size
            fn = min(st + batch_size, len(candidates))
            batch = candidates[st:fn]

            for i, it in enumerate(batch, 1):
                it["batch_id"] = i

            current_pct = 40 + int((b_idx / batches_count) * 50)
            with STATE_LOCK:
                APP_STATE["progress"] = current_pct
                APP_STATE["status_text"] = f"ИИ-анализ пачки {b_idx + 1}/{batches_count} ({len(batch)} тем) через {resolved_model}..."

            try:
                batch_found = engine.evaluate_batch_with_ai(
                    batch,
                    model=resolved_model,
                    ollama_url=ollama_url,
                )
                selected_pool.extend(batch_found)
            except Exception as e:
                print(f"Ошибка пачки: {e}", file=sys.stderr)

        # Сортировка по баллу
        selected_pool.sort(key=lambda x: x["score"], reverse=True)
        top_news = selected_pool[:top_count]

        # Обновление кэша
        for it in fresh_items:
            history.add(it["guid"])
        if not no_cache:
            engine.save_seen_cache(history)

        # Сохранение в Markdown автоматически
        today_str = datetime.now().strftime("%Y-%m-%d")
        engine.write_markdown_file(top_news, resolved_model, SCRIPT_DIR / f"ixbt_digest_{today_str}.md")

        with STATE_LOCK:
            APP_STATE["results"] = top_news
            APP_STATE["running"] = False
            APP_STATE["progress"] = 100
            APP_STATE["status_text"] = f"Готово! Отобрано {len(top_news)} лучших открытий и технологий."
            APP_STATE["last_updated"] = datetime.now().strftime("%H:%M:%S")

    except Exception as e:
        with STATE_LOCK:
            APP_STATE["running"] = False
            APP_STATE["error"] = str(e)
            APP_STATE["status_text"] = f"Ошибка: {e}"


# --- HTML/CSS/JS шаблон страницы -------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="ru" class="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>iXBT AI Digest — Новости науки и технологий</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(23, 32, 51, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --accent: #38bdf8;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(56, 189, 248, 0.12) 0px, transparent 50%);
      background-attachment: fixed;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      color: var(--text);
      min-height: 100vh;
      line-height: 1.6;
    }

    .container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 2rem 1.25rem 5rem;
    }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--card-border);
      flex-wrap: wrap;
      gap: 1rem;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 1rem;
    }

    .logo-icon {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      background: linear-gradient(135deg, #6366f1, #38bdf8);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.5rem;
      box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
    }

    .brand h1 {
      font-size: 1.5rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(to right, #fff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand p {
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .status-pill {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.4rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 500;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.25);
      color: #34d399;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 10px #10b981;
    }

    /* Control Panel */
    .panel {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }

    .controls-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1.25rem;
      margin-bottom: 1.5rem;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    .form-group label {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    select, input[type="number"] {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: var(--text);
      padding: 0.65rem 0.9rem;
      border-radius: 10px;
      font-size: 0.9rem;
      font-family: inherit;
      outline: none;
      transition: all 0.2s ease;
    }

    select:focus, input[type="number"]:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
    }

    .checkboxes {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      align-items: center;
      padding-top: 0.5rem;
    }

    .checkbox-label {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.85rem;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
    }

    .checkbox-label input {
      accent-color: var(--primary);
      width: 16px;
      height: 16px;
    }

    .actions-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
      padding-top: 1rem;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    .btn-primary {
      background: linear-gradient(135deg, #6366f1, #3b82f6);
      color: #fff;
      font-weight: 600;
      font-size: 0.95rem;
      padding: 0.75rem 1.6rem;
      border-radius: 12px;
      border: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.6rem;
      box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
      transition: all 0.2s ease;
    }

    .btn-primary:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 12px 30px rgba(99, 102, 241, 0.55);
    }

    .btn-primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      color: var(--text);
      font-weight: 500;
      font-size: 0.85rem;
      padding: 0.65rem 1.1rem;
      border-radius: 10px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.2s;
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
    }

    .secondary-btns {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    /* Progress bar */
    .progress-box {
      margin-top: 1.25rem;
      display: none;
    }

    .progress-track {
      width: 100%;
      height: 6px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 9999px;
      overflow: hidden;
      margin-bottom: 0.5rem;
    }

    .progress-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #6366f1, #38bdf8);
      border-radius: 9999px;
      transition: width 0.4s ease;
    }

    .progress-info {
      display: flex;
      justify-content: space-between;
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    /* Filter pills */
    .filters-bar {
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1.5rem;
      overflow-x: auto;
      padding-bottom: 0.5rem;
    }

    .filter-btn {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: var(--text-muted);
      padding: 0.45rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      white-space: nowrap;
      transition: all 0.2s;
    }

    .filter-btn.active, .filter-btn:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(99, 102, 241, 0.4);
      color: #a5b4fc;
    }

    /* News Cards */
    .cards-list {
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }

    .card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.5rem;
      transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
      position: relative;
    }

    .card:hover {
      transform: translateY(-2px);
      border-color: rgba(99, 102, 241, 0.35);
      box-shadow: 0 15px 35px rgba(0, 0, 0, 0.35);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.8rem;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .badge-group {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .cat-badge {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
      background: rgba(56, 189, 248, 0.12);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.25);
    }

    .score-badge {
      font-size: 0.8rem;
      font-weight: 700;
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }

    .score-breakthrough {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .score-high {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .pub-date {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 0.8rem;
      line-height: 1.4;
    }

    .card-title a {
      color: var(--text);
      text-decoration: none;
      transition: color 0.2s;
    }

    .card-title a:hover {
      color: var(--accent);
    }

    .card-snippet {
      font-size: 0.9rem;
      color: #cbd5e1;
      margin-bottom: 1rem;
      line-height: 1.5;
    }

    /* AI Insight Box */
    .ai-insight {
      background: rgba(99, 102, 241, 0.08);
      border-left: 3px solid var(--primary);
      border-radius: 0 10px 10px 0;
      padding: 0.85rem 1rem;
      margin-bottom: 1rem;
    }

    .ai-insight-title {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.8rem;
      font-weight: 700;
      color: #a5b4fc;
      margin-bottom: 0.35rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .ai-insight-text {
      font-size: 0.88rem;
      color: #e2e8f0;
      line-height: 1.5;
    }

    .card-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 0.75rem;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .read-link {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent);
      text-decoration: none;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    .read-link:hover {
      text-decoration: underline;
    }

    .copy-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      cursor: pointer;
      font-size: 0.8rem;
      display: flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.3rem 0.5rem;
      border-radius: 6px;
      transition: all 0.2s;
    }

    .copy-btn:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.08);
    }

    /* Empty state */
    .empty-state {
      text-align: center;
      padding: 4rem 1rem;
      color: var(--text-muted);
    }

    .empty-icon {
      font-size: 3rem;
      margin-bottom: 1rem;
    }

    /* Toast */
    .toast {
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      background: #1e293b;
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: #fff;
      padding: 0.75rem 1.25rem;
      border-radius: 10px;
      font-size: 0.85rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
      display: none;
      z-index: 100;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="logo-icon">🚀</div>
        <div>
          <h1>iXBT Tech & Science AI Digest</h1>
          <p>Интеллектуальный радар прорывных технологий и научных открытий</p>
        </div>
      </div>
      <div id="statusBadge" class="status-pill">
        <div class="status-dot"></div>
        <span id="ollamaStatusText">Ollama подключена</span>
      </div>
    </header>

    <div class="panel">
      <div class="controls-grid">
        <div class="form-group">
          <label>Модель ИИ (Ollama)</label>
          <select id="modelSelect">
            <option value="gemma3:4b">gemma3:4b (Рекомендуется)</option>
            <option value="spark">spark-x2.5-heretic:4b</option>
            <option value="qwen2.5:3b">qwen2.5:3b (Быстрая)</option>
            <option value="qwen3:4b">qwen3:4b</option>
            <option value="llama3.2:latest">llama3.2:latest</option>
            <option value="qwen2.5:1.5b">qwen2.5:1.5b (Ультрабыстрая)</option>
          </select>
        </div>

        <div class="form-group">
          <label>Количество лучших новостей (Топ)</label>
          <select id="topSelect">
            <option value="3">Топ 3</option>
            <option value="5" selected>Топ 5 (Оптимально)</option>
            <option value="8">Топ 8</option>
            <option value="12">Топ 12</option>
          </select>
        </div>

        <div class="form-group">
          <label>Глубина проверки (свежих тем)</label>
          <select id="limitSelect">
            <option value="15">15 новостей (Быстро, ~30с)</option>
            <option value="30" selected>30 новостей (Стандарт, ~1.5м)</option>
            <option value="50">50 новостей (Все последние)</option>
          </select>
        </div>
      </div>

      <div class="checkboxes">
        <label class="checkbox-label">
          <input type="checkbox" id="allFeedsCheck"> Включать лонгриды и обзоры iXBT
        </label>
        <label class="checkbox-label">
          <input type="checkbox" id="noiseFilterCheck" checked> Отсекать потребительский шум (скидки, чехлы, авто)
        </label>
        <label class="checkbox-label">
          <input type="checkbox" id="noCacheCheck"> Перепроверить всё (игнорировать историю)
        </label>
      </div>

      <div class="progress-box" id="progressBox">
        <div class="progress-track">
          <div class="progress-fill" id="progressFill"></div>
        </div>
        <div class="progress-info">
          <span id="progressText">Подготовка...</span>
          <span id="progressPct">0%</span>
        </div>
      </div>

      <div class="actions-bar">
        <button class="btn-primary" id="startBtn" onclick="startScan()">
          <span>⚡</span> <span id="btnLabel">Запустить ИИ-анализ</span>
        </button>

        <div class="secondary-btns">
          <button class="btn-secondary" onclick="exportMarkdown()">
            <span>💾</span> Скачать Markdown
          </button>
          <button class="btn-secondary" onclick="openMdView()">
            <span>🌐</span> Открыть в браузере (HTML)
          </button>
        </div>
      </div>
    </div>

    <!-- Category filter bar -->
    <div class="filters-bar" id="filtersBar">
      <button class="filter-btn active" onclick="filterByCategory('all')">Все категории</button>
    </div>

    <!-- News cards container -->
    <div class="cards-list" id="cardsList">
      <div class="empty-state">
        <div class="empty-icon">🛰️</div>
        <h3>Дайджест пока не сформирован</h3>
        <p>Нажмите кнопку «Запустить ИИ-анализ», чтобы найти самые интересные открытия на iXBT.com</p>
      </div>
    </div>
  </div>

  <div class="toast" id="toast">Текст скопирован!</div>

  <script>
    let pollInterval = null;
    let currentResults = [];
    let activeFilter = 'all';

    // Загрузка моделей и статуса при старте
    window.onload = async () => {
      await loadModels();
      await checkStatus();
    };

    async function loadModels() {
      try {
        const res = await fetch('/api/models');
        const data = await res.json();
        const select = document.getElementById('modelSelect');
        if (data.models && data.models.length > 0) {
          select.innerHTML = '';
          data.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            let label = m;
            if (m.includes('gemma3:4b')) label += ' (Рекомендуется)';
            if (m.includes('spark')) label += ' (Thinking-модель)';
            if (m.includes('qwen2.5:3b')) label += ' (Сбалансированная)';
            if (m.includes('1.5b') || m.includes('1.7b')) label += ' (Быстрая)';
            opt.textContent = label;
            if (m === 'gemma3:4b' || m.includes('gemma3')) opt.selected = true;
            select.appendChild(opt);
          });
        }
      } catch (e) {
        console.error('Ошибка загрузки моделей:', e);
      }
    }

    async function startScan() {
      const model = document.getElementById('modelSelect').value;
      const top = parseInt(document.getElementById('topSelect').value, 10);
      const limit = parseInt(document.getElementById('limitSelect').value, 10);
      const allFeeds = document.getElementById('allFeedsCheck').checked;
      const raw = !document.getElementById('noiseFilterCheck').checked;
      const noCache = document.getElementById('noCacheCheck').checked;

      const btn = document.getElementById('startBtn');
      btn.disabled = true;
      document.getElementById('progressBox').style.display = 'block';

      try {
        await fetch('/api/scan', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ model, top, limit, all_feeds: allFeeds, raw, no_cache: noCache })
        });

        if (pollInterval) clearInterval(pollInterval);
        pollInterval = setInterval(checkStatus, 1500);
      } catch (e) {
        showToast('Ошибка запуска: ' + e);
        btn.disabled = false;
      }
    }

    async function checkStatus() {
      try {
        const res = await fetch('/api/status');
        const state = await res.json();

        const btn = document.getElementById('startBtn');
        const progressBox = document.getElementById('progressBox');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const progressPct = document.getElementById('progressPct');

        if (state.running) {
          btn.disabled = true;
          document.getElementById('btnLabel').textContent = 'ИИ анализирует...';
          progressBox.style.display = 'block';
          progressFill.style.width = state.progress + '%';
          progressPct.textContent = state.progress + '%';
          progressText.textContent = state.status_text;
        } else {
          btn.disabled = false;
          document.getElementById('btnLabel').textContent = 'Запустить ИИ-анализ';
          if (state.progress === 100) {
            progressFill.style.width = '100%';
            progressPct.textContent = '100%';
            progressText.textContent = state.status_text;
            setTimeout(() => { progressBox.style.display = 'none'; }, 3000);
          }
          if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
          }
        }

        if (state.results && state.results.length > 0) {
          currentResults = state.results;
          renderCards(currentResults);
          updateCategoryFilters(currentResults);
        }
      } catch (e) {
        console.error('Ошибка проверки статуса:', e);
      }
    }

    function updateCategoryFilters(items) {
      const filtersBar = document.getElementById('filtersBar');
      const categories = ['all'];
      items.forEach(it => {
        if (it.category && !categories.includes(it.category)) {
          categories.push(it.category);
        }
      });

      filtersBar.innerHTML = '';
      categories.forEach(cat => {
        const btn = document.createElement('button');
        btn.className = 'filter-btn' + (cat === activeFilter ? ' active' : '');
        btn.textContent = cat === 'all' ? `Все публикации (${items.length})` : cat;
        btn.onclick = () => filterByCategory(cat);
        filtersBar.appendChild(btn);
      });
    }

    function filterByCategory(cat) {
      activeFilter = cat;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');

      if (cat === 'all') {
        renderCards(currentResults);
      } else {
        renderCards(currentResults.filter(it => it.category === cat));
      }
    }

    function renderCards(items) {
      const container = document.getElementById('cardsList');
      if (!items || items.length === 0) {
        container.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <h3>Нет публикаций в этой категории</h3>
          </div>`;
        return;
      }

      container.innerHTML = items.map((it, idx) => {
        const isTop = it.score >= 9;
        const scoreClass = isTop ? 'score-breakthrough' : 'score-high';
        const scoreLabel = isTop ? '★ ' + Math.round(it.score) + '/10 Прорыв!' : '★ ' + Math.round(it.score) + '/10';

        return `
          <div class="card">
            <div class="card-header">
              <div class="badge-group">
                <span class="cat-badge">${escapeHtml(it.category || 'Технологии')}</span>
                <span class="score-badge ${scoreClass}">${scoreLabel}</span>
              </div>
              <span class="pub-date">${escapeHtml(it.pub_date || '')}</span>
            </div>

            <h3 class="card-title">
              <a href="${escapeHtml(it.link)}" target="_blank" rel="noopener">${idx + 1}. ${escapeHtml(it.title)}</a>
            </h3>

            ${it.summary ? `<p class="card-snippet">${escapeHtml(it.summary)}</p>` : ''}

            <div class="ai-insight">
              <div class="ai-insight-title">💡 В чём значимость открытия (ИИ-анализ)</div>
              <div class="ai-insight-text">${escapeHtml(it.why_important || it.summary || '')}</div>
            </div>

            <div class="card-footer">
              <a href="${escapeHtml(it.link)}" target="_blank" rel="noopener" class="read-link">
                Читать на iXBT.com ↗
              </a>
              <button class="copy-btn" onclick="copyCardText(${idx})">
                📋 Копировать текст
              </button>
            </div>
          </div>
        `;
      }).join('');
    }

    function copyCardText(idx) {
      const it = currentResults[idx];
      if (!it) return;
      const text = `${it.title}\\nОценка: ${it.score}/10 [${it.category}]\\n${it.why_important}\\nСсылка: ${it.link}`;
      navigator.clipboard.writeText(text);
      showToast('Скопировано в буфер обмена!');
    }

    async function exportMarkdown() {
      try {
        const res = await fetch('/api/export_markdown', { method: 'POST' });
        const data = await res.json();
        showToast('Отчет сохранен: ' + data.path);
      } catch (e) {
        showToast('Ошибка сохранения: ' + e);
      }
    }

    async function openMdView() {
      try {
        const res = await fetch('/api/preview_html', { method: 'POST' });
        const data = await res.json();
        if (data.url) {
          window.open(data.url, '_blank');
        } else {
          showToast(data.message || 'Сгенерировано');
        }
      } catch (e) {
        showToast('Ошибка mdview: ' + e);
      }
    }

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.style.display = 'block';
      setTimeout(() => { t.style.display = 'none'; }, 3500);
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }
  </script>
</body>
</html>"""


# --- Обработчик HTTP-запросов ----------------------------------------------
class DigestServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Отключаем шумные логи каждого GET
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET(head_only=True)

    def do_GET(self, head_only=False):
        url = urllib.parse.urlparse(self.path)

        if url.path == "/" or url.path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)
            return

        if url.path == "/api/status":
            with STATE_LOCK:
                self.send_json(APP_STATE)
            return

        if url.path == "/api/models":
            try:
                models = engine.get_installed_ollama_models(engine.DEFAULT_OLLAMA_URL)
            except Exception:
                models = []
            self.send_json({"models": models})
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            params = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            params = {}

        if url.path == "/api/scan":
            with STATE_LOCK:
                if APP_STATE["running"]:
                    self.send_json({"error": "Анализ уже запущен"}, status=400)
                    return

            model = params.get("model", "gemma3:4b")
            top_count = params.get("top", 5)
            limit = params.get("limit", 30)
            all_feeds = params.get("all_feeds", False)
            no_cache = params.get("no_cache", False)
            raw_mode = params.get("raw", False)

            thread = threading.Thread(
                target=run_scan_worker,
                args=(model, top_count, limit, all_feeds, no_cache, raw_mode),
                daemon=True,
            )
            thread.start()

            self.send_json({"status": "started"})
            return

        if url.path == "/api/export_markdown":
            with STATE_LOCK:
                res = APP_STATE["results"]
                model_name = APP_STATE["model_used"]

            today_str = datetime.now().strftime("%Y-%m-%d")
            out_file = SCRIPT_DIR / f"ixbt_digest_{today_str}.md"
            engine.write_markdown_file(res, model_name, out_file)
            self.send_json({"status": "ok", "path": str(out_file)})
            return

        if url.path == "/api/preview_html":
            with STATE_LOCK:
                res = APP_STATE["results"]
                model_name = APP_STATE["model_used"]

            today_str = datetime.now().strftime("%Y-%m-%d")
            out_file = SCRIPT_DIR / f"ixbt_digest_{today_str}.md"
            engine.write_markdown_file(res, model_name, out_file)

            mdview_script = SCRIPT_DIR / "mdview.py"
            html_path = None
            if mdview_script.exists():
                try:
                    proc = subprocess.run(
                        ["python3", str(mdview_script), str(out_file)],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    html_path = proc.stdout.strip()
                except Exception as e:
                    print(f"mdview error: {e}")

            self.send_json({
                "status": "ok",
                "message": f"Сгенерирован HTML: {html_path}" if html_path else "Отчет сохранен",
                "html_file": html_path,
            })
            return

        self.send_response(404)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Визуальный веб-интерфейс для iXBT AI Digest")
    parser.add_argument("--port", type=int, default=7860, help="Порт веб-сервера (по умолчанию 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Хост (по умолчанию 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать браузер автоматически")

    args = parser.parse_args()

    # Предзагружаем результаты из последнего сохраненного дайджеста
    load_cached_results()

    server_address = (args.host, args.port)
    try:
        httpd = ThreadingHTTPServer(server_address, DigestServerHandler)
    except OSError as e:
        print(f"Порт {args.port} занят, пробуем 7861... ({e})")
        server_address = (args.host, args.port + 1)
        httpd = ThreadingHTTPServer(server_address, DigestServerHandler)

    actual_port = server_address[1]
    url = f"http://localhost:{actual_port}"

    print(f"\n========================================================")
    print(f" 🚀 iXBT Tech & Science AI Digest — Веб-интерфейс")
    print(f" Адрес: {url}")
    print(f" Нажмите Ctrl+C для остановки сервера")
    print(f"========================================================\n")

    if not args.no_browser:
        # Открываем в браузере в фоновом потоке через секунду
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
        httpd.server_close()


if __name__ == "__main__":
    main()
